#!/usr/bin/env python3
"""
blimp_sanity.py — Pseudo log-likelihood (PLL) sanity check on BLiMP and BLiMP-supplement

Overview
        Evaluates a masked language model (BERT-style) using pseudo log-likelihood:
        mask one position at a time and sum the log-prob of the original token. The
        script supports Hugging Face BLiMP subsets and an optional BLiMP-supplement
        JSONL corpus with sentence_good/sentence_bad pairs.

Usage
        python blimp_sanity.py --model_path <checkpoint_dir> \
                [--subset SUBJECT_VERB_AGREEMENT] [--max_examples 200] \
                [--device cuda] [--split auto|train|test|validation] \
                [--normalize none|per_token] [--dump 5] \
                [--benchmark blimp|blimp_supplement|both] \
                [--supplement_dir /path/to/jsonl] [--supplement_task task_name]

Arguments (selected)
        --model_path            Path to a directory with config.json + weights (and tokenizer)
        --subset                Single BLiMP subset; if omitted, runs all available
        --max_examples          Cap examples per subset/task for speed
        --normalize             "per_token" divides total PLL by token count (length norm)
        --dump                  Save worst-K examples by gap for diagnostics
        --benchmark             Which benchmark(s) to run: BLiMP, supplement, or both
        --supplement_dir        Folder containing <task>.jsonl files for supplement
        --supplement_task       Only run this supplement task name (filename stem)

Innovations & efficiency
        - Vocab clamp: if tokenizer vocab > model vocab, IDs ≥ model.vocab_size are mapped
            to [UNK] to avoid index errors when tokenizers/models mismatch.
        - Two loading paths: tries AutoModelForMaskedLM; falls back to manual import of the
            saved modeling/config files for custom architectures.
        - Per-token normalization option reports PLL per token for length-robust comparisons.
        - Worst-K example dump shows token-level contributors for quick error analysis.
"""
import argparse, math, os, csv, json
from typing import List, Tuple, Dict, Any, Optional
import torch
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from tqdm import tqdm

def pick_split(dataset_name: str, subset: str, split_arg: str):
    """Pick a valid split for a HF dataset subset.

    If split_arg != "auto", returns it verbatim; else probes validation, test, then train.
    Raises if none are available.
    """
    if split_arg != "auto":
        return split_arg
    for sp in ("validation", "test", "train"):
        try:
            _ = load_dataset(dataset_name, subset, split=sp)
            return sp
        except Exception:
            continue
    raise RuntimeError(f"No split found for {dataset_name}/{subset} (tried validation/test/train)")

def pll_score_with_tokens(model, tok, text: str, device: torch.device, normalize: str = "none"):
    """Compute pseudo log-likelihood for one text and return details.

    Steps: tokenize text; for positions 1..L-2 (skip specials) mask token i; compute
    log softmax and collect the log-probability of the original token. Sum over i.

    Safety: If tokenizer produces IDs not present in model's vocab, clamp them to UNK
    to avoid out-of-range indexing.

    Returns:
        total (float): summed (or per-token) PLL score
        contribs (List[tuple]): per-position contributions (pos, token_id, logp, token_str)
        enc (BatchEncoding): tokenized batch encoding for downstream stats
    """
    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"]
    # Safety: clamp token ids to model vocab size if tokenizer vocab > model vocab
    try:
        vocab_limit = int(getattr(model.config, "vocab_size", input_ids.max().item() + 1))
    except Exception:
        vocab_limit = None
    if vocab_limit is not None:
        unk_id = tok.unk_token_id
        if unk_id is None:
            unk_id = 0
        hi = input_ids.ge(vocab_limit)
        if bool(hi.any().item()):
            input_ids = input_ids.masked_fill(hi, int(unk_id))
            # also update enc to reflect replacements
            enc["input_ids"] = input_ids
    input_ids = input_ids.to(device)
    attn = enc["attention_mask"].to(device)
    L = input_ids.size(1)
    mask_id = tok.mask_token_id
    if mask_id is None:
        raise ValueError("Tokenizer has no mask_token_id")

    contribs: List[Tuple[int, int, float, str]] = []  # (pos, token_id, logp, token_str)
    total = 0.0
    steps = 0
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), enabled=(device.type == "cuda")):
        for i in range(1, L - 1):
            orig = int(input_ids[0, i].item())
            if orig == tok.pad_token_id:
                continue
            masked = input_ids.clone()
            masked[0, i] = mask_id
            out = model(input_ids=masked, attention_mask=attn)
            logits = out.logits[0, i]  # vocab
            logp = torch.log_softmax(logits, dim=-1)[orig]
            val = float(logp.item())
            contribs.append((i, orig, val, tok.convert_ids_to_tokens([orig])[0]))
            total += val
            steps += 1

    if normalize == "per_token" and steps > 0:
        total = total / steps
    return total, contribs, enc

def find_blimp_supplement_dir(explicit: Optional[str] = None) -> Optional[str]:
    """Try to locate the BLiMP supplement JSONL folder.
    Preference order: explicit -> ./blimp_supplement -> ../evaluation-pipeline-2024-fresh/evaluation_data/supplement_filtered
    -> ./evaluation-pipeline-2024-fresh/evaluation_data/supplement_filtered
    """
    """Return a directory path containing BLiMP-supplement JSONL files.

    Accepts an explicit path (dir or file in dir). Otherwise searches common
    local locations near the repository root.
    """
    if explicit:
        if os.path.isdir(explicit):
            return explicit
        # allow file path pointing to a jsonl file; use its dir
        if os.path.isfile(explicit):
            return os.path.dirname(os.path.abspath(explicit))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "blimp_supplement"),
        os.path.normpath(os.path.join(here, "..", "evaluation-pipeline-2024-fresh", "evaluation_data", "supplement_filtered")),
        os.path.join(here, "evaluation-pipeline-2024-fresh", "evaluation_data", "supplement_filtered"),
        os.path.join(os.getcwd(), "blimp_supplement"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None

def get_available_supplement_tasks(supp_dir: str) -> List[str]:
    """List available task names (filename stems) from a supplement directory."""
    tasks: List[str] = []
    try:
        for fn in os.listdir(supp_dir):
            if fn.endswith(".jsonl"):
                tasks.append(os.path.splitext(fn)[0])
    except Exception:
        pass
    return sorted(tasks)

def run_supplement_subset(
    model,
    tok,
    task_name: str,
    jsonl_path: str,
    device: torch.device,
    limit: int,
    normalize: str,
    dump: int = 0,
):
    """Run PLL evaluation on a supplement JSONL file with sentence_good/sentence_bad pairs.

    Returns a dict with fields: task, n, acc, oov_rate, normalize, examples, source
    where acc is the proportion of pairs with PLL(good) > PLL(bad).
    """
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"Supplement file not found: {jsonl_path}")
    rows_raw: List[Dict[str, Any]] = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # must contain sentence_good/sentence_bad
                if "sentence_good" in obj and "sentence_bad" in obj:
                    rows_raw.append(obj)
            except Exception:
                continue
            if limit > 0 and len(rows_raw) >= limit:
                break

    wins = 0
    total = 0
    oov_tokens = 0
    total_tokens = 0
    gaps: List[Tuple[float, Dict[str, Any]]] = []

    for row in tqdm(rows_raw, desc=f"supplement:{task_name}"):
        good = row["sentence_good"]
        bad = row["sentence_bad"]
        s_good, contrib_g, enc_g = pll_score_with_tokens(model, tok, good, device, normalize=normalize)
        s_bad, contrib_b, enc_b = pll_score_with_tokens(model, tok, bad, device, normalize=normalize)
        gap = s_good - s_bad
        wins += int(gap > 0.0)
        total += 1
        # OOV stats
        unk = tok.unk_token_id
        if unk is not None and unk >= 0:
            oov_tokens += int((enc_g["input_ids"] == unk).sum().item() + (enc_b["input_ids"] == unk).sum().item())
        total_tokens += int(enc_g["input_ids"].numel() + enc_b["input_ids"].numel())
        if dump > 0:
            gaps.append((gap, {
                "good": good,
                "bad": bad,
                "gap": gap,
                "s_good": s_good,
                "s_bad": s_bad,
                "good_top_neg_tokens": sorted(contrib_g, key=lambda t: t[2])[:3],
                "bad_top_pos_tokens": sorted(contrib_b, key=lambda t: -t[2])[:3],
            }))

    acc = (wins / max(1, total)) if total > 0 else 0.0
    oov_rate = oov_tokens / max(1, total_tokens)
    worst: List[Dict[str, Any]] = []
    if dump > 0 and gaps:
        worst = [info for _, info in sorted(gaps, key=lambda x: x[0])[:dump]]
    return {
        "task": task_name,
        "n": total,
        "acc": acc,
        "oov_rate": oov_rate,
        "normalize": normalize,
        "examples": worst,
        "source": os.path.basename(jsonl_path),
    }

def ensure_subsets_list(dataset_name: str) -> List[str]:
    """Return BLiMP subset names, falling back to a static list if necessary."""
    try:
        return get_dataset_config_names(dataset_name)
    except Exception:
        # fallback to a static list (common BLiMP subsets)
        return [
            "adjunct_island", "anaphor_agreement", "argument_structure",
            "binding", "control_raising", "determiner_noun_agreement",
            "ellipsis", "filler_gap", "irregular_forms", "npi_licensing",
            "quantifiers", "subject_verb_agreement", "tough_movement",
            "wh_questions", "wh_vs_that_with_gap", "wh_vs_that_no_gap",
            "wh_vs_that_with_gap_long_distance", "wh_vs_that_no_gap_long_distance",
        ]

def run_subset(model, tok, subset: str, split: str, device: torch.device, limit: int, normalize: str, dump: int = 0):
    """Run PLL evaluation on one BLiMP subset split.

    Returns a dict with fields: subset, split, n, acc, oov_rate, normalize, examples.
    """
    ds = load_dataset("blimp", subset, split=split)
    n = max(limit, len(ds)) if limit > 0 else len(ds)
    wins = 0
    total = 0
    oov_tokens = 0
    total_tokens = 0
    gaps: List[Tuple[float, Dict[str, Any]]] = []  # (good-bad gap, info)

    for row in tqdm(ds.select(range(n)), desc=f"{subset}/{split}"):
        good = row["sentence_good"]
        bad = row["sentence_bad"]
        s_good, contrib_g, enc_g = pll_score_with_tokens(model, tok, good, device, normalize=normalize)
        s_bad,  contrib_b, enc_b = pll_score_with_tokens(model, tok, bad,  device, normalize=normalize)
        gap = s_good - s_bad
        wins += int(gap > 0.0)
        total += 1
        # OOV stats
        unk = tok.unk_token_id
        if unk is not None and unk >= 0:
            oov_tokens += int((enc_g["input_ids"] == unk).sum().item() + (enc_b["input_ids"] == unk).sum().item())
        total_tokens += int(enc_g["input_ids"].numel() + enc_b["input_ids"].numel())
        if dump > 0:
            gaps.append((gap, {
                "good": good,
                "bad": bad,
                "gap": gap,
                "s_good": s_good,
                "s_bad": s_bad,
                "good_top_neg_tokens": sorted(contrib_g, key=lambda t: t[2])[:3],
                "bad_top_pos_tokens": sorted(contrib_b, key=lambda t: -t[2])[:3],
            }))

    acc = wins / max(1, total)
    oov_rate = oov_tokens / max(1, total_tokens)
    worst: List[Dict[str, Any]] = []
    if dump > 0 and gaps:
        worst = [info for _, info in sorted(gaps, key=lambda x: x[0])[:dump]]
    return {
        "subset": subset,
        "split": split,
        "n": total,
        "acc": acc,
        "oov_rate": oov_rate,
        "normalize": normalize,
        "examples": worst,
    }

def main():
    """CLI entry point to evaluate a model on BLiMP and/or BLiMP-supplement.

    Example
        python blimp_sanity.py --model_path ./model_vault/model_bl_bert_ltgds_regular \
            --benchmark both --max_examples 200 --normalize per_token --dump 5
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--subset", default=None, help="If unset, run all BLiMP subsets available")
    ap.add_argument("--max_examples", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="auto", help="auto | train | test | validation")
    ap.add_argument("--normalize", default="none", choices=["none", "per_token"])
    ap.add_argument("--dump", type=int, default=5, help="dump worst-K examples per subset")
    ap.add_argument("--save_csv", default="outputs/blimp_pll_report.csv")
    ap.add_argument("--save_json", default="outputs/blimp_pll_examples.jsonl")
    # New: supplement options
    ap.add_argument("--benchmark", default="blimp", choices=["blimp", "blimp_supplement", "both"],
                    help="Which benchmark(s) to run")
    ap.add_argument("--supplement_dir", default=None, help="Path to folder containing BLiMP supplement JSONL files")
    ap.add_argument("--supplement_task", default=None, help="If set, only run this supplement task (filename without .jsonl)")
    ap.add_argument("--supplement_save_csv", default="outputs/blimp_supplement_pll_report.csv")
    ap.add_argument("--supplement_save_json", default="outputs/blimp_supplement_pll_examples.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.supplement_save_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.supplement_save_json), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Prefer fast, but fall back to slow if the tokenizer.json lacks a compatible normalizer
    try:
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] Fast tokenizer load failed: {e}\n[INFO] Falling back to use_fast=False")
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    # Load without accelerate dispatch to avoid meta tensors
    try:
        # Prefer loading state_dict explicitly to keep from_pretrained on eager tensors
        sd = None
        pt_path = os.path.join(args.model_path, "pytorch_model.bin")
        sf_path = os.path.join(args.model_path, "model.safetensors")
        if os.path.exists(pt_path):
            sd = torch.load(pt_path, map_location="cpu")
        elif os.path.exists(sf_path):
            try:
                from safetensors.torch import load_file as load_safetensors
                sd = load_safetensors(sf_path)
            except Exception:
                sd = None
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            device_map=None,
            use_safetensors=False,
            state_dict=sd,
        )
        model = model.to(device).eval()
    except Exception as e_auto:
        print(f"[WARN] AutoModel load failed: {e_auto}\n[INFO] Falling back to manual load from modeling_ltgbert.py")
        # Manual load: import model class from checkpoint and load state_dict
        import importlib.util
        modeling_fp = os.path.join(args.model_path, "modeling_ltgbert.py")
        if not os.path.exists(modeling_fp):
            raise RuntimeError(f"modeling_ltgbert.py not found in {args.model_path}")
        spec = importlib.util.spec_from_file_location("ltg_modeling_local", modeling_fp)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to prepare import spec for modeling_ltgbert.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Load config via AutoConfig (trust_remote_code uses configuration_ltgbert.py if present)
        cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        # Expect class LtgBertForMaskedLM
        if not hasattr(mod, "LtgBertForMaskedLM"):
            raise RuntimeError("LtgBertForMaskedLM not found in modeling_ltgbert.py")
        ModelCls = getattr(mod, "LtgBertForMaskedLM")
        model = ModelCls(cfg)
        # Load state dict
        sd = None
        pt_path = os.path.join(args.model_path, "pytorch_model.bin")
        sf_path = os.path.join(args.model_path, "model.safetensors")
        if os.path.exists(pt_path):
            sd = torch.load(pt_path, map_location="cpu")
        elif os.path.exists(sf_path):
            try:
                from safetensors.torch import load_file as load_safetensors
                sd = load_safetensors(sf_path)
            except Exception as e_st:
                raise RuntimeError(f"Failed to load safetensors: {e_st}")
        else:
            raise RuntimeError("No pytorch_model.bin or model.safetensors found in checkpoint")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[WARN] Missing keys when loading state_dict: {missing[:8]}{'...' if len(missing)>8 else ''}")
        if unexpected:
            print(f"[WARN] Unexpected keys when loading state_dict: {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")
        if hasattr(model, "tie_weights"):
            try:
                model.tie_weights()
            except Exception:
                pass
        model = model.to(device).eval()
    if hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
        except Exception:
            pass

    # Run BLiMP (HF) benchmark if requested
    rows: List[Dict[str, Any]] = []
    if args.benchmark in ("blimp", "both"):
        subsets = [args.subset] if args.subset else ensure_subsets_list("blimp")
        with open(args.save_json, "w") as jf:
            for subset in subsets:
                try:
                    split = pick_split("blimp", subset, args.split)
                except Exception as e:
                    print(f"[WARN] Skipping {subset}: {e}")
                    continue
                res = run_subset(model, tok, subset, split, device, args.max_examples, args.normalize, dump=args.dump)
                rows.append(res)
                for ex in res.get("examples", []):
                    jf.write(json.dumps({"subset": subset, "split": split, **ex}, ensure_ascii=False) + "\n")
        with open("outputs/blimp_pll_summary.jsonl", "w") as sf:
            for r in rows:
                sf.write(json.dumps({
                    "subset": r["subset"],
                    "split": r["split"],
                    "acc": r["acc"],
                    "oov_rate": r["oov_rate"],
                    "n": r["n"]
                }) + "\n")
                print(f"Subset={r['subset']:<35} split={r['split']:<10} n={r['n']:<5} PLL-acc={r['acc']:.4f}  OOV={r['oov_rate']:.2%}")
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["subset", "split", "n", "acc", "oov_rate", "normalize"])
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in w.fieldnames})
        if rows:
            macro = sum(r["acc"] for r in rows) / len(rows)
            print(f"\n[BLiMP Summary] subsets={len(rows)}  macro-PLL-acc={macro:.4f}  saved: {args.save_csv}, {args.save_json}")

    # Run BLiMP supplement benchmark if requested
    rows_sup: List[Dict[str, Any]] = []
    if args.benchmark in ("blimp_supplement", "both"):
        supp_dir = find_blimp_supplement_dir(args.supplement_dir)
        if not supp_dir:
            raise RuntimeError("Could not locate BLiMP supplement folder. Pass --supplement_dir pointing to JSONL files.")
        tasks = [args.supplement_task] if args.supplement_task else get_available_supplement_tasks(supp_dir)
        if not tasks:
            raise RuntimeError(f"No supplement tasks found in {supp_dir}")
        with open(args.supplement_save_json, "w") as jf:
            for task in tasks:
                fpath = os.path.join(supp_dir, f"{task}.jsonl")
                if not os.path.isfile(fpath):
                    print(f"[WARN] Supplement file missing, skipping: {fpath}")
                    continue
                res = run_supplement_subset(model, tok, task, fpath, device, args.max_examples, args.normalize, dump=args.dump)
                rows_sup.append(res)
                for ex in res.get("examples", []):
                    jf.write(json.dumps({"task": task, **ex}, ensure_ascii=False) + "\n")
        # summary + csv
        with open("outputs/blimp_supplement_pll_summary.jsonl", "w") as sf:
            for r in rows_sup:
                sf.write(json.dumps({
                    "task": r["task"],
                    "acc": r["acc"],
                    "oov_rate": r["oov_rate"],
                    "n": r["n"],
                    "source": r.get("source", "")
                }) + "\n")
                print(f"SuppTask={r['task']:<28} n={r['n']:<5} PLL-acc={r['acc']:.4f}  OOV={r['oov_rate']:.2%}")
        with open(args.supplement_save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["task", "n", "acc", "oov_rate", "normalize", "source"])
            w.writeheader()
            for r in rows_sup:
                w.writerow({k: r.get(k, "") for k in w.fieldnames})
        if rows_sup:
            macro_sup = sum(r["acc"] for r in rows_sup) / len(rows_sup)
            print(f"\n[BLiMP Supplement Summary] tasks={len(rows_sup)}  macro-PLL-acc={macro_sup:.4f}  saved: {args.supplement_save_csv}, {args.supplement_save_json}")

    # quick overall print if both
    if args.benchmark == "both":
        total_tasks = (len(rows) if rows else 0) + (len(rows_sup) if rows_sup else 0)
        if total_tasks:
            macro_all = (sum(r["acc"] for r in rows) + sum(r["acc"] for r in rows_sup)) / total_tasks
            print(f"\n[Overall] total_parts={total_tasks} macro-PLL-acc={macro_all:.4f}")

if __name__ == "__main__":
    main()