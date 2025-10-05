"""
Metrics utilities for training-time monitoring.

This module provides helpers to track per-epoch distributions (sequence lengths,
truncation rate, uniqueness) and compute per-step metrics such as MLM accuracy,
masking split, throughput, and gradient statistics for logging.
"""

import time
from typing import Dict, Any, Tuple
import torch

from thin_logger import compute_protected_ids


def start_epoch_tracking(config: dict, tokenizer) -> Dict[str, Any]:
    """Initialize and return a dictionary for per-epoch tracking state.

    Keys include:
      - seq_len_cfg: configured sequence length budget
      - length_hist: histogram of effective lengths in [0..seq_len_cfg]
      - truncation_hits: count of samples reaching the max length
      - unique_hashes: capped set used to estimate repeats
      - protected_ids: structural token ids to exclude from masking leak stats
    """
    seq_len_cfg = int(config.get("seq_length", 128))
    try:
        mask_token_id = tokenizer.token_to_id("[MASK]")
    except Exception:
        mask_token_id = None
    state = {
        "seq_len_cfg": seq_len_cfg,
        "length_hist": [0] * (seq_len_cfg + 1),
        "truncation_hits": 0,
        "unique_hashes": set(),
        "unique_cap": int(config.get("unique_cap", 200000)),
        "mask_token_id": mask_token_id,
        "protected_ids": compute_protected_ids(tokenizer),
    }
    return state


def update_epoch_distributions(state: Dict[str, Any], batch: Dict[str, torch.Tensor]) -> None:
    """Update sequence length histogram, truncation count, and uniqueness estimator."""
    seq_len_cfg = state["seq_len_cfg"]
    length_hist = state["length_hist"]
    trunc_hits = state["truncation_hits"]

    eff_lens = batch["attention_mask"].sum(dim=1).detach().cpu().tolist()
    for L in eff_lens:
        Li = int(L)
        if 0 <= Li <= seq_len_cfg:
            length_hist[Li] += 1
        else:
            length_hist[min(max(Li, 0), seq_len_cfg)] += 1
        if Li >= seq_len_cfg:
            trunc_hits += 1
    state["truncation_hits"] = trunc_hits

    # Uniqueness estimator (hash of rows, capped)
    uniq_set = state["unique_hashes"]
    cap = state["unique_cap"]
    if len(uniq_set) < cap:
        ids_cpu = batch["input_ids"].detach().cpu()
        for row in ids_cpu:
            h = int((row.long() * 1315423911).sum().item()) & 0xFFFFFFFFFFFFFFFF
            uniq_set.add(h)


def compute_train_step_metrics(
    state: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
    logits: torch.Tensor,
    scheduler,
    monitor,
    step: int,
    epoch: int,
    last_log_time: float,
    prev_step_end: float,
) -> Tuple[Dict[str, Any], float, float]:
    """Compute step-level metrics for logging.

    Returns:
        (metrics: Dict[str, Any], new_last_log_time: float, new_prev_step_end: float)
    """
    with torch.no_grad():
        labels = batch["labels"]
        mask = labels != -100
        masked_count = int(mask.sum().item())
        if masked_count > 0:
            preds = logits.argmax(dim=-1)
            correct = int((preds[mask] == labels[mask]).sum().item())
            mlm_acc = correct / masked_count
        else:
            mlm_acc = 0.0

        prot_leak = 0
        prot_ids = state.get("protected_ids") or set()
        if prot_ids:
            prot_mask = torch.zeros_like(mask, dtype=torch.bool)
            for pid in prot_ids:
                prot_mask |= (labels == pid)
            prot_leak = int((prot_mask & mask).sum().item())

        # Masking split among masked positions
        mask_token = int(state.get("mask_token_id")) if state.get("mask_token_id") is not None else -1
        masked_input = batch["input_ids"][mask]
        masked_labels = labels[mask]
        mask_token_count = int((masked_input == mask_token).sum().item()) if mask_token >= 0 else 0
        kept_count = int((masked_input == masked_labels).sum().item()) if masked_count > 0 else 0
        random_count = max(masked_count - mask_token_count - kept_count, 0)

        # Throughput and timing
        bsz = batch["input_ids"].size(0)
        seq_len = batch["input_ids"].size(1)
        now = time.time()
        dt = max(1e-6, now - last_log_time)
        toks_per_s = (bsz * seq_len) / dt
        smp_per_s = bsz / dt
        step_end = time.time()
        step_time = step_end - (prev_step_end if prev_step_end else now)
        data_time = max(0.0, (now - (prev_step_end if prev_step_end else now)))

        # LR and grad signals
        lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else None
        grad_norm = monitor.grad_norm_history[-1] if monitor.grad_norm_history else 0.0
        clipped = 1 if grad_norm >= 0.98 * getattr(monitor, "max_grad_norm", 1.0) else 0

        metrics = {
            "mlm_acc": float(mlm_acc),
            "lr": float(lr) if lr is not None else None,
            "tokens_per_sec": float(toks_per_s),
            "samples_per_sec": float(smp_per_s),
            "step_time_sec": float(step_time),
            "data_time_sec": float(data_time),
            "grad_norm": float(grad_norm),
            "grad_clipped": int(clipped),
            "seq_len": int(seq_len),
            "masked_count": int(masked_count),
            "mask_token_count": int(mask_token_count),
            "keep_count": int(kept_count),
            "random_count": int(random_count),
            "protected_label_leak": int(prot_leak),
            "epoch": int(epoch),
        }

        return metrics, now, step_end


def finalize_epoch_summaries(state: Dict[str, Any], samples_seen: int) -> Dict[str, Dict[str, Any]]:
    """Compute length distribution summary and uniqueness summary at epoch end."""
    length_hist = state["length_hist"]
    seq_len_cfg = state["seq_len_cfg"]
    total = sum(length_hist)

    lengths = {}
    if total > 0:
        def percentile(p):
            tgt = p * total
            s = 0
            for i, c in enumerate(length_hist):
                s += c
                if s >= tgt:
                    return i
            return seq_len_cfg
        avg_len = sum(i * c for i, c in enumerate(length_hist)) / total
        p50 = percentile(0.50)
        p90 = percentile(0.90)
        p99 = percentile(0.99)
        trunc_rate = state["truncation_hits"] / total
        lengths = {
            "avg_len": float(avg_len),
            "p50": int(p50),
            "p90": int(p90),
            "p99": int(p99),
            "truncation_rate": float(trunc_rate),
        }

    uniq = len(state["unique_hashes"]) if state["unique_hashes"] else 0
    repeat_ratio = 1.0 - (uniq / samples_seen) if samples_seen > 0 else 0.0
    data = {
        "unique_sequences": int(uniq),
        "samples_seen": int(samples_seen),
        "repeat_ratio": float(repeat_ratio),
    }
    return {"lengths": lengths, "data": data}
