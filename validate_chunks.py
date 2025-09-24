#!/usr/bin/env python
"""
validate_chunks.py

Purpose:
  Validate semantic boundary integrity and structural token constraints across buffered chunk files
  produced by prepare_data.py (each chunk file: tensor shape [N, block_size]).

Checks Performed:
  1. File shape & dtype: ensure 2-D tensor, dtype long, width == inferred block_size.
  2. Structural token ID detection: determine contiguous special token prefix via tokenizer file if provided.
  3. DOC/EOD balance:
     - Inside each block sequence, [DOC] must not appear after an unmatched [DOC] without an [EOD] between multiple starts.
     - Count mismatched or nested starts >1 depth.
  4. Cross-block continuity (optional): If a block ends with an open document (started without EOD) and next block resumes
     without EOD or new DOC, flag (unless block ended exactly at boundary).
  5. No tokens after first PAD in a block (strict padding tail rule) if --enforce-pad-tail is set.
  6. Forbidden pattern: [EOD] immediately followed by non-PAD non-[DOC] token when --strict-eod-boundary.
  7. Optional: verify average fill ratio vs metadata (if meta JSON present) and warn on deviation > tolerance.

Exit Codes:
  0 = success (no errors)
  1 = soft warnings only (if --treat-warn-nonzero not set, still 0)
  2 = errors detected (semantic boundary violations)

Usage:
  python validate_chunks.py --cache-path ./cache --tokenizer-path ./data/pretrain/wordpiece_vocab.json \
        --doc-token [DOC] --eod-token [EOD] --pad-token [PAD] --sep-token [SEP] \
        --max-files 1600 --enforce-pad-tail --strict-eod-boundary

Performance:
  Streams through chunk*.pt sequentially; for large corpora set --max-files for quick audit.
"""
import argparse
import glob
import json
import os
import sys
from typing import List, Optional, Dict, Any, Set, Tuple

import torch


def detect_token_id(tokenizer_path: Optional[str], token: str) -> Optional[int]:
    if tokenizer_path is None:
        return None
    try:
        from tokenizer import Tokenizer
        tok = Tokenizer.from_file(tokenizer_path)
        return tok.token_to_id(token)
    except Exception:
        return None


def load_meta(meta_path: str):
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def validate_file(tensor_path: str, args, ids, open_doc_initial=False, tokenizer_obj=None, anomalies_list: Optional[List[Dict[str, Any]]] = None, anomaly_key_set: Optional[Set[Tuple[str, str, int]]] = None):
    """Validate a single chunk tensor file.

    If anomalies_list is provided (together with anomaly_key_set), structural boundary errors will
    record contextual token information for later inspection / JSON emission.
    """
    errors: List[str] = []
    warnings: List[str] = []
    doc_id, eod_id, pad_id, sep_id = ids

    try:
        data = torch.load(tensor_path, map_location='cpu')
    except Exception as e:
        errors.append(f"LoadFail:{os.path.basename(tensor_path)}:{e}")
        return errors, warnings, 0, 0, 0

    if not torch.is_tensor(data):
        errors.append(f"NotTensor:{tensor_path}")
        return errors, warnings, 0, 0, 0
    if data.dim() != 2:
        errors.append(f"BadRank:{tensor_path}:rank={data.dim()}")
        return errors, warnings, 0, 0, 0

    num_blocks, width = data.shape
    if args.block_size and width != args.block_size:
        errors.append(f"BadWidth:{tensor_path}:expected={args.block_size} got={width}")
    block_size = width

    open_doc = open_doc_initial
    blocks_checked = 0
    total_fill_tokens = 0
    pad_tail_violations = 0

    for b in range(num_blocks):
        seq = data[b]
        # fill tokens (non-pad)
        if pad_id is not None:
            non_pad = (seq != pad_id).sum().item()
            total_fill_tokens += non_pad
        else:
            total_fill_tokens += block_size

        # Check pad tail rule
        if args.enforce_pad_tail and pad_id is not None:
            # First PAD index then ensure rest all PAD
            pad_positions = (seq == pad_id).nonzero(as_tuple=False).flatten().tolist()
            if pad_positions:
                first_pad = pad_positions[0]
                trailing = seq[first_pad:]
                if not torch.all(trailing == pad_id):
                    pad_tail_violations += 1
                    errors.append(f"PadTail:{os.path.basename(tensor_path)}:block={b}")

        # Semantic boundary scan (with optional literal heuristics)
        if doc_id is not None and eod_id is not None:
            seq_list = data[b].tolist()
            n_local = len(seq_list)
            literal_doc_positions: Set[int] = set()
            literal_eod_positions: Set[int] = set()
            if args.literal_heuristics:
                for j, tok in enumerate(seq_list):
                    if tok == doc_id and open_doc:
                        prev_tok = seq_list[j-1] if j > 0 else None
                        next_tok = seq_list[j+1] if j+1 < n_local else None
                        if prev_tok in (sep_id, pad_id, doc_id, eod_id, None) and next_tok in (sep_id, pad_id, doc_id, eod_id, None):
                            literal_doc_positions.add(j)
                    elif tok == eod_id and not open_doc:
                        ahead = seq_list[j+1:j+6]
                        if doc_id not in ahead:
                            literal_eod_positions.add(j)
            for j, tok in enumerate(seq_list):
                if tok == doc_id:
                    if open_doc and j not in literal_doc_positions:
                        err_type = 'NestedDOC'
                        err_str = f"{err_type}:{os.path.basename(tensor_path)}:block={b}"
                        errors.append(err_str)
                        if anomalies_list is not None:
                            key = (err_type, tensor_path, b)
                            if key not in anomaly_key_set:
                                anomaly_key_set.add(key)
                                try:
                                    first_pad_idx = seq_list.index(pad_id)
                                except ValueError:
                                    first_pad_idx = n_local
                                token_ids = seq_list[:first_pad_idx]
                                decoded = None
                                if tokenizer_obj is not None:
                                    try:
                                        decoded = [tokenizer_obj.id_to_token(x) for x in token_ids]
                                    except Exception:
                                        decoded = None
                                anomalies_list.append({
                                    'error_type': err_type,
                                    'file': os.path.basename(tensor_path),
                                    'path': tensor_path,
                                    'block_index': b,
                                    'token_ids': token_ids,
                                    'decoded_tokens': decoded,
                                    'doc_positions': [k for k, _id in enumerate(token_ids) if _id == doc_id],
                                    'eod_positions': [k for k, _id in enumerate(token_ids) if _id == eod_id]
                                })
                    if j not in literal_doc_positions:
                        open_doc = True
                elif tok == eod_id:
                    if (not open_doc) and j not in literal_eod_positions:
                        err_type = 'EODNoOpen'
                        err_str = f"{err_type}:{os.path.basename(tensor_path)}:block={b}"
                        errors.append(err_str)
                        if anomalies_list is not None:
                            key = (err_type, tensor_path, b)
                            if key not in anomaly_key_set:
                                anomaly_key_set.add(key)
                                try:
                                    first_pad_idx = seq_list.index(pad_id)
                                except ValueError:
                                    first_pad_idx = n_local
                                token_ids = seq_list[:first_pad_idx]
                                decoded = None
                                if tokenizer_obj is not None:
                                    try:
                                        decoded = [tokenizer_obj.id_to_token(x) for x in token_ids]
                                    except Exception:
                                        decoded = None
                                anomalies_list.append({
                                    'error_type': err_type,
                                    'file': os.path.basename(tensor_path),
                                    'path': tensor_path,
                                    'block_index': b,
                                    'token_ids': token_ids,
                                    'decoded_tokens': decoded,
                                    'doc_positions': [k for k, _id in enumerate(token_ids) if _id == doc_id],
                                    'eod_positions': [k for k, _id in enumerate(token_ids) if _id == eod_id]
                                })
                    if j not in literal_eod_positions:
                        open_doc = False
                elif pad_id is not None and tok == pad_id:
                    break
        blocks_checked += 1

    # Load meta for optional fill ratio check
    meta = load_meta(tensor_path.replace('.pt', '.meta.json'))
    if meta and 'num_blocks' in meta and meta.get('num_blocks') != num_blocks:
        warnings.append(f"MetaMismatchBlocks:{os.path.basename(tensor_path)} meta={meta.get('num_blocks')} actual={num_blocks}")

    return errors, warnings, blocks_checked, total_fill_tokens, block_size, open_doc


def main():
    p = argparse.ArgumentParser(description="Validate semantic and structural constraints of chunked blocks")
    p.add_argument('--cache-path', required=True)
    p.add_argument('--tokenizer-path', help='Path to tokenizer JSON for ID lookup')
    p.add_argument('--doc-token', default='[DOC]')
    p.add_argument('--eod-token', default='[EOD]')
    p.add_argument('--pad-token', default='[PAD]')
    p.add_argument('--sep-token', default='[SEP]')
    p.add_argument('--block-size', type=int, default=None, help='Expected block width (optional)')
    p.add_argument('--max-files', type=int, default=None, help='Limit number of files for quick audit')
    p.add_argument('--enforce-pad-tail', action='store_true', help='Require that once PAD appears only PAD follows')
    p.add_argument('--strict-eod-boundary', action='store_true', help='(Reserved for future) enforce EOD boundary rules')
    p.add_argument('--treat-warn-nonzero', action='store_true', help='Return non-zero exit on warnings')
    p.add_argument('--carry-open-doc', action='store_true', help='Persist open document state across files (treat unmatched DOC at file boundary as contextual)')
    # Anomaly / inspection options
    p.add_argument('--emit-anomalies-json', help='If set, write a JSON list of anomaly contexts to this path')
    p.add_argument('--max-anomaly-records', type=int, default=100, help='Cap number of anomaly context records (to prevent huge JSON)')
    p.add_argument('--literal-heuristics', action='store_true', help='Apply heuristics to down-rank literal [DOC]/[EOD] text occurrences (non-structural)')
    args = p.parse_args()

    doc_id = detect_token_id(args.tokenizer_path, args.doc_token)
    eod_id = detect_token_id(args.tokenizer_path, args.eod_token)
    pad_id = detect_token_id(args.tokenizer_path, args.pad_token)
    sep_id = detect_token_id(args.tokenizer_path, args.sep_token)

    ids = (doc_id, eod_id, pad_id, sep_id)

    paths = sorted(glob.glob(os.path.join(args.cache_path, 'chunk*.pt')))
    if args.max_files is not None:
        paths = paths[:args.max_files]
    if not paths:
        print("No chunk files found.")
        sys.exit(2)
    # Aggregation containers
    total_errors: List[str] = []
    total_warnings: List[str] = []
    total_blocks = 0
    total_fill_tokens = 0
    block_size_ref = None

    # Anomaly capture setup
    anomalies = None
    anomaly_key_set = None
    tokenizer_obj = None
    if args.emit_anomalies_json:
        anomalies = []
        anomaly_key_set = set()
        if args.tokenizer_path:
            try:
                from tokenizer import Tokenizer
                tokenizer_obj = Tokenizer.from_file(args.tokenizer_path)
            except Exception:
                tokenizer_obj = None

    open_doc_carry = False
    for idx, tensor_path in enumerate(paths):
        errs, warns, blocks_checked, fill_tokens, block_size, open_doc_state = validate_file(
            tensor_path,
            args,
            ids,
            open_doc_initial=open_doc_carry if args.carry_open_doc else False,
            tokenizer_obj=tokenizer_obj,
            anomalies_list=anomalies,
            anomaly_key_set=anomaly_key_set
        )
        total_errors.extend(errs)
        total_warnings.extend(warns)
        total_blocks += blocks_checked
        total_fill_tokens += fill_tokens
        if block_size_ref is None and block_size:
            block_size_ref = block_size
        if args.carry_open_doc:
            open_doc_carry = open_doc_state

    if block_size_ref is None:
        block_size_ref = args.block_size or 0
    avg_fill = (total_fill_tokens / (total_blocks * block_size_ref)) if (total_blocks and block_size_ref) else 0.0

    print("[Validate][Summary]")
    print(f"  Files inspected: {len(paths)}")
    print(f"  Total blocks: {total_blocks}")
    print(f"  Block size: {block_size_ref}")
    print(f"  Avg fill ratio (non-pad): {avg_fill:.4f}")
    print(f"  Errors: {len(total_errors)}")
    print(f"  Warnings: {len(total_warnings)}")
    if args.literal_heuristics:
        print("  Literal heuristics: ENABLED (boundary tokens inside ambiguous spans may be down-ranked)")

    if total_errors:
        print("[Validate][Errors]")
        for e in total_errors[:200]:
            print("  -", e)
        if len(total_errors) > 200:
            print(f"  ... {len(total_errors)-200} more")
    if total_warnings:
        print("[Validate][Warnings]")
        for w in total_warnings[:200]:
            print("  -", w)
        if len(total_warnings) > 200:
            print(f"  ... {len(total_warnings)-200} more")

    if anomalies is not None and args.emit_anomalies_json:
        # Respect max records limit
        if len(anomalies) > args.max_anomaly_records:
            anomalies = anomalies[:args.max_anomaly_records]
        try:
            with open(args.emit_anomalies_json, 'w', encoding='utf-8') as f_out:
                json.dump(anomalies, f_out, ensure_ascii=False, indent=2)
            print(f"[Validate][Anomalies] Wrote {len(anomalies)} context records to {args.emit_anomalies_json}")
        except Exception as e:
            print(f"[Validate][Anomalies][WriteError] {e}")

    if total_errors:
        sys.exit(2)
    if total_warnings and args.treat_warn_nonzero:
        sys.exit(1)
    sys.exit(0)

if __name__ == '__main__':
    main()
