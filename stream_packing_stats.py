#!/usr/bin/env python3
"""Streaming packing stats collector.

Purpose: Run pack_sentences in streaming mini-batches of raw lines to avoid
materializing the entire corpus in memory before tokenization and reduce wall clock
before first stats appear.

Strategy:
  - Accumulate lines up to a block (e.g. N raw lines or doc boundary counts) then call
    pack_sentences on that slice with collect_stats=True
  - Merge stats counters cumulatively.
  - Write periodic JSON lines snapshots so long corpus runs can be monitored.
  - Final output: merged stats + derived rates.

Note: Because heuristics can carry dangling sentences across packs *within* a call,
calling pack_sentences on slices may slightly under-estimate dangling_carried if a
dangling fragment would have carried across slice boundaries. To mitigate, we keep
"carryover_buffer" of last few non-terminal sentences from previous slice and prepend
them to the next slice.
"""
import argparse, os, json, glob, sys
from typing import List
from tokenizer import Tokenizer
from sentence_packing import pack_sentences

def iter_lines(data_dir):
    pattern = os.path.join(data_dir, "**/*.md")
    for fp in glob.glob(pattern, recursive=True):
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
        except Exception as e:
            print(f"[Warn] Could not read {fp}: {e}", file=sys.stderr)

def merge_stats(total, part):
    if total is None: return part
    for k,v in part.items():
        total[k] = total.get(k,0) + v
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--max-length', type=int, default=512)
    ap.add_argument('--slice-lines', type=int, default=120000, help='How many raw non-marker lines per slice (approx)')
    ap.add_argument('--snapshot-every', type=int, default=5, help='Write snapshot after this many slices')
    ap.add_argument('--out-json', default='stream_packing_stats.json')
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)

    slice_buf: List[str] = []
    stats_total = None
    slices = 0
    carryover_prefix: List[str] = []  # holds dangling carried fragment(s) from prior slice

    def flush_slice():
        nonlocal slice_buf, stats_total, slices, carryover_prefix
        if not slice_buf: return
        # Inject DOC/EOD markers around the slice to preserve doc semantics lightly; treat each slice as pseudo-doc boundary for stats safety
        lines = ['[DOC]'] + carryover_prefix + slice_buf + ['[EOD]']
        result = pack_sentences(lines, tok, args.max_length, add_doc_and_speaker=True, collect_stats=True, drop_short_trailing=True)
        segments, stats = result
        # For approximate cross-slice carry: if last segment ends non-terminal, extract its last sentence as prefix for next slice
        last_seg = segments[-1] if segments else ''
        last_nonterm = False
        if last_seg:
            # Determine last sentence
            parts = [p.strip() for p in last_seg.split('[SEP]') if p.strip()]
            if parts:
                last = parts[-1].replace('[CLS]', '').strip()
                if last and not last[-1:] in '.!?':
                    last_nonterm = True
                    carryover_prefix = [last]
                else:
                    carryover_prefix = []
        stats_total = merge_stats(stats_total, stats)
        slices += 1
        if slices % args.snapshot_every == 0:
            packs = max(stats_total.get('packs',1),1)
            snap = {
                'slices': slices,
                'packs': stats_total.get('packs',0),
                'terminal_end_rate': stats_total.get('terminal_end',0)/packs,
                'nonterminal_end_rate': stats_total.get('nonterminal_end',0)/packs,
                'dangling_carried_rate': stats_total.get('dangling_carried',0)/packs,
                'borrow_move_rate': stats_total.get('borrow_moves',0)/packs,
                'short_trailing_dropped_rate': stats_total.get('short_trailing_dropped',0)/packs,
            }
            print(json.dumps({'snapshot': snap}))
        slice_buf = []

    for i,line in enumerate(iter_lines(args.data_dir)):
        slice_buf.append(line)
        if len(slice_buf) >= args.slice_lines:
            flush_slice()
    flush_slice()

    if not stats_total:
        print('{}')
        return
    packs = max(stats_total.get('packs',1),1)
    final = {
        'packs': stats_total.get('packs',0),
        'terminal_end': stats_total.get('terminal_end',0),
        'nonterminal_end': stats_total.get('nonterminal_end',0),
        'dangling_carried': stats_total.get('dangling_carried',0),
        'borrow_moves': stats_total.get('borrow_moves',0),
        'short_trailing_dropped': stats_total.get('short_trailing_dropped',0),
        'terminal_end_rate': stats_total.get('terminal_end',0)/packs,
        'nonterminal_end_rate': stats_total.get('nonterminal_end',0)/packs,
        'dangling_carried_rate': stats_total.get('dangling_carried',0)/packs,
        'borrow_move_rate': stats_total.get('borrow_moves',0)/packs,
        'short_trailing_dropped_rate': stats_total.get('short_trailing_dropped',0)/packs,
        'slices': slices
    }
    with open(args.out_json,'w') as f:
        json.dump(final,f,indent=2)
    print(json.dumps({'final': final}, indent=2))

if __name__ == '__main__':
    main()
