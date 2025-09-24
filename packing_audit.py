#!/usr/bin/env python3
"""Audit statistics for packed sentence segments.

Reads a directory of cached chunk .pt files OR a raw packed text file list and
computes:
  - number of samples
  - token length distribution
  - punctuation-final rate (last sentence in sample ends .!? or filler)
  - average sentences per sample
  - fraction samples starting with [DOC] and ending before [EOD]
  - speaker token prevalence
  - histogram buckets for fill ratio vs max length

Usage:
  python packing_audit.py --tokenizer tokenizer.json --max-len 512 --packed-list packed.txt
  python packing_audit.py --tokenizer tokenizer.json --cache-dir cache/
"""
import argparse, os, json, math, statistics
from collections import Counter, defaultdict
import torch

FILLER_TERMINALS = {"yeah","mm","erm","uh","er","ah","hmm"}

def load_packed_from_cache(cache_dir):
    samples = []
    for fn in sorted(os.listdir(cache_dir)):
        if fn.startswith('chunk') and fn.endswith('.pt'):
            try:
                obj = torch.load(os.path.join(cache_dir, fn), map_location='cpu')
                # Expect list of dicts with 'text'
                for ex in obj:
                    if isinstance(ex, dict) and 'text' in ex:
                        samples.append(ex['text'])
            except Exception as e:
                print(f"[Warn] Failed to load {fn}: {e}")
    return samples

def load_packed_list(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

def is_terminal(sent: str):
    if not sent: return False
    if sent[-1] in '.!?': return True
    if sent.lower().strip('"\'') in FILLER_TERMINALS: return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--cache-dir')
    ap.add_argument('--packed-list')
    ap.add_argument('--max-len', type=int, default=512)
    ap.add_argument('--jsonl-out')
    args = ap.parse_args()

    from tokenizer import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)

    if args.cache_dir:
        samples = load_packed_from_cache(args.cache_dir)
    else:
        samples = load_packed_list(args.packed_list)
    print(f"Loaded {len(samples)} packed samples")

    lengths=[]; sent_counts=[]; punct_final=0; doc_starts=0; eod_tokens=0; speaker_tokens=0
    fill_buckets=Counter()
    for s in samples:
        ids = tok.encode(s).ids
        lengths.append(len(ids))
        # sentence count ~ count of [SEP]
        sent_counts.append(s.count('[SEP]'))
        # approximate last sentence text
        parts=[p.strip() for p in s.split('[SEP]') if p.strip()]
        if parts:
            last=parts[-1].replace('[CLS]','').strip()
            if is_terminal(last):
                punct_final+=1
        if s.startswith('[CLS] [DOC]'):
            doc_starts+=1
        if '[EOD]' in s:
            eod_tokens+=1
        if '[SPK]' in s:
            speaker_tokens+=1
        ratio=min(len(ids)/args.max_len,1.0)
        bucket=int(ratio*10)
        fill_buckets[bucket]+=1

    report={
        'num_samples': len(samples),
        'avg_len': statistics.mean(lengths) if lengths else 0,
        'median_len': statistics.median(lengths) if lengths else 0,
        'p95_len': sorted(lengths)[int(0.95*len(lengths))] if lengths else 0,
        'avg_sentences_per_sample': statistics.mean(sent_counts) if sent_counts else 0,
        'punct_final_rate': punct_final/len(samples) if samples else 0,
        'doc_start_fraction': doc_starts/len(samples) if samples else 0,
        'eod_token_fraction': eod_tokens/len(samples) if samples else 0,
        'speaker_token_fraction': speaker_tokens/len(samples) if samples else 0,
        'fill_ratio_hist': {f'{b/10:.1f}-{(b+1)/10:.1f}': fill_buckets.get(b,0) for b in range(10)}
    }
    print(json.dumps(report, indent=2))
    if args.jsonl_out:
        with open(args.jsonl_out,'w') as f:
            f.write(json.dumps(report)+'\n')

if __name__=='__main__':
    main()
