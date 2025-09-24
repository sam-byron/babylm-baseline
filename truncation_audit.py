#!/usr/bin/env python
"""Structured truncation / boundary audit.

Generates a JSON report summarizing how often training lines terminate
mid-sentence vs at punctuation, and how frequently key function/complementizer
and wh tokens appear immediately before a segment boundary. Helps diagnose
whether truncation is biasing lexical priors (e.g., favoring 'that').

Metrics:
  sampled: number of lines analyzed
  punct_final_rate: fraction ending with sentence-final punctuation (.,!,?)
  unk_tail_rate: fraction whose last real token is [UNK]
  function_tail_rate: last real token in FUNCTION_WORDS
  mid_sentence_rate: 1 - punct_final_rate (approx; may include dialogue fragments)
  mid_subword_tail_rate: heuristic continuing-subword tail rate
  top_tail_tokens: most common last real tokens (excluding specials)
  boundary_counts: per tracked token {"total": occurrences anywhere, "boundary_after": occurrences followed by truncation boundary}
  boundary_after_rate: boundary_after/total for each tracked token

Usage:
  python truncation_audit.py --max-samples 8000 --out audit_trunc.json
"""
from __future__ import annotations
import argparse, json, glob, re, random
from collections import Counter, defaultdict
from tokenizer import Tokenizer

SPECIALS = {"[CLS]", "[SEP]", "[PAD]", "[MASK]"}
SENT_PUNCT = set(".!?")
PUNCT_RE = re.compile(r"[.!?]+$")
QUOTE_CHARS = set("'\"")

# Core tokens to monitor for boundary adjacency
COMPLEMENTIZERS = {"that", "which"}
WH_WORDS = {"what","who","whom","whose","which","where","when","why","how"}
FUNCTION_WORDS = {
    'that','who','what','which','to','of','with','by','for','in','on','as','at','and','but','or','if','because','while','whom','whose'
}
TRACKED = sorted(COMPLEMENTIZERS | WH_WORDS)

tok = Tokenizer.from_file("./tokenizer.json")

def load_lines(limit:int) -> list[str]:
    lines=[]
    for f in glob.glob('data/pretrain/bnc/**/*.md', recursive=True):
        with open(f,'r',encoding='utf-8') as h:
            for line in h:
                line=line.strip()
                if line:
                    lines.append(f"[CLS] {line} [SEP]")
                if len(lines) >= limit: break
        if len(lines)>=limit: break
    return lines

def extract_last_real(tokens:list[str]):
    for t in reversed(tokens):
        u = t.strip()
        if not u: continue
        if u.upper() in SPECIALS: continue
        return u
    return None

def classify_tail(tok_text:str):
    norm = tok_text.lower()
    is_punct_final = bool(PUNCT_RE.search(tok_text))
    # Heuristic mid-subword: leading punctuation or leftover prefix artifacts
    mid_sub = (tok_text.startswith('##') or (not tok_text[0].isalnum() and tok_text[0] not in QUOTE_CHARS))
    return norm, is_punct_final, mid_sub

def audit(lines, max_samples:int):
    sample = random.sample(lines, min(len(lines), max_samples))
    tail_counter = Counter()
    punct_final = 0
    mid_subword = 0
    function_tail = 0
    unk_tail = 0

    # For boundary adjacency: token occurrences and how many end up at boundary
    token_total = Counter()
    token_boundary_after = Counter()

    total=0
    for s in sample:
        parts = s.strip().split()
        # collect last real
        last_real = extract_last_real(parts)
        if last_real is None:
            continue
        total += 1
        norm, is_punct_final, mid_sub = classify_tail(last_real)
        tail_counter[norm] += 1
        if is_punct_final: punct_final += 1
        if mid_sub: mid_subword += 1
        if norm in FUNCTION_WORDS: function_tail += 1
        if norm == '[unk]': unk_tail += 1
        # Count internal occurrences to approximate adjacency potential
        # We treat the boundary as after the last_real token always.
        tokens_lower = [w.lower() for w in parts if w.upper() not in SPECIALS]
        for i, t in enumerate(tokens_lower):
            if t in TRACKED:
                token_total[t] += 1
                if i == len(tokens_lower)-1:
                    token_boundary_after[t] += 1

    if total == 0:
        return {"sampled":0}
    def rate(x): return x/total

    report = {
        "sampled": total,
        "punct_final_rate": rate(punct_final),
        "mid_sentence_rate": 1 - rate(punct_final),
        "function_tail_rate": rate(function_tail),
        "unk_tail_rate": rate(unk_tail),
        "mid_subword_tail_rate": rate(mid_subword),
        "top_tail_tokens": tail_counter.most_common(25),
        "boundary_counts": {}
    }
    for t in TRACKED:
        tot = token_total[t]
        bnd = token_boundary_after[t]
        if tot:
            report["boundary_counts"][t] = {"total": tot, "boundary_after": bnd, "boundary_after_rate": bnd / tot}
        else:
            report["boundary_counts"][t] = {"total": 0, "boundary_after": 0, "boundary_after_rate": None}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-samples', type=int, default=8000)
    ap.add_argument('--out', type=str, default='truncation_audit.json')
    ap.add_argument('--pretty', action='store_true', help='Pretty-print JSON to stdout')
    args = ap.parse_args()
    lines = load_lines(args.max_samples)
    report = audit(lines, args.max_samples)
    with open(args.out,'w') as h:
        json.dump(report, h, indent=2)
    if args.pretty:
        print(json.dumps(report, indent=2))
    else:
        print(f"Wrote {args.out} (sampled={report.get('sampled',0)})")

if __name__ == '__main__':
    main()
