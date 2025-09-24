import json, random, re
from tokenizer import Tokenizer
from sentence_packing import pack_sentences
from truncation_audit import PUNCT_RE, SPECIALS
from pathlib import Path

# Simple smoke test: load a few raw lines, pack, compute ending punctuation rate

def iter_raw_lines(limit=4000):
    import glob
    count=0
    for f in glob.glob('data/pretrain/bnc/**/*.md', recursive=True):
        with open(f,'r',encoding='utf-8') as h:
            for line in h:
                line=line.strip()
                if line:
                    yield line
                    count+=1
                if count>=limit:
                    return

def last_real_token(seq: str):
    toks = seq.strip().split()
    for t in reversed(toks):
        if t.upper() in SPECIALS: continue
        return t
    return None

if __name__ == "__main__":
    tok = Tokenizer.from_file('./tokenizer.json')
    lines = list(iter_raw_lines())
    packed = pack_sentences(lines, tok, 512)
    sample = random.sample(packed, min(2000, len(packed)))
    punct=0
    total=0
    over=0
    for seq in sample:
        # length check
        ids = tok.encode(seq.replace('[CLS]','').replace('[SEP]','')).ids
        if len(ids) > 512:
            over+=1
        lr = last_real_token(seq)
        if lr:
            total+=1
            if PUNCT_RE.search(lr):
                punct+=1
    print(f"Packed sequences: {len(packed)}")
    print(f"Sampled: {total}")
    print(f"Approx punct_final_rate: {punct/total:.2%}")
    print(f"Sequences over budget: {over}")
    # Show a couple examples
    for ex in sample[:3]:
        print('---')
        print(ex[:300])
