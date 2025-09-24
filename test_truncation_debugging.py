"""Quick diagnostics for truncation artifacts.

Measures properties of the LAST REAL token in each sampled training line,
excluding artificial wrappers like [CLS] / [SEP] / [PAD]. The previous
implementation accidentally treated the injected [SEP] marker as the final
token 100% of the time, producing zero rates for punctuation / function tails.
"""

from collections import Counter
import random, argparse, statistics as stats
from tokenizer import Tokenizer
import glob
import re

tok = Tokenizer.from_file("./tokenizer.json")

SPECIALS = {"[CLS]", "[SEP]", "[PAD]", "[MASK]"}

DEFAULT_FUNCTION_WORDS = {
    'that','who','what','which','to','of','with','by','for','in','on','as','at','and','but','or','if','because','while'
}

PUNCT_RE = re.compile(r"[.!?]+$")

def extract_last_real_token(text_tokens):
    """Return last non-special token string or None.
    text_tokens: list of raw whitespace tokens (already stripped)
    """
    for tok_ in reversed(text_tokens):
        tt = tok_.strip()
        if not tt:
            continue
        if tt.upper() in SPECIALS:
            continue
        return tok_
    return None

def analyze(samples, max_samples=5000, function_words=None, top_k=15):
    if function_words is None:
        function_words = DEFAULT_FUNCTION_WORDS
    tail_counter = Counter()
    punct_final = 0
    mid_subword = 0
    function_tail = 0
    total = 0
    tail_lengths = []  # length in characters of trailing fragment (last token)

    for s in random.sample(samples, min(len(samples), max_samples)):
        raw_tokens = s.strip().split()
        last_real = extract_last_real_token(raw_tokens)
        if last_real is None:
            continue
        total += 1
        norm = last_real.lower()
        tail_counter[norm] += 1
        if PUNCT_RE.search(last_real):
            punct_final += 1
        # WordPiece mid-subword heuristic (continuing piece) -- our tokenizer uses empty prefix
        # so we approximate by: token not alnum at start OR very short fragment w/out punctuation
        if not last_real[0].isalnum() and last_real[0] not in '"\'' or last_real.startswith("##"):
            mid_subword += 1
        if norm in function_words:
            function_tail += 1
        tail_lengths.append(len(last_real))

    if total == 0:
        print("No valid samples after filtering specials.")
        return

    def pct(x):
        return f"{(x/total):.3%}"

    print(f"Samples considered: {total}")
    print(f"Punct-final: {pct(punct_final)}")
    print(f"Function-tail: {pct(function_tail)}")
    print(f"Mid-subword heuristic: {pct(mid_subword)}")
    print("Top tail tokens (excluding specials):")
    for tok_, cnt in tail_counter.most_common(top_k):
        print(f"  {tok_:>15} : {cnt:6d} ({cnt/total:.2%})")
    if tail_lengths:
        print("Tail length chars: mean={:.2f} median={} p90={} max={}".format(
            stats.mean(tail_lengths), int(stats.median(tail_lengths)),
            sorted(tail_lengths)[int(0.90*len(tail_lengths))-1], max(tail_lengths)))

def load_lines(limit=8000):
    lines = []
    for f in glob.glob('data/pretrain/bnc/**/*.md', recursive=True):
        with open(f,'r',encoding='utf-8') as h:
            for line in h:
                line=line.strip()
                if line:
                    # preserve injected wrappers so we can test skipping
                    lines.append(f"[CLS] {line} [SEP]")
                if len(lines) >= limit: break
        if len(lines)>=limit: break
    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-samples', type=int, default=5000)
    ap.add_argument('--function-words', type=str, default=None,
                    help='Comma separated override list of function words to monitor.')
    ap.add_argument('--top-k', type=int, default=15, help='Top tail tokens to display')
    args = ap.parse_args()
    if args.function_words:
        fw = {w.strip().lower() for w in args.function_words.split(',') if w.strip()}
    else:
        fw = None
    lines = load_lines()
    analyze(lines, max_samples=args.max_samples, function_words=fw, top_k=args.top_k)

if __name__ == '__main__':
    main()