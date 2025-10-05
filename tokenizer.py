"""
tokenizer.py — Train a WordPiece tokenizer with extended special tokens

Overview
    Initializes and trains a WordPiece tokenizer with additional structural tokens
    such as [DOC], [EOD], and [SPK]. Provides an F95 utility to compute the token
    frequency threshold covering 95% of token occurrences.

Usage
    python tokenizer.py --data-dir data/pretrain/bnc --vocab_path data/pretrain/bpe.json \
                                            --vocab_size 16384 --min_frequency 10

Innovations & efficiency
    - ByteLevel pre-tokenization and decoding for robust raw text handling.
    - Reproducibility manifest (sha256, args) saved alongside the tokenizer.
    - Lightweight streaming over .md files to avoid loading entire corpora in memory.
"""
import argparse
from collections import Counter
import glob
import os
import hashlib
import json
from datetime import datetime

from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers import Tokenizer, pre_tokenizers, decoders, processors


def initialize_tokenizer(args):
    # Extended special tokens (fresh start): added document, end-of-document, speaker markers
    special_tokens = [
        "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[PAR]", "[TAB]",
        "[DOC]", "[EOD]", "[SPK]"
    ]

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=True),
        pre_tokenizers.Digits(individual_digits=True)
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = WordPieceTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        min_frequency=args.min_frequency,
        continuing_subword_prefix=''
    )

    return tokenizer, trainer


def calculate_f95(tokenizer, line_iter):
    """Compute the 95% cumulative frequency threshold token count.

    Builds a frequency table of emitted tokens (post-tokenization) by streaming
    an iterator of lines. Returns the frequency value at the 95th percentile and
    the full sorted frequency list.
    """
    counter = Counter()
    total_lines = 0
    for sentence in line_iter:
        sentence = sentence.strip()
        if not sentence:
            continue
        counter.update(tokenizer.encode(sentence).tokens)
        total_lines += 1
    sorted_subwords = counter.most_common()
    print(f"[F95] Processed {total_lines} lines. Unique subwords encountered: {len(sorted_subwords)}")
    if not sorted_subwords:
        return 0, []
    print("100 most common subwords:\n" + '\n'.join(str(x) for x in sorted_subwords[:100]) + '\n')
    subword95 = sorted_subwords[len(sorted_subwords) * 95 // 100]
    return subword95[1], sorted_subwords


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT sharding')
    # Deprecated: --input_path retained for backward compatibility; prefer --data-dir
    parser.add_argument('--input_path', type=str, default=None, help='(Deprecated) Single input filename; use --data-dir instead')
    parser.add_argument('--data-dir', type=str, default='data/pretrain/bnc', help='Root directory containing .md files recursively')
    parser.add_argument('--vocab_path', type=str, default="data/pretrain/bpe.json", help='Specify the output filename')
    parser.add_argument('--vocab_size', type=int, default=2**14, help='Number of subwords in the trained tokenizer')
    parser.add_argument('--min_frequency', type=int, default=10, help='Minimal number of occurences of every candidate subword')
    parser.add_argument('--max-lines', type=int, default=None, help='Optional cap on number of lines for faster experimentation')
    parser.add_argument('--shuffle-files', action='store_true', help='Shuffle file order before streaming (may impact reproducibility)')
    args = parser.parse_args()

    print(f"Initializing a WordPiece tokenizer", flush=True)
    tokenizer, trainer = initialize_tokenizer(args)

    print("Training the tokenizer (streaming all .md files)", flush=True)

    def iter_all_md_lines(root_dir: str):
        pattern = os.path.join(root_dir, '**', '*.md')
        files = glob.glob(pattern, recursive=True)
        if not files:
            raise SystemExit(f"No .md files found under {root_dir}")
        if args.shuffle_files:
            import random
            random.Random(13).shuffle(files)  # deterministic shuffle
        line_count = 0
        for fp in files:
            try:
                with open(fp, 'r', encoding='utf-8') as fh:
                    for raw in fh:
                        line = raw.strip()
                        if not line:
                            continue
                        # Remove explicit [TAB] token markers if present in raw data
                        line = line.replace('[TAB] ', '').strip()
                        if not line:
                            continue
                        yield line
                        line_count += 1
                        if args.max_lines and line_count >= args.max_lines:
                            return
            except Exception as e:
                print(f"[Warn] Failed reading {fp}: {e}")

    # Choose data source: data-dir preferred; fallback to single file if provided and no data-dir
    data_source_desc = f"directory {args.data_dir}" if args.data_dir else f"file {args.input_path}"
    print(f"[Data] Streaming from {data_source_desc}")
    def single_file_iterator(file_path: str):
        with open(file_path, 'r', encoding='utf-8') as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                yield line.replace('[TAB] ', '').strip()
    line_iter = iter_all_md_lines(args.data_dir) if args.data_dir else single_file_iterator(args.input_path)

    tokenizer.train_from_iterator(line_iter, trainer)

    print("Saving the tokenizer", flush=True)
    tokenizer.save(args.vocab_path)

    # Produce a reproducibility manifest (hash + args + special tokens)
    try:
        with open(args.vocab_path, 'rb') as f_tok:
            tok_bytes = f_tok.read()
        sha256 = hashlib.sha256(tok_bytes).hexdigest()
        manifest = {
            "created_utc": datetime.utcnow().isoformat() + 'Z',
            "vocab_path": args.vocab_path,
            "data_dir": args.data_dir,
            "input_path": args.input_path,
            "vocab_size_requested": args.vocab_size,
            "min_frequency": args.min_frequency,
            "max_lines": args.max_lines,
            "shuffle_files": args.shuffle_files,
            "special_tokens": list(tokenizer.get_vocab().keys())[:20],  # preview only
            "sha256": sha256
        }
        with open(args.vocab_path + '.manifest.json', 'w') as f_man:
            json.dump(manifest, f_man, indent=2)
        print(f"Wrote manifest {args.vocab_path + '.manifest.json'} (sha256={sha256[:12]}...)")
    except Exception as e:
        print(f"Warning: failed to write tokenizer manifest: {e}")

    print("TEST")
    print("Trying to load the tokenizer...")
    tokenizer = Tokenizer.from_file(args.vocab_path)
    print("Success!")

    # Recompute F95 over the same corpus (may be expensive)
    print("Computing F95 over corpus (second streaming pass)...", flush=True)
    f95_value, freq_list = calculate_f95(tokenizer, iter_all_md_lines(args.data_dir))
    with open(args.vocab_path + '_freqs', 'w') as f_freq:
        for subword, freq in freq_list:
            f_freq.write(f"{subword}: {freq}\n")
    print(f"F_{95}% is {f95_value}\n")

    print("Samples from the tokenizer:")

    def test(tokenizer, text):
        subwords = tokenizer.encode(text).tokens
        return ' '.join(subwords)

    texts = [
        """One of the most impressive long term hobby projects is Robert's Rocket Project. He started building a 100 lbf liquid engine in 2001, fired a regeneratively cooled version in 2007, started building a regen 250 lbf in 2008.""",
        """what are examples of interfaces that allow you to manage sets of queries (SQL, splunk, lucene/elastic, xpath, whatever other language)?""",
        """### Increasingly seeing a big schism between what I think my research is & what others think it is. I don't do qualitative work and I'm not trained in anthro or theories of race or gender. I can't supervise students with these interests! I'm a sociophonetician who works on prosody!""",
        """The Northern Lights season is here... Taking these pictures is an art itself and requires preparation, so The Local spoke to an expert to find out how to take awe-inspiring snaps of the Northern Lights.""",
        """Some people have SOTA facial recognition abilities: "At the very upper end of the performance scale, a cohort of just 1-2% of the population are 'super-recognisers'-people who can memorise and recall unfamiliar faces, even after the briefest glimpse.\""""
    ]

    for text in texts:
        print(f"INPUT:  {text}\nTOKENS: {test(tokenizer, text)}\n", flush=True)
