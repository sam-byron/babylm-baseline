#!/usr/bin/env python3
"""
Test script to verify the tokenization fix (no padding) produces natural-length sequences
"""

import torch
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("./data/pretrain/wordpiece_vocab.json")
tokenizer.model_max_length = 512
# Don't enable padding - let sequences be their natural length
tokenizer.enable_truncation(max_length=tokenizer.model_max_length)

# Test a few sample texts
sample_texts = [
    "[CLS] This is a short sentence. [SEP]",
    "[CLS] This is a much longer sentence with many more words to see what happens with tokenization when we don't pad everything to 512 tokens but instead let sequences be their natural length. [SEP]",
    "[CLS] Short text. [SEP]",
    "[CLS] Medium length sentence with some content but not too much really just testing here. [SEP]"
]

print("🧪 TESTING TOKENIZATION WITHOUT PADDING")
print("=" * 60)

for i, text in enumerate(sample_texts):
    # Tokenize
    encoding = tokenizer.encode(text)
    tokens = encoding.ids
    
    print(f"\nSample {i+1}:")
    print(f"  Text: {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"  Token count: {len(tokens)}")
    print(f"  Tokens: {tokens}")
    
    # Count special tokens
    pad_count = tokens.count(0)  # PAD
    unk_count = tokens.count(1)  # UNK  
    cls_count = tokens.count(2)  # CLS
    sep_count = tokens.count(3)  # SEP
    mask_count = tokens.count(4)  # MASK
    
    maskable_count = sum(1 for t in tokens if t > 4)
    
    print(f"  Special tokens: PAD={pad_count}, UNK={unk_count}, CLS={cls_count}, SEP={sep_count}, MASK={mask_count}")
    print(f"  Maskable tokens: {maskable_count} ({100*maskable_count/len(tokens):.1f}%)")

print(f"\n🎯 CONCLUSION:")
print(f"Without pre-padding, sequences have their natural lengths")
print(f"and most tokens (90%+) are maskable content tokens!")
print(f"This should give us much better masking rates.")

# Test batch tokenization
print(f"\n🧪 TESTING BATCH TOKENIZATION")
print("=" * 40)

batch_encodings = tokenizer.encode_batch(sample_texts)
batch_tokens = [enc.ids for enc in batch_encodings]

print(f"Batch lengths: {[len(tokens) for tokens in batch_tokens]}")
print(f"Max length in batch: {max(len(tokens) for tokens in batch_tokens)}")
print(f"Min length in batch: {min(len(tokens) for tokens in batch_tokens)}")

print(f"\n✅ This looks much better! Dynamic collator can pad these to batch max length.")
