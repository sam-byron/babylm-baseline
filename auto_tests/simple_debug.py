#!/usr/bin/env python3
"""
Simple debug for wordpiece tokenizer behavior.
"""

from tokenizers import Tokenizer
from dynamic_collator import create_dynamic_collator

def debug_wordpiece_tokenizer():
    print("🔍 DEBUGGING WORDPIECE TOKENIZER")
    print("=" * 50)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file("./data/pretrain/wordpiece_vocab.json")
    
    # Test text
    test_text = "The quick brown fox"
    tokens = tokenizer.encode(test_text)
    
    print(f"Text: '{test_text}'")
    print(f"Token IDs: {tokens.ids}")
    print(f"Token strings: {tokens.tokens}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    
    # Check special tokens
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    print(f"\nSpecial token IDs:")
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        print(f"  {token}: {token_id}")
    
    # Create collator and check its special token detection
    config = {"masking_strategy": "span", "mask_p": 0.15, "random_p": 0.1, "keep_p": 0.1}
    collator = create_dynamic_collator(config, tokenizer)
    print(f"\nCollator special token IDs: {collator.special_token_ids}")
    print(f"Vocab size: {collator.vocab_size}")
    
    # Test sequence with CLS/SEP
    cls_token_id = tokenizer.token_to_id("[CLS]") or 101
    sep_token_id = tokenizer.token_to_id("[SEP]") or 102
    test_seq = [cls_token_id] + tokens.ids + [sep_token_id]
    
    print(f"\nFull sequence: {test_seq}")
    print(f"Length: {len(test_seq)}")
    
    # Check which tokens are considered special
    special_count = sum(1 for t in test_seq if t in collator.special_token_ids)
    maskable_count = len(test_seq) - special_count
    
    print(f"Special tokens detected: {special_count}")
    print(f"Maskable tokens: {maskable_count}")
    print(f"Expected masks (15%): {maskable_count * 0.15:.2f}")

if __name__ == "__main__":
    debug_wordpiece_tokenizer()
