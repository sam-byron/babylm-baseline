#!/usr/bin/env python3
"""
Debug the masking rate issue by examining token counts and masking logic.
"""

import torch
import numpy as np
import random
from tokenizers import Tokenizer
from dynamic_collator import create_dynamic_collator


def debug_masking_rates():
    """Debug why masking rates are higher than expected."""
    print("🔍 DEBUGGING MASKING RATES")
    print("=" * 50)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file("./data/pretrain/wordpiece_vocab.json")
    
    # Configuration
    config = {
        "masking_strategy": "span",
        "mask_p": 0.15,
        "random_p": 0.1,
        "keep_p": 0.1,
        "geometric_p": 0.2,
    }
    
    # Create collator
    collator = create_dynamic_collator(config, tokenizer)
    
    # Test text
    test_text = "The quick brown fox jumps over the lazy dog and runs through the forest."
    tokens = tokenizer.encode(test_text).ids
    cls_token_id = tokenizer.token_to_id("[CLS]") or 1
    sep_token_id = tokenizer.token_to_id("[SEP]") or 2
    test_seq = [cls_token_id] + tokens + [sep_token_id]
    
    print(f"📊 Token Analysis:")
    print(f"   Original text: {test_text}")
    print(f"   Total tokens: {len(test_seq)}")
    print(f"   Special token IDs: {collator.special_token_ids}")
    print(f"   CLS token: {cls_token_id}, SEP token: {sep_token_id}")
    
    # Analyze what tokens are considered maskable
    special_count = sum(1 for t in test_seq if t in collator.special_token_ids)
    maskable_count = len(test_seq) - special_count
    
    print(f"   Special tokens in sequence: {special_count}")
    print(f"   Maskable tokens: {maskable_count}")
    print(f"   Expected to mask (15%): {maskable_count * 0.15:.2f}")
    
    # Show token details
    print(f"\n🔤 Token Details:")
    for i, token_id in enumerate(test_seq):
        token_text = tokenizer.id_to_token(token_id)
        is_special = token_id in collator.special_token_ids
        print(f"   {i:2d}: {token_id:5d} '{token_text}' {'[SPECIAL]' if is_special else '[MASKABLE]'}")
    
    # Test masking multiple times
    print(f"\n🎭 Masking Behavior Analysis:")
    for round_num in range(5):
        result = collator([test_seq])
        
        input_ids = result['input_ids'][0]
        labels = result['labels'][0]
        attention_mask = result['attention_mask'][0]
        
        # Count tokens
        total_tokens = torch.sum(attention_mask == 1).item()
        special_in_result = sum(1 for t in input_ids[:total_tokens] if t.item() in collator.special_token_ids)
        maskable_in_result = total_tokens - special_in_result
        masked_positions = labels != -100
        num_masked = torch.sum(masked_positions).item()
        
        print(f"\n   Round {round_num + 1}:")
        print(f"     Total tokens: {total_tokens}")
        print(f"     Special tokens: {special_in_result}")
        print(f"     Maskable tokens: {maskable_in_result}")
        print(f"     Actually masked: {num_masked}")
        print(f"     Masking rate: {num_masked/maskable_in_result*100:.2f}%")
        
        # Show what was masked
        masked_details = []
        for i, (inp, lbl) in enumerate(zip(input_ids, labels)):
            if lbl != -100:  # This position was masked
                original = tokenizer.id_to_token(lbl.item())
                current = tokenizer.id_to_token(inp.item())
                masked_details.append(f"{original}→{current}")
        
        print(f"     Masked tokens: {masked_details}")
        
        # Exact calculation check
        exact_target = maskable_in_result * 0.15
        print(f"     Expected (15% of {maskable_in_result}): {exact_target:.2f}")
        print(f"     Difference: {num_masked - exact_target:.2f}")


if __name__ == "__main__":
    debug_masking_rates()
