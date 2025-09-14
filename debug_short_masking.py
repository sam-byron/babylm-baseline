#!/usr/bin/env python3

import torch
import json
from tokenizers import Tokenizer
from dynamic_collator import create_dynamic_collator

def test_specific_short_sequence():
    """Test masking on a specifically crafted short sequence."""
    print("🔍 DEBUGGING SHORT SEQUENCE MASKING")
    print("=" * 50)
    
    # Setup
    with open("model_babylm_ltg_bert.json", "r") as f:
        config = json.load(f)
    
    tokenizer = Tokenizer.from_file("data/pretrain/wordpiece_vocab.json")
    collate_fn = create_dynamic_collator(config, tokenizer)
    
    # Create a very short sequence: [CLS] "The cat sat" [SEP]
    # This should have only 3 content tokens
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    
    # Get some word tokens - find them in the vocabulary
    # Let's use simple tokens that should exist
    the_id = tokenizer.encode("The").ids[0] if tokenizer.encode("The").ids else 100
    cat_id = tokenizer.encode("cat").ids[0] if tokenizer.encode("cat").ids else 101
    sat_id = tokenizer.encode("sat").ids[0] if tokenizer.encode("sat").ids else 102
    
    tokens = [cls_id, the_id, cat_id, sat_id, sep_id]
    
    print(f"Test sequence: {tokens}")
    print(f"Content tokens: 3 (positions 1, 2, 3)")
    print(f"Expected masking with new logic: 0 tokens (sequence too short)")
    
    # Convert to tensor
    input_tensor = torch.tensor([tokens], dtype=torch.long)
    
    # Test masking 10 times to see distribution
    mask_counts = []
    for i in range(10):
        batch = collate_fn([torch.tensor(tokens, dtype=torch.long)])
        labels = batch['labels'][0]
        input_ids = batch['input_ids'][0]
        
        # Count masked positions
        masked_positions = (labels != -100).sum().item()
        mask_token_count = (input_ids == tokenizer.token_to_id("[MASK]")).sum().item()
        
        mask_counts.append(masked_positions)
        
        if i < 3:  # Show first 3 examples
            print(f"\\nTrial {i+1}:")
            print(f"  Input:  {input_ids[:5].tolist()}")
            print(f"  Labels: {labels[:5].tolist()}")
            print(f"  Masked positions: {masked_positions}")
            print(f"  [MASK] tokens: {mask_token_count}")
            print(f"  Masking rate: {masked_positions/3*100:.1f}%")
    
    avg_masked = sum(mask_counts) / len(mask_counts)
    print(f"\\n📊 Average masked tokens: {avg_masked:.1f}")
    print(f"📊 Average masking rate: {avg_masked/3*100:.1f}%")
    
    if avg_masked < 0.5:
        print("✅ Short sequence masking working correctly!")
    else:
        print("❌ Short sequences still being over-masked")

if __name__ == "__main__":
    test_specific_short_sequence()
