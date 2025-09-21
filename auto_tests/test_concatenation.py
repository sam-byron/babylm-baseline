#!/usr/bin/env python3


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import json
from pathlib import Path
import glob
from data_loader import ChunkedDataset
from tokenizers import Tokenizer

def test_concatenation_logic():
    """Test the new sequence concatenation logic in ChunkedDataset."""
    print("🔗 TESTING NEW CONCATENATION LOGIC")
    print("=" * 60)
    
    # Load config and tokenizer
    with open("model_babylm_ltg_bert.json", "r") as f:
        config = json.load(f)
    
    tokenizer = Tokenizer.from_file("data/pretrain/wordpiece_vocab.json")
    
    # Get chunk files for testing - use more for better statistics
    all_chunk_paths = sorted(glob.glob("model_babylm_bert_ltg/chunk*.pt"))
    chunk_paths = all_chunk_paths[:50]  # Test with first 50 chunks for better statistics
    print(f"Testing with {len(chunk_paths)} chunks (out of {len(all_chunk_paths)} total)")
    
    # Create ChunkedDataset
    block_size = config["block_size"]
    pad_id = tokenizer.token_to_id("[PAD]")
    
    print(f"\n📊 Creating ChunkedDataset with block_size={block_size}")
    dataset = ChunkedDataset(chunk_paths, block_size=block_size, tokenizer=tokenizer, pad_token_id=pad_id)
    
    # Test a few samples
    print(f"\n🎯 Dataset Statistics:")
    print(f"  Total blocks: {len(dataset):,}")
    print(f"  Block lengths: {len(dataset.block_lengths):,}")
    
    # Analyze block length distribution
    full_blocks = sum(1 for length in dataset.block_lengths if length == block_size)
    partial_blocks = len(dataset.block_lengths) - full_blocks
    avg_length = sum(dataset.block_lengths) / len(dataset.block_lengths) if dataset.block_lengths else 0
    
    print(f"  Full blocks (512 tokens): {full_blocks:,} ({full_blocks/len(dataset)*100:.1f}%)")
    print(f"  Partial blocks: {partial_blocks:,} ({partial_blocks/len(dataset)*100:.1f}%)")
    print(f"  Average block length: {avg_length:.1f} tokens")
    
    # Test individual samples
    print(f"\n🔍 Testing individual samples:")
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        original_length = dataset.block_lengths[i]
        pad_tokens = (sample == pad_id).sum().item()
        real_tokens = block_size - pad_tokens
        
        print(f"  Sample {i:2d}: length={len(sample):3d}, real_tokens={real_tokens:3d}, "
              f"pad_tokens={pad_tokens:3d} ({pad_tokens/block_size*100:.1f}%), "
              f"original_length={original_length:3d}")
        
        # Verify the sample is properly padded
        if len(sample) != block_size:
            print(f"    ❌ ERROR: Sample length {len(sample)} != block_size {block_size}")
        
        # Check if padding is correct
        expected_pad = max(0, block_size - original_length)
        if pad_tokens != expected_pad:
            print(f"    ❌ ERROR: Expected {expected_pad} pad tokens, got {pad_tokens}")
        else:
            print(f"    ✅ Padding correct")
    
    # Efficiency analysis
    total_real_tokens = sum(dataset.block_lengths)
    total_possible_tokens = len(dataset) * block_size
    efficiency = total_real_tokens / total_possible_tokens * 100 if total_possible_tokens > 0 else 0
    
    print(f"\n📈 Efficiency Analysis:")
    print(f"  Total real tokens: {total_real_tokens:,}")
    print(f"  Total possible tokens: {total_possible_tokens:,}")
    print(f"  Token efficiency: {efficiency:.1f}%")
    
    # Compare with expected efficiency (should be much better than 14% we had before)
    if efficiency > 80:
        print(f"  ✅ EXCELLENT: Token efficiency > 80%")
    elif efficiency > 60:
        print(f"  ✅ GOOD: Token efficiency > 60%")
    elif efficiency > 40:
        print(f"  ⚠️  FAIR: Token efficiency > 40%")
    else:
        print(f"  ❌ POOR: Token efficiency < 40%")
    
    # Test multiple batches for masking statistics
    print(f"\n🎭 Testing with dynamic masking (multiple batches):")
    from dynamic_collator import create_dynamic_collator
    
    # Create collator
    collate_fn = create_dynamic_collator(config, tokenizer)
    
    # Test multiple small batches to get better masking statistics
    num_test_batches = 10
    batch_size = 8
    masking_rates = []
    
    mask_token_id = tokenizer.token_to_id("[MASK]")
    
    for batch_idx in range(num_test_batches):
        start_idx = batch_idx * batch_size
        if start_idx + batch_size > len(dataset):
            break
            
        batch_samples = [dataset[start_idx + i] for i in range(batch_size)]
        batch = collate_fn(batch_samples)
        
        # Analyze masking for this batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        for sample_idx in range(input_ids.shape[0]):
            sample_input_ids = input_ids[sample_idx]
            sample_attention_mask = attention_mask[sample_idx]
            
            masked_positions = (sample_input_ids == mask_token_id).sum().item()
            real_tokens = sample_attention_mask.sum().item()
            masking_rate = masked_positions / real_tokens * 100 if real_tokens > 0 else 0
            masking_rates.append(masking_rate)
    
    # Statistics
    if masking_rates:
        avg_masking_rate = sum(masking_rates) / len(masking_rates)
        min_masking_rate = min(masking_rates)
        max_masking_rate = max(masking_rates)
        
        print(f"  Tested {len(masking_rates)} samples across {num_test_batches} batches")
        print(f"  Average masking rate: {avg_masking_rate:.2f}%")
        print(f"  Min masking rate: {min_masking_rate:.2f}%")
        print(f"  Max masking rate: {max_masking_rate:.2f}%")
        print(f"  Expected range: 12-18%")
        
        if 12 <= avg_masking_rate <= 18:
            print(f"  ✅ Average masking rate within expected range")
        else:
            print(f"  ⚠️  Average masking rate outside expected range")
            
        # Show distribution
        in_range = sum(1 for rate in masking_rates if 12 <= rate <= 18)
        print(f"  Samples in range: {in_range}/{len(masking_rates)} ({in_range/len(masking_rates)*100:.1f}%)")
    else:
        print("  ❌ No samples tested")
    
    print(f"\n🎉 Concatenation test completed!")
    return dataset

if __name__ == "__main__":
    test_concatenation_logic()
