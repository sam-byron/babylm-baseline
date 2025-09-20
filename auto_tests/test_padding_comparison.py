#!/usr/bin/env python3
"""
Test dynamic padding approach where sequences keep natural lengths
and are only padded to the batch maximum
"""

import torch
from tokenizers import Tokenizer
from dynamic_collator import create_dynamic_collator
import json

def test_dynamic_padding_approach():
    print("🧪 TESTING DYNAMIC PADDING APPROACH")
    print("=" * 60)
    
    # Load config and tokenizer
    with open("model_babylm_ltg_bert_FIXED.json", "r") as f:
        config = json.load(f)
        
    tokenizer = Tokenizer.from_file(config.get("tokenizer_path"))
    tokenizer.model_max_length = config.get("max_position_embeddings", 512)
    tokenizer.enable_truncation(max_length=tokenizer.model_max_length)
    
    # Load raw sequences directly from chunk (no ChunkedDataset padding)
    chunk = torch.load('model_babylm_bert_ltg/chunk184.pt', map_location='cpu')
    
    print(f"📊 Loaded {len(chunk)} raw sequences")
    
    # Test with different batch sizes
    batch_sizes = [3, 5, 8]
    
    for batch_size in batch_sizes:
        print(f"\\n🎭 Testing batch size {batch_size}")
        
        # Get natural length sequences
        raw_sequences = chunk[:batch_size]
        natural_lengths = [len(seq) for seq in raw_sequences]
        
        print(f"  Natural lengths: {natural_lengths}")
        print(f"  Max length: {max(natural_lengths)}")
        print(f"  Min length: {min(natural_lengths)}")
        
        # Convert to tensors for the collator
        batch = [torch.tensor(seq, dtype=torch.long) for seq in raw_sequences]
        
        # Re-enable dynamic padding in collator for this test
        # We need to temporarily modify the collator to handle padding
        
        print(f"  💭 With dynamic padding, batch would be padded to {max(natural_lengths)} tokens")
        
        # Calculate efficiency
        total_natural_tokens = sum(natural_lengths)
        total_padded_tokens = max(natural_lengths) * batch_size
        efficiency = total_natural_tokens / total_padded_tokens
        
        print(f"  📈 Efficiency: {total_natural_tokens}/{total_padded_tokens} = {efficiency:.1%}")
        
        # Calculate what percentage would be maskable
        maskable_counts = []
        for seq in raw_sequences:
            seq_tensor = torch.tensor(seq)
            maskable = (seq_tensor > 4).sum().item()
            maskable_counts.append(maskable)
        
        total_maskable = sum(maskable_counts)
        maskable_ratio = total_maskable / total_natural_tokens
        
        print(f"  🎯 Maskable tokens: {total_maskable}/{total_natural_tokens} = {maskable_ratio:.1%}")
        
        if efficiency > 0.5:
            print(f"    ✅ Good efficiency (>{50}%)")
        else:
            print(f"    ⚠️  Low efficiency (<{50}%)")

def compare_approaches():
    print(f"\\n" + "="*70)
    print("📊 COMPARISON: FIXED vs DYNAMIC PADDING")
    print("="*70)
    
    chunk = torch.load('model_babylm_bert_ltg/chunk184.pt', map_location='cpu')
    
    # Analyze 100 sequences
    sample_sequences = chunk[:100]
    lengths = [len(seq) for seq in sample_sequences]
    
    print(f"\\n🔍 Analysis of 100 sequences:")
    print(f"  Natural lengths: {min(lengths)}-{max(lengths)} (avg: {sum(lengths)/len(lengths):.1f})")
    
    # Fixed padding to 512
    total_natural = sum(lengths)
    total_fixed_512 = 512 * len(lengths)
    fixed_efficiency = total_natural / total_fixed_512
    
    print(f"\\n📌 FIXED PADDING (block_size=512):")
    print(f"  Total tokens: {total_natural} -> {total_fixed_512}")
    print(f"  Efficiency: {fixed_efficiency:.1%}")
    print(f"  Waste: {100-fixed_efficiency*100:.1f}% padding tokens")
    
    # Dynamic padding (batch max)
    batch_size = 8
    dynamic_wastes = []
    
    for i in range(0, len(sample_sequences), batch_size):
        batch_seqs = sample_sequences[i:i+batch_size]
        if len(batch_seqs) < batch_size:
            continue
            
        batch_lengths = [len(seq) for seq in batch_seqs]
        batch_max = max(batch_lengths)
        batch_natural = sum(batch_lengths)
        batch_padded = batch_max * len(batch_seqs)
        
        batch_efficiency = batch_natural / batch_padded
        dynamic_wastes.append(batch_efficiency)
    
    avg_dynamic_efficiency = sum(dynamic_wastes) / len(dynamic_wastes)
    
    print(f"\\n🔄 DYNAMIC PADDING (batch max, batch_size={batch_size}):")
    print(f"  Average efficiency: {avg_dynamic_efficiency:.1%}")
    print(f"  Average waste: {100-avg_dynamic_efficiency*100:.1f}% padding tokens")
    
    print(f"\\n🎯 RECOMMENDATION:")
    if avg_dynamic_efficiency > fixed_efficiency * 1.5:
        print(f"  ✅ Use DYNAMIC padding - {avg_dynamic_efficiency/fixed_efficiency:.1f}x more efficient")
    else:
        print(f"  ⚖️  Both approaches have similar efficiency")

if __name__ == "__main__":
    test_dynamic_padding_approach()
    compare_approaches()
