#!/usr/bin/env python3
"""
Test dynamic collator with properly tokenized (non-padded) chunks
"""


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from tokenizers import Tokenizer
from dynamic_collator import create_dynamic_collator
import json

def test_with_fixed_chunks():
    print("🧪 TESTING DYNAMIC COLLATOR WITH FIXED CHUNKS")
    print("=" * 60)
    
    # Load config and tokenizer
    with open("model_babylm_ltg_bert_FIXED.json", "r") as f:
        config = json.load(f)
        
    tokenizer = Tokenizer.from_file(config.get("tokenizer_path"))
    tokenizer.model_max_length = config.get("max_position_embeddings", 512)
    tokenizer.enable_truncation(max_length=tokenizer.model_max_length)
    
    # Create dynamic collator
    collator = create_dynamic_collator(config, tokenizer)
    
    # Load the newly regenerated chunk (with natural lengths)
    chunk = torch.load('model_babylm_bert_ltg/chunk184.pt', map_location='cpu')
    
    print(f"📊 Loaded chunk with {len(chunk)} sequences")
    print(f"Sequence lengths: {[len(seq) for seq in chunk[:10]]}...")
    
    # Test with different batch sizes
    for batch_size in [3, 5, 8]:
        print(f"\\n🎭 Testing batch size {batch_size}")
        
        # Get a batch of sequences
        batch_sequences = chunk[:batch_size]
        
        # Convert to list of tensors (what the data loader would provide)
        batch = [torch.tensor(seq, dtype=torch.long) for seq in batch_sequences]
        
        print(f"  Input lengths: {[len(seq) for seq in batch]}")
        
        # Apply dynamic masking collator
        result = collator(batch)
        
        # Analyze results
        input_ids = result['input_ids']
        attention_mask = result['attention_mask']  
        labels = result['labels']
        
        print(f"  Output batch shape: {input_ids.shape}")
        print(f"  Max length padded to: {input_ids.shape[1]}")
        
        # Count masking statistics
        total_maskable = 0
        total_masked = 0
        mask_operations = {'[MASK]': 0, 'Random': 0, 'Kept': 0}
        
        for i in range(len(batch)):
            seq_len = len(batch[i])
            original_seq = batch[i]
            masked_seq = input_ids[i]
            labels_seq = labels[i]
            
            # Count maskable tokens (> 4, non-special)
            maskable_positions = (original_seq > 4)
            maskable_count = maskable_positions.sum().item()
            
            # Count actually masked positions (where labels != -100)
            masked_positions = (labels_seq != -100)
            masked_count = masked_positions.sum().item()
            
            total_maskable += maskable_count
            total_masked += masked_count
            
            # Analyze what happened to masked tokens
            for pos in range(seq_len):
                if labels_seq[pos] != -100:  # This position was masked
                    original_token = original_seq[pos].item()
                    masked_token = masked_seq[pos].item()
                    
                    if masked_token == 4:  # [MASK]
                        mask_operations['[MASK]'] += 1
                    elif masked_token == original_token:  # Kept original
                        mask_operations['Kept'] += 1
                    else:  # Random replacement
                        mask_operations['Random'] += 1
            
            print(f"    Seq {i+1}: {maskable_count} maskable, {masked_count} masked ({100*masked_count/maskable_count:.1f}%)")
        
        # Overall statistics
        masking_rate = total_masked / total_maskable if total_maskable > 0 else 0
        total_ops = sum(mask_operations.values())
        
        print(f"  📈 Overall masking: {total_masked}/{total_maskable} = {masking_rate:.1%}")
        
        if total_ops > 0:
            mask_pct = 100 * mask_operations['[MASK]'] / total_ops
            random_pct = 100 * mask_operations['Random'] / total_ops  
            kept_pct = 100 * mask_operations['Kept'] / total_ops
            print(f"  🎭 Operations: [MASK]={mask_pct:.1f}%, Random={random_pct:.1f}%, Kept={kept_pct:.1f}%")
        
        # Check if masking rate is reasonable (should be close to 15%)
        if 12 <= masking_rate * 100 <= 18:
            print(f"  ✅ Masking rate looks good!")
        else:
            print(f"  ⚠️  Masking rate seems off (target ~15%)")

if __name__ == "__main__":
    test_with_fixed_chunks()
