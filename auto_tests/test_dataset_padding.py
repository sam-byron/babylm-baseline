#!/usr/bin/env python3
"""
Test the updated configuration where ChunkedDataset handles padding 
and dynamic_collator only handles masking
"""

import torch
from tokenizers import Tokenizer
from dynamic_collator import create_dynamic_collator
from data_loader import ChunkedDataset
import json

def test_dataset_padding_workflow():
    print("🧪 TESTING CHUNKED DATASET PADDING + DYNAMIC MASKING")
    print("=" * 70)
    
    # Load config and tokenizer
    with open("model_babylm_ltg_bert_FIXED.json", "r") as f:
        config = json.load(f)
        
    tokenizer = Tokenizer.from_file(config.get("tokenizer_path"))
    tokenizer.model_max_length = config.get("max_position_embeddings", 512)
    tokenizer.enable_truncation(max_length=tokenizer.model_max_length)
    
    # Get pad token ID
    pad_id = tokenizer.token_to_id("[PAD]")
    print(f"PAD token ID: {pad_id}")
    
    # Create ChunkedDataset (should handle padding to block_size)
    block_size = config["block_size"]  # 512
    chunk_paths = ['model_babylm_bert_ltg/chunk184.pt']  # Use our regenerated chunk
    
    dataset = ChunkedDataset(
        chunk_paths=chunk_paths, 
        block_size=block_size, 
        pad_token_id=pad_id
    )
    
    print(f"\\n📊 Dataset created with {len(dataset)} samples")
    print(f"Block size (padding target): {block_size}")
    
    # Test individual samples from dataset
    print(f"\\n🔍 Testing individual dataset samples:")
    for i in range(3):
        sample = dataset[i]
        pad_count = (sample == pad_id).sum().item()
        maskable_count = (sample > 4).sum().item()
        
        print(f"  Sample {i+1}: length={len(sample)}, PAD={pad_count} ({100*pad_count/len(sample):.1f}%), maskable={maskable_count} ({100*maskable_count/len(sample):.1f}%)")
        
        if len(sample) != block_size:
            print(f"    ❌ ERROR: Sample length {len(sample)} != block_size {block_size}")
        else:
            print(f"    ✅ Correct length (matches block_size)")
    
    # Test batch processing with dynamic collator
    print(f"\\n🎭 Testing dynamic collator with dataset samples:")
    collator = create_dynamic_collator(config, tokenizer)
    
    # Create a batch of samples from the dataset
    batch_size = 3
    batch = [dataset[i] for i in range(batch_size)]
    
    print(f"Input batch shapes: {[sample.shape for sample in batch]}")
    
    # All should be the same length since ChunkedDataset pads them
    lengths = [len(sample) for sample in batch]
    if len(set(lengths)) == 1:
        print(f"✅ All sequences have the same length: {lengths[0]}")
    else:
        print(f"❌ ERROR: Sequences have different lengths: {lengths}")
        return
    
    # Apply dynamic masking collator
    try:
        result = collator(batch)
        
        input_ids = result['input_ids']
        attention_mask = result['attention_mask']
        labels = result['labels']
        
        print(f"\\n📈 Collator results:")
        print(f"  Output batch shape: {input_ids.shape}")
        print(f"  Attention mask shape: {attention_mask.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        # Analyze masking
        total_maskable = 0
        total_masked = 0
        
        for i in range(batch_size):
            original_seq = batch[i]
            labels_seq = labels[i]
            
            # Count maskable tokens (> 4, excluding special tokens)
            maskable_count = (original_seq > 4).sum().item()
            # Count masked positions (where labels != -100)
            masked_count = (labels_seq != -100).sum().item()
            
            total_maskable += maskable_count
            total_masked += masked_count
            
            print(f"  Seq {i+1}: {maskable_count} maskable, {masked_count} masked ({100*masked_count/maskable_count:.1f}%)")
        
        overall_rate = total_masked / total_maskable if total_maskable > 0 else 0
        print(f"  📊 Overall masking rate: {overall_rate:.1%}")
        
        if 12 <= overall_rate * 100 <= 18:
            print(f"  ✅ Masking rate looks good!")
        else:
            print(f"  ⚠️  Masking rate outside expected range (12-18%)")
            
    except Exception as e:
        print(f"❌ ERROR in dynamic collator: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_padding_workflow()
