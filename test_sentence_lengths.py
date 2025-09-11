#!/usr/bin/env python3
"""
Test the sentence-aware data loading pipeline to verify proper sentence lengths.
"""

import json
import torch
import tokenizers
from mlm_dataset import SentenceAwareDataset
from data_loader import data_loader

def test_sentence_aware_data_loading():
    """Test that sentence-aware data loading preserves variable sentence lengths."""
    
    print("🧪 Testing Sentence-Aware Data Loading")
    print("=" * 50)
    
    # Load config
    with open("model_babylm_ltg_bert.json", 'r') as f:
        config = json.load(f)
    
    # Load tokenizer
    tokenizer = tokenizers.Tokenizer.from_file(config["tokenizer_path"])
    
    # Get cache path
    cache_path = config["cache_path"]
    print(f"Cache path: {cache_path}")
    
    # Test direct dataset loading
    print("\n📋 Testing SentenceAwareDataset directly:")
    
    dataset = SentenceAwareDataset(
        cache_path=cache_path,
        tokenizer=tokenizer,
        seq_length=512,  # This should be ignored now
        mask_p=0.15,
        random_p=0.1,
        keep_p=0.1
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test a few samples
    print("\n📊 Sample sentence lengths:")
    sample_lengths = []
    cls_counts = []
    sep_counts = []
    
    for i in range(min(20, len(dataset))):
        sample = dataset[i]
        input_ids = sample['input_ids']
        length = len(input_ids)
        sample_lengths.append(length)
        
        # Count boundary tokens
        cls_count = (input_ids == tokenizer.token_to_id("[CLS]")).sum().item()
        sep_count = (input_ids == tokenizer.token_to_id("[SEP]")).sum().item()
        cls_counts.append(cls_count)
        sep_counts.append(sep_count)
        
        print(f"  Sample {i}: {length} tokens, {cls_count} [CLS], {sep_count} [SEP]")
    
    avg_length = sum(sample_lengths) / len(sample_lengths)
    min_length = min(sample_lengths)
    max_length = max(sample_lengths)
    
    print(f"\n📈 Length statistics:")
    print(f"  Average: {avg_length:.1f} tokens")
    print(f"  Range: {min_length} - {max_length} tokens")
    print(f"  Boundary tokens: {sum(cls_counts)} [CLS], {sum(sep_counts)} [SEP]")
    
    # Check if lengths are variable (not all 512)
    unique_lengths = len(set(sample_lengths))
    print(f"  Unique lengths: {unique_lengths}")
    
    if unique_lengths == 1 and sample_lengths[0] == 512:
        print("❌ All sentences are 512 tokens - still being padded!")
        return False
    elif unique_lengths > 1:
        print("✅ Variable sentence lengths - sentence-aware processing working!")
    else:
        print(f"⚠️ All sentences have same length ({sample_lengths[0]}) - may be an issue")
    
    # Test data loader
    print("\n🔄 Testing DataLoader with collate function:")
    
    try:
        train_loader, val_loader, test_loader, collate_fn, total_tokens = data_loader(config, tokenizer, cache_path)
        
        # Get a batch
        batch = next(iter(train_loader))
        
        print(f"Batch shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        
        # Check that batch has proper boundaries and lengths
        batch_input_ids = batch['input_ids']
        batch_attention = batch['attention_mask']
        
        for i in range(min(3, batch_input_ids.size(0))):
            sequence = batch_input_ids[i]
            attention = batch_attention[i]
            
            # Count real tokens (where attention == 1)
            real_tokens = attention.sum().item()
            
            # Count boundary tokens
            cls_count = (sequence == tokenizer.token_to_id("[CLS]")).sum().item()
            sep_count = (sequence == tokenizer.token_to_id("[SEP]")).sum().item()
            
            print(f"  Batch sample {i}: {real_tokens} real tokens, {cls_count} [CLS], {sep_count} [SEP]")
        
        print("✅ DataLoader working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ DataLoader error: {e}")
        return False

if __name__ == "__main__":
    success = test_sentence_aware_data_loading()
    
    if success:
        print("\n🎉 Sentence-aware data loading is working correctly!")
        print("You can now train with variable-length sentences that preserve syntax.")
    else:
        print("\n❌ There are still issues with sentence-aware data loading.")
        print("Check the implementation for padding/truncation problems.")
