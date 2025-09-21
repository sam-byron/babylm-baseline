#!/usr/bin/env python3
"""
Simple runner script to test the data loader with your current configuration.
"""


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_basic_tests():
    """Run basic tests without heavy dependencies."""
    try:
        from test_data_loader import DataLoaderTester
        
        # Use your existing config
        config_path = "model_babylm_ltg_bert.json"
        
        print("🧪 Starting basic data loader tests...")
        tester = DataLoaderTester(config_path, num_test_batches=5)
        
        # Run individual test components
        print("\n1. Setting up tokenizer...")
        tester.setup_tokenizer()
        
        print("\n2. Testing ChunkedDataset...")
        dataset = tester.test_chunked_dataset()
        
        print("\n3. Testing TokenBudgetBatchSampler...")
        tester.test_token_budget_batch_sampler(dataset)
        
        print("\n4. Testing masking strategies...")
        tester.test_masking_strategies()
        
        print("\n5. Testing data loader batches...")
        train_loader, batch_stats = tester.test_data_loader_batches()
        
        print("\n6. Testing batch consistency...")
        tester.test_batch_consistency(train_loader, num_tests=3)
        
        print("\n✅ Basic tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_quick_batch_analysis():
    """Quick analysis of a few batches."""
    try:
        import json
        import torch
        import random
        from data_loader import data_loader
        from tokenizers import Tokenizer
        
        # Load config
        with open("model_babylm_ltg_bert.json", "r") as f:
            config = json.load(f)
        
        # Load tokenizer
        tokenizer = Tokenizer.from_file(config["tokenizer_path"])
        
        print("🔍 Quick batch analysis...")
        
        # Create data loader
        train_loader, val_loader, test_loader, collate_fn, total_tokens = data_loader(
            config, tokenizer, config["cache_path"]
        )
        
        print(f"📊 Data loader created:")
        print(f"  - Total training tokens: {total_tokens:,}")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Validation batches: {len(val_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        
        # Analyze a few random batches
        print(f"\n🎯 Analyzing 3 random training batches:")
        
        # Sample 3 random batches
        data_iter = iter(train_loader)
        batches = []
        for i, batch in enumerate(data_iter):
            batches.append(batch)
            if i >= 10:  # Collect first 10 batches
                break
        
        # Randomly sample 3
        sample_batches = random.sample(batches, min(3, len(batches)))
        
        for i, batch in enumerate(sample_batches):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch.get('labels')
            
            batch_size, seq_len = input_ids.shape
            real_tokens = torch.sum(attention_mask == 1).item()
            pad_tokens = torch.sum(input_ids == tokenizer.token_to_id("[PAD]")).item()
            
            print(f"  Batch {i+1}:")
            print(f"    - Shape: {input_ids.shape}")
            print(f"    - Real tokens: {real_tokens:,}")
            print(f"    - Padding tokens: {pad_tokens:,}")
            print(f"    - Padding ratio: {pad_tokens/(batch_size*seq_len)*100:.1f}%")
            
            if labels is not None:
                masked_tokens = torch.sum(labels != -100).item()
                print(f"    - Masked tokens: {masked_tokens:,}")
                print(f"    - Masking rate: {masked_tokens/real_tokens*100:.1f}%")
            
            # Show a few example tokens
            first_seq = input_ids[0]
            non_pad_mask = first_seq != tokenizer.token_to_id("[PAD]")
            non_pad_tokens = first_seq[non_pad_mask][:10]  # First 10 non-pad tokens
            
            token_texts = [tokenizer.id_to_token(tid.item()) for tid in non_pad_tokens]
            print(f"    - First 10 tokens: {token_texts}")
        
        print("\n✅ Quick analysis completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Quick analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run data loader tests")
    parser.add_argument("--mode", choices=["basic", "quick"], default="quick", 
                       help="Test mode: 'basic' for full tests, 'quick' for fast analysis")
    
    args = parser.parse_args()
    
    if args.mode == "basic":
        success = run_basic_tests()
    else:
        success = run_quick_batch_analysis()
    
    sys.exit(0 if success else 1)
