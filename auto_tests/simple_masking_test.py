#!/usr/bin/env python3


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import json
from tokenizers import Tokenizer

def simple_collator_test():
    """Simple test to identify the key difference between dynamic and static masking."""
    print("🔍 SIMPLE DYNAMIC VS STATIC TEST")
    print("=" * 50)
    
    # Load configuration
    with open("model_babylm_ltg_bert.json", "r") as f:
        config = json.load(f)
    
    # Test what happens when we switch the flag
    print(f"Current config use_dynamic_masking: {config.get('use_dynamic_masking', False)}")
    
    # Create a simple test sequence
    tokenizer = Tokenizer.from_file("data/pretrain/wordpiece_vocab.json")
    
    # Simple sequence: [CLS] word1 word2 word3 ... [SEP] [PAD] [PAD] ...
    test_sequence = (
        [tokenizer.token_to_id("[CLS]")] +
        list(range(100, 110)) +  # 10 content words
        [tokenizer.token_to_id("[SEP]")] +
        [tokenizer.token_to_id("[PAD]")] * 500  # padding
    )
    
    test_tensor = torch.tensor(test_sequence, dtype=torch.long)
    print(f"Test sequence length: {len(test_sequence)}")
    print(f"Content tokens: 10 (positions 1-10)")
    
    # Test both approaches manually
    print(f"\\n📊 MANUAL MASKING COMPARISON:")
    
    # 1. Try importing dynamic collator
    try:
        from dynamic_collator import create_dynamic_collator
        dynamic_collator = create_dynamic_collator(config, tokenizer)
        print(f"✅ Dynamic collator imported successfully")
        
        # Test it
        try:
            batch = dynamic_collator([test_tensor])
            d_labels = batch['labels'][0]
            d_input = batch['input_ids'][0]
            d_attention = batch['attention_mask'][0]
            
            d_masked = (d_labels != -100).sum().item()
            d_mask_tokens = (d_input == tokenizer.token_to_id('[MASK]')).sum().item()
            
            print(f"Dynamic result: {d_masked} masked ({d_mask_tokens} [MASK])")
        except Exception as e:
            print(f"❌ Dynamic collator failed: {e}")
    except Exception as e:
        print(f"❌ Could not import dynamic collator: {e}")
    
    # 2. Try static approach
    try:
        from mlm_dataset import SpanMaskingStrategy
        
        static_strategy = SpanMaskingStrategy(
            mask_p=config.get("mask_p", 0.15),
            tokenizer=tokenizer,
            n_special_tokens=6,
            random_p=config.get("random_p", 0.1),
            keep_p=config.get("keep_p", 0.1)
        )
        
        print(f"✅ Static masking strategy created successfully")
        
        # Test it
        input_ids = test_tensor.clone()
        input_ids, labels = static_strategy(input_ids)
        
        s_masked = (labels != -100).sum().item()
        s_mask_tokens = (input_ids == tokenizer.token_to_id('[MASK]')).sum().item()
        
        print(f"Static result: {s_masked} masked ({s_mask_tokens} [MASK])")
        
    except Exception as e:
        print(f"❌ Static masking failed: {e}")
    
    # Most likely hypothesis based on your observation
    print(f"\\n💡 HYPOTHESIS:")
    print(f"If dynamic masking causes loss plateau but static doesn't:")
    print(f"1. 🐌 Dynamic masking is significantly slower (confirmed)")
    print(f"2. 🎭 Dynamic masking has different masking patterns")  
    print(f"3. 🔧 Dynamic masking has a bug in implementation")
    print(f"4. ⚡ Dynamic masking uses different data loading path")
    print(f"\\n🎯 Your logic is sound: if masking is deterministic,")
    print(f"   performance should be similar. The issue is likely in:")
    print(f"   - Implementation bug")
    print(f"   - Different data flow")
    print(f"   - Computational overhead affecting gradient updates")

if __name__ == "__main__":
    simple_collator_test()
