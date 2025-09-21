#!/usr/bin/env python3
"""
Test your current data loading pipeline to verify MLM masking is working correctly.
"""


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import json
import torch
from tokenizer import Tokenizer
from data_loader import data_loader

def test_current_data_pipeline():
    """Test your existing data pipeline to see what's happening"""
    print("🔍 TESTING YOUR CURRENT DATA PIPELINE")
    print("=" * 50)
    
    # Load your config
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config = json.load(f)
    
    # Load your tokenizer
    tokenizer = Tokenizer.from_file(config.get("tokenizer_path"))
    tokenizer.model_max_length = config.get("max_position_embeddings", 512)
    tokenizer.enable_padding(length=tokenizer.model_max_length)
    
    print(f"✅ Loaded tokenizer from: {config.get('tokenizer_path')}")
    print(f"✅ Masking strategy: {config.get('masking_strategy', 'span')}")
    print(f"✅ Mask probability: {config.get('mask_p', 0.15)}")
    
    try:
        # Load data using your current pipeline
        train_loader, val_loader, test_loader, collate_fn, total_tokens = data_loader(
            config, tokenizer, config["cache_path"]
        )
        print(f"✅ Data loaders created successfully")
        print(f"✅ Total training tokens: {total_tokens:,}")
        
        # Get one batch and analyze it
        print(f"\n🔍 Analyzing first training batch...")
        batch = next(iter(train_loader))
        
        print(f"Batch keys: {list(batch.keys())}")
        
        # Check input_ids
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Analyze masking statistics
        for i in range(min(2, input_ids.size(0))):  # Check first 2 samples
            sample_input = input_ids[i]
            sample_labels = labels[i]
            sample_attention = attention_mask[i]
            
            # Count real tokens (non-padding)
            real_tokens = (sample_attention == 1).sum().item()
            
            # Count masked tokens (labels != -100)
            masked_tokens = (sample_labels != -100).sum().item()
            
            # Count different token types in input
            mask_token_id = tokenizer.token_to_id("[MASK]")
            mask_count = (sample_input == mask_token_id).sum().item()
            
            print(f"\n📊 Sample {i+1} analysis:")
            print(f"   Real tokens: {real_tokens}/{len(sample_input)}")
            print(f"   Masked tokens (labels != -100): {masked_tokens}")
            print(f"   Masking ratio: {masked_tokens/real_tokens*100:.1f}%")
            print(f"   [MASK] tokens in input: {mask_count}")
            
            # Show some example masked positions
            masked_positions = torch.where(sample_labels != -100)[0][:5]  # First 5 masked positions
            print(f"   First few masked positions: {masked_positions.tolist()}")
            
            for pos in masked_positions:
                original_token = sample_labels[pos].item()
                input_token = sample_input[pos].item()
                original_word = tokenizer.id_to_token(original_token) if original_token != -100 else "N/A"
                input_word = tokenizer.id_to_token(input_token)
                print(f"     Pos {pos}: '{original_word}' → '{input_word}'")
        
        # Overall statistics
        total_real_tokens = (attention_mask == 1).sum().item()
        total_masked_tokens = (labels != -100).sum().item()
        overall_mask_ratio = total_masked_tokens / total_real_tokens * 100
        
        print(f"\n📈 BATCH STATISTICS:")
        print(f"   Total real tokens: {total_real_tokens:,}")
        print(f"   Total masked tokens: {total_masked_tokens:,}")
        print(f"   Overall masking ratio: {overall_mask_ratio:.2f}%")
        
        if overall_mask_ratio < 5:
            print(f"❌ PROBLEM: Masking ratio is too low! Should be ~15%")
            return False
        elif overall_mask_ratio > 30:
            print(f"❌ PROBLEM: Masking ratio is too high! Should be ~15%")
            return False
        else:
            print(f"✅ Masking ratio looks good!")
            return True
            
    except Exception as e:
        print(f"❌ ERROR in data pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_training_batch():
    """Debug what happens in actual training"""
    print("\n🔍 DEBUGGING TRAINING SCENARIO")
    print("=" * 50)
    
    # Simulate exactly what happens in your training
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config = json.load(f)
    
    tokenizer = Tokenizer.from_file(config.get("tokenizer_path"))
    tokenizer.model_max_length = config.get("max_position_embeddings", 512)
    tokenizer.enable_padding(length=tokenizer.model_max_length)
    
    train_loader, _, _, _, _ = data_loader(config, tokenizer, config["cache_path"])
    
    # Get a batch like your training does
    batch = next(iter(train_loader))
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    
    print(f"Batch from data loader:")
    print(f"  input_ids range: [{input_ids.min().item()}, {input_ids.max().item()}]")
    print(f"  labels range: [{labels.min().item()}, {labels.max().item()}]")
    print(f"  Labels != -100: {(labels != -100).sum().item()} / {labels.numel()}")
    
    # Test with your model
    from ltg_bert import LtgBertForMaskedLM
    from ltg_bert_config import LtgBertConfig
    
    model_config = LtgBertConfig(
        vocab_size=config['vocab_size'],
        hidden_size=config['hidden_size'],
        num_hidden_layers=config['num_hidden_layers'],
        num_attention_heads=config['num_attention_heads'],
        intermediate_size=config['intermediate_size'],
        max_position_embeddings=config.get('max_position_embeddings', config.get('block_size', 512)),
        position_bucket_size=config['position_bucket_size'],
        layer_norm_eps=config['layer_norm_eps'],
        hidden_dropout_prob=config['hidden_dropout_prob'],
        attention_probs_dropout_prob=config['attention_probs_dropout_prob']
    )
    
    model = LtgBertForMaskedLM(model_config)
    model.eval()
    
    # Forward pass - take just first sample to avoid memory issues
    with torch.no_grad():
        sample_input = input_ids[:1]  # First sample only
        sample_attention = attention_mask[:1]
        sample_labels = labels[:1]
        
        print(f"\nTesting model forward pass:")
        print(f"  Input shape: {sample_input.shape}")
        print(f"  Sample labels != -100: {(sample_labels != -100).sum().item()}")
        
        outputs = model(input_ids=sample_input, attention_mask=sample_attention, labels=sample_labels)
        
        print(f"  Model output loss: {outputs.loss.item():.4f}")
        print(f"  Prediction scores shape: {outputs.logits.shape}")
        
        if outputs.logits.shape[0] == (sample_labels != -100).sum().item():
            print(f"✅ Model correctly processes only masked tokens!")
            return True
        else:
            print(f"❌ Model output shape mismatch:")
            print(f"   Expected: {(sample_labels != -100).sum().item()} (masked tokens)")
            print(f"   Got: {outputs.logits.shape[0]}")
            return False

def main():
    print("🧪 TESTING YOUR COMPLETE DATA PIPELINE")
    print("=" * 60)
    
    # Test 1: Current data pipeline
    pipeline_ok = test_current_data_pipeline()
    
    # Test 2: Training scenario
    training_ok = debug_training_batch()
    
    print("\n" + "=" * 60)
    print("🎯 FINAL DIAGNOSIS:")
    
    if pipeline_ok and training_ok:
        print("✅ YOUR DATA PIPELINE IS WORKING CORRECTLY!")
        print("   The gradient explosion is NOT due to masking issues.")
        print("   The problem must be elsewhere (likely the ultra-conservative settings).")
        print("\n🚀 RECOMMENDED ACTION:")
        print("   1. Revert to more reasonable hyperparameters:")
        print("      - learning_rate: 1e-4 to 5e-4")
        print("      - max_grad_norm: 1.0")
        print("      - batch_size: 8-16")
        print("   2. Your masking is already correct!")
    else:
        print("❌ Found issues in your data pipeline")
        print("   → Need to fix the masking implementation")

if __name__ == "__main__":
    main()
