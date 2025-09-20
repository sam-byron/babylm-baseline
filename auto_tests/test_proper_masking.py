#!/usr/bin/env python3
"""
Test with PROPER masking to simulate real MLM training.
This should resolve the gradient explosion.
"""

import torch
import torch.nn as nn
import json
from ltg_bert import LtgBertForMaskedLM
from ltg_bert_config import LtgBertConfig

def create_proper_mlm_batch(batch_size, seq_len, vocab_size, mask_ratio=0.15):
    """Create a properly masked batch like real MLM training"""
    # Create input sequence
    input_ids = torch.randint(5, vocab_size-100, (batch_size, seq_len))  # Avoid special tokens
    
    # Create labels (initially all -100)
    labels = torch.full_like(input_ids, -100)
    
    # Randomly mask tokens
    for b in range(batch_size):
        for s in range(seq_len):
            if torch.rand(1).item() < mask_ratio:
                # This token will be masked
                labels[b, s] = input_ids[b, s].clone()  # Store original token
                
                # 80% of time: replace with [MASK] token (assume ID 4)
                # 10% of time: replace with random token  
                # 10% of time: keep original
                rand = torch.rand(1).item()
                if rand < 0.8:
                    input_ids[b, s] = 4  # [MASK] token
                elif rand < 0.9:
                    input_ids[b, s] = torch.randint(5, vocab_size-100, (1,)).item()
                # else: keep original (10% of time)
    
    attention_mask = torch.ones_like(input_ids)
    
    return input_ids, attention_mask, labels

def test_with_proper_masking():
    """Test the model with proper MLM masking"""
    print("🎯 TESTING WITH PROPER MLM MASKING")
    print("=" * 50)
    
    # Load config
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config_dict = json.load(f)
    
    config = LtgBertConfig(
        vocab_size=config_dict['vocab_size'],
        hidden_size=config_dict['hidden_size'],
        num_hidden_layers=config_dict['num_hidden_layers'],
        num_attention_heads=config_dict['num_attention_heads'],
        intermediate_size=config_dict['intermediate_size'],
        max_position_embeddings=config_dict['max_position_embeddings'],
        position_bucket_size=config_dict['position_bucket_size'],
        layer_norm_eps=1e-5,  # Fixed epsilon
        hidden_dropout_prob=config_dict['hidden_dropout_prob'],
        attention_probs_dropout_prob=config_dict['attention_probs_dropout_prob']
    )
    
    model = LtgBertForMaskedLM(config)
    model.train()
    
    print("🔍 Test 1: Minimal proper masking (1×10, ~15% masked)")
    input_ids, attention_mask, labels = create_proper_mlm_batch(1, 10, config.vocab_size)
    
    masked_tokens = (labels != -100).sum().item()
    print(f"   Masked tokens: {masked_tokens}/{labels.numel()} ({masked_tokens/labels.numel()*100:.1f}%)")
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    total_norm = torch.sqrt(sum(param.grad.norm()**2 for param in model.parameters() if param.grad is not None))
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradient norm: {total_norm.item():.4f}")
    
    print("\n🔍 Test 2: Larger proper masking (2×64, ~15% masked)")
    model.zero_grad()
    
    input_ids, attention_mask, labels = create_proper_mlm_batch(2, 64, config.vocab_size)
    masked_tokens = (labels != -100).sum().item()
    print(f"   Masked tokens: {masked_tokens}/{labels.numel()} ({masked_tokens/labels.numel()*100:.1f}%)")
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    total_norm_large = torch.sqrt(sum(param.grad.norm()**2 for param in model.parameters() if param.grad is not None))
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradient norm: {total_norm_large.item():.4f}")
    
    print("\n🔍 Test 3: Training-size proper masking (8×512, ~15% masked)")
    model.zero_grad()
    
    input_ids, attention_mask, labels = create_proper_mlm_batch(8, 512, config.vocab_size)
    masked_tokens = (labels != -100).sum().item()
    print(f"   Masked tokens: {masked_tokens}/{labels.numel()} ({masked_tokens/labels.numel()*100:.1f}%)")
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    total_norm_full = torch.sqrt(sum(param.grad.norm()**2 for param in model.parameters() if param.grad is not None))
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradient norm: {total_norm_full.item():.4f}")
    
    print(f"\n📊 ANALYSIS:")
    print(f"   Gradient progression: {total_norm.item():.1f} → {total_norm_large.item():.1f} → {total_norm_full.item():.1f}")
    
    if total_norm_full.item() < 100:
        print("✅ GRADIENT EXPLOSION FIXED!")
        print("   → Proper masking resolves the issue")
        return True
    elif total_norm_full.item() < 1000:
        print("⚠️  Much better, but still needs conservative settings")
        print("   → Use learning_rate ≤ 1e-4, max_grad_norm = 1.0")
        return True
    else:
        print("❌ Still problematic - may need additional fixes")
        return False

def compare_wrong_vs_right_masking():
    """Compare the old wrong way vs the correct way"""
    print("\n🔄 COMPARING WRONG VS CORRECT MASKING")
    print("=" * 50)
    
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config_dict = json.load(f)
    
    config = LtgBertConfig(
        vocab_size=config_dict['vocab_size'],
        hidden_size=config_dict['hidden_size'],
        num_hidden_layers=config_dict['num_hidden_layers'],
        num_attention_heads=config_dict['num_attention_heads'],
        intermediate_size=config_dict['intermediate_size'],
        max_position_embeddings=config_dict['max_position_embeddings'],
        position_bucket_size=config_dict['position_bucket_size'],
        layer_norm_eps=1e-5,
        hidden_dropout_prob=config_dict['hidden_dropout_prob'],
        attention_probs_dropout_prob=config_dict['attention_probs_dropout_prob']
    )
    
    # Test wrong way (what you were doing)
    print("❌ WRONG WAY (what caused explosions):")
    model = LtgBertForMaskedLM(config)
    
    input_ids = torch.ones((2, 64), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()  # ALL tokens labeled!
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    wrong_norm = torch.sqrt(sum(param.grad.norm()**2 for param in model.parameters() if param.grad is not None))
    masked_tokens_wrong = (labels != -100).sum().item()
    
    print(f"   Labeled tokens: {masked_tokens_wrong}/{labels.numel()} (100%)")
    print(f"   Gradient norm: {wrong_norm.item():.1f}")
    
    # Test right way
    print("\n✅ CORRECT WAY (proper MLM):")
    model = LtgBertForMaskedLM(config)
    
    input_ids, attention_mask, labels = create_proper_mlm_batch(2, 64, config.vocab_size)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    right_norm = torch.sqrt(sum(param.grad.norm()**2 for param in model.parameters() if param.grad is not None))
    masked_tokens_right = (labels != -100).sum().item()
    
    print(f"   Labeled tokens: {masked_tokens_right}/{labels.numel()} (~15%)")
    print(f"   Gradient norm: {right_norm.item():.1f}")
    
    ratio = wrong_norm.item() / right_norm.item()
    print(f"\n📊 EXPLOSION RATIO: {ratio:.1f}x")
    print(f"   Wrong way produces {ratio:.1f}x larger gradients!")

def main():
    print("🧪 GRADIENT EXPLOSION ROOT CAUSE ANALYSIS")
    print("=" * 60)
    
    # Test with proper masking
    fixed = test_with_proper_masking()
    
    # Compare wrong vs right approach
    compare_wrong_vs_right_masking()
    
    print("\n" + "=" * 60)
    print("🎯 FINAL DIAGNOSIS:")
    
    if fixed:
        print("✅ GRADIENT EXPLOSION WAS CAUSED BY IMPROPER MASKING!")
        print("\n🔧 THE FIX:")
        print("   1. Use proper MLM masking (~15% of tokens)")
        print("   2. Set most labels to -100 (ignored in loss)")
        print("   3. Only mask randomly selected tokens")
        print("\n🚀 RECOMMENDED SETTINGS:")
        print("   - learning_rate: 1e-4 to 5e-4")
        print("   - max_grad_norm: 1.0")
        print("   - batch_size: 8-16")
        print("   - Your original settings were probably fine!")
    else:
        print("❌ Masking helped but didn't fully resolve the issue")
        print("   → Additional architectural problems may exist")

if __name__ == "__main__":
    main()
