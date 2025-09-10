#!/usr/bin/env python3
"""
Quick diagnostic for gradient explosion issues.
Run this to check if the problem is in model, data, or training setup.
"""

import torch
import torch.nn as nn
import json
from ltg_bert import LtgBertForMaskedLM
from ltg_bert_config import LtgBertConfig

def quick_gradient_test():
    """Quick test with minimal data"""
    print("🔍 QUICK GRADIENT DIAGNOSTIC")
    print("-" * 30)
    
    # Load config
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config_dict = json.load(f)
    
    # Create proper config object
    config = LtgBertConfig(
        vocab_size=config_dict['vocab_size'],
        hidden_size=config_dict['hidden_size'],
        num_hidden_layers=config_dict['num_hidden_layers'],
        num_attention_heads=config_dict['num_attention_heads'],
        intermediate_size=config_dict['intermediate_size'],
        max_position_embeddings=config_dict['max_position_embeddings'],
        position_bucket_size=config_dict['position_bucket_size'],
        layer_norm_eps=config_dict['layer_norm_eps'],
        hidden_dropout_prob=config_dict['hidden_dropout_prob'],
        attention_probs_dropout_prob=config_dict['attention_probs_dropout_prob']
    )
    
    print(f"Model config: {config.hidden_size}h, {config.num_hidden_layers}L, {config.vocab_size}V")
    
    # Create model
    model = LtgBertForMaskedLM(config)
    model.train()
    
    # Create MINIMAL test data
    batch_size = 1
    seq_len = 10  # Very short sequence
    vocab_size = config.vocab_size
    
    # Simple input (all token 1)
    input_ids = torch.ones((batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    print(f"Test input: batch_size={batch_size}, seq_len={seq_len}")
    print(f"Input range: [{input_ids.min().item()}, {input_ids.max().item()}]")
    
    # Forward pass
    print("\n1️⃣ Forward pass...")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    logits = outputs.logits
    
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"   Logits mean: {logits.mean().item():.3f}")
    
    if torch.isnan(loss) or torch.isinf(loss):
        print("❌ PROBLEM: Loss is NaN/Inf")
        return False
    
    if logits.abs().max() > 100:
        print(f"⚠️  Very large logits: {logits.abs().max().item():.1f}")
    
    # Backward pass
    print("\n2️⃣ Backward pass...")
    loss.backward()
    
    # Check gradients
    grad_info = []
    total_norm_sq = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_info.append((name, grad_norm))
            total_norm_sq += grad_norm ** 2
            
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"❌ PROBLEM: NaN/Inf gradients in {name}")
                return False
    
    total_norm = torch.sqrt(torch.tensor(total_norm_sq)).item()
    
    print(f"   Total gradient norm: {total_norm:.4f}")
    
    # Show top 5 largest gradient norms
    grad_info.sort(key=lambda x: x[1], reverse=True)
    print("   Top 5 gradient norms:")
    for name, norm in grad_info[:5]:
        print(f"      {name}: {norm:.4f}")
    
    # Test with gradient clipping
    print("\n3️⃣ Testing gradient clipping...")
    
    # Reload model for fresh gradients
    model = LtgBertForMaskedLM(config)
    model.train()
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    # Apply clipping
    original_norm = torch.sqrt(sum(param.grad.norm()**2 for param in model.parameters() if param.grad is not None))
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    clipped_norm = torch.sqrt(sum(param.grad.norm()**2 for param in model.parameters() if param.grad is not None))
    
    print(f"   Original norm: {original_norm.item():.4f}")
    print(f"   Clipped norm: {clipped_norm.item():.4f}")
    print(f"   Clipping ratio: {clipped_norm.item() / original_norm.item():.6f}")
    
    # Verdict
    print(f"\n📊 VERDICT:")
    if total_norm > 1000:
        print("❌ GRADIENT EXPLOSION confirmed even with minimal data!")
        print("   → Problem is likely in model architecture or initialization")
        print("   → Check weight initialization, layer norms, or activation functions")
    elif total_norm > 10:
        print("⚠️  Large gradients with minimal data")
        print("   → May become problematic with real batch sizes")
        print("   → Consider even lower learning rates")
    else:
        print("✅ Gradients seem reasonable with minimal data")
        print("   → Problem may be with batch size, sequence length, or real data")
    
    return total_norm < 1000

if __name__ == "__main__":
    quick_gradient_test()
