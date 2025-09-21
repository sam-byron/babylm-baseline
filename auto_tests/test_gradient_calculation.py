#!/usr/bin/env python3
"""
Comprehensive gradient calculation testing for LTG BERT model.
Tests for numerical stability, gradient flow, and potential bugs.
"""


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import numpy as np
import json
from transformers import AutoTokenizer
from ltg_bert import LtgBertForMaskedLM
from ltg_bert_config import LtgBertConfig
from data_loader import data_loader

def test_model_initialization():
    """Test if model weights are initialized properly"""
    print("🔍 Testing Model Initialization...")
    
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config_dict = json.load(f)
    
    config = LtgBertConfig(**config_dict)
    model = LtgBertForMaskedLM(config)
    
    # Check for NaN/Inf in initial weights
    has_nan = False
    has_inf = False
    weight_stats = []
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ NaN found in {name}")
            has_nan = True
        if torch.isinf(param).any():
            print(f"❌ Inf found in {name}")
            has_inf = True
            
        weight_stats.append({
            'name': name,
            'shape': list(param.shape),
            'mean': param.mean().item(),
            'std': param.std().item(),
            'min': param.min().item(),
            'max': param.max().item()
        })
    
    print(f"✅ Initial weights check: NaN={has_nan}, Inf={has_inf}")
    
    # Check weight magnitude - very large initial weights can cause explosions
    large_weights = [w for w in weight_stats if abs(w['max']) > 10 or abs(w['min']) < -10]
    if large_weights:
        print("⚠️  Large initial weights detected:")
        for w in large_weights:
            print(f"   {w['name']}: range=[{w['min']:.3f}, {w['max']:.3f}]")
    
    return model, weight_stats

def test_forward_pass():
    """Test forward pass for numerical stability"""
    print("\n🔍 Testing Forward Pass...")
    
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config_dict = json.load(f)
    
    config = LtgBertConfig(**config_dict)
    model = LtgBertForMaskedLM(config)
    
    # Create simple test input
    batch_size = 2
    seq_len = 128
    vocab_size = config.vocab_size
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Check for NaN/Inf in outputs
            if torch.isnan(logits).any():
                print("❌ NaN in forward pass logits")
                return False
            if torch.isinf(logits).any():
                print("❌ Inf in forward pass logits")
                return False
                
            print(f"✅ Forward pass successful")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
            print(f"   Logits mean: {logits.mean().item():.3f}")
            
            # Very large logits can indicate problems
            if logits.abs().max() > 100:
                print(f"⚠️  Very large logits detected: max={logits.abs().max().item():.1f}")
                
            return True
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return False

def test_gradient_computation():
    """Test gradient computation with simple loss"""
    print("\n🔍 Testing Gradient Computation...")
    
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config_dict = json.load(f)
    
    config = LtgBertConfig(**config_dict)
    model = LtgBertForMaskedLM(config)
    model.train()
    
    # Create simple test data
    batch_size = 2
    seq_len = 128
    vocab_size = config.vocab_size
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    
    print(f"   Initial loss: {loss.item():.4f}")
    
    # Check loss for NaN/Inf
    if torch.isnan(loss):
        print("❌ Loss is NaN")
        return False
    if torch.isinf(loss):
        print("❌ Loss is Inf")
        return False
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_norms = []
    nan_grads = []
    inf_grads = []
    large_grads = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append((name, grad_norm))
            
            if torch.isnan(param.grad).any():
                nan_grads.append(name)
            if torch.isinf(param.grad).any():
                inf_grads.append(name)
            if grad_norm > 1000:  # Arbitrary threshold for "large"
                large_grads.append((name, grad_norm))
    
    print(f"✅ Gradient computation completed")
    print(f"   Total parameters with gradients: {len(grad_norms)}")
    
    if nan_grads:
        print(f"❌ NaN gradients in: {nan_grads}")
    if inf_grads:
        print(f"❌ Inf gradients in: {inf_grads}")
    if large_grads:
        print("⚠️  Large gradients detected:")
        for name, norm in large_grads:
            print(f"   {name}: {norm:.1f}")
    
    # Compute overall gradient norm
    total_norm = torch.sqrt(sum(param.grad.norm()**2 for param in model.parameters() if param.grad is not None))
    print(f"   Total gradient norm: {total_norm.item():.4f}")
    
    return total_norm.item(), grad_norms

def test_data_loading():
    """Test if data loading produces valid inputs"""
    print("\n🔍 Testing Data Loading...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('./data/pretrain/wordpiece_vocab.json', 
                                                local_files_only=True, trust_remote_code=True)
        
        with open('model_babylm_ltg_bert.json', 'r') as f:
            config_dict = json.load(f)
        
        config = LtgBertConfig(**config_dict)
        # Create data loader with small batch
        train_loader, val_loader, test_loader, collate_fn, total_train_batches = data_loader(
            config=config,
            tokenizer=tokenizer,
            cache_path=config.get('cache_path', 'cache_test')
        )
        
        # Get first batch
        batch = next(iter(data_loader))
        
        print(f"✅ Data loading successful")
        print(f"   Batch keys: {list(batch.keys())}")
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                if key == 'input_ids':
                    print(f"      range=[{value.min().item()}, {value.max().item()}]")
                    # Check for invalid token IDs
                    if value.min() < 0 or value.max() >= config.vocab_size:
                        print(f"❌ Invalid token IDs detected! vocab_size={config.vocab_size}")
        
        return True, batch
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False, None

def test_loss_scaling():
    """Test if loss scaling is causing issues"""
    print("\n🔍 Testing Loss Scaling...")
    
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config_dict = json.load(f)
    
    config = LtgBertConfig(**config_dict)
    model = LtgBertForMaskedLM(config)
    
    # Test with different loss scaling
    batch_size = 2
    seq_len = 128
    vocab_size = config.vocab_size
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    scales = [1.0, 0.1, 0.01]
    
    for scale in scales:
        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss * scale
        loss.backward()
        
        total_norm = torch.sqrt(sum(param.grad.norm()**2 for param in model.parameters() if param.grad is not None))
        print(f"   Loss scale {scale}: gradient norm = {total_norm.item():.4f}")

def test_learning_rate_sensitivity():
    """Test model sensitivity to different learning rates"""
    print("\n🔍 Testing Learning Rate Sensitivity...")
    
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config_dict = json.load(f)
    
    config = LtgBertConfig(**config_dict)
    model = LtgBertForMaskedLM(config)
    
    batch_size = 2
    seq_len = 128
    vocab_size = config.vocab_size
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    learning_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    for lr in learning_rates:
        # Create fresh model copy
        model_copy = LtgBertForMaskedLM(config)
        model_copy.load_state_dict(model.state_dict())
        
        optimizer = torch.optim.AdamW(model_copy.parameters(), lr=lr)
        
        # Forward + backward
        outputs = model_copy(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        # Check gradient norm before optimizer step
        total_norm = torch.sqrt(sum(param.grad.norm()**2 for param in model_copy.parameters() if param.grad is not None))
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model_copy.parameters(), max_norm=1.0)
        clipped_norm = torch.sqrt(sum(param.grad.norm()**2 for param in model_copy.parameters() if param.grad is not None))
        
        print(f"   LR {lr}: loss={loss.item():.4f}, grad_norm={total_norm.item():.4f}, clipped={clipped_norm.item():.4f}")

def main():
    """Run all gradient tests"""
    print("🧪 GRADIENT CALCULATION TESTING")
    print("=" * 50)
    
    # Test 1: Model initialization
    model, weight_stats = test_model_initialization()
    
    # Test 2: Forward pass
    forward_ok = test_forward_pass()
    
    # Test 3: Basic gradient computation
    if forward_ok:
        grad_norm, grad_norms = test_gradient_computation()
        
        # If gradients are already large with simple test, we have a problem
        if grad_norm > 100:
            print(f"⚠️  ISSUE: Even simple gradient computation gives large norm: {grad_norm:.1f}")
    
    # Test 4: Data loading
    data_ok, batch = test_data_loading()
    
    # Test 5: Loss scaling
    test_loss_scaling()
    
    # Test 6: Learning rate sensitivity
    test_learning_rate_sensitivity()
    
    print("\n" + "=" * 50)
    print("🏁 TESTING COMPLETE")
    
    # Summary recommendations
    print("\n💡 RECOMMENDATIONS:")
    if grad_norm > 100:
        print("❌ Large gradients detected even with simple test data")
        print("   → Check model architecture for numerical instabilities")
        print("   → Consider different weight initialization")
    else:
        print("✅ Basic gradient computation seems stable")
        print("   → Issue may be with real training data or batch size")

if __name__ == "__main__":
    main()
