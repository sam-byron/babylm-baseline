#!/usr/bin/env python3

import torch
import json
from ltg_bert import LtgBertForMaskedLM
from ltg_bert_config import LtgBertConfig
import torch.nn as nn

def test_model_initialization():
    """Test if the model has proper weight initialization"""
    
    # Load config
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config = json.load(f)
    
    print("🔍 Testing Model Initialization")
    print("=" * 50)
    
    # Create model
    bert_config = LtgBertConfig(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        max_position_embeddings=config["max_position_embeddings"],
        layer_norm_eps=config["layer_norm_eps"],
        attention_probs_dropout_prob=config["attention_probs_dropout_prob"],
        hidden_dropout_prob=config["hidden_dropout_prob"]
    )
    
    model = LtgBertForMaskedLM(bert_config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Check weight statistics
    print("\n📊 Weight Statistics:")
    print("-" * 30)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            mean_val = param.data.mean().item()
            std_val = param.data.std().item()
            min_val = param.data.min().item()
            max_val = param.data.max().item()
            
            print(f"{name:40} | mean: {mean_val:8.4f} | std: {std_val:8.4f} | min: {min_val:8.4f} | max: {max_val:8.4f}")
            
            # Check for problematic initializations
            if abs(mean_val) > 1.0:
                print(f"  ⚠️  WARNING: Large mean value!")
            if std_val > 1.0:
                print(f"  ⚠️  WARNING: Large std deviation!")
            if abs(min_val) > 10.0 or abs(max_val) > 10.0:
                print(f"  ⚠️  WARNING: Extreme values!")
    
    # Test forward pass with dummy data
    print("\n🔬 Testing Forward Pass:")
    print("-" * 30)
    
    dummy_input = torch.randint(0, config["vocab_size"], (2, 512))
    dummy_labels = torch.randint(0, config["vocab_size"], (2, 512))
    
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(dummy_input, labels=dummy_labels)
            loss = outputs.loss
            logits = outputs.logits
            
            print(f"✅ Forward pass successful!")
            print(f"   Loss: {loss.item():.4f}")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Logits mean: {logits.mean().item():.4f}")
            print(f"   Logits std: {logits.std().item():.4f}")
            print(f"   Logits min: {logits.min().item():.4f}")
            print(f"   Logits max: {logits.max().item():.4f}")
            
            # Check for problematic outputs
            if torch.isnan(loss):
                print("  ⚠️  WARNING: NaN loss!")
            if torch.isinf(loss):
                print("  ⚠️  WARNING: Infinite loss!")
            if loss.item() > 20.0:
                print(f"  ⚠️  WARNING: Very high initial loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
    
    # Test gradient computation
    print("\n🎯 Testing Gradient Computation:")
    print("-" * 30)
    
    model.train()
    try:
        outputs = model(dummy_input, labels=dummy_labels)
        loss = outputs.loss
        loss.backward()
        
        total_grad_norm = 0.0
        param_count = 0
        problematic_gradients = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += grad_norm ** 2
                param_count += 1
                
                if grad_norm > 100.0:
                    problematic_gradients.append((name, grad_norm))
        
        total_grad_norm = total_grad_norm ** 0.5
        
        print(f"✅ Gradient computation successful!")
        print(f"   Total gradient norm: {total_grad_norm:.4f}")
        print(f"   Parameters with gradients: {param_count}")
        
        if problematic_gradients:
            print(f"   ⚠️  WARNING: {len(problematic_gradients)} parameters with high gradients:")
            for name, grad_norm in problematic_gradients[:5]:
                print(f"      {name}: {grad_norm:.2f}")
        
        if total_grad_norm > 1000.0:
            print(f"  ⚠️  WARNING: Very high total gradient norm: {total_grad_norm:.4f}")
        
    except Exception as e:
        print(f"❌ Gradient computation failed: {e}")

if __name__ == "__main__":
    test_model_initialization()
