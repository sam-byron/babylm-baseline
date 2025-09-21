#!/usr/bin/env python3
"""
Test the custom weight initialization to see if it's causing gradient explosions.
"""


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import math
import json
from ltg_bert import LtgBertForMaskedLM
from ltg_bert_config import LtgBertConfig

def analyze_initialization():
    """Analyze the custom weight initialization"""
    print("🔍 ANALYZING WEIGHT INITIALIZATION")
    print("-" * 40)
    
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config_dict = json.load(f)
    
    # Create proper config object
    config = LtgBertConfig(
        vocab_size=config_dict['vocab_size'],
        hidden_size=config_dict['hidden_size'],
        num_hidden_layers=config_dict['num_hidden_layers'],
        num_attention_heads=config_dict['num_attention_heads'],
        intermediate_size=config_dict['intermediate_size'],
        max_position_embeddings=config_dict.get('max_position_embeddings', config_dict.get('block_size', 512)),
        position_bucket_size=config_dict['position_bucket_size'],
        layer_norm_eps=config_dict['layer_norm_eps'],
        hidden_dropout_prob=config_dict['hidden_dropout_prob'],
        attention_probs_dropout_prob=config_dict['attention_probs_dropout_prob']
    )
    
    # Create model
    model = LtgBertForMaskedLM(config)
    
    print(f"Model parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Analyze weight distributions
    print("\n📊 Weight Statistics:")
    large_weights = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            std = param.std().item()
            mean = param.mean().item()
            min_val = param.min().item()
            max_val = param.max().item()
            
            print(f"\n{name}:")
            print(f"  Shape: {list(param.shape)}")
            print(f"  Mean: {mean:.6f}, Std: {std:.6f}")
            print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
            
            # Check for problematic initialization
            if abs(max_val) > 1.0 or abs(min_val) > 1.0:
                large_weights.append((name, min_val, max_val))
            
            # Check custom initialization formula
            if 'Linear' in str(type(param)) or 'weight' in name:
                if 'embedding' in name.lower():
                    expected_std = math.sqrt(2.0 / (5.0 * param.shape[-1]))
                elif len(param.shape) >= 2:  # Linear layer
                    expected_std = math.sqrt(2.0 / (5.0 * param.shape[0]))  # in_features
                else:
                    expected_std = None
                
                if expected_std:
                    print(f"  Expected std: {expected_std:.6f} (actual: {std:.6f})")
                    if abs(std - expected_std) > 0.001:
                        print(f"  ⚠️  Std mismatch!")
    
    if large_weights:
        print(f"\n⚠️  LARGE WEIGHTS DETECTED:")
        for name, min_val, max_val in large_weights:
            print(f"   {name}: [{min_val:.3f}, {max_val:.3f}]")
    
    return large_weights

def test_standard_vs_custom_init():
    """Compare standard vs custom initialization"""
    print("\n🔄 COMPARING INITIALIZATION METHODS")
    print("-" * 40)
    
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config_dict = json.load(f)
    
    # Create proper config object
    config = LtgBertConfig(
        vocab_size=config_dict['vocab_size'],
        hidden_size=config_dict['hidden_size'],
        num_hidden_layers=config_dict['num_hidden_layers'],
        num_attention_heads=config_dict['num_attention_heads'],
        intermediate_size=config_dict['intermediate_size'],
        max_position_embeddings=config_dict.get('max_position_embeddings', config_dict.get('block_size', 512)),
        position_bucket_size=config_dict['position_bucket_size'],
        layer_norm_eps=config_dict['layer_norm_eps'],
        hidden_dropout_prob=config_dict['hidden_dropout_prob'],
        attention_probs_dropout_prob=config_dict['attention_probs_dropout_prob']
    )
    
    # Test with standard initialization
    print("\n1️⃣ Standard PyTorch initialization:")
    model_std = LtgBertForMaskedLM(config)
    
    # Reinitialize with standard method
    for module in model_std.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, 0.0, 0.02)
    
    # Test gradients with both
    input_ids = torch.ones((1, 10), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    # Test custom init
    print("\n2️⃣ Testing custom initialization:")
    model_custom = LtgBertForMaskedLM(config)  # Uses custom init
    outputs = model_custom(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    custom_norm = torch.sqrt(sum(param.grad.norm()**2 for param in model_custom.parameters() if param.grad is not None))
    print(f"   Custom init gradient norm: {custom_norm.item():.4f}")
    
    # Test standard init
    print("\n3️⃣ Testing standard initialization:")
    outputs = model_std(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    std_norm = torch.sqrt(sum(param.grad.norm()**2 for param in model_std.parameters() if param.grad is not None))
    print(f"   Standard init gradient norm: {std_norm.item():.4f}")
    
    print(f"\n📊 Ratio (custom/standard): {custom_norm.item() / std_norm.item():.2f}")
    
    if custom_norm > std_norm * 10:
        print("❌ Custom initialization produces much larger gradients!")
        return False
    else:
        print("✅ Initialization comparison looks reasonable")
        return True

def check_layer_norm_issues():
    """Check if LayerNorm epsilon is causing issues"""
    print("\n🔍 CHECKING LAYER NORM SETTINGS")
    print("-" * 40)
    
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config_dict = json.load(f)
    
    print(f"LayerNorm epsilon from config: {config_dict.get('layer_norm_eps', 'Not specified')}")
    
    # Create proper config object
    config = LtgBertConfig(
        vocab_size=config_dict['vocab_size'],
        hidden_size=config_dict['hidden_size'],
        num_hidden_layers=config_dict['num_hidden_layers'],
        num_attention_heads=config_dict['num_attention_heads'],
        intermediate_size=config_dict['intermediate_size'],
        max_position_embeddings=config_dict.get('max_position_embeddings', config_dict.get('block_size', 512)),
        position_bucket_size=config_dict['position_bucket_size'],
        layer_norm_eps=config_dict['layer_norm_eps'],
        hidden_dropout_prob=config_dict['hidden_dropout_prob'],
        attention_probs_dropout_prob=config_dict['attention_probs_dropout_prob']
    )
    
    model = LtgBertForMaskedLM(config)
    
    # Find all LayerNorm modules
    ln_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            ln_modules.append((name, module))
            print(f"LayerNorm '{name}': eps={module.eps}")
    
    print(f"Total LayerNorm modules: {len(ln_modules)}")
    
    # Check if epsilon is too small (can cause numerical instability)
    small_eps = [name for name, module in ln_modules if module.eps < 1e-8]
    if small_eps:
        print(f"⚠️  Very small epsilon in: {small_eps}")
        print("   This might cause numerical instability!")

def main():
    print("🧪 WEIGHT INITIALIZATION DIAGNOSTIC")
    print("=" * 50)
    
    # Test 1: Analyze current initialization
    large_weights = analyze_initialization()
    
    # Test 2: Compare with standard initialization
    init_ok = test_standard_vs_custom_init()
    
    # Test 3: Check LayerNorm settings
    check_layer_norm_issues()
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 DIAGNOSTIC SUMMARY")
    
    if large_weights:
        print("❌ ISSUE: Large initial weights detected")
        print("   → Custom initialization may be too aggressive")
        print("   → Consider using standard PyTorch initialization")
    
    if not init_ok:
        print("❌ ISSUE: Custom initialization produces much larger gradients")
        print("   → Switch to standard initialization methods")
    
    print("\n💡 RECOMMENDATIONS:")
    print("1. Try disabling custom initialization")
    print("2. Use standard PyTorch init: kaiming_uniform for linear layers")
    print("3. Use smaller std for embeddings (0.02 instead of computed value)")
    print("4. Increase LayerNorm epsilon if it's very small")

if __name__ == "__main__":
    main()
