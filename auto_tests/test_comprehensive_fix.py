#!/usr/bin/env python3
"""
Create a fixed version of the model with proper initialization and settings.
This should resolve the gradient explosion issues.
"""


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import json
from ltg_bert import LtgBertForMaskedLM
from ltg_bert_config import LtgBertConfig

def apply_proper_initialization(model):
    """Apply proper weight initialization that's known to work well"""
    print("🔧 Applying proper weight initialization...")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Use standard Xavier/Glorot initialization for linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Use small normal initialization for embeddings
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm initialization - check if weights exist
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

def test_fixed_model():
    """Test the model with proper initialization"""
    print("🧪 TESTING FIXED MODEL")
    print("=" * 40)
    
    # Load config and apply fixes
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config_dict = json.load(f)
    
    # Create config with safer settings
    config = LtgBertConfig(
        vocab_size=config_dict['vocab_size'],
        hidden_size=config_dict['hidden_size'],
        num_hidden_layers=config_dict['num_hidden_layers'],
        num_attention_heads=config_dict['num_attention_heads'],
        intermediate_size=config_dict['intermediate_size'],
        max_position_embeddings=config_dict.get('max_position_embeddings', config_dict.get('block_size', 512)),
        position_bucket_size=config_dict['position_bucket_size'],
        layer_norm_eps=1e-5,  # Fixed epsilon
        hidden_dropout_prob=config_dict['hidden_dropout_prob'],
        attention_probs_dropout_prob=config_dict['attention_probs_dropout_prob']
    )
    
    print(f"✅ LayerNorm epsilon: {config.layer_norm_eps}")
    
    # Create model
    model = LtgBertForMaskedLM(config)
    
    # Apply proper initialization
    apply_proper_initialization(model)
    
    # Test with minimal data
    batch_size = 1
    seq_len = 10
    input_ids = torch.ones((batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    
    print(f"\n🔍 Testing with minimal data: {batch_size}×{seq_len}")
    
    # Forward pass
    model.train()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    
    print(f"   Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradient norm
    total_norm = torch.sqrt(sum(param.grad.norm()**2 for param in model.parameters() if param.grad is not None))
    print(f"   Gradient norm: {total_norm.item():.4f}")
    
    # Test with slightly larger data
    print(f"\n🔍 Testing with larger data: 2×64")
    model.zero_grad()
    
    input_ids_large = torch.ones((2, 64), dtype=torch.long)
    attention_mask_large = torch.ones_like(input_ids_large)
    labels_large = input_ids_large.clone()
    
    outputs = model(input_ids=input_ids_large, attention_mask=attention_mask_large, labels=labels_large)
    loss = outputs.loss
    loss.backward()
    
    total_norm_large = torch.sqrt(sum(param.grad.norm()**2 for param in model.parameters() if param.grad is not None))
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradient norm: {total_norm_large.item():.4f}")
    
    # Analysis
    scale_factor = total_norm_large.item() / total_norm.item()
    expected_scale = (2 * 64) / (1 * 10)  # 12.8x more data
    
    print(f"\n📊 Scaling Analysis:")
    print(f"   Data scale factor: {expected_scale:.1f}x")
    print(f"   Gradient scale factor: {scale_factor:.1f}x")
    print(f"   Ratio: {scale_factor / expected_scale:.2f}")
    
    if scale_factor > expected_scale * 2:
        print("⚠️  Gradients scale worse than linearly with data size")
    else:
        print("✅ Gradient scaling looks reasonable")
    
    # Predict gradient norm for full training
    full_training_data = 8 * 512  # Your actual training setup
    predicted_norm = total_norm.item() * (full_training_data / 10)
    
    print(f"\n🔮 Predictions for full training (8×512):")
    print(f"   Predicted gradient norm: {predicted_norm:.1f}")
    
    if predicted_norm > 1000:
        print("❌ Still likely to explode with full training data")
        print("   → Need even more conservative approach")
        return False
    elif predicted_norm > 100:
        print("⚠️  Will need very aggressive gradient clipping")
        print("   → Recommend max_grad_norm = 0.1")
        return True
    else:
        print("✅ Should be stable with normal gradient clipping")
        print("   → Can use max_grad_norm = 1.0")
        return True

def create_fixed_config():
    """Create a configuration file with all the fixes applied"""
    print("\n🔧 Creating fixed configuration...")
    
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config = json.load(f)
    
    # Apply all fixes
    config['layer_norm_eps'] = 1e-5  # Fix LayerNorm epsilon
    config['learning_rate'] = 5e-6   # Ultra-conservative learning rate
    config['max_grad_norm'] = 0.1    # Aggressive gradient clipping
    config['warmup_steps_proportion'] = 0.3  # Longer warmup
    config['weight_decay'] = 0.0001  # Minimal weight decay
    config['batch_size'] = 4         # Smaller batches
    config['grad_accum'] = 24        # Maintain effective batch size
    
    # Save fixed config
    with open('model_babylm_ltg_bert_FIXED.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("   ✅ Saved fixed config to: model_babylm_ltg_bert_FIXED.json")
    print("   📋 Key changes:")
    print(f"      - LayerNorm epsilon: {config['layer_norm_eps']}")
    print(f"      - Learning rate: {config['learning_rate']}")
    print(f"      - Max grad norm: {config['max_grad_norm']}")
    print(f"      - Batch size: {config['batch_size']}")
    print(f"      - Grad accumulation: {config['grad_accum']}")

def main():
    print("🛠️  COMPREHENSIVE GRADIENT EXPLOSION FIX")
    print("=" * 50)
    
    # Test the fixed model
    stable = test_fixed_model()
    
    # Create fixed configuration
    create_fixed_config()
    
    print("\n" + "=" * 50)
    print("🎯 FINAL RECOMMENDATIONS:")
    
    if stable:
        print("✅ Fixed model shows better gradient behavior")
        print("🚀 NEXT STEPS:")
        print("   1. Use the fixed config: model_babylm_ltg_bert_FIXED.json")
        print("   2. Restart training with proper initialization")
        print("   3. Monitor first 10 steps for gradient norms < 1.0")
    else:
        print("❌ Model still shows instability")
        print("🔍 ADDITIONAL DEBUGGING NEEDED:")
        print("   1. Check model architecture for bugs")
        print("   2. Consider using a different base model")
        print("   3. Try much smaller model for testing")

if __name__ == "__main__":
    main()
