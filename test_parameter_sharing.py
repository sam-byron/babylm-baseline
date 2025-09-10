#!/usr/bin/env python3
"""
Test script to demonstrate ALBERT-style parameter sharing
"""

from ltg_bert_config import LtgBertConfig
from ltg_bert import LtgBertForMaskedLM

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def test_parameter_sharing():
    """Compare parameter counts with and without sharing"""
    
    # Standard model (no parameter sharing)
    config_standard = LtgBertConfig(
        vocab_size=16384,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=2048,
        share_layer_weights=False
    )
    
    # ALBERT-style model (with parameter sharing)
    config_shared = LtgBertConfig(
        vocab_size=16384,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=2048,
        share_layer_weights=True
    )
    
    print("🔍 Creating models...")
    model_standard = LtgBertForMaskedLM(config_standard)
    model_shared = LtgBertForMaskedLM(config_shared)
    
    # Count parameters
    total_std, trainable_std = count_parameters(model_standard)
    total_shared, trainable_shared = count_parameters(model_shared)
    
    print("\n📊 Parameter Comparison:")
    print(f"Standard Model:     {total_std:,} total, {trainable_std:,} trainable")
    print(f"Shared Model:       {total_shared:,} total, {trainable_shared:,} trainable")
    print(f"Reduction:          {total_std - total_shared:,} parameters ({((total_std - total_shared) / total_std * 100):.1f}%)")
    print(f"Memory Reduction:   {(total_std - total_shared) * 4 / (1024**2):.1f} MB (fp32)")
    
    # Show model sizes
    std_mb = total_std * 4 / (1024**2)  # Assume fp32
    shared_mb = total_shared * 4 / (1024**2)
    
    print(f"\n💾 Model Sizes:")
    print(f"Standard Model:     {std_mb:.1f} MB")
    print(f"Shared Model:       {shared_mb:.1f} MB")
    print(f"Size Ratio:         {shared_mb/std_mb:.2f}x")
    
    print("\n✅ Benefits of ALBERT-style parameter sharing:")
    print("   • Significantly fewer parameters")
    print("   • Reduced memory usage")
    print("   • Better generalization for limited data")
    print("   • Faster training due to fewer parameters")
    print("   • Regularization effect from weight reuse")

if __name__ == "__main__":
    test_parameter_sharing()
