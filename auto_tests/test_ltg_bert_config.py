#!/usr/bin/env python3
"""
Test script for LtgBertConfig and LtgBertForMaskedLM
"""


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ltg_bert_config import LtgBertConfig
from ltg_bert import LtgBertForMaskedLM
import torch
import json

def test_config_creation():
    """Test basic configuration creation"""
    print("Testing LtgBertConfig creation...")
    
    # Test default configuration
    config = LtgBertConfig()
    print(f"Default config: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")
    
    # Test custom configuration
    config_custom = LtgBertConfig(
        vocab_size=32000,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024
    )
    print(f"Custom config: vocab_size={config_custom.vocab_size}, hidden_size={config_custom.hidden_size}")
    
    # Test configuration from dictionary (like JSON loading)
    config_dict = {
        "vocab_size": 16384,
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "intermediate_size": 1024,
        "position_bucket_size": 32,
        "max_position_embeddings": 512,
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "layer_norm_eps": 1e-7
    }
    
    config_from_dict = LtgBertConfig(**config_dict)
    print(f"Config from dict: vocab_size={config_from_dict.vocab_size}, hidden_size={config_from_dict.hidden_size}")
    
    # Test configuration serialization
    config_json = config_from_dict.to_json_string()
    print("Configuration serializes to JSON successfully")
    
    return config_from_dict

def test_model_creation():
    """Test model creation with custom configuration"""
    print("\nTesting LtgBertForMaskedLM creation...")
    
    # Create configuration
    config = LtgBertConfig(
        vocab_size=16384,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        intermediate_size=1024
    )
    
    # Create model
    model = LtgBertForMaskedLM(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = input_ids.clone()
    
    # Mask some tokens for MLM
    mask_indices = torch.rand(batch_size, seq_len) < 0.15
    labels[~mask_indices] = -100
    
    print(f"Testing forward pass with input shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    print(f"Forward pass successful!")
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}")
    
    return model

def test_auto_registration():
    """Test auto-registration with transformers"""
    print("\nTesting AutoConfig and AutoModel registration...")
    
    try:
        from transformers import AutoConfig, AutoModelForMaskedLM
        
        # Test config creation via AutoConfig
        config = AutoConfig.from_pretrained("./", trust_remote_code=True)
        print("AutoConfig loaded successfully")
        
        # For now we can't test AutoModelForMaskedLM.from_pretrained without saved files
        print("Auto-registration test requires saved model files")
        
    except Exception as e:
        print(f"Auto-registration test skipped: {e}")

def test_compatibility_with_existing_configs():
    """Test loading configurations from existing JSON files"""
    print("\nTesting compatibility with existing config files...")
    
    try:
        # Test with base config
        with open("configs/base.json", "r") as f:
            base_config_dict = json.load(f)
        
        config_base = LtgBertConfig(**base_config_dict)
        model_base = LtgBertForMaskedLM(config_base)
        print(f"Base config loaded: {config_base.hidden_size}d, {config_base.num_hidden_layers} layers")
        
        # Test with small config
        with open("configs/small.json", "r") as f:
            small_config_dict = json.load(f)
        
        config_small = LtgBertConfig(**small_config_dict)
        model_small = LtgBertForMaskedLM(config_small)
        print(f"Small config loaded: {config_small.hidden_size}d, {config_small.num_hidden_layers} layers")
        
        print("Existing config files are compatible!")
        
    except Exception as e:
        print(f"Error loading existing configs: {e}")

if __name__ == "__main__":
    print("=== Testing LtgBertConfig and LtgBertForMaskedLM ===")
    
    config = test_config_creation()
    model = test_model_creation()
    test_auto_registration()
    test_compatibility_with_existing_configs()
    
    print("\n=== All tests completed ===")
