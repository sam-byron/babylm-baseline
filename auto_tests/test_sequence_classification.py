#!/usr/bin/env python3
"""
Test script for LtgBertForSequenceClassification model
"""

import torch
from ltg_bert_config import LtgBertConfig
from ltg_bert import LtgBertForSequenceClassification
from transformers import AutoConfig, AutoModelForSequenceClassification


def test_manual_instantiation():
    """Test creating the model manually"""
    print("Testing manual instantiation...")
    
    config = LtgBertConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=128,
        num_labels=3  # 3-class classification
    )
    
    model = LtgBertForSequenceClassification(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, config.num_labels, (batch_size,))
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Loss: {outputs.loss}")
    print("Manual instantiation test passed!")
    return True


def test_auto_model():
    """Test loading via AutoModelForSequenceClassification"""
    print("\nTesting AutoModelForSequenceClassification...")
    
    try:
        # Create config
        config = LtgBertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            max_position_embeddings=128,
            num_labels=2  # Binary classification
        )
        
        # This should work after registration
        model = AutoModelForSequenceClassification.from_config(config)
        print(f"AutoModel created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, config.num_labels, (batch_size,))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        print(f"Output logits shape: {outputs.logits.shape}")
        print(f"Loss: {outputs.loss}")
        print("AutoModelForSequenceClassification test passed!")
        return True
        
    except Exception as e:
        print(f"AutoModelForSequenceClassification test failed: {e}")
        return False


def test_different_problem_types():
    """Test different problem types (regression, multi-label)"""
    print("\nTesting different problem types...")
    
    # Test regression
    config = LtgBertConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=128,
        num_labels=1  # Regression
    )
    
    model = LtgBertForSequenceClassification(config)
    
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randn(batch_size, 1)  # Continuous labels for regression
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    print(f"Regression - Logits shape: {outputs.logits.shape}, Loss: {outputs.loss}")
    
    # Test multi-label classification
    config.num_labels = 3
    config.problem_type = "multi_label_classification"
    model = LtgBertForSequenceClassification(config)
    
    labels = torch.randint(0, 2, (batch_size, 3)).float()  # Binary labels for each class
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    print(f"Multi-label - Logits shape: {outputs.logits.shape}, Loss: {outputs.loss}")
    print("Problem types test passed!")
    return True


if __name__ == "__main__":
    print("Testing LtgBertForSequenceClassification...")
    
    success = True
    success &= test_manual_instantiation()
    success &= test_auto_model()
    success &= test_different_problem_types()
    
    if success:
        print("\n✅ All tests passed! LtgBertForSequenceClassification is working correctly.")
    else:
        print("\n❌ Some tests failed.")
