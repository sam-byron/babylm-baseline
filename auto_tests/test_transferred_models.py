#!/usr/bin/env python3
"""
Test script to verify the transferred classification models work correctly.
"""


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from transformers import AutoModelForSequenceClassification, AutoConfig
from ltg_bert_config import LtgBertConfig
from ltg_bert import LtgBertForSequenceClassification


def test_transferred_model(model_path, model_name):
    """Test a transferred model."""
    print(f"\n🧪 Testing {model_name}")
    print("=" * 50)
    
    # Load the model
    print(f"📥 Loading model from: {model_path}")
    config = LtgBertConfig.from_pretrained(model_path)
    model = LtgBertForSequenceClassification.from_pretrained(model_path)
    
    print(f"✅ Model loaded successfully!")
    print(f"   - Vocab size: {config.vocab_size}")
    print(f"   - Hidden size: {config.hidden_size}")
    print(f"   - Num labels: {config.num_labels}")
    print(f"   - Problem type: {config.problem_type}")
    print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("🚀 Testing forward pass...")
    batch_size, seq_len = 2, 64
    
    # Create dummy input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    if config.problem_type == "regression":
        labels = torch.randn(batch_size)  # Continuous labels for regression
    else:
        labels = torch.randint(0, config.num_labels, (batch_size,))  # Class labels
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    print(f"   ✅ Forward pass successful!")
    print(f"   - Input shape: {input_ids.shape}")
    print(f"   - Output logits shape: {outputs.logits.shape}")
    print(f"   - Loss: {outputs.loss:.4f}")
    
    if config.problem_type == "regression":
        print(f"   - Predictions: {outputs.logits.squeeze()}")
    else:
        predictions = torch.argmax(outputs.logits, dim=-1)
        print(f"   - Predictions: {predictions}")
    
    print(f"   ✅ {model_name} is working correctly!")
    return True


def test_automodel_loading(model_path, model_name):
    """Test loading via AutoModelForSequenceClassification."""
    print(f"\n🤖 Testing AutoModel loading for {model_name}")
    print("-" * 50)
    
    try:
        # This should work with our registration
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print(f"✅ AutoModelForSequenceClassification loading successful!")
        print(f"   - Model type: {type(model).__name__}")
        
        # Quick forward pass test
        config = model.config
        input_ids = torch.randint(0, config.vocab_size, (1, 32))
        attention_mask = torch.ones(1, 32)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        print(f"   - Quick test passed: {outputs.logits.shape}")
        return True
        
    except Exception as e:
        print(f"❌ AutoModel loading failed: {e}")
        return False


def main():
    """Test all transferred models."""
    print("🧪 Testing Transferred Classification Models")
    print("=" * 60)
    
    models_to_test = [
        ("model_vault/sentiment_classifier", "Sentiment Classifier (Binary)"),
        ("model_vault/topic_classifier", "Topic Classifier (8-class)")
    ]
    
    all_passed = True
    
    for model_path, model_name in models_to_test:
        try:
            # Test direct loading
            success = test_transferred_model(model_path, model_name)
            all_passed = all_passed and success
            
            # Test AutoModel loading
            success = test_automodel_loading(model_path, model_name)
            all_passed = all_passed and success
            
        except Exception as e:
            print(f"❌ Test failed for {model_name}: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Your transferred models are ready for fine-tuning!")
        print("\n📋 Next Steps:")
        print("1. Prepare your classification dataset")
        print("2. Use transformers Trainer for fine-tuning")
        print("3. Evaluate on your test set")
        print("\n🚀 Your models are ready for downstream tasks!")
    else:
        print("❌ Some tests failed. Please check the output above.")


if __name__ == "__main__":
    main()
