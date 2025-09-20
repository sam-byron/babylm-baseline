#!/usr/bin/env python3
"""
Simple test to verify a classification model works correctly.
"""

import torch
from ltg_bert_config import LtgBertConfig
from ltg_bert import LtgBertForSequenceClassification


def test_simple_classification():
    """Test creating and using a classification model directly."""
    print("🧪 Simple Classification Test")
    print("=" * 40)
    
    # Create a config for 8-class classification
    config = LtgBertConfig(
        vocab_size=16384,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=2048,
        max_position_embeddings=512,
        num_labels=8,  # 8-class classification
        problem_type="single_label_classification"
    )
    
    print(f"✅ Created config with {config.num_labels} labels")
    
    # Create model
    model = LtgBertForSequenceClassification(config)
    print(f"✅ Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, config.num_labels, (batch_size,))
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Labels: {labels}")
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    print(f"✅ Forward pass successful!")
    print(f"   - Logits shape: {outputs.logits.shape}")
    print(f"   - Loss: {outputs.loss:.4f}")
    print(f"   - Predictions: {torch.argmax(outputs.logits, dim=-1)}")
    
    return model


if __name__ == "__main__":
    model = test_simple_classification()
    print("\n🎉 Simple test passed! The classification model works correctly.")
