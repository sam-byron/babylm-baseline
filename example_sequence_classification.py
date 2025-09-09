#!/usr/bin/env python3
"""
Usage example for LtgBertForSequenceClassification model
"""

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from ltg_bert_config import LtgBertConfig
from ltg_bert import LtgBertForSequenceClassification


def example_usage():
    """
    Example showing how to use LtgBertForSequenceClassification for text classification
    """
    print("🚀 LtgBertForSequenceClassification Usage Example")
    print("=" * 50)
    
    # Method 1: Direct instantiation
    print("\n1. Direct Model Instantiation:")
    config = LtgBertConfig(
        vocab_size=30522,  # BERT-like vocab size
        hidden_size=256,   # Smaller for demo
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=512,
        max_position_embeddings=512,
        num_labels=2,  # Binary classification (e.g., sentiment analysis)
        classifier_dropout=0.1
    )
    
    model = LtgBertForSequenceClassification(config)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Method 2: Using AutoModelForSequenceClassification (recommended)
    print("\n2. Using AutoModelForSequenceClassification:")
    auto_model = AutoModelForSequenceClassification.from_config(config)
    print(f"✓ AutoModel created with {sum(p.numel() for p in auto_model.parameters()):,} parameters")
    
    # Example forward pass
    print("\n3. Example Forward Pass:")
    batch_size, seq_len = 4, 128
    
    # Simulate tokenized text input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, config.num_labels, (batch_size,))  # Ground truth labels
    
    # Forward pass with loss computation
    with torch.no_grad():
        outputs = auto_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    print(f"✓ Input shape: {input_ids.shape}")
    print(f"✓ Output logits shape: {outputs.logits.shape}")
    print(f"✓ Loss: {outputs.loss:.4f}")
    print(f"✓ Predictions: {torch.argmax(outputs.logits, dim=-1)}")
    
    # Different classification scenarios
    print("\n4. Different Classification Tasks:")
    
    # Multi-class classification (3 classes)
    config.num_labels = 3
    model_multiclass = LtgBertForSequenceClassification(config)
    labels_multiclass = torch.randint(0, 3, (batch_size,))
    
    with torch.no_grad():
        outputs = model_multiclass(input_ids, attention_mask, labels_multiclass)
    print(f"✓ Multi-class (3 classes) - Logits: {outputs.logits.shape}, Loss: {outputs.loss:.4f}")
    
    # Regression task
    config.num_labels = 1
    model_regression = LtgBertForSequenceClassification(config)
    labels_regression = torch.randn(batch_size)  # Continuous labels for regression (1D)
    
    with torch.no_grad():
        outputs = model_regression(input_ids, attention_mask, labels_regression)
    print(f"✓ Regression - Logits: {outputs.logits.shape}, Loss: {outputs.loss:.4f}")
    
    # Multi-label classification
    config.num_labels = 4
    config.problem_type = "multi_label_classification"
    model_multilabel = LtgBertForSequenceClassification(config)
    labels_multilabel = torch.randint(0, 2, (batch_size, 4)).float()  # Multiple binary labels
    
    with torch.no_grad():
        outputs = model_multilabel(input_ids, attention_mask, labels_multilabel)
    print(f"✓ Multi-label (4 labels) - Logits: {outputs.logits.shape}, Loss: {outputs.loss:.4f}")
    
    print("\n5. Key Features:")
    print("✓ Compatible with transformers AutoModelForSequenceClassification")
    print("✓ Supports binary, multi-class, multi-label, and regression tasks")
    print("✓ Automatic loss computation based on problem type")
    print("✓ Uses [CLS] token for classification (first token)")
    print("✓ Configurable dropout for regularization")
    print("✓ Shares the same architecture as LtgBertForMaskedLM")
    
    print("\n🎉 Usage example completed successfully!")


def integration_with_existing_workflow():
    """
    Show how this integrates with existing MLM training workflow
    """
    print("\n" + "=" * 50)
    print("🔄 Integration with Existing Workflow")
    print("=" * 50)
    
    print("\n1. Loading pre-trained MLM model for fine-tuning:")
    
    # Simulate loading from a checkpoint
    config = LtgBertConfig(
        vocab_size=16384,  # Same as your existing model
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=2048,
        max_position_embeddings=512,
        num_labels=2  # Add classification head
    )
    
    print("✓ Configuration matches your existing MLM model")
    
    # For fine-tuning, you would:
    # 1. Load pre-trained MLM weights
    # 2. Initialize classification head randomly
    # 3. Fine-tune on classification task
    
    classification_model = LtgBertForSequenceClassification(config)
    print(f"✓ Classification model ready with {sum(p.numel() for p in classification_model.parameters()):,} parameters")
    
    print("\n2. Weight sharing between models:")
    print("✓ Embedding layer: Shared")
    print("✓ Transformer layers: Shared") 
    print("✓ Classification head: New (randomly initialized)")
    
    print("\n3. Typical fine-tuning workflow:")
    print("   a) Load pre-trained MLM checkpoint")
    print("   b) Create LtgBertForSequenceClassification with same config")
    print("   c) Load shared weights (embedding + transformer)")
    print("   d) Fine-tune on classification dataset")
    
    print("\n✓ Ready for classification tasks!")


if __name__ == "__main__":
    example_usage()
    integration_with_existing_workflow()
