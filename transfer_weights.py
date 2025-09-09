#!/usr/bin/env python3
"""
Weight Transfer Utility for LtgBert Models

This script transfers pre-trained weights from LtgBertForMaskedLM to LtgBertForSequenceClassification.
Only the shared components (embedding + transformer) are transferred. The classification head
is randomly initialized and ready for fine-tuning.
"""

import torch
import os
from transformers import AutoConfig, AutoModelForMaskedLM, AutoModelForSequenceClassification
from ltg_bert_config import LtgBertConfig
from ltg_bert import LtgBertForMaskedLM, LtgBertForSequenceClassification


def transfer_weights_to_classification(
    mlm_model_path,
    num_labels,
    output_path=None,
    classifier_dropout=None,
    problem_type=None
):
    """
    Transfer weights from a pre-trained LtgBertForMaskedLM to LtgBertForSequenceClassification.
    
    Args:
        mlm_model_path (str): Path to the pre-trained MLM model directory
        num_labels (int): Number of labels for classification task
        output_path (str, optional): Path to save the classification model. If None, returns model in memory.
        classifier_dropout (float, optional): Dropout rate for classification head
        problem_type (str, optional): Type of problem ('single_label_classification', 'multi_label_classification', 'regression')
    
    Returns:
        LtgBertForSequenceClassification: The model with transferred weights
    """
    
    print(f"🔄 Transferring weights from MLM model: {mlm_model_path}")
    
    # Load the pre-trained MLM model
    print("📥 Loading pre-trained MLM model...")
    mlm_config = LtgBertConfig.from_pretrained(mlm_model_path)
    mlm_model = LtgBertForMaskedLM.from_pretrained(mlm_model_path)
    
    print(f"✅ Loaded MLM model with config:")
    print(f"   - Vocab size: {mlm_config.vocab_size}")
    print(f"   - Hidden size: {mlm_config.hidden_size}")
    print(f"   - Num layers: {mlm_config.num_hidden_layers}")
    print(f"   - Attention heads: {mlm_config.num_attention_heads}")
    
    # Create classification config based on MLM config
    print("⚙️ Creating classification model config...")
    classification_config = LtgBertConfig(
        vocab_size=mlm_config.vocab_size,
        hidden_size=mlm_config.hidden_size,
        num_hidden_layers=mlm_config.num_hidden_layers,
        num_attention_heads=mlm_config.num_attention_heads,
        intermediate_size=mlm_config.intermediate_size,
        hidden_dropout_prob=mlm_config.hidden_dropout_prob,
        attention_probs_dropout_prob=mlm_config.attention_probs_dropout_prob,
        max_position_embeddings=mlm_config.max_position_embeddings,
        position_bucket_size=mlm_config.position_bucket_size,
        layer_norm_eps=mlm_config.layer_norm_eps,
        use_cache=mlm_config.use_cache,
        classifier_dropout=classifier_dropout or mlm_config.hidden_dropout_prob,
        num_labels=num_labels,
        problem_type=problem_type
    )
    
    # Create classification model
    print(f"🏗️ Creating classification model with {num_labels} labels...")
    classification_model = LtgBertForSequenceClassification(classification_config)
    
    # Transfer shared weights
    print("🔄 Transferring shared weights...")
    
    # 1. Transfer embedding weights
    print("   📝 Transferring embedding layer...")
    classification_model.embedding.load_state_dict(mlm_model.embedding.state_dict())
    
    # 2. Transfer transformer weights
    print("   🔧 Transferring transformer layers...")
    classification_model.transformer.load_state_dict(mlm_model.transformer.state_dict())
    
    print("✅ Weight transfer completed!")
    print(f"   - Shared components: ✅ Transferred")
    print(f"   - Classification head: 🎲 Randomly initialized (ready for fine-tuning)")
    
    # Verify the transfer worked
    print("🔍 Verifying weight transfer...")
    verify_weight_transfer(mlm_model, classification_model)
    
    # Save if output path provided
    if output_path:
        print(f"💾 Saving classification model to: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        
        # Ensure the config has the correct num_labels before saving
        classification_model.config.num_labels = num_labels
        
        classification_model.save_pretrained(output_path)
        print("✅ Model saved successfully!")
    
    return classification_model


def verify_weight_transfer(mlm_model, classification_model):
    """Verify that weights were transferred correctly."""
    
    # Check embedding weights
    mlm_embed_weight = mlm_model.embedding.word_embedding.weight
    cls_embed_weight = classification_model.embedding.word_embedding.weight
    
    if torch.allclose(mlm_embed_weight, cls_embed_weight):
        print("   ✅ Embedding weights match")
    else:
        print("   ❌ Embedding weights don't match!")
        return False
    
    # Check first transformer layer weights as a sample
    mlm_first_layer = mlm_model.transformer.layers[0]
    cls_first_layer = classification_model.transformer.layers[0]
    
    # Check attention weights
    mlm_attn_weight = mlm_first_layer.attention.in_proj_qk.weight
    cls_attn_weight = cls_first_layer.attention.in_proj_qk.weight
    
    if torch.allclose(mlm_attn_weight, cls_attn_weight):
        print("   ✅ Transformer weights match")
    else:
        print("   ❌ Transformer weights don't match!")
        return False
    
    print("   ✅ Weight transfer verification passed!")
    return True


def load_classification_model_for_finetuning(model_path, num_labels_override=None):
    """
    Load a classification model for fine-tuning.
    
    Args:
        model_path (str): Path to the classification model
        num_labels_override (int, optional): Override the number of labels if needed
    
    Returns:
        LtgBertForSequenceClassification: Ready for fine-tuning
    """
    
    print(f"📥 Loading classification model from: {model_path}")
    
    config = LtgBertConfig.from_pretrained(model_path)
    if num_labels_override:
        config.num_labels = num_labels_override
        print(f"🔧 Overriding num_labels to: {num_labels_override}")
    
    model = LtgBertForSequenceClassification.from_pretrained(model_path, config=config)
    
    print(f"✅ Loaded classification model:")
    print(f"   - Labels: {config.num_labels}")
    print(f"   - Problem type: {config.problem_type}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def compare_model_sizes(mlm_model, classification_model):
    """Compare the parameter counts of both models."""
    
    mlm_params = sum(p.numel() for p in mlm_model.parameters())
    cls_params = sum(p.numel() for p in classification_model.parameters())
    
    print(f"\n📊 Model Comparison:")
    print(f"   MLM model parameters: {mlm_params:,}")
    print(f"   Classification model parameters: {cls_params:,}")
    print(f"   Difference: {cls_params - mlm_params:,} (classification head)")
    
    # Calculate shared vs new parameters
    shared_params = mlm_params - sum(p.numel() for p in mlm_model.classifier.parameters())
    new_params = cls_params - shared_params
    
    print(f"\n🔄 Parameter Transfer Summary:")
    print(f"   Shared (transferred): {shared_params:,} parameters")
    print(f"   New (classification head): {new_params:,} parameters")
    print(f"   Transfer ratio: {shared_params/cls_params*100:.1f}% of parameters transferred")


# Example usage functions
def example_sentiment_analysis():
    """Example: Transfer weights for binary sentiment analysis."""
    
    print("🎭 Example: Binary Sentiment Analysis")
    print("=" * 50)
    
    mlm_path = "model_babylm_bert_ltg/checkpoint"
    output_path = "model_vault/sentiment_classifier"
    
    # Transfer for binary classification (positive/negative sentiment)
    model = transfer_weights_to_classification(
        mlm_model_path=mlm_path,
        num_labels=2,  # Binary: positive/negative
        output_path=output_path,
        problem_type="single_label_classification"
    )
    
    print(f"\n🎉 Sentiment classifier ready!")
    print(f"📁 Saved to: {output_path}")
    print("🚀 Ready for fine-tuning on sentiment data!")
    
    return model


def example_topic_classification():
    """Example: Transfer weights for multi-class topic classification."""
    
    print("📚 Example: Topic Classification")
    print("=" * 50)
    
    mlm_path = "model_babylm_bert_ltg/checkpoint"
    output_path = "model_vault/topic_classifier"
    
    # Transfer for multi-class classification (e.g., news categories)
    model = transfer_weights_to_classification(
        mlm_model_path=mlm_path,
        num_labels=8,  # Example: 8 topic categories
        output_path=output_path,
        problem_type="single_label_classification"
    )
    
    print(f"\n🎉 Topic classifier ready!")
    print(f"📁 Saved to: {output_path}")
    print("🚀 Ready for fine-tuning on topic classification data!")
    
    return model


def example_regression():
    """Example: Transfer weights for regression task."""
    
    print("📈 Example: Regression Task")
    print("=" * 50)
    
    mlm_path = "model_babylm_bert_ltg/checkpoint"
    output_path = "model_vault/regression_model"
    
    # Transfer for regression (e.g., predicting review scores)
    model = transfer_weights_to_classification(
        mlm_model_path=mlm_path,
        num_labels=1,  # Single continuous output
        output_path=output_path,
        problem_type="regression"
    )
    
    print(f"\n🎉 Regression model ready!")
    print(f"📁 Saved to: {output_path}")
    print("🚀 Ready for fine-tuning on regression data!")
    
    return model


if __name__ == "__main__":
    print("🔄 LtgBert Weight Transfer Utility")
    print("=" * 50)
    
    # Check if pre-trained model exists
    mlm_path = "model_babylm_bert_ltg/checkpoint"
    if not os.path.exists(mlm_path):
        print(f"❌ Pre-trained MLM model not found at: {mlm_path}")
        print("Please ensure you have a trained model saved there.")
        exit(1)
    
    print("🎯 Available Examples:")
    print("1. Sentiment Analysis (Binary Classification)")
    print("2. Topic Classification (Multi-class)")
    print("3. Regression Task")
    print()
    
    choice = input("Select example (1-3) or 'q' to quit: ").strip()
    
    if choice == "1":
        example_sentiment_analysis()
    elif choice == "2":
        example_topic_classification()
    elif choice == "3":
        example_regression()
    elif choice.lower() == "q":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Running sentiment analysis example...")
        example_sentiment_analysis()
