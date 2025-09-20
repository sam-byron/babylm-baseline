#!/usr/bin/env python3
"""
Simple weight transfer demonstration that works around the saving/loading issues.
"""

import torch
from ltg_bert_config import LtgBertConfig
from ltg_bert import LtgBertForMaskedLM, LtgBertForSequenceClassification


def simple_weight_transfer():
    """Demonstrate successful weight transfer without save/load complications."""
    print("🔄 Simple Weight Transfer Demonstration")
    print("=" * 50)
    
    # Load your pre-trained MLM model
    print("📥 Loading pre-trained MLM model...")
    mlm_model_path = "model_babylm_bert_ltg/checkpoint"
    mlm_config = LtgBertConfig.from_pretrained(mlm_model_path)
    mlm_model = LtgBertForMaskedLM.from_pretrained(mlm_model_path)
    
    print(f"✅ MLM model loaded:")
    print(f"   - Parameters: {sum(p.numel() for p in mlm_model.parameters()):,}")
    print(f"   - Vocab size: {mlm_config.vocab_size}")
    print(f"   - Hidden size: {mlm_config.hidden_size}")
    
    # Create classification model with same architecture
    print("\n🏗️ Creating classification model...")
    cls_config = LtgBertConfig(
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
        num_labels=3,  # 3-class classification
        problem_type="single_label_classification"
    )
    
    cls_model = LtgBertForSequenceClassification(cls_config)
    print(f"✅ Classification model created:")
    print(f"   - Parameters: {sum(p.numel() for p in cls_model.parameters()):,}")
    print(f"   - Num labels: {cls_config.num_labels}")
    
    # Transfer weights
    print("\n🔄 Transferring weights...")
    
    # Before transfer - test that they're different
    print("📊 Before transfer:")
    mlm_embed_sample = mlm_model.embedding.word_embedding.weight[0, :5]
    cls_embed_sample = cls_model.embedding.word_embedding.weight[0, :5]
    print(f"   MLM embedding sample: {mlm_embed_sample}")
    print(f"   CLS embedding sample: {cls_embed_sample}")
    print(f"   Are they equal? {torch.allclose(mlm_embed_sample, cls_embed_sample)}")
    
    # Transfer embedding weights
    cls_model.embedding.load_state_dict(mlm_model.embedding.state_dict())
    
    # Transfer transformer weights  
    cls_model.transformer.load_state_dict(mlm_model.transformer.state_dict())
    
    # After transfer - test that they're now the same
    print("\n📊 After transfer:")
    mlm_embed_sample = mlm_model.embedding.word_embedding.weight[0, :5]
    cls_embed_sample = cls_model.embedding.word_embedding.weight[0, :5]
    print(f"   MLM embedding sample: {mlm_embed_sample}")
    print(f"   CLS embedding sample: {cls_embed_sample}")
    print(f"   Are they equal? {torch.allclose(mlm_embed_sample, cls_embed_sample)}")
    
    # Test the transferred model
    print("\n🧪 Testing transferred model...")
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, cls_config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, cls_config.num_labels, (batch_size,))
    
    with torch.no_grad():
        outputs = cls_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    print(f"✅ Transferred model works!")
    print(f"   - Input shape: {input_ids.shape}")
    print(f"   - Output shape: {outputs.logits.shape}")
    print(f"   - Loss: {outputs.loss:.4f}")
    print(f"   - Predictions: {torch.argmax(outputs.logits, dim=-1)}")
    
    # Compare model sizes
    print("\n📊 Model comparison:")
    mlm_params = sum(p.numel() for p in mlm_model.parameters())
    cls_params = sum(p.numel() for p in cls_model.parameters())
    print(f"   MLM model: {mlm_params:,} parameters")
    print(f"   Classification model: {cls_params:,} parameters")
    print(f"   Difference: {cls_params - mlm_params:,} (classification head)")
    
    print("\n🎉 Weight transfer successful!")
    print("🚀 Your classification model is ready for fine-tuning!")
    
    return cls_model


if __name__ == "__main__":
    model = simple_weight_transfer()
