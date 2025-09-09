#!/usr/bin/env python3
"""
Transfer weights from pre-trained LtgBertForMaskedLM to LtgBertForSequenceClassification
This allows you to use your pre-trained model for classification tasks without retraining!
"""

import torch
import os
import json
from ltg_bert_config import LtgBertConfig
from ltg_bert import LtgBertForMaskedLM, LtgBertForSequenceClassification


def load_mlm_model(checkpoint_path):
    """Load the pre-trained MLM model from checkpoint"""
    print(f"Loading MLM model from: {checkpoint_path}")
    
    # Load config
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = LtgBertConfig(**config_dict)
    print(f"✓ Config loaded: {config.hidden_size}D, {config.num_hidden_layers} layers")
    
    # Load weights - try safetensors first, then pytorch_model.bin
    model_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(model_path):
        print("✓ Loading from model.safetensors")
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
    else:
        model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        print("✓ Loading from pytorch_model.bin")
        state_dict = torch.load(model_path, map_location='cpu')
    
    print(f"✓ Checkpoint loaded with {len(state_dict)} parameters")
    
    return state_dict, config


def transfer_shared_weights(state_dict, classification_model):
    """Transfer only the shared weights (embedding + transformer) from state dict"""
    print("Transferring shared weights...")
    
    # Extract embedding weights
    embedding_keys = [k for k in state_dict.keys() if k.startswith('embedding.')]
    embedding_state = {k: state_dict[k] for k in embedding_keys}
    
    if embedding_state:
        classification_model.embedding.load_state_dict(embedding_state, strict=True)
        print("✓ Embedding weights transferred")
    else:
        print("❌ No embedding weights found in checkpoint")
        return False
    
    # Extract transformer weights
    transformer_keys = [k for k in state_dict.keys() if k.startswith('transformer.')]
    transformer_state = {k: state_dict[k] for k in transformer_keys}
    
    if transformer_state:
        classification_model.transformer.load_state_dict(transformer_state, strict=True)
        print("✓ Transformer weights transferred")
    else:
        print("❌ No transformer weights found in checkpoint")
        return False
    
    # Classification head is randomly initialized (as it should be for fine-tuning)
    print("✓ Classification head randomly initialized for fine-tuning")
    
    return True


def verify_transfer(state_dict, classification_model):
    """Verify that weights were transferred correctly"""
    print("Verifying weight transfer...")
    
    # Check embedding weights
    checkpoint_emb = state_dict['embedding.word_embedding.weight']
    cls_emb = classification_model.embedding.word_embedding.weight
    
    if torch.allclose(checkpoint_emb, cls_emb):
        print("✓ Embedding weights match perfectly")
    else:
        print("❌ Embedding weights don't match!")
        return False
    
    # Check first transformer layer weights as a sample
    checkpoint_layer0 = state_dict['transformer.layers.0.attention.in_proj_qk.weight']
    cls_layer0 = classification_model.transformer.layers[0].attention.in_proj_qk.weight
    
    if torch.allclose(checkpoint_layer0, cls_layer0):
        print("✓ Transformer weights match perfectly")
    else:
        print("❌ Transformer weights don't match!")
        return False
    
    print("✓ Weight transfer verification successful!")
    return True


def create_classification_model(config, num_labels, problem_type=None):
    """Create a classification model with the same config"""
    print(f"Creating classification model for {num_labels} labels")
    
    # Update config for classification
    classification_config = LtgBertConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        max_position_embeddings=config.max_position_embeddings,
        position_bucket_size=config.position_bucket_size,
        layer_norm_eps=config.layer_norm_eps,
        classifier_dropout=config.hidden_dropout_prob,  # Use same dropout
        num_labels=num_labels,
        problem_type=problem_type
    )
    
    classification_model = LtgBertForSequenceClassification(classification_config)
    print(f"✓ Classification model created with {sum(p.numel() for p in classification_model.parameters()):,} parameters")
    
    return classification_model


def transfer_weights(mlm_model, classification_model):
    """Transfer shared weights from MLM to classification model"""
    print("Transferring weights...")
    
    # Transfer embedding weights
    classification_model.embedding.load_state_dict(mlm_model.embedding.state_dict())
    print("✓ Embedding weights transferred")
    
    # Transfer transformer weights
    classification_model.transformer.load_state_dict(mlm_model.transformer.state_dict())
    print("✓ Transformer weights transferred")
    
    # Classification head is randomly initialized (as it should be for fine-tuning)
    print("✓ Classification head randomly initialized for fine-tuning")
    
    return classification_model


def save_classification_model(model, save_path):
    """Save the classification model"""
    print(f"Saving classification model to: {save_path}")
    
    os.makedirs(save_path, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    
    # Save config
    model.config.save_pretrained(save_path)
    
    print(f"✓ Model saved to {save_path}")


def verify_transfer(mlm_model, classification_model):
    """Verify that weights were transferred correctly"""
    print("Verifying weight transfer...")
    
    # Check embedding weights
    mlm_emb = mlm_model.embedding.word_embedding.weight
    cls_emb = classification_model.embedding.word_embedding.weight
    
    if torch.allclose(mlm_emb, cls_emb):
        print("✓ Embedding weights match perfectly")
    else:
        print("❌ Embedding weights don't match!")
        return False
    
    # Check first transformer layer weights as a sample
    mlm_layer0 = mlm_model.transformer.layers[0].attention.in_proj_qk.weight
    cls_layer0 = classification_model.transformer.layers[0].attention.in_proj_qk.weight
    
    if torch.allclose(mlm_layer0, cls_layer0):
        print("✓ Transformer weights match perfectly")
    else:
        print("❌ Transformer weights don't match!")
        return False
    
    print("✓ Weight transfer verification successful!")
    return True


def main():
    """Main function to transfer weights from MLM to classification model"""
    print("🚀 LTG BERT Weight Transfer Utility")
    print("=" * 50)
    
    # Paths
    mlm_checkpoint_path = "model_babylm_bert_ltg/checkpoint"
    classification_save_path = "model_babylm_bert_ltg_classification"
    
    # Configuration for classification task
    num_labels = 2  # Change this for your specific task
    problem_type = None  # Let the model auto-detect, or set explicitly
    
    try:
        # Step 1: Load pre-trained MLM checkpoint
        state_dict, config = load_mlm_model(mlm_checkpoint_path)
        
        # Step 2: Create classification model
        classification_model = create_classification_model(config, num_labels, problem_type)
        
        # Step 3: Transfer shared weights
        if transfer_shared_weights(state_dict, classification_model):
            # Step 4: Verify transfer
            if verify_transfer(state_dict, classification_model):
                # Step 5: Save classification model
                save_classification_model(classification_model, classification_save_path)
                
                print("\n🎉 Weight transfer completed successfully!")
                print(f"✓ Classification model ready at: {classification_save_path}")
                print(f"✓ Model has {num_labels} labels for classification")
                print("\n📝 Next steps:")
                print("1. Load the classification model for fine-tuning")
                print("2. Prepare your classification dataset")
                print("3. Fine-tune only the classification head (or full model)")
                print("4. The shared weights (embedding + transformer) are already pre-trained!")
                
            else:
                print("❌ Weight transfer verification failed!")
        else:
            print("❌ Weight transfer failed!")
            
    except Exception as e:
        print(f"❌ Error during weight transfer: {e}")
        raise


def create_usage_example():
    """Create an example of how to use the transferred model"""
    example_code = '''
# Example: How to use the transferred classification model

import torch
from transformers import AutoTokenizer
from ltg_bert import LtgBertForSequenceClassification
from ltg_bert_config import LtgBertConfig

# Load the transferred model
model_path = "model_babylm_bert_ltg_classification"
config = LtgBertConfig.from_pretrained(model_path)
model = LtgBertForSequenceClassification.from_pretrained(model_path, config=config)

# Load tokenizer (use the same one from your MLM training)
tokenizer_path = "model_babylm_bert_ltg/checkpoint"  # Has tokenizer.json
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Example inference (after fine-tuning)
texts = ["This is a positive example", "This is a negative example"]
# inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Forward pass
# with torch.no_grad():
#     outputs = model(**inputs)
#     predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     predicted_classes = torch.argmax(predictions, dim=-1)

print("Model ready for fine-tuning on your classification task!")
'''
    
    with open("classification_model_usage_example.py", "w") as f:
        f.write(example_code)
    
    print("✓ Usage example saved to: classification_model_usage_example.py")


if __name__ == "__main__":
    main()
    create_usage_example()
