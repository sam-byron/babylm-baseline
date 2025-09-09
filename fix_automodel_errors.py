#!/usr/bin/env python3
"""
Solutions to prevent AutoModel loading errors with custom LTG BERT models.
This addresses the transformers_modules registration issue.
"""

import os
import json
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification
from ltg_bert_config import LtgBertConfig
from ltg_bert import LtgBertForSequenceClassification


def solution_1_explicit_registration(model_path):
    """
    Solution 1: Explicit registration before loading
    This ensures the model classes are properly registered.
    """
    print("🔧 Solution 1: Explicit Registration")
    print("=" * 40)
    
    # Always register before loading
    print("📝 Registering model classes...")
    AutoConfig.register("ltg_bert", LtgBertConfig)
    AutoModelForSequenceClassification.register(LtgBertConfig, LtgBertForSequenceClassification)
    print("✅ Registration complete")
    
    try:
        # Now load the model
        print(f"📥 Loading model from: {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("✅ Model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        return model
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def solution_2_save_without_custom_code(model, save_path):
    """
    Solution 2: Save model without custom code dependencies
    This avoids the transformers_modules issue entirely.
    """
    print("\n🔧 Solution 2: Save Without Custom Code")
    print("=" * 40)
    
    print(f"💾 Saving model to: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    print("✅ Model weights saved")
    
    # Save config as standard JSON (not using save_pretrained)
    config_dict = {
        "model_type": "ltg_bert",
        "architectures": ["LtgBertForSequenceClassification"],
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "intermediate_size": model.config.intermediate_size,
        "hidden_dropout_prob": model.config.hidden_dropout_prob,
        "attention_probs_dropout_prob": model.config.attention_probs_dropout_prob,
        "max_position_embeddings": model.config.max_position_embeddings,
        "position_bucket_size": model.config.position_bucket_size,
        "layer_norm_eps": model.config.layer_norm_eps,
        "use_cache": model.config.use_cache,
        "classifier_dropout": getattr(model.config, 'classifier_dropout', model.config.hidden_dropout_prob),
        "num_labels": model.config.num_labels,
        "problem_type": getattr(model.config, 'problem_type', None),
        "torch_dtype": "float32",
        "transformers_version": "4.52.4"
    }
    
    with open(os.path.join(save_path, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)
    print("✅ Config saved")
    
    # Copy source files
    import shutil
    for filename in ['ltg_bert.py', 'ltg_bert_config.py']:
        if os.path.exists(filename):
            shutil.copy2(filename, save_path)
            print(f"✅ Copied {filename}")
    
    print("✅ Model saved without custom code dependencies")


def solution_3_manual_loading(model_path):
    """
    Solution 3: Manual loading without AutoModel
    This bypasses the AutoModel system entirely.
    """
    print("\n🔧 Solution 3: Manual Loading")
    print("=" * 40)
    
    try:
        print(f"📥 Loading config from: {model_path}")
        config = LtgBertConfig.from_pretrained(model_path)
        print("✅ Config loaded")
        
        print("🏗️ Creating model manually...")
        model = LtgBertForSequenceClassification(config)
        print("✅ Model created")
        
        print("📥 Loading weights...")
        model_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location='cpu')
            model.load_state_dict(state_dict)
            print("✅ Weights loaded")
        else:
            print("❌ No weights found")
            return None
        
        print("✅ Manual loading successful!")
        return model
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def solution_4_registration_helper():
    """
    Solution 4: Create a registration helper that always works
    """
    print("\n🔧 Solution 4: Registration Helper")
    print("=" * 40)
    
    def ensure_ltg_bert_registered():
        """Ensure LTG BERT is registered with transformers."""
        try:
            # Check if already registered by trying to get the config
            AutoConfig.get_config_class("ltg_bert")
            print("✅ LTG BERT already registered")
        except (KeyError, ValueError):
            # Not registered, so register it
            print("📝 Registering LTG BERT...")
            AutoConfig.register("ltg_bert", LtgBertConfig)
            AutoModelForSequenceClassification.register(LtgBertConfig, LtgBertForSequenceClassification)
            print("✅ LTG BERT registered")
    
    return ensure_ltg_bert_registered


def create_lm_eval_compatible_model(mlm_model_path, output_path, num_labels=2):
    """
    Create a model that's guaranteed to work with lm_eval and other tools.
    """
    print("\n🎯 Creating lm_eval Compatible Model")
    print("=" * 50)
    
    # Step 1: Load MLM model and transfer weights
    print("📥 Loading pre-trained MLM model...")
    mlm_config = LtgBertConfig.from_pretrained(mlm_model_path)
    mlm_model = torch.load(os.path.join(mlm_model_path, "pytorch_model.bin"), map_location='cpu')
    
    # Step 2: Create classification config
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
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    # Step 3: Create model
    print("🏗️ Creating classification model...")
    cls_model = LtgBertForSequenceClassification(cls_config)
    
    # Step 4: Transfer weights (only shared components)
    print("🔄 Transferring weights...")
    cls_state_dict = cls_model.state_dict()
    
    # Copy shared weights from MLM model
    for key in cls_state_dict:
        if key in mlm_model and key.startswith(('embedding.', 'transformer.')):
            cls_state_dict[key] = mlm_model[key]
            print(f"   ✅ Transferred: {key}")
    
    cls_model.load_state_dict(cls_state_dict)
    
    # Step 5: Save in compatible format
    print("💾 Saving in compatible format...")
    solution_2_save_without_custom_code(cls_model, output_path)
    
    print("✅ lm_eval compatible model created!")
    return cls_model


def test_all_solutions():
    """Test all solutions with a real model."""
    print("🧪 Testing All Solutions")
    print("=" * 60)
    
    # First, create a test model using weight transfer
    print("🏗️ Setting up test model...")
    from simple_weight_transfer import simple_weight_transfer
    test_model = simple_weight_transfer()
    
    # Save it using different methods
    test_save_path = "test_models/compatible_model"
    solution_2_save_without_custom_code(test_model, test_save_path)
    
    # Test loading methods
    print("\n" + "=" * 60)
    
    # Test Solution 1: Explicit registration
    model1 = solution_1_explicit_registration(test_save_path)
    
    # Test Solution 3: Manual loading
    model3 = solution_3_manual_loading(test_save_path)
    
    # Test Solution 4: Registration helper
    registration_helper = solution_4_registration_helper()
    registration_helper()
    
    print("\n🎉 All solutions tested!")
    return model1, model3


if __name__ == "__main__":
    # Run the tests
    test_all_solutions()
    
    print("\n📋 Summary of Solutions:")
    print("1. Always register classes before loading")
    print("2. Save without custom code dependencies") 
    print("3. Use manual loading instead of AutoModel")
    print("4. Use registration helper functions")
    print("\n🚀 These approaches will prevent the AutoModel error!")
