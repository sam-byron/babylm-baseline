#!/usr/bin/env python3
"""
LM-Eval Compatible Model Creation Guide

This script shows you how to create models that work with lm_eval and other 
evaluation frameworks without the AutoModel registration errors.
"""

import os
import json
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification
from ltg_bert_config import LtgBertConfig
from ltg_bert import LtgBertForSequenceClassification


def create_lm_eval_model(source_model_path, output_path, task_type="classification"):
    """
    Create a model specifically designed to work with lm_eval.
    
    Args:
        source_model_path: Path to your trained MLM model
        output_path: Where to save the lm_eval compatible model
        task_type: "classification" or "generation"
    """
    print(f"🎯 Creating lm_eval compatible model for {task_type}")
    print("=" * 50)
    
    # Step 1: Ensure registration
    print("📝 Registering model classes...")
    try:
        AutoConfig.register("ltg_bert", LtgBertConfig)
        if task_type == "classification":
            AutoModelForSequenceClassification.register(LtgBertConfig, LtgBertForSequenceClassification)
        print("✅ Registration successful")
    except Exception as e:
        print(f"⚠️ Registration warning: {e}")
    
    # Step 2: Load source model
    print("📥 Loading source model...")
    config = LtgBertConfig.from_pretrained(source_model_path)
    
    if task_type == "classification":
        # For classification tasks
        config.num_labels = 2  # Adjust as needed
        config.problem_type = "single_label_classification"
        model = LtgBertForSequenceClassification(config)
        
        # Load MLM weights
        mlm_state_dict = torch.load(os.path.join(source_model_path, "pytorch_model.bin"), map_location='cpu')
        
        # Transfer shared weights
        model_state_dict = model.state_dict()
        for key in model_state_dict:
            if key in mlm_state_dict and key.startswith(('embedding.', 'transformer.')):
                model_state_dict[key] = mlm_state_dict[key]
        
        model.load_state_dict(model_state_dict)
        print("✅ Classification model created with transferred weights")
        
    else:
        # For generation tasks, use MLM model as-is
        from ltg_bert import LtgBertForMaskedLM
        model = LtgBertForMaskedLM.from_pretrained(source_model_path)
        print("✅ Generation model loaded")
    
    # Step 3: Save in lm_eval compatible format
    print("💾 Saving in lm_eval compatible format...")
    os.makedirs(output_path, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
    
    # Create a clean config without custom code references
    clean_config = {
        "model_type": "ltg_bert",
        "architectures": [model.__class__.__name__],
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "intermediate_size": config.intermediate_size,
        "hidden_dropout_prob": config.hidden_dropout_prob,
        "attention_probs_dropout_prob": config.attention_probs_dropout_prob,
        "max_position_embeddings": config.max_position_embeddings,
        "position_bucket_size": config.position_bucket_size,
        "layer_norm_eps": config.layer_norm_eps,
        "use_cache": config.use_cache,
        "torch_dtype": "float32",
        "transformers_version": "4.52.4"
    }
    
    if task_type == "classification":
        clean_config.update({
            "num_labels": config.num_labels,
            "problem_type": config.problem_type,
            "classifier_dropout": getattr(config, 'classifier_dropout', config.hidden_dropout_prob)
        })
    
    with open(os.path.join(output_path, "config.json"), 'w') as f:
        json.dump(clean_config, f, indent=2)
    
    # Copy source files for custom code loading
    import shutil
    source_files = ['ltg_bert.py', 'ltg_bert_config.py']
    for filename in source_files:
        if os.path.exists(filename):
            shutil.copy2(filename, output_path)
            print(f"✅ Copied {filename}")
    
    # Copy tokenizer files if they exist
    tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'special_tokens_map.json']
    for filename in tokenizer_files:
        source_file = os.path.join(source_model_path, filename)
        if os.path.exists(source_file):
            shutil.copy2(source_file, output_path)
            print(f"✅ Copied {filename}")
    
    print("✅ lm_eval compatible model saved!")
    return model


def create_lm_eval_script(model_path):
    """Create a script that properly loads the model for lm_eval."""
    
    script_content = f'''#!/usr/bin/env python3
"""
LM-Eval loading script for LTG BERT models.
Use this script to ensure proper model loading with lm_eval.
"""

import sys
import os

# Add current directory to path for custom modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoConfig, AutoModelForSequenceClassification
from ltg_bert_config import LtgBertConfig
from ltg_bert import LtgBertForSequenceClassification

# Register the model classes
print("Registering LTG BERT classes...")
AutoConfig.register("ltg_bert", LtgBertConfig)
AutoModelForSequenceClassification.register(LtgBertConfig, LtgBertForSequenceClassification)

def load_model(model_path="{model_path}"):
    """Load the LTG BERT model for evaluation."""
    print(f"Loading LTG BERT model from: {{model_path}}")
    
    # Method 1: Try AutoModel loading
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("✅ Loaded via AutoModelForSequenceClassification")
        return model
    except Exception as e:
        print(f"AutoModel loading failed: {{e}}")
    
    # Method 2: Manual loading
    try:
        config = LtgBertConfig.from_pretrained(model_path)
        model = LtgBertForSequenceClassification(config)
        
        import torch
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu')
        model.load_state_dict(state_dict)
        
        print("✅ Loaded via manual loading")
        return model
    except Exception as e:
        print(f"Manual loading failed: {{e}}")
        raise

if __name__ == "__main__":
    model = load_model()
    print(f"Model type: {{type(model).__name__}}")
    print(f"Config: {{model.config}}")
'''
    
    script_path = os.path.join(model_path, "load_for_eval.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Created loading script: {script_path}")


def test_lm_eval_compatibility(model_path):
    """Test that the model can be loaded correctly for evaluation."""
    print(f"🧪 Testing lm_eval compatibility for: {model_path}")
    print("=" * 50)
    
    # Test 1: Registration
    print("📝 Testing registration...")
    try:
        AutoConfig.register("ltg_bert", LtgBertConfig)
        AutoModelForSequenceClassification.register(LtgBertConfig, LtgBertForSequenceClassification)
        print("✅ Registration successful")
    except Exception as e:
        print(f"❌ Registration failed: {e}")
        return False
    
    # Test 2: AutoModel loading
    print("🤖 Testing AutoModel loading...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("✅ AutoModel loading successful")
    except Exception as e:
        print(f"❌ AutoModel loading failed: {e}")
        print("   This is the error you'd see in lm_eval")
        
        # Test fallback: Manual loading
        print("🔧 Testing manual loading fallback...")
        try:
            config = LtgBertConfig.from_pretrained(model_path)
            model = LtgBertForSequenceClassification(config)
            
            state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu')
            model.load_state_dict(state_dict)
            print("✅ Manual loading successful")
        except Exception as e2:
            print(f"❌ Manual loading also failed: {e2}")
            return False
    
    # Test 3: Model functionality
    print("🚀 Testing model functionality...")
    try:
        # Quick forward pass test
        import torch
        input_ids = torch.randint(0, model.config.vocab_size, (1, 10))
        attention_mask = torch.ones(1, 10)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        print(f"✅ Model works! Output shape: {outputs.logits.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Model functionality test failed: {e}")
        return False


def main():
    """Main function to create lm_eval compatible models."""
    print("🎯 LM-Eval Compatibility Setup")
    print("=" * 60)
    
    source_mlm_path = "model_babylm_bert_ltg/checkpoint"
    
    # Create classification model for lm_eval
    classification_output = "lm_eval_models/classification"
    if os.path.exists(source_mlm_path):
        print("Creating classification model for lm_eval...")
        create_lm_eval_model(source_mlm_path, classification_output, "classification")
        create_lm_eval_script(classification_output)
        
        # Test compatibility
        is_compatible = test_lm_eval_compatibility(classification_output)
        
        if is_compatible:
            print("\n🎉 SUCCESS! Your model is lm_eval compatible!")
            print(f"📁 Model saved to: {classification_output}")
            print("📋 To use with lm_eval:")
            print(f"   cd {classification_output}")
            print("   python load_for_eval.py  # Test the model")
            print("   # Then use the model path in lm_eval")
        else:
            print("\n❌ Compatibility issues detected. See error messages above.")
    
    else:
        print(f"❌ Source MLM model not found at: {source_mlm_path}")
        print("   Make sure you have a trained model there first.")


if __name__ == "__main__":
    main()
