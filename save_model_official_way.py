#!/usr/bin/env python3
"""
Official HuggingFace-compatible model saving script.

This saves your model in the standard way that works with all HuggingFace tools
and evaluation frameworks like lm_eval.
"""

import os
import shutil
from transformers import AutoConfig, AutoModelForSequenceClassification
from ltg_bert_config import LtgBertConfig
from ltg_bert import LtgBertForSequenceClassification, LtgBertForMaskedLM
import torch


def save_model_official_way(source_model_path, output_path, task_type="classification"):
    """
    Save a model the official HuggingFace way for use with lm_eval and other tools.
    
    Args:
        source_model_path: Path to your trained MLM model
        output_path: Where to save the official model
        task_type: "classification" or "mlm"
    """
    print(f"🎯 Saving model the official HuggingFace way")
    print("=" * 50)
    
    # Step 1: Load the model and config
    print("📥 Loading source model...")
    config = LtgBertConfig.from_pretrained(source_model_path)
    
    if task_type == "classification":
        # For classification, create new model and transfer weights
        config.num_labels = 2  # Adjust as needed
        config.problem_type = "single_label_classification"
        model = LtgBertForSequenceClassification(config)
        
        # Load MLM weights and transfer shared layers
        mlm_state_dict = torch.load(os.path.join(source_model_path, "pytorch_model.bin"), map_location='cpu')
        model_state_dict = model.state_dict()
        
        # Transfer embedding and transformer weights
        for key in model_state_dict:
            if key in mlm_state_dict and key.startswith(('embedding.', 'transformer.')):
                model_state_dict[key] = mlm_state_dict[key]
        
        model.load_state_dict(model_state_dict)
        print("✅ Classification model created with transferred weights")
        
    else:
        # For MLM, just load directly
        model = LtgBertForMaskedLM.from_pretrained(source_model_path)
        print("✅ MLM model loaded")
    
    # Step 2: Save with proper naming convention
    print("💾 Saving model the official way...")
    os.makedirs(output_path, exist_ok=True)
    
    # This is the key part - register for auto class!
    config.register_for_auto_class()
    if task_type == "classification":
        model.register_for_auto_class("AutoModelForSequenceClassification")
    else:
        model.register_for_auto_class("AutoModelForMaskedLM")
    
    # Save model and config
    model.save_pretrained(output_path)
    config.save_pretrained(output_path)
    
    # Step 3: Copy source files with proper naming
    print("📁 Copying source files with proper naming...")
    
    # Copy with the required naming convention
    dest_modeling = os.path.join(output_path, f"modeling_{config.model_type}.py")
    dest_config = os.path.join(output_path, f"configuration_{config.model_type}.py")
    
    if os.path.exists("ltg_bert.py"):
        shutil.copy2("ltg_bert.py", dest_modeling)
        print(f"✅ Copied ltg_bert.py -> {dest_modeling}")
    
    if os.path.exists("ltg_bert_config.py"):
        shutil.copy2("ltg_bert_config.py", dest_config)
        print(f"✅ Copied ltg_bert_config.py -> {dest_config}")
    
    # Copy tokenizer files
    tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'special_tokens_map.json']
    for filename in tokenizer_files:
        source_file = os.path.join(source_model_path, filename)
        if os.path.exists(source_file):
            shutil.copy2(source_file, output_path)
            print(f"✅ Copied {filename}")
    
    print(f"✅ Model saved to: {output_path}")
    return output_path


def test_official_loading(model_path):
    """Test loading the model the official way."""
    print(f"🧪 Testing official loading: {model_path}")
    print("=" * 50)
    
    try:
        # This is how lm_eval will load it
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Config type: {type(config).__name__}")
        
        # Test forward pass
        import torch
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        attention_mask = torch.ones(1, 10)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        print(f"✅ Forward pass works! Output shape: {outputs.logits.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        return False


def main():
    """Save model the official HuggingFace way."""
    print("🎯 Official HuggingFace Model Saving")
    print("=" * 60)
    
    source_mlm_path = "model_babylm_bert_ltg/checkpoint"
    official_output = "official_models/classification"
    
    if os.path.exists(source_mlm_path):
        # Save the model the official way
        save_model_official_way(source_mlm_path, official_output, "classification")
        
        # Test that it works
        success = test_official_loading(official_output)
        
        if success:
            print("\n🎉 SUCCESS! Your model is saved the official way!")
            print(f"📁 Model saved to: {official_output}")
            print("📋 To use with lm_eval:")
            print(f"   lm_eval --model hf \\")
            print(f"       --model_args pretrained={official_output},trust_remote_code=True \\")
            print(f"       --tasks your_tasks")
            print("\n💡 That's it! No special scripts, no complex registration.")
            print("   Just use trust_remote_code=True - this is the official way!")
        else:
            print("\n❌ Something went wrong. Check the error messages above.")
    
    else:
        print(f"❌ Source MLM model not found at: {source_mlm_path}")
        print("   Make sure you have a trained model there first.")


if __name__ == "__main__":
    main()
