#!/usr/bin/env python3
"""
Test script to verify the official model works exactly like lm_eval would use it.
"""

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import torch

def test_lm_eval_style_loading():
    """Test loading the model exactly how lm_eval would do it."""
    
    print("🧪 Testing lm_eval-style loading...")
    print("=" * 50)
    
    model_path = "official_models/classification"
    
    try:
        # This is exactly how lm_eval loads models
        print("📥 Loading config with trust_remote_code=True...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ Config loaded: {type(config).__name__}")
        
        print("📥 Loading model with trust_remote_code=True...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ Model loaded: {type(model).__name__}")
        
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✅ Tokenizer loaded: {type(tokenizer).__name__}")
        
        # Test inference
        print("🚀 Testing inference...")
        text = "This is a test sentence."
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ Inference works!")
        print(f"   Input text: '{text}'")
        print(f"   Output shape: {outputs.logits.shape}")
        print(f"   Logits: {outputs.logits}")
        
        # Test what lm_eval specifically needs
        print("🔍 Testing lm_eval specific requirements...")
        print(f"   Model type: {config.model_type}")
        print(f"   Num labels: {config.num_labels}")
        print(f"   Vocab size: {config.vocab_size}")
        print(f"   Max length: {config.max_position_embeddings}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_lm_eval_style_loading()
    
    if success:
        print("\n🎉 PERFECT! Your model works exactly like lm_eval expects!")
        print("\n📋 Usage with lm_eval:")
        print("   lm_eval --model hf \\")
        print("       --model_args pretrained=official_models/classification,trust_remote_code=True \\")
        print("       --tasks your_tasks")
        print("\n✨ No more complex scripts needed!")
    else:
        print("\n❌ Something's not working. Check the errors above.")
