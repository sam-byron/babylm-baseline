#!/usr/bin/env python3
"""
Test script to verify the model loads correctly for MLM evaluation
"""

from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
import torch

def test_model_loading():
    """Test that the model loads correctly as a masked language model"""
    checkpoint_path = "model_babylm_bert_ltg/checkpoint"
    
    print("Loading config...")
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    print(f"✓ Config loaded: {type(config)}")
    print(f"  Model type: {config.model_type}")
    print(f"  Architectures: {config.architectures}")
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    print(f"✓ Tokenizer loaded: {type(tokenizer)}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    print("\nLoading model...")
    model = AutoModelForMaskedLM.from_pretrained(checkpoint_path, trust_remote_code=True)
    print(f"✓ Model loaded: {type(model)}")
    
    print("\nTesting MLM inference...")
    # Test with a simple masked sentence
    text = "The quick brown [MASK] jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # Find the masked token position
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_logits = logits[0, mask_token_index, :]
    
    # Get top predictions
    top_tokens = torch.topk(mask_logits, 5, dim=1).indices[0].tolist()
    
    print(f"✓ MLM inference successful!")
    print(f"  Input: {text}")
    print(f"  Top predictions for [MASK]:")
    for i, token_id in enumerate(top_tokens):
        token = tokenizer.decode([token_id])
        print(f"    {i+1}. {token}")
    
    return True

if __name__ == "__main__":
    try:
        test_model_loading()
        print("\n🎉 SUCCESS: Model loads and works correctly for MLM!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
