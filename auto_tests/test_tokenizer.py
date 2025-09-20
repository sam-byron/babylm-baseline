#!/usr/bin/env python3

import os
import json
from tokenizer import Tokenizer

def test_tokenizer():
    """Test that our tokenizer is working correctly"""
    
    # Load configuration
    with open("model_babylm_ltg_bert.json", "r") as f:
        config = json.load(f)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(config["tokenizer_path"])
    tokenizer.enable_padding(length=config["max_position_embeddings"])
    tokenizer.enable_truncation(max_length=config["max_position_embeddings"])
    
    # Test text samples
    test_texts = [
        "Hello world, this is a test sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is fascinating!",
        "This tokenizer should handle various texts properly."
    ]
    
    print("Testing individual encoding...")
    for i, text in enumerate(test_texts):
        encoding = tokenizer.encode(text)
        print(f"Text {i}: '{text}'")
        print(f"  Tokens: {encoding.tokens[:10]}...")  # First 10 tokens
        print(f"  IDs: {encoding.ids[:10]}...")       # First 10 IDs
        print(f"  Length: {len(encoding.ids)}")
        print()
    
    print("Testing batch encoding...")
    batch_encodings = tokenizer.encode_batch(test_texts)
    
    for i, encoding in enumerate(batch_encodings):
        print(f"Batch text {i}: tokens={len(encoding.ids)}, unique_tokens={len(set(encoding.ids))}")
        # Check for excessive UNK tokens
        unk_count = encoding.ids.count(tokenizer.token_to_id('[UNK]')) if tokenizer.token_to_id('[UNK]') else 0
        print(f"  UNK tokens: {unk_count}/{len(encoding.ids)} ({100*unk_count/len(encoding.ids):.1f}%)")
        print()
    
    print("Tokenizer test completed successfully!")
    return True

if __name__ == "__main__":
    test_tokenizer()
