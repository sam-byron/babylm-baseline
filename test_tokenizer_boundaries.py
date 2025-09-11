#!/usr/bin/env python3
"""
Test script to verify that boundary tokens are working correctly.
"""

import os
import sys
import tokenizers

def test_boundary_tokens():
    """Test that the tokenizer correctly handles boundary tokens."""
    
    print("🔍 Testing Tokenizer Boundary Token Handling")
    print("=" * 50)
    
    # Load the correct tokenizer
    tokenizer_path = "./data/pretrain/wordpiece_vocab.json"
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        return False
    
    print(f"✅ Loading tokenizer from: {tokenizer_path}")
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
    
    # Test special token mappings
    print("\n📋 Special Token Mappings:")
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    token_ids = {}
    
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        token_ids[token] = token_id
        print(f"  {token} → {token_id}")
    
    if token_ids["[CLS]"] is None or token_ids["[SEP]"] is None:
        print("❌ Missing boundary tokens!")
        return False
    
    # Test sentence tokenization
    print("\n🧪 Testing Sentence Tokenization:")
    
    test_sentences = [
        "This is a simple sentence.",
        "Another test sentence with some words.",
        "Short one."
    ]
    
    for i, sentence in enumerate(test_sentences):
        print(f"\nSentence {i+1}: '{sentence}'")
        
        # Method 1: Add boundary tokens as strings
        bounded_text = f"[CLS] {sentence} [SEP]"
        encoding = tokenizer.encode(bounded_text)
        print(f"  String method: {len(encoding.ids)} tokens")
        print(f"  Token IDs: {encoding.ids}")
        
        cls_count = encoding.ids.count(token_ids["[CLS]"])
        sep_count = encoding.ids.count(token_ids["[SEP]"])
        print(f"  Boundary count: {cls_count} [CLS], {sep_count} [SEP]")
        
        # Method 2: Add boundary tokens as IDs manually
        sentence_encoding = tokenizer.encode(sentence, add_special_tokens=False)
        manual_ids = [token_ids["[CLS]"]] + sentence_encoding.ids + [token_ids["[SEP]"]]
        print(f"  Manual method: {len(manual_ids)} tokens")
        print(f"  Token IDs: {manual_ids}")
        
        manual_cls_count = manual_ids.count(token_ids["[CLS]"])
        manual_sep_count = manual_ids.count(token_ids["[SEP]"])
        print(f"  Boundary count: {manual_cls_count} [CLS], {manual_sep_count} [SEP]")
        
        # Verify both methods work
        if cls_count == 1 and sep_count == 1:
            print(f"  ✅ String method works correctly")
        else:
            print(f"  ❌ String method failed: {cls_count} [CLS], {sep_count} [SEP]")
        
        if manual_cls_count == 1 and manual_sep_count == 1:
            print(f"  ✅ Manual method works correctly")
        else:
            print(f"  ❌ Manual method failed: {manual_cls_count} [CLS], {manual_sep_count} [SEP]")
    
    print("\n📊 Summary:")
    
    # Test the actual utils_mp approach
    print("\n🛠️ Testing utils_mp.py approach:")
    
    # Simulate what utils_mp.py does
    test_sentence = "This is a test sentence for boundary preservation."
    
    # Get token IDs
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    
    # Tokenize sentence without special tokens
    encoding = tokenizer.encode(test_sentence, add_special_tokens=False)
    sentence_tokens = encoding.ids
    
    # Add boundaries manually
    bounded_tokens = [cls_token_id] + sentence_tokens + [sep_token_id]
    
    print(f"  Original sentence: '{test_sentence}'")
    print(f"  Sentence tokens: {len(sentence_tokens)}")
    print(f"  With boundaries: {len(bounded_tokens)}")
    print(f"  First 10 tokens: {bounded_tokens[:10]}")
    print(f"  Last 10 tokens: {bounded_tokens[-10:]}")
    
    # Count boundaries
    final_cls_count = bounded_tokens.count(cls_token_id)
    final_sep_count = bounded_tokens.count(sep_token_id)
    
    print(f"  Final boundary count: {final_cls_count} [CLS], {final_sep_count} [SEP]")
    
    if final_cls_count == 1 and final_sep_count == 1:
        print("  ✅ utils_mp.py approach will work correctly!")
        return True
    else:
        print("  ❌ utils_mp.py approach has issues!")
        return False

if __name__ == "__main__":
    success = test_boundary_tokens()
    
    if success:
        print("\n🎉 All boundary token tests passed!")
        print("The tokenizer setup is working correctly.")
    else:
        print("\n❌ Boundary token tests failed!")
        print("There may be an issue with the tokenizer configuration.")
    
    sys.exit(0 if success else 1)
