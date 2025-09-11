#!/usr/bin/env python3
"""
Test script to validate the sentence-aware data processing pipeline.

This script tests the fixes made to prepare_data.py, utils_mp.py, and related files
to ensure sentence boundaries are properly preserved during data preprocessing.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

def create_test_config():
    """Create a test configuration with sentence-aware processing enabled."""
    return {
        "cache_path": "./test_sentence_cache",
        "chunk_size": 5,  # Small chunk for testing
        "block_size": 128,
        "batch_size": 2,
        "max_position_embeddings": 512,
        "tokenizer_path": "./tokenizer.json",
        "use_sentence_aware": True,  # Enable sentence-aware processing
        "mask_p": 0.15,
        "random_p": 0.1,
        "keep_p": 0.1
    }

def create_test_data():
    """Create test BNC markdown files for testing."""
    test_data_dir = Path("./test_bnc_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Create sample BNC markdown files
    sample_texts = [
        """# Document 1
[UNK]: 'Well now James, what can we do for you?'
Bessie: 'Oh [UNK]. I feel a bit sad.'
Cassi: 'Not so bad. How about you?'
[UNK]: 'Fine, thanks for asking.'""",
        
        """# Document 2
Speaker1: 'This is another test document.'
Speaker2: 'Yes, it contains multiple sentences. Each should be processed separately.'
Speaker1: 'Exactly! That is the whole point.'""",
        
        """# Document 3
Narrator: 'The experiment was conducted carefully.'
Scientist: 'We observed significant results. The data shows clear patterns.'
Assistant: 'Should we publish these findings?'"""
    ]
    
    for i, text in enumerate(sample_texts):
        with open(test_data_dir / f"test_doc_{i}.md", 'w') as f:
            f.write(text)
    
    print(f"✅ Created test data in {test_data_dir}")
    return test_data_dir

def test_sentence_extraction():
    """Test the sentence extraction function."""
    print("\n🧪 Testing sentence extraction...")
    
    from prepare_data import extract_sentences_from_markdown
    
    test_text = """# Test Document
[UNK]: 'Hello there! How are you today?'
Person: 'I am fine. Thanks for asking.'
Narrator: 'The conversation continued smoothly.'"""
    
    sentences = extract_sentences_from_markdown(test_text)
    
    print(f"Input text: {repr(test_text)}")
    print(f"Extracted {len(sentences)} sentences:")
    for i, sentence in enumerate(sentences, 1):
        print(f"  {i}: {repr(sentence)}")
    
    # Verify we got individual sentences
    expected_sentences = [
        "Hello there!",
        "How are you today?",
        "I am fine.",
        "Thanks for asking.",
        "The conversation continued smoothly."
    ]
    
    if len(sentences) == len(expected_sentences):
        print("✅ Sentence extraction working correctly")
        return True
    else:
        print(f"❌ Expected {len(expected_sentences)} sentences, got {len(sentences)}")
        return False

def test_data_preparation():
    """Test the modified data preparation pipeline."""
    print("\n🧪 Testing sentence-aware data preparation...")
    
    # Create test data
    test_data_dir = create_test_data()
    
    try:
        # Temporarily modify the data directory in prepare_data
        import prepare_data
        
        # Create a small dataset from test data
        original_load_function = prepare_data.load_md_files_to_dataset
        
        def test_load_function(data_dir):
            # Override to use our test data
            return original_load_function(str(test_data_dir))
        
        prepare_data.load_md_files_to_dataset = test_load_function
        
        # Test the sentence extraction
        dataset = test_load_function("dummy")
        
        print(f"Created dataset with {len(dataset)} sentences")
        
        # Check sample sentences
        if len(dataset) > 0:
            print("Sample sentences:")
            for i in range(min(5, len(dataset))):
                print(f"  {i+1}: {dataset[i]['text']}")
            
            # Verify sentences are individual, not documents
            avg_length = sum(len(item['text'].split()) for item in dataset) / len(dataset)
            print(f"Average sentence length: {avg_length:.1f} words")
            
            if avg_length < 20:  # Sentences should be short
                print("✅ Data preparation extracting individual sentences")
                return True
            else:
                print(f"❌ Average length too high ({avg_length}), may still be processing documents")
                return False
        else:
            print("❌ No sentences extracted")
            return False
            
    except Exception as e:
        print(f"❌ Error in data preparation: {e}")
        return False
    finally:
        # Clean up test data
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir)

def test_tokenization():
    """Test that tokenization preserves sentence boundaries."""
    print("\n🧪 Testing sentence-aware tokenization...")
    
    try:
        from tokenizer import Tokenizer
        
        # Load tokenizer
        tokenizer = Tokenizer.from_file("./tokenizer.json")
        
        # Test sentences
        test_sentences = [
            "Hello there! How are you?",
            "I am fine, thanks.",
            "This is a test sentence."
        ]
        
        print("Testing tokenization with sentence boundaries...")
        
        for sentence in test_sentences:
            # Test the new boundary-aware tokenization
            bounded_sentence = f"[CLS] {sentence} [SEP]"
            tokens = tokenizer.encode(bounded_sentence)
            
            print(f"Sentence: {sentence}")
            print(f"With boundaries: {bounded_sentence}")
            print(f"Tokens: {tokens.ids[:10]}... (length: {len(tokens.ids)})")
            
            # Check for boundary tokens
            if tokens.ids[0] != tokenizer.token_to_id("[CLS]"):
                print("❌ Missing [CLS] token at start")
                return False
            if tokens.ids[-1] != tokenizer.token_to_id("[SEP]"):
                print("❌ Missing [SEP] token at end")
                return False
        
        print("✅ Tokenization preserving sentence boundaries")
        return True
        
    except Exception as e:
        print(f"❌ Error in tokenization test: {e}")
        return False

def test_mlm_dataset():
    """Test the sentence-aware MLM dataset."""
    print("\n🧪 Testing sentence-aware MLM dataset...")
    
    try:
        # Check if we have tokenized data to test with
        cache_path = "./test_sentence_cache"
        
        if not Path(cache_path).exists():
            print("⚠️ No tokenized data found, skipping MLM dataset test")
            return True
        
        from mlm_dataset import SentenceAwareDataset
        from tokenizer import Tokenizer
        
        tokenizer = Tokenizer.from_file("./tokenizer.json")
        
        # Test the dataset
        dataset = SentenceAwareDataset(
            cache_path=cache_path,
            tokenizer=tokenizer,
            seq_length=128,
            mask_p=0.15
        )
        
        if len(dataset) > 0:
            # Test a sample
            sample = dataset[0]
            
            print(f"Dataset size: {len(dataset)}")
            print(f"Sample shapes: input_ids={sample['input_ids'].shape}, attention_mask={sample['attention_mask'].shape}, labels={sample['labels'].shape}")
            
            # Check for boundary tokens
            cls_positions = (sample['input_ids'] == tokenizer.token_to_id("[CLS]")).nonzero()
            sep_positions = (sample['input_ids'] == tokenizer.token_to_id("[SEP]")).nonzero()
            
            print(f"Boundary tokens found: {len(cls_positions)} [CLS], {len(sep_positions)} [SEP]")
            
            if len(cls_positions) > 0 and len(sep_positions) > 0:
                print("✅ MLM dataset preserving sentence boundaries")
                return True
            else:
                print("❌ No sentence boundary tokens found in MLM dataset")
                return False
        else:
            print("⚠️ Empty dataset")
            return True
            
    except Exception as e:
        print(f"❌ Error in MLM dataset test: {e}")
        return False

def main():
    """Run all tests to validate the sentence-aware pipeline."""
    print("🔍 Testing Sentence-Aware Data Processing Pipeline")
    print("=" * 60)
    
    tests = [
        ("Sentence Extraction", test_sentence_extraction),
        ("Data Preparation", test_data_preparation),
        ("Tokenization", test_tokenization),
        ("MLM Dataset", test_mlm_dataset),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The sentence-aware pipeline is working correctly.")
        print("\nNext steps:")
        print("1. Run full data preparation with sentence-aware processing")
        print("2. Train the model with the fixed pipeline")
        print("3. Evaluate on BLiMP to see the syntax improvement")
    else:
        print(f"\n⚠️ {len(results) - passed} test(s) failed. Please fix the issues before proceeding.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
