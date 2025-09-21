#!/usr/bin/env python3
"""
Generate mock BLiMP data for testing purposes.
"""

import pickle
import gzip
import os
import json
from typing import Dict, List, Tuple

def generate_mock_blimp_data() -> Dict:
    """Generate realistic mock BLiMP data for testing."""
    
    mock_data = {
        "anaphor_agreement": [
            {"sentence_good": "The woman likes herself", "sentence_bad": "The woman likes himself"},
            {"sentence_good": "The man hurt himself", "sentence_bad": "The man hurt herself"},
            {"sentence_good": "The children enjoyed themselves", "sentence_bad": "The children enjoyed himself"},
        ],
        
        "determiner_noun_agreement": [
            {"sentence_good": "This book is interesting", "sentence_bad": "These book is interesting"},
            {"sentence_good": "Those cats are sleeping", "sentence_bad": "That cats are sleeping"},
            {"sentence_good": "Many students study hard", "sentence_bad": "Much students study hard"},
        ],
        
        "subject_verb_agreement": [
            {"sentence_good": "The cat runs quickly", "sentence_bad": "The cat run quickly"},
            {"sentence_good": "The dogs run fast", "sentence_bad": "The dogs runs fast"},
            {"sentence_good": "She walks to school", "sentence_bad": "She walk to school"},
        ],
        
        "irregular_plurals": [
            {"sentence_good": "The children play outside", "sentence_bad": "The childs play outside"},
            {"sentence_good": "Many people like music", "sentence_bad": "Many peoples like music"},
            {"sentence_good": "The mice are small", "sentence_bad": "The mouses are small"},
        ],
        
        "npi_licensing": [
            {"sentence_good": "She didn't see anyone", "sentence_bad": "She saw anyone"},
            {"sentence_good": "Nobody likes this food", "sentence_bad": "Somebody likes this food"},
            {"sentence_good": "He never goes anywhere", "sentence_bad": "He always goes anywhere"},
        ],
        
        "quantifier_scope": [
            {"sentence_good": "Every student read some books", "sentence_bad": "Some books read every student"},
            {"sentence_good": "All cats like most fish", "sentence_bad": "Most fish like all cats"},
            {"sentence_good": "Many people visit few places", "sentence_bad": "Few places visit many people"},
        ]
    }
    
    return mock_data

def generate_mock_tokenizer_vocab() -> Dict[str, int]:
    """Generate a mock tokenizer vocabulary."""
    
    # Common tokens for BLiMP evaluation
    vocab = {
        "[PAD]": 0,
        "[UNK]": 100,
        "[CLS]": 101,
        "[SEP]": 102,
        "[MASK]": 103,
        
        # Common words from BLiMP
        "the": 1996, "a": 1037, "an": 2019, "this": 2023, "that": 2008, "these": 2122, "those": 2216,
        "cat": 4937, "cats": 8870, "dog": 3899, "dogs": 6077, "book": 2338, "books": 2808,
        "woman": 2450, "women": 2607, "man": 2158, "men": 2588, "child": 2775, "children": 2336,
        "student": 3076, "students": 2493, "people": 2111, "person": 2711,
        
        # Verbs
        "run": 3216, "runs": 3216, "walk": 3328, "walks": 7365, "like": 2066, "likes": 7777,
        "see": 2156, "sees": 5224, "go": 2175, "goes": 3115, "study": 2817, "studies": 2817,
        "play": 2377, "plays": 3248, "hurt": 3424, "hurts": 3424, "enjoy": 3746, "enjoys": 3746,
        "visit": 3942, "visits": 3942, "read": 2191, "reads": 7173,
        
        # Pronouns and reflexives
        "she": 2016, "he": 2002, "they": 2027, "himself": 2370, "herself": 2841, "themselves": 2213,
        "anyone": 3087, "someone": 2619, "nobody": 6919, "somebody": 6219,
        
        # Adjectives and adverbs
        "quickly": 5221, "fast": 2698, "slowly": 3254, "hard": 2524, "small": 2235, "big": 2502,
        "interesting": 5875, "good": 2204, "bad": 2919,
        
        # Prepositions and others
        "to": 2000, "outside": 2648, "anywhere": 6638, "somewhere": 8730, "never": 2196, "always": 2467,
        "is": 2003, "are": 2024, "didn": 2134, "t": 1005,
        
        # Quantifiers
        "every": 2296, "all": 2035, "some": 2070, "many": 2116, "much": 2172, "most": 2087,
        "few": 2261, "several": 2195,
        
        # Numbers
        "one": 2028, "two": 2048, "three": 2093, "four": 2176, "five": 2274,
    }
    
    return vocab

def save_mock_blimp_data(output_path: str):
    """Save mock BLiMP data to a pickle file."""
    data = generate_mock_blimp_data()
    
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved mock BLiMP data to {output_path}")
    print(f"Groups: {list(data.keys())}")
    print(f"Total examples: {sum(len(group) for group in data.values())}")

def save_mock_tokenizer_vocab(output_path: str):
    """Save mock tokenizer vocabulary to a JSON file."""
    vocab = generate_mock_tokenizer_vocab()
    
    with open(output_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"Saved mock tokenizer vocabulary to {output_path}")
    print(f"Vocabulary size: {len(vocab)}")

def create_test_data_directory(base_path: str = "test_data"):
    """Create a test data directory with mock files."""
    os.makedirs(base_path, exist_ok=True)
    
    # Save BLiMP data
    blimp_path = os.path.join(base_path, "mock_blimp.pkl.gz")
    save_mock_blimp_data(blimp_path)
    
    # Save tokenizer vocab
    vocab_path = os.path.join(base_path, "mock_vocab.json")
    save_mock_tokenizer_vocab(vocab_path)
    
    # Create a simple test config
    config = {
        "blimp_path": blimp_path,
        "vocab_path": vocab_path,
        "device": "cpu",
        "batch_size": 4
    }
    
    config_path = os.path.join(base_path, "test_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTest data directory created at: {base_path}")
    print("Files created:")
    print(f"  - {blimp_path}")
    print(f"  - {vocab_path}")
    print(f"  - {config_path}")
    
    return base_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate mock data for BLiMP testing")
    parser.add_argument("--output-dir", "-o", default="test_data", 
                       help="Output directory for test data")
    parser.add_argument("--blimp-only", action="store_true", 
                       help="Generate only BLiMP data file")
    parser.add_argument("--vocab-only", action="store_true", 
                       help="Generate only vocabulary file")
    
    args = parser.parse_args()
    
    if args.blimp_only:
        output_path = os.path.join(args.output_dir, "mock_blimp.pkl.gz")
        os.makedirs(args.output_dir, exist_ok=True)
        save_mock_blimp_data(output_path)
    elif args.vocab_only:
        output_path = os.path.join(args.output_dir, "mock_vocab.json")
        os.makedirs(args.output_dir, exist_ok=True)
        save_mock_tokenizer_vocab(output_path)
    else:
        create_test_data_directory(args.output_dir)