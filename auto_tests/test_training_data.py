#!/usr/bin/env python3


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import json
from tokenizer import Tokenizer

def test_training_data():
    """Test that the training pipeline can use the new tokenized data"""
    
    # Load configuration
    with open("model_babylm_ltg_bert.json", "r") as f:
        config = json.load(f)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(config["tokenizer_path"])
    
    # Load the tokenized data
    chunk_file = "model_babylm_bert_ltg/chunk0.pt"
    data = torch.load(chunk_file, map_location="cpu")
    
    print(f"Loaded {len(data)} sequences from {chunk_file}")
    
    # Analyze a few sequences
    vocab_stats = {}
    total_tokens = 0
    unk_tokens = 0
    
    unk_id = tokenizer.token_to_id('[UNK]') if tokenizer.token_to_id('[UNK]') else None
    
    for i in range(min(10, len(data))):
        seq = data[i]
        total_tokens += len(seq)
        
        # Count unique tokens
        for token_id in seq:
            vocab_stats[token_id] = vocab_stats.get(token_id, 0) + 1
            if unk_id is not None and token_id == unk_id:
                unk_tokens += 1
        
        print(f"Sequence {i}: {len(seq)} tokens, {len(set(seq))} unique")
    
    print(f"\nVocabulary statistics (first 10 sequences):")
    print(f"Total tokens: {total_tokens}")
    print(f"Unique tokens: {len(vocab_stats)}")
    print(f"UNK tokens: {unk_tokens} ({100*unk_tokens/total_tokens:.2f}%)")
    
    # Show most common tokens
    sorted_tokens = sorted(vocab_stats.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 most frequent tokens:")
    for token_id, count in sorted_tokens[:10]:
        try:
            token_str = tokenizer.id_to_token(token_id)
            print(f"  {token_id}: '{token_str}' ({count} times)")
        except:
            print(f"  {token_id}: <unknown> ({count} times)")
    
    print("\nTraining data test completed successfully!")
    return True

if __name__ == "__main__":
    test_training_data()
