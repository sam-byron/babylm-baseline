#!/usr/bin/env python3
"""
Programmatically create all necessary tokenizer files for transformers compatibility.

Generates:
 - vocab.txt (sorted by id)
 - tokenizer_config.json (BERT-compatible defaults)
 - special_tokens_map.json (auto-detected specials)
 - tokenizer.json (copied if provided and not present)

Also updates config.json with an auto_map entry for AutoTokenizer so
AutoTokenizer.from_pretrained can resolve a standard tokenizer class
without relying on the custom config class.
"""

import argparse
import json
import os
from tokenizers import Tokenizer as HFTokenizer
from training_utils import generate_special_tokens_map

def create_tokenizer_files(checkpoint_path: str, tokenizer_path: str = "./tokenizer.json"):
    """
    Create all necessary tokenizer files for transformers compatibility
    """
    print(f"Creating tokenizer files in {checkpoint_path}")
    
    # Load your custom tokenizer
    tokenizer = HFTokenizer.from_file(tokenizer_path)
    
    # Get vocabulary from the tokenizer
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create vocab.txt file (required for BERT-style tokenizers)
    vocab_file = os.path.join(checkpoint_path, "vocab.txt")
    with open(vocab_file, 'w', encoding='utf-8') as f:
        # Sort by token ID to maintain order
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        for token, _ in sorted_vocab:
            f.write(token + '\n')
    print(f"Created {vocab_file}")
    
    # Create tokenizer_config.json
    tokenizer_config = {
        "clean_up_tokenization_spaces": True,
        "cls_token": "[CLS]",
        "do_lower_case": True,
        "mask_token": "[MASK]",
        "model_max_length": 512,
        "pad_token": "[PAD]",
        "sep_token": "[SEP]",
        "strip_accents": None,
        "tokenize_chinese_chars": True,
        "tokenizer_class": "BertTokenizer",
        "unk_token": "[UNK]",
        "vocab_size": vocab_size
    }
    
    tokenizer_config_file = os.path.join(checkpoint_path, "tokenizer_config.json")
    with open(tokenizer_config_file, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"Created {tokenizer_config_file}")
    
    # Create special_tokens_map.json using the refactored utility
    generate_special_tokens_map(tokenizer, checkpoint_path)
    
    # Copy the existing tokenizer.json if it doesn't exist in checkpoint
    tokenizer_json_dest = os.path.join(checkpoint_path, "tokenizer.json")
    if not os.path.exists(tokenizer_json_dest) and os.path.exists(tokenizer_path):
        import shutil
        shutil.copy2(tokenizer_path, tokenizer_json_dest)
        print(f"Copied {tokenizer_path} to {tokenizer_json_dest}")
    
    print("All tokenizer files created successfully!")

def update_config_for_tokenizer(checkpoint_path: str):
    """
    Update config.json to include tokenizer auto_map
    """
    config_path = os.path.join(checkpoint_path, "config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Add tokenizer to auto_map if not present
        if "auto_map" not in config:
            config["auto_map"] = {}
        
        # Use standard BertTokenizer since we're creating BERT-compatible files
        config["auto_map"]["AutoTokenizer"] = [
            "transformers.BertTokenizer",
            "transformers.BertTokenizerFast",
        ]
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Updated {config_path} with tokenizer auto_map")
    else:
        print(f"Warning: {config_path} not found")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True, help="Directory containing model/config.json where tokenizer files will be written")
    ap.add_argument("--tokenizer_path", default="./tokenizer.json", help="Path to tokenizer.json from tokenizers library")
    args = ap.parse_args()

    # Create all tokenizer files
    create_tokenizer_files(args.checkpoint_path, args.tokenizer_path)

    # Update config.json
    update_config_for_tokenizer(args.checkpoint_path)

    print(f"\nAll files created in {args.checkpoint_path}:")
    files = os.listdir(args.checkpoint_path)
    for file in sorted(files):
        print(f"  - {file}")
