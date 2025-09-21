#!/usr/bin/env python3
"""
Regenerate specific chunks with corrected tokenization (no padding)
"""

import json
import torch
import os
import gc
from tokenizers import Tokenizer
from utils_mp import process_and_save_chunk

def regenerate_specific_chunks(config_path, chunk_indices):
    """Regenerate specific chunk indices with corrected tokenization."""
    
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Load tokenizer with correct settings (no padding)
    tokenizer = Tokenizer.from_file(config.get("tokenizer_path"))
    tokenizer.model_max_length = config.get("max_position_embeddings", 512)
    tokenizer.enable_truncation(max_length=tokenizer.model_max_length)
    # Note: NO padding enabled here
    
    cache_path = config["cache_path"]
    chunk_size = config["chunk_size"]
    
    print(f"🔧 REGENERATING CHUNKS: {chunk_indices}")
    print(f"Cache path: {cache_path}")
    print(f"Chunk size: {chunk_size}")
    print(f"Tokenizer: {config.get('tokenizer_path')}")
    print("=" * 60)
    
    # Load the original text data
    from prepare_data import BertDataset
    data_dir = "./data/pretrain/bnc" 
    ds = BertDataset(data_dir, tokenizer)
    ds.shuffle()  # Same shuffle as original
    
    print(f"Dataset loaded: {len(ds)} samples")
    
    # Process each requested chunk
    for chunk_idx in chunk_indices:
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(ds))
        
        if start_idx >= len(ds):
            print(f"❌ Chunk {chunk_idx} starts beyond dataset length ({start_idx} >= {len(ds)})")
            continue
            
        print(f"\\n📦 Processing chunk {chunk_idx}: samples {start_idx}-{end_idx-1}")
        
        # Get the chunk data
        chunk_samples = []
        for i in range(start_idx, end_idx):
            chunk_samples.append(ds[i])
            
        # Create argument tuple for processing function
        chunk_arg = (chunk_samples, chunk_idx, cache_path, tokenizer)
        
        # Process and save the chunk
        try:
            result = process_and_save_chunk(chunk_arg)
            if result is not None:
                print(f"✅ Successfully regenerated chunk {chunk_idx}")
            else:
                print(f"❌ Failed to regenerate chunk {chunk_idx}")
        except Exception as e:
            print(f"❌ Error regenerating chunk {chunk_idx}: {e}")
        
        # Clean up
        del chunk_samples
        gc.collect()
    
    print(f"\\n🎯 Regeneration complete!")

if __name__ == "__main__":
    # Regenerate the chunks we backed up for testing
    config_path = "model_babylm_ltg_bert_FIXED.json"
    test_chunks = [184, 444]  # The chunks we want to test
    
    regenerate_specific_chunks(config_path, test_chunks)
