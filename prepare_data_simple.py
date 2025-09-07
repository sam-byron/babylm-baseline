#!/usr/bin/env python3

import os
import json
import glob
import torch
import gc
from tokenizer import Tokenizer
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count

def load_md_files(data_dir):
    """Load all .md files from the BNC directory"""
    pattern = os.path.join(data_dir, "**/*.md")
    md_files = glob.glob(pattern, recursive=True)
    print(f"Found {len(md_files)} .md files")
    
    texts = []
    for file_path in md_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # Only add non-empty content
                    texts.append(content)
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    print(f"Loaded {len(texts)} text segments")
    return texts

def process_chunk_simple(args):
    """Simplified chunk processing function"""
    chunk_texts, chunk_index, cache_path, tokenizer_path, max_length = args
    
    # Load tokenizer in the worker process
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_padding(length=max_length)
    tokenizer.enable_truncation(max_length=max_length)
    
    try:
        # Tokenize the chunk
        encodings = tokenizer.encode_batch(chunk_texts)
        
        # Extract token IDs
        tokenized_chunk = []
        for encoding in encodings:
            if hasattr(encoding, 'ids'):
                tokenized_chunk.append(encoding.ids)
            else:
                tokenized_chunk.append(encoding)
        
        # Calculate stats
        total_tokens = sum(len(seq) for seq in tokenized_chunk)
        unique_tokens = set()
        for seq in tokenized_chunk:
            unique_tokens.update(seq)
        
        unk_token_id = tokenizer.token_to_id('[UNK]')
        unk_count = 0
        if unk_token_id is not None:
            for seq in tokenized_chunk:
                unk_count += seq.count(unk_token_id)
        
        print(f"Chunk {chunk_index}: {len(tokenized_chunk)} sequences, "
              f"{total_tokens} total tokens, {len(unique_tokens)} unique tokens, "
              f"{unk_count} UNK tokens ({100*unk_count/total_tokens:.1f}%)")
        
        # Save chunk
        chunk_file = os.path.join(cache_path, f"chunk{chunk_index}.pt")
        torch.save(tokenized_chunk, chunk_file)
        print(f"Saved chunk {chunk_index} to {chunk_file}")
        
        return chunk_index, len(tokenized_chunk), total_tokens, len(unique_tokens), unk_count
        
    except Exception as e:
        print(f"Error processing chunk {chunk_index}: {e}")
        return None

def prepare_data_simple():
    """Simplified data preparation without datasets library"""
    
    # Load configuration
    with open("model_babylm_ltg_bert.json", "r") as f:
        config = json.load(f)
    
    cache_path = config["cache_path"]
    chunk_size = config["chunk_size"]
    tokenizer_path = config["tokenizer_path"]
    max_length = config["max_position_embeddings"]
    
    # Create cache directory
    os.makedirs(cache_path, exist_ok=True)
    print(f"Cache directory: {cache_path}")
    
    # Load text data
    data_dir = "./data/pretrain/bnc"
    texts = load_md_files(data_dir)
    
    if not texts:
        print("No text data found!")
        return
    
    # Split into chunks
    chunks = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        chunks.append((chunk, len(chunks), cache_path, tokenizer_path, max_length))
    
    print(f"Created {len(chunks)} chunks of size ~{chunk_size}")
    
    # Process chunks in parallel
    num_workers = min(cpu_count(), 8)  # Limit to avoid memory issues
    print(f"Processing with {num_workers} workers...")
    
    with Pool(num_workers) as pool:
        results = pool.map(process_chunk_simple, chunks)
    
    # Summary
    valid_results = [r for r in results if r is not None]
    if valid_results:
        total_sequences = sum(r[1] for r in valid_results)
        total_tokens = sum(r[2] for r in valid_results)
        total_unique = len(set().union(*[range(r[3]) for r in valid_results]))  # Approximation
        total_unk = sum(r[4] for r in valid_results)
        
        print(f"\nProcessing complete!")
        print(f"Total chunks: {len(valid_results)}")
        print(f"Total sequences: {total_sequences}")
        print(f"Total tokens: {total_tokens}")
        print(f"Total UNK tokens: {total_unk} ({100*total_unk/total_tokens:.2f}%)")
    
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    prepare_data_simple()
