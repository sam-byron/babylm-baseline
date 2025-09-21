import gc
import os
import random
import torch
from itertools import islice
from multiprocessing import Pool
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

from concurrent.futures import ThreadPoolExecutor

# global thread pool for parallel disk writes
_SAVE_POOL = ThreadPoolExecutor(max_workers=64)

def process_and_save_chunk(arg, tokenizer_unused=None):
    """
    Tokenize and save a single chunk.
    Note: tokenizer_unused parameter is kept for compatibility but tokenizer comes from arg
    """
    sample_chunk, chunk_index, cache_path, tokenizer = arg
    
    # Extract text from the samples
    sample_chunk = [sample["text"] for sample in sample_chunk]
    if not isinstance(sample_chunk, list) or not all(isinstance(t, str) for t in sample_chunk):
        print(f"Expected list[str], got {type(sample_chunk)}")
        raise ValueError(f"Expected list[str], got {type(sample_chunk)}")
    
    try:
        # Use encode_batch which returns list of Encoding objects
        tokenized_batch = tokenizer.encode_batch(sample_chunk)
        
        # Extract the actual token IDs from each Encoding object
        individual_sequences = []
        for encoding in tokenized_batch:
            if hasattr(encoding, 'ids'):
                individual_sequences.append(encoding.ids)
            else:
                # Fallback if encoding is already a list of IDs
                individual_sequences.append(encoding)
        
        if not individual_sequences:
            raise ValueError("Tokenizer did not produce any output. Check the input or tokenizer implementation.")
        
        # FIXED: Concatenate all sequences into one long sequence per chunk
        # This is what ChunkedDataset expects - long sequences to split into blocks
        concatenated_sequence = []
        for seq in individual_sequences:
            concatenated_sequence.extend(seq)
        
        # Store as a single long sequence (ChunkedDataset will split it into blocks)
        tokenized_chunk = [concatenated_sequence]
        
        # Debug: Print stats about the concatenated chunk
        total_tokens = len(concatenated_sequence)
        unique_tokens = set(concatenated_sequence)
        
        print(f"Chunk {chunk_index}: {len(individual_sequences)} sentences → 1 concatenated sequence, {total_tokens} total tokens, {len(unique_tokens)} unique tokens", flush=True)
        
    except Exception as e:
        print(f"Error tokenizing chunk {chunk_index}: {e}", flush=True)
        return None
    
    chunk_file_path = os.path.join(cache_path, f"chunk{chunk_index}.pt")
    
    # Save synchronously to ensure the file is written before the function returns
    torch.save(tokenized_chunk, chunk_file_path)
    print(f"Saved chunk {chunk_index} → {chunk_file_path}", flush=True)
    
    print(f"Tokenization complete for chunk {chunk_index} with {len(tokenized_chunk)} sequences", flush=True)
    
    # Clean up memory
    del tokenized_chunk, tokenized_batch
    gc.collect()

    return chunk_index

