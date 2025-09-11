import gc
import os
import random
import torch
from itertools import islice
from multiprocessing import Pool, current_process
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

def save_checkpoint(epoch, model, optimizer, scheduler, scaler, checkpoint_path="checkpoint.pt"):
     # if model is wrapped (DDP, DistributedDataParallel, etc.) it has .module
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}", flush=True)

def load_checkpoint(checkpoint_path="checkpoint.pt"):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint
    else:
        print("No checkpoint found. Starting from scratch.")
        return None
    
def batch_generator_sequential(dataset, chunk_size, max_samples=2**31):
    """Generate batches of samples from a non-streaming dataset more efficiently."""
    for i in range(0, min(len(dataset), max_samples), chunk_size):
        yield dataset[i:i + chunk_size]

def batch_generator(dataset, chunk_size, max_samples):
    """Generate batches of samples from a streaming dataset more efficiently."""
    iterator = iter(dataset)  # Create an iterator from the dataset
    num_samples = 0

    while num_samples < max_samples:
        # Use islice to fetch a batch of size `batch_size`
        batch = list(islice(iterator, chunk_size))
        if not batch:
            break  # Stop if there are no more samples
        yield batch
        num_samples += len(batch)

def fetch_batch(args):
    dataset, start, end = args
    return list(islice(dataset, start, end))

def batch_generator_parallel(dataset, batch_size, max_samples, num_workers, start_index=0):
    """Generate batches in parallel using multiprocessing, starting from a specific index."""
    # Calculate the total number of batches, starting from the given start_index
    total_batches = (max_samples + batch_size - 1) // batch_size
    args = [
        (dataset, start_index + i * batch_size, start_index + (i + 1) * batch_size)
        for i in range(total_batches)
    ]

    with Pool(num_workers) as pool:
        for batch in pool.imap(fetch_batch, args):
            yield batch

def tokenize_sample(sample, tokenizer):
    """Tokenize a single sample."""

    if not isinstance(sample, list) or not all(isinstance(x, str) for x in sample):
        raise ValueError(f"Expected list[str], got {type(sample)}")
    return tokenizer(sample, add_special_tokens=False, truncation=True, return_tensors="pt", padding=True)  # Tokenize the sample and return the input IDs

def tokenize_samples(samples, tokenizer):
    """Tokenize a batch of samples."""
    if isinstance(samples, list):
        # If the samples are a list of strings, tokenize each string
        texts = samples
    else:
        raise ValueError(f"Unexpected samples format: samples")
    #     print(f"Using batch_encode_plus for better performance with large batches")
    texts = [text + tokenizer.eos_token for sublist in texts for text in sublist]  # Add EOS token to each text
    
    return tokenizer.encode(
        texts,
        add_special_tokens=False,
        truncation=True,
        padding=False,  # No padding for now, we will handle it later
        # return_tensors='pt'  # Return as PyTorch tensors
        )
def load_chunk(chunk_path):
    """Helper function to load a single chunk."""
    print(f"Loading chunk from {chunk_path}...", flush=True)
    return torch.load(chunk_path, map_location="cpu")

def load_chunk_safe(path):
    try:
        return load_chunk(path)
    except (RuntimeError, EOFError) as e:
        print(f"[Warning] failed to load {path}: {e}")
        return []

from concurrent.futures import ThreadPoolExecutor

# global thread pool for parallel disk writes
_SAVE_POOL = ThreadPoolExecutor(max_workers=64)

def load_chunk_async(chunk_path):
    """
    Load a single chunk asynchronously using the ELECTRA pattern.
    This function is optimized for loading existing tokenized chunks.
    """
    try:
        import torch
        import os
        import gc
        
        # Get file info for progress tracking
        filename = os.path.basename(chunk_path)
        file_size = os.path.getsize(chunk_path)
        
        print(f"Loading {filename} ({file_size/(1024*1024):.1f}MB)...", flush=True)
        
        # Load chunk data with memory optimization
        chunk_data = torch.load(chunk_path, map_location='cpu', weights_only=False)
        sentences = []
        
        # Process sentences efficiently
        if isinstance(chunk_data, list):
            for sentence_tokens in chunk_data:
                if isinstance(sentence_tokens, (list, torch.Tensor)) and len(sentence_tokens) > 0:
                    # Convert to tensor if needed
                    if not isinstance(sentence_tokens, torch.Tensor):
                        sentence_tokens = torch.tensor(sentence_tokens, dtype=torch.long)
                    elif sentence_tokens.dtype != torch.long:
                        sentence_tokens = sentence_tokens.long()
                    
                    # Basic validation
                    if len(sentence_tokens) > 0:
                        sentences.append(sentence_tokens)
        
        # Progress indicator
        print(f"✓ {filename}: {len(sentences)} sentences loaded", flush=True)
        
        # Clean up intermediate data
        del chunk_data
        gc.collect()
        
        return sentences
        
    except Exception as e:
        print(f"❌ Error loading {chunk_path}: {e}", flush=True)
        # Force cleanup on error
        import gc
        gc.collect()
        return []  # Return empty list for easier handling


def process_and_save_chunk(arg, tokenizer_unused=None):
    """
    Load and save a single chunk - ELECTRA pattern for sentence-aware loading.
    This function matches the working ELECTRA implementation exactly.
    """
    # Unpack arguments - but this is for loading existing chunks, not tokenizing new ones
    chunk_path, chunk_index, cache_path = arg
    
    try:
        # Load the chunk using the existing load_chunk_safe function
        print(f"Loading chunk {chunk_index} from {chunk_path}...", flush=True)
        tokenized_chunk = load_chunk_safe(chunk_path)
        
        if not tokenized_chunk:
            print(f"Warning: Empty chunk loaded from {chunk_path}")
            return None
        
        # For sentence-aware loading, we expect the chunk to already be tokenized properly
        # Just validate the structure
        if isinstance(tokenized_chunk, list) and len(tokenized_chunk) > 0:
            # Count sentences and tokens for progress tracking
            total_sentences = len(tokenized_chunk)
            total_tokens = sum(len(seq) if hasattr(seq, '__len__') else 0 for seq in tokenized_chunk)
            avg_length = total_tokens / total_sentences if total_sentences > 0 else 0
            
            print(f"Chunk {chunk_index}: {total_sentences} sentences, "
                  f"avg length: {avg_length:.1f} tokens", flush=True)
        
        # Use async saving with ThreadPoolExecutor like ELECTRA pattern
        chunk_file_path = os.path.join(cache_path, f"chunk{chunk_index}.pt")
        fut = _SAVE_POOL.submit(torch.save, tokenized_chunk, chunk_file_path)
        fut.add_done_callback(lambda f: print(f"Saved chunk {chunk_index} → {chunk_file_path}", flush=True))
        
        print(f"Processing complete for chunk {chunk_index} with {len(tokenized_chunk)} sentences", flush=True)
        
        # Clean up memory - this is critical for memory management
        result_size = len(tokenized_chunk)
        del tokenized_chunk
        gc.collect()  # Force garbage collection to free memory
        
        return fut  # Return future like ELECTRA pattern
        
    except Exception as e:
        print(f"Error processing chunk {chunk_index} from {chunk_path}: {e}", flush=True)
        return None

