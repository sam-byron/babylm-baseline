from datetime import datetime
from collections import OrderedDict
import gc
import math
# from torch.utils.data import Dataset
import os
from pathlib import Path
import random
import glob
from functools import partial
import hashlib
import pickle
from utils_mp import load_chunk, load_chunk_safe
from multiprocessing import Pool
from itertools import chain
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler
from collator import Collator
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datasets import concatenate_datasets
from transformers import AutoTokenizer, get_scheduler, DataCollatorForWholeWordMask
import argparse
import json
from typing import Union
from typing import List, Iterator
from transformers import DataCollatorForLanguageModeling
import sys
from transformers import (
    ElectraConfig,
    ElectraForPreTraining,
    ElectraTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ===== Simple ANSI color helper =====
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"

# helper to compute total real tokens in a Dataset
def compute_total_tokens(ds: Union[Dataset, object]) -> int:
    """
    If `ds` has an index_map of (path, seq_idx, start, end) tuples,
    sum up (end - start). Otherwise assume __getitem__ returns a dict
    with 'input_ids' and sum their lengths.
    """
    if hasattr(ds, "index_map"):
        return sum(end - start for (_, _, start, end) in ds.index_map)
    # fallback for map‐style datasets returning dicts
    return sum(len(ex) for ex in ds)

# helper to build index entries for one chunk file in parallel
def _index_for_path(args):
    path, block_size = args
    seqs = torch.load(path, map_location="cpu")
    entries = []
    for seq_i, seq in enumerate(seqs):
        length = len(seq)
        # We iterate up to length - 1. This elegantly ensures that any
        # created block will have at least one token and a subsequent
        # token to predict. It naturally skips sequences with length <= 1.
        for i in range(0, length - 1, block_size):
            start = i
            # The end of the slice is the minimum of the full sequence length
            # or the start of the next block.
            end = min(length, i + block_size)
            entries.append((path, seq_i, start, end))
    return entries

class ChunkedDataset(Dataset):
    # FIX 1: Change the default dtype to torch.long for compatibility with embedding layers.
    def __init__(self, chunk_paths, block_size, dtype=torch.long, pad_token_id=None, cache_size=50):
        # shuffle once
        self.chunk_paths = list(chunk_paths)
        # random.shuffle(self.chunk_paths)
        self.block_size   = block_size
        self.dtype        = dtype
        self.pad_token_id = pad_token_id or 0
        self.cache_size   = cache_size
        # path -> loaded list of sequences
        self._chunk_cache = OrderedDict()
        
        # Cache directory for index map
        self.cache_dir = Path("./cache_index")
        self.cache_dir.mkdir(exist_ok=True)
        
        # build a flat index of every block
        self._build_index()

    def _get_cache_key(self):
        """Generate a cache key based on chunk paths, modification times, and block size."""
        # Create a hash of all chunk paths and their modification times
        path_data = []
        for path in sorted(self.chunk_paths):
            try:
                mtime = os.path.getmtime(path)
                path_data.append((path, mtime))
            except OSError:
                # If file doesn't exist, use current timestamp
                path_data.append((path, time.time()))
        
        # Include block size in the hash
        cache_input = (tuple(path_data), self.block_size)
        cache_str = str(cache_input).encode('utf-8')
        return hashlib.md5(cache_str).hexdigest()

    def _build_index(self):
        """Create self.index_map = [ (path, seq_idx, start, end), ... ] with caching."""
        cache_key = self._get_cache_key()
        cache_file = self.cache_dir / f"index_map_{cache_key}.pkl"
        
        # Try to load from cache first
        if cache_file.exists():
            try:
                print(f"{C.CYAN}Loading cached index map from {cache_file}{C.RESET}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.index_map = cached_data['index_map']
                    self.block_lengths = cached_data['block_lengths']
                    print(f"{C.GREEN}Loaded {len(self.index_map)} index entries from cache{C.RESET}")
                    return
            except (pickle.PickleError, KeyError, EOFError) as e:
                print(f"{C.YELLOW}Cache file corrupted, rebuilding index: {e}{C.RESET}")
                # Remove corrupted cache file
                try:
                    cache_file.unlink()
                except OSError:
                    pass
        
        # Build index from scratch
        print(f"{C.BLUE}Building index map from scratch...{C.RESET}")
        args = [(path, self.block_size) for path in self.chunk_paths]
        max_workers = min(mp.cpu_count(), len(args))
        self.index_map = []
        with mp.Pool(processes=max_workers) as pool:
            for entries in pool.imap_unordered(_index_for_path, args):
                self.index_map.extend(entries)

        # improve cache locality: group by (path, seq_i, start)
        self.index_map.sort(key=lambda e: (e[0], e[1], e[2]))
        # Optional: shuffle by path (keeps locality), not by individual samples
        # from itertools import groupby
        # groups = []
        # for _, g in groupby(self.index_map, key=lambda e: e[0]):
        #     groups.append(list(g))
        # random.shuffle(groups)
        # self.index_map = [x for grp in groups for x in grp]

        # Precompute the true length of each block (before padding)
        self.block_lengths = [end - start for (_, _, start, end) in self.index_map]
        
        # Cache the results
        try:
            cache_data = {
                'index_map': self.index_map,
                'block_lengths': self.block_lengths,
                'cache_key': cache_key,
                'created_at': time.time()
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"{C.GREEN}Cached index map to {cache_file}{C.RESET}")
            
            # Clean up old cache files
            self._cleanup_old_cache_files()
            
        except (OSError, pickle.PickleError) as e:
            print(f"{C.YELLOW}Warning: Failed to cache index map: {e}{C.RESET}")
        
        print(f"{C.CYAN}Built {len(self.index_map)} index entries{C.RESET}")

    def _cleanup_old_cache_files(self, keep_latest=5):
        """Remove old cache files, keeping only the most recent ones."""
        try:
            cache_files = list(self.cache_dir.glob("index_map_*.pkl"))
            if len(cache_files) <= keep_latest:
                return
            
            # Sort by modification time, newest first
            cache_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Remove old files
            for old_file in cache_files[keep_latest:]:
                try:
                    old_file.unlink()
                    print(f"{C.DIM}Removed old cache file: {old_file.name}{C.RESET}")
                except OSError:
                    pass
        except Exception as e:
            print(f"{C.YELLOW}Warning: Failed to cleanup cache files: {e}{C.RESET}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        path, seq_i, start, end = self.index_map[idx]
        # simple LRU cache
        if path not in self._chunk_cache:
            data = torch.load(path, map_location="cpu")
            self._chunk_cache[path] = data
            # evict oldest if over capacity
            if len(self._chunk_cache) > self.cache_size:
                self._chunk_cache.popitem(last=False)
        seq = self._chunk_cache[path][seq_i]
        sub = seq[start:end] if isinstance(seq, torch.Tensor) else seq[start:end]
        lst = sub.tolist() if isinstance(sub, torch.Tensor) else sub
        # if len(lst) < self.block_size:
        #     lst = lst + [self.pad_token_id] * (self.block_size - len(lst))
        return torch.tensor(lst, dtype=self.dtype)


def create_and_cache_splits(config):
    """Create train/val/test splits once and cache them."""
    
    cache_path = config["cache_path"]
    # Create splits directory
    splits_dir = Path(cache_path) / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    # Check if splits already exist
    splits_file = splits_dir / "dataset_splits.json"
    if splits_file.exists():
        print(f"{C.CYAN}Dataset splits already exist, loading cached splits...{C.RESET}")
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        return splits['train_paths'], splits['val_paths'], splits['test_paths']
    
    # Create new splits
    print(f"{C.BLUE}Creating new dataset splits...{C.RESET}")
    chunk_paths = sorted(glob.glob(os.path.join(cache_path, "chunk*.pt")))
    
    # Shuffle once and save the order
    random.shuffle(chunk_paths)
    
    train_frac = config.get("train_frac", 0.85)
    val_frac   = config.get("val_frac", 0.05)
    
    N = len(chunk_paths)
    idx1 = int(train_frac * N)
    idx2 = int((train_frac + val_frac) * N)

    train_paths = chunk_paths[:idx1]
    val_paths   = chunk_paths[idx1:idx2]
    test_paths  = chunk_paths[idx2:]
    
    # Cache the splits
    splits = {
        'train_paths': train_paths,
        'val_paths': val_paths,
        'test_paths': test_paths,
        'created_at': str(datetime.now()),
        'config': {
            'train_frac': train_frac,
            'val_frac': val_frac,
            'total_chunks': N
        }
    }
    
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"{C.GREEN}Cached dataset splits to {splits_file}{C.RESET}")
    print(f"{C.CYAN}Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}{C.RESET}")
    
    return train_paths, val_paths, test_paths

class TokenBudgetBatchSampler(BatchSampler):
    """
    A BatchSampler that creates batches of indices where the total number of
    tokens (based on sample lengths) does not exceed a specified budget.

    To minimize padding and wasted computation, it sorts samples by length
    before creating batches.
    """
    def __init__(self, lengths: List[int], max_tokens: int, shuffle: bool = True):
        """
        Args:
            lengths: A list of integers representing the length of each sample
                     in the dataset.
            max_tokens: The maximum number of tokens allowed in a single batch.
            shuffle: If True, the order of batches is shuffled at the beginning
                     of each epoch.
        """
        if not isinstance(lengths, list) or not all(isinstance(l, int) for l in lengths):
            raise TypeError("lengths must be a list of integers.")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")

        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.batches = self._create_batches()

    def _create_batches(self) -> List[List[int]]:
        # Create a list of (index, length) tuples and sort by length
        indices_with_lengths = sorted(enumerate(self.lengths), key=lambda x: x[1])

        batches = []
        current_batch = []
        current_token_count = 0

        for index, length in indices_with_lengths:
            if length > self.max_tokens:
                print(f"{C.YELLOW}Warning: Sample {index} with length {length} is larger than max_tokens {self.max_tokens} and will be skipped.{C.RESET}")
                continue

            if current_token_count + length > self.max_tokens and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_token_count = 0

            current_batch.append(index)
            current_token_count += length

        if current_batch:
            batches.append(current_batch)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            random.shuffle(self.batches)
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)

# Update the data_loader function to use PreloadedDataset
def data_loader(config, tokenizer, cache_path):
    block_size = config["block_size"]
    batch_size = config["batch_size"]

    # Load or create cached splits
    train_paths, val_paths, test_paths = create_and_cache_splits(config)

    # Get pad token ID from tokenizers.Tokenizer object
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise ValueError("PAD token not found in tokenizer vocabulary")
    print(f"{C.BLUE}Creating ChunkedDataset instances...{C.RESET}")
    train_ds = ChunkedDataset(train_paths, block_size=block_size, pad_token_id=pad_id)
    val_ds   = ChunkedDataset(val_paths,   block_size=block_size, pad_token_id=pad_id)
    test_ds  = ChunkedDataset(test_paths,  block_size=block_size, pad_token_id=pad_id)

    # Compute total tokens for each split
    total_tokens_train = sum(train_ds.block_lengths)
    total_tokens_val   = sum(val_ds.block_lengths)
    total_tokens_test  = sum(test_ds.block_lengths)
    print(f"{C.CYAN}Total tokens → train: {total_tokens_train:,}, val: {total_tokens_val:,}, test: {total_tokens_test:,}{C.RESET}")

    # Print length of datasets
    print(f"{C.CYAN}Train dataset length: {len(train_ds)}{C.RESET}")
    print(f"{C.CYAN}Val dataset length: {len(val_ds)}{C.RESET}")
    print(f"{C.CYAN}Test dataset length: {len(test_ds)}{C.RESET}")

    # Get special token IDs for our custom tokenizer
    mask_token_id = tokenizer.token_to_id("[MASK]")
    if mask_token_id is None:
        raise ValueError("MASK token not found in tokenizer vocabulary")
    
    # Custom MLM collator for tokenizers.Tokenizer
    def collate_fn_with_mask(examples):
        # examples is a list of token ID lists
        # Convert to tensors properly - check if already tensor or list
        input_ids = []
        for ex in examples:
            if isinstance(ex, torch.Tensor):
                input_ids.append(ex.detach().clone())
            else:
                input_ids.append(torch.tensor(ex, dtype=torch.long))
        
        # Pad sequences to the same length
        max_len = max(len(seq) for seq in input_ids)
        padded_input_ids = []
        attention_masks = []
        
        for seq in input_ids:
            # Pad sequence
            padding_length = max_len - len(seq)
            padded_seq = torch.cat([seq, torch.full((padding_length,), pad_id, dtype=torch.long)])
            padded_input_ids.append(padded_seq)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.cat([torch.ones(len(seq), dtype=torch.long), 
                                      torch.zeros(padding_length, dtype=torch.long)])
            attention_masks.append(attention_mask)
        
        # Stack into batch tensors
        input_ids_batch = torch.stack(padded_input_ids)
        attention_mask_batch = torch.stack(attention_masks)
        
        # Apply MLM masking (15% probability)
        labels = input_ids_batch.clone()
        
        # Create random mask for 15% of tokens (excluding special tokens)
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = (labels == pad_id) | (labels == tokenizer.token_to_id("[CLS]")) | (labels == tokenizer.token_to_id("[SEP]"))
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # 80% of the time, replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids_batch[indices_replaced] = mask_token_id
        
        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(tokenizer.get_vocab_size(), labels.shape, dtype=torch.long)
        input_ids_batch[indices_random] = random_words[indices_random]
        
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        
        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels,
        }

    # build dynamic, token‐based training batches
    max_tokens = config.get("max_tokens", config["block_size"] * config["batch_size"])
    print(f"{C.BLUE}Creating DataLoader with dynamic token batching (max_tokens={max_tokens})...{C.RESET}")
    lengths = train_ds.block_lengths

    train_batch_sampler = TokenBudgetBatchSampler(
        lengths=lengths, 
        max_tokens=max_tokens, 
        shuffle=True
    )

    
    
    train_loader = DataLoader(
        train_ds,
        # batch_sampler=train_batch_sampler,
        batch_size=config["batch_size"],
        num_workers=2,  # Reduced to avoid deadlocks
        pin_memory=True,
        collate_fn=collate_fn_with_mask,  # or collate_fn_with_mask if you prefer explicit mask
        prefetch_factor=2,  # Reduced prefetch factor
        persistent_workers=False,  # Disabled to avoid multiprocessing issues
        drop_last=True,  # optional: move drop_last here if you want strict batch shapes
    )

    print(f"  {C.GREEN}→ {len(train_loader)} train batches{C.RESET}")

    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True,
                               collate_fn=collate_fn_with_mask, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True,
                               collate_fn=collate_fn_with_mask, drop_last=True)

    print(f"{C.CYAN}Data preparation complete. Train files: {len(train_paths)}, Val files: {len(val_paths)}, Test files: {len(test_paths)}{C.RESET}")

    return train_loader, val_loader, test_loader, collate_fn_with_mask, total_tokens_train
    # return val_loader, val_loader, test_loader, collate_fn, total_tokens_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iter Data Loader Script")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)

    block_size = config["block_size"]
    batch_size = config["batch_size"]

    # Load or create cached splits
    train_paths, val_paths, test_paths = create_and_cache_splits(config)