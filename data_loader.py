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
from multiprocessing import Pool
from itertools import chain
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler
import time
import argparse
import json
from typing import Union
from typing import List, Iterator
import sys
# Import masking strategies from mlm_dataset
from mlm_dataset import SpanMaskingStrategy, SubwordMaskingStrategy, WholeWordMaskingStrategy

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

class ChunkedDataset(Dataset):
    # FIX 1: Change the default dtype to torch.long for compatibility with embedding layers.
    def __init__(self, chunk_paths, block_size, tokenizer, dtype=torch.long, pad_token_id=None, cache_size=50):
        # shuffle once
        self.chunk_paths = list(chunk_paths)
        # random.shuffle(self.chunk_paths)
        self.block_size   = block_size
        self.tokenizer    = tokenizer
        self.cls_index = self.tokenizer.token_to_id("[CLS]")
        self.sep_index = self.tokenizer.token_to_id("[SEP]")
        self.dtype        = dtype
        self.pad_token_id = pad_token_id or 0
        self.cache_size   = cache_size
        # path -> concatenated sequence data
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
    
    # helper to build index entries for one chunk file in parallel (concatenation mode)
    def _index_for_path(self, args):
        path, block_size = args
        seqs = torch.load(path, map_location="cpu")
        
        # Chain sequences together until we reach target block_size
        entries = []
        all_tokens = []
        current_block = []
        current_pos = 0
        
        for seq in seqs:
            seq_tokens = seq.tolist() if isinstance(seq, torch.Tensor) else seq
            
            # Add sequence to current block
            current_block.extend(seq_tokens)
            
            # While current block has enough tokens for a full block, create training blocks
            while len(current_block) >= block_size:  # Reserve space for [CLS] and [SEP]
                # Extract exactly block_size tokens for the training block
                block_tokens = current_block[:block_size]
                block_tokens = block_tokens
                all_tokens.extend(block_tokens)
                
                # Create index entry
                start = current_pos
                end = current_pos + block_size
                entries.append((path, 0, start, end))
                current_pos += block_size
                
                # Keep remaining tokens for next block
                current_block = current_block[block_size:]
        
        # Handle remaining tokens (if any) - add them if we have at least 2 tokens for MLM
        if len(current_block) >= 2:
            current_block = current_block
            all_tokens.extend(current_block)
            start = current_pos
            end = current_pos + len(current_block)
            entries.append((path, 0, start, end))
        
        return entries, all_tokens

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
                    self._concatenated_data = cached_data.get('concatenated_data', {})
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
        print(f"{C.BLUE}Building index map from scratch with sequence concatenation...{C.RESET}")
        args = [(path, self.block_size) for path in self.chunk_paths]
        max_workers = min(mp.cpu_count(), len(args))
        self.index_map = []
        self._concatenated_data = {}  # Store concatenated sequences
        
        with mp.Pool(processes=max_workers) as pool:
            for entries, concatenated_tokens in pool.imap_unordered(self._index_for_path, args):
                if entries:  # Only process if we got valid entries
                    path = entries[0][0]  # Get path from first entry
                    self._concatenated_data[path] = concatenated_tokens
                    self.index_map.extend(entries)

        # improve cache locality: group by (path, seq_i, start)
        self.index_map.sort(key=lambda e: (e[0], e[1], e[2]))
        
        # Precompute the true length of each block (before padding)
        self.block_lengths = [end - start for (_, _, start, end) in self.index_map]
        
        print(f"{C.MAGENTA}Concatenation statistics:{C.RESET}")
        total_blocks = len(self.index_map)
        total_tokens = sum(self.block_lengths)
        avg_block_length = total_tokens / total_blocks if total_blocks > 0 else 0
        full_blocks = sum(1 for length in self.block_lengths if length == self.block_size)
        print(f"  Total blocks: {total_blocks:,}")
        print(f"  Full blocks ({self.block_size} tokens): {full_blocks:,} ({full_blocks/total_blocks*100:.1f}%)")
        print(f"  Average block length: {avg_block_length:.1f} tokens")
        print(f"  Total tokens: {total_tokens:,}")
        
        # Cache the results
        try:
            cache_data = {
                'index_map': self.index_map,
                'block_lengths': self.block_lengths,
                'concatenated_data': self._concatenated_data,
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
        
        print(f"{C.CYAN}Built {len(self.index_map)} index entries with concatenation{C.RESET}")

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
        
        # Get concatenated data for this chunk
        if path not in self._chunk_cache:
            # Load from cached concatenated data if available
            if hasattr(self, '_concatenated_data') and path in self._concatenated_data:
                concatenated_tokens = self._concatenated_data[path]
            else:
                # Fallback: load and concatenate on-the-fly
                seqs = torch.load(path, map_location="cpu")
                concatenated_tokens = []
                for seq in seqs:
                    if isinstance(seq, torch.Tensor):
                        concatenated_tokens.extend(seq.tolist())
                    else:
                        concatenated_tokens.extend(seq)
            
            self._chunk_cache[path] = concatenated_tokens
            # evict oldest if over capacity
            if len(self._chunk_cache) > self.cache_size:
                self._chunk_cache.popitem(last=False)
        
        # Get the block from the concatenated sequence
        concatenated_seq = self._chunk_cache[path]
        block_tokens = concatenated_seq[start:end]
        
        # Pad block to block_size if needed
        if len(block_tokens) < self.block_size:
            block_tokens = block_tokens + [self.pad_token_id] * (self.block_size - len(block_tokens))
        
        return torch.tensor(block_tokens, dtype=self.dtype)


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
    train_ds = ChunkedDataset(train_paths, block_size=block_size, tokenizer=tokenizer, pad_token_id=pad_id)
    val_ds   = ChunkedDataset(val_paths,   block_size=block_size, tokenizer=tokenizer, pad_token_id=pad_id)
    test_ds  = ChunkedDataset(test_paths,  block_size=block_size, tokenizer=tokenizer, pad_token_id=pad_id)

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
    
    # Check if dynamic masking is enabled
    use_dynamic_masking = config.get("use_dynamic_masking", False)
    
    if use_dynamic_masking:
        print(f"{C.MAGENTA}🎭 Using RoBERTa-style Dynamic Masking{C.RESET}")
        from dynamic_collator import create_dynamic_collator
        
        # Create dynamic masking collator
        collate_fn = create_dynamic_collator(config, tokenizer)
        
        # For dynamic masking, we don't need to pre-apply masking to datasets
        # The collator will handle masking on-the-fly
        
    else:
        print(f"{C.BLUE}Using static masking strategy{C.RESET}")
        
        # Get masking strategy from config (default to "span")
        masking_strategy_name = config.get("masking_strategy", "span")
        
        # Initialize the appropriate masking strategy
        n_special_tokens = 6  # [PAD], [UNK], [CLS], [SEP], [MASK], [unused0]
        mask_p = config.get("mask_p", 0.15)  # 15% masking probability
        random_p = config.get("random_p", 0.1)  # 10% random token replacement
        keep_p = config.get("keep_p", 0.1)  # 10% keep original token
        
        masking_strategies = {
            "span": SpanMaskingStrategy,
            "subword": SubwordMaskingStrategy,
            "whole_word": WholeWordMaskingStrategy,
        }
        
        if masking_strategy_name not in masking_strategies:
            raise ValueError(f"Unknown masking strategy: {masking_strategy_name}. Choose from {list(masking_strategies.keys())}")
        
        masking_strategy_class = masking_strategies[masking_strategy_name]
        masking_strategy = masking_strategy_class(
            mask_p=mask_p,
            tokenizer=tokenizer,
            n_special_tokens=n_special_tokens,
            padding_label_id=-100,
            random_p=random_p,
            keep_p=keep_p,
        )
        
        print(f"{C.BLUE}Using {masking_strategy_name} masking strategy (mask_p={mask_p}, random_p={random_p}, keep_p={keep_p}){C.RESET}")
        
        # Custom MLM collator using the chosen masking strategy (static)
        def collate_fn(examples):
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
            all_labels = []
            
            for seq in input_ids:
                # Pad sequence
                padding_length = max_len - len(seq)
                padded_seq = torch.cat([seq, torch.full((padding_length,), pad_id, dtype=torch.long)])
                
                # Apply the chosen masking strategy to the padded sequence
                masked_tokens, labels = masking_strategy(padded_seq)

                padded_input_ids.append(masked_tokens)
                all_labels.append(labels)
                
                # Create attention mask (1 for real tokens, 0 for padding)
                attention_mask = torch.cat([torch.ones(len(seq), dtype=torch.long), 
                                          torch.zeros(padding_length, dtype=torch.long)])
                attention_masks.append(attention_mask)
            
            # Stack into batch tensors
            input_ids_batch = torch.stack(padded_input_ids)
            attention_mask_batch = torch.stack(attention_masks)
            labels_batch = torch.stack(all_labels)
            
            return {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "labels": labels_batch,
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
        batch_size=config["batch_size"],
        num_workers=2,  # Reduced to avoid deadlocks
        pin_memory=True,
        collate_fn=collate_fn,  # Use configurable masking strategy
        prefetch_factor=2,  # Reduced prefetch factor
        persistent_workers=False,  # Disabled to avoid multiprocessing issues
        drop_last=True,  # optional: move drop_last here if you want strict batch shapes
    )

    print(f"  {C.GREEN}→ {len(train_loader)} train batches{C.RESET}")

    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True,
                               collate_fn=collate_fn, drop_last=True)  # Use configurable masking for validation too
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True,
                               collate_fn=collate_fn, drop_last=True)  # Use configurable masking for test too

    print(f"{C.CYAN}Data preparation complete. Train files: {len(train_paths)}, Val files: {len(val_paths)}, Test files: {len(test_paths)}{C.RESET}")

    return train_loader, val_loader, test_loader, collate_fn, total_tokens_train
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