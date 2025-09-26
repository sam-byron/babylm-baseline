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
from typing import List, Iterator, Any
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

STRUCTURAL_TOKENS = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]","[PAR]","[TAB]","[DOC]","[EOD]","[SPK]"]
def get_special_token_ids(tokenizer):
    ids = []
    for tok in STRUCTURAL_TOKENS:
        tid = tokenizer.token_to_id(tok)
        if tid is not None:
            ids.append(tid)
    return sorted(set(ids))

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
    """Dataset over pre-generated fixed-size training blocks stored in chunk*.pt files.

    Two operational modes:
      1. Preblocked mode (preferred): Each chunk file is a 2-D LongTensor [N, block_size]. We simply
         index rows without any re-concatenation.
      2. Legacy concatenation mode: Each chunk file stores a list/1-D sequences requiring chaining
         into uniform blocks (old pipeline). Retained for backward compatibility.
    """
    def __init__(self, chunk_paths, block_size, tokenizer, dtype=torch.long, pad_token_id=None, cache_size=50):
        self.chunk_paths = list(chunk_paths)
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.cls_index = self.tokenizer.token_to_id("[CLS]")
        self.sep_index = self.tokenizer.token_to_id("[SEP]")
        self.dtype = dtype
        self.pad_token_id = pad_token_id or 0
        self.cache_size = cache_size
    # path -> loaded tensor (2-D) OR concatenated flat list (legacy)
        self._chunk_cache: "OrderedDict[str, Any]" = OrderedDict()

        # Defer cache_dir until we know dataset root; place under chunk folder for stability

        # Detect format from first available file
        sample_path = None
        for pth in self.chunk_paths:
            if os.path.exists(pth):
                sample_path = pth
                break
        if sample_path is None:
            raise ValueError("No existing chunk paths provided to ChunkedDataset")

        # Set cache directory relative to dataset root for reproducibility
        root_dir = Path(sample_path).parent
        self.cache_dir = root_dir / "cache_index"
        self.cache_dir.mkdir(exist_ok=True)

        sample_obj = torch.load(sample_path, map_location="cpu")
        self.preblocked = torch.is_tensor(sample_obj) and sample_obj.dim() == 2
        if self.preblocked and self.block_size is not None and sample_obj.shape[1] != self.block_size:
            raise ValueError(f"Block size mismatch: arg {self.block_size} vs file width {sample_obj.shape[1]}")
        if self.preblocked:
            # Standard fast path
            self._build_index_preblocked()

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
    


    def _build_index_preblocked(self):
        """Index rows directly for preblocked 2-D tensors."""
        cache_key = self._get_cache_key() + "_preblocked"
        cache_file = self.cache_dir / f"index_map_{cache_key}.pkl"
        if cache_file.exists():
            try:
                print(f"{C.CYAN}Loading cached (preblocked) index map from {cache_file}{C.RESET}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self.index_map = cached_data['index_map']
                self.block_lengths = cached_data['block_lengths']
                print(f"{C.GREEN}Loaded {len(self.index_map)} row entries from cache{C.RESET}")
                return
            except Exception as e:
                print(f"{C.YELLOW}Cache load failed, rebuilding preblocked index: {e}{C.RESET}")
                try: cache_file.unlink()
                except Exception: pass
        print(f"{C.BLUE}Building preblocked index (row-level) ...{C.RESET}")
        self.index_map = []
        self.block_lengths = []
        total_rows = 0
        for pth in self.chunk_paths:
            try:
                tensor = torch.load(pth, map_location='cpu')
                if not (torch.is_tensor(tensor) and tensor.dim() == 2):
                    raise ValueError("Encountered non-2D tensor in preblocked mode; mixed formats not supported")
                n_rows, width = tensor.shape
                if width != self.block_size:
                    raise ValueError(f"Width mismatch in {pth}: expected {self.block_size} got {width}")
                # store (path, row_idx, start, end) with synthetic offsets for compatibility
                for r in range(n_rows):
                    start = r * self.block_size
                    end = start + self.block_size
                    self.index_map.append((pth, r, start, end))
                    self.block_lengths.append(self.block_size)
                total_rows += n_rows
            except Exception as e:
                print(f"{C.RED}Failed indexing {pth}: {e}{C.RESET}")
        print(f"{C.MAGENTA}Preblocked stats:{C.RESET} rows={total_rows:,} block_size={self.block_size}")
        # Cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'index_map': self.index_map, 'block_lengths': self.block_lengths}, f)
            print(f"{C.GREEN}Cached preblocked index to {cache_file}{C.RESET}")
        except Exception as e:
            print(f"{C.YELLOW}Warning: failed to cache preblocked index: {e}{C.RESET}")

    
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
        path, row_or_seq, start, end = self.index_map[idx]
        if self.preblocked:
            # Load entire 2-D tensor into cache lazily
            if path not in self._chunk_cache:
                tensor = torch.load(path, map_location='cpu')
                if not (torch.is_tensor(tensor) and tensor.dim() == 2):
                    raise ValueError(f"Expected 2-D tensor in preblocked mode for {path}")
                self._chunk_cache[path] = tensor
                if len(self._chunk_cache) > self.cache_size:
                    self._chunk_cache.popitem(last=False)
            tensor = self._chunk_cache[path]
            self._chunk_cache.move_to_end(path, last=True)
            block = tensor[row_or_seq].clone()
            # Ensure dtype long
            if block.dtype != torch.long:
                block = block.long()
            return block


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

# Update the data_loader function to use sentence-aware dataset when available
def data_loader(config, tokenizer, cache_path):
    block_size = config["block_size"]
    batch_size = config["batch_size"]
    
    # Check if we should use sentence-aware processing
    use_sentence_aware = config.get("use_sentence_aware", True)
    
    if use_sentence_aware:
        print(f"{C.MAGENTA}🎯 Using Sentence-Aware Dataset for improved syntax learning{C.RESET}")
        return _create_sentence_aware_loader(config, tokenizer, cache_path)

def _create_sentence_aware_loader(config, tokenizer, cache_path):
    """Create data loaders using the sentence-aware dataset."""
    from mlm_dataset import SentenceAwareDataset
    
    # Create the sentence-aware dataset
    print(f"{C.CYAN}Loading sentence-aware dataset from {cache_path}...{C.RESET}")
    
    try:
        # Use the new sentence-aware dataset
        full_dataset = SentenceAwareDataset(
            cache_path=cache_path,
            tokenizer=tokenizer,
            seq_length=config.get("block_size", 512),
            mask_p=config.get("mask_p", 0.15),
            random_p=config.get("random_p", 0.1),
            keep_p=config.get("keep_p", 0.1)
        )
        
        print(f"{C.GREEN}Loaded {len(full_dataset)} sentences for sentence-aware training{C.RESET}")
        
        # Split the dataset
        total_size = len(full_dataset)
        train_size = int(0.85 * total_size)
        val_size = int(0.05 * total_size)
        test_size = total_size - train_size - val_size
        
        # Create train/val/test splits
        import torch.utils.data as data
        train_dataset, val_dataset, test_dataset = data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        print(f"{C.CYAN}Dataset splits → train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}{C.RESET}")
        
        # Always use dynamic masking for sentence-aware mode for now
        from dynamic_collator import create_dynamic_collator
        base_collate = create_dynamic_collator(config, tokenizer)

        def sentence_dynamic_collate_fn(batch):
            """Adapter that extracts raw input_ids from dataset items and delegates to dynamic collator."""
            seqs = []
            for item in batch:
                if isinstance(item, dict):
                    seqs.append(item.get('input_ids', item))
                else:
                    seqs.append(item)
            return base_collate(seqs)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,  # moderate multiprocessing
            pin_memory=True,
            collate_fn=sentence_dynamic_collate_fn,
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=sentence_dynamic_collate_fn,
            drop_last=True,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=sentence_dynamic_collate_fn,
            drop_last=True,
        )
        
        # Compute total tokens precisely (train split)
        if hasattr(full_dataset, "sequences") and hasattr(train_dataset, "indices"):
            # Fast path: sum lengths directly from underlying storage using subset indices
            total_tokens_train = sum(int(len(full_dataset.sequences[i])) for i in train_dataset.indices)
            avg_sentence_length = total_tokens_train / max(1, len(train_dataset))
        else:
            # Fallback: iterate the subset (slower but exact)
            total_tokens_train = 0
            for item in train_dataset:
                seq = item['input_ids'] if isinstance(item, dict) else item
                total_tokens_train += int(len(seq))
            avg_sentence_length = total_tokens_train / max(1, len(train_dataset))
        
        print(f"{C.GREEN}Sentence-aware data loaders created successfully{C.RESET}")
        print(f"{C.CYAN}Average sentence length: {avg_sentence_length:.1f} tokens{C.RESET}")
        print(f"{C.CYAN}Total training tokens (exact): {total_tokens_train:,}{C.RESET}")
        
        return train_loader, val_loader, test_loader, sentence_dynamic_collate_fn, total_tokens_train
        
    except Exception as e:
        print(f"{C.RED}Error creating sentence-aware dataset: {e}{C.RESET}")



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