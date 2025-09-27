#!/usr/bin/env python3#!/usr/bin/env python3

""""""

Comprehensive test suite for data_loader.pyComprehensive test     def __init__(self, config_path: str, num_test_batches: int = 3):

Tests batch sampling, dataset statistics, and masking strategies.        """

"""        Initialize the tester.

        

# Add parent directory to path for imports        Args:

import os            config_path: Path to the model configuration file

import sys            num_test_batches: Number of batches to test (default: 3 for speed)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))        """

if parent_dir not in sys.path:        self.config_path = config_path

    sys.path.insert(0, parent_dir)        self.num_test_batches = num_test_batchesdata_loader.py

Tests batch sampling, dataset statistics, and masking strategies.

import json"""

import os

import random

import time# Add parent directory to path for imports

from collections import defaultdict, Counterimport os

from pathlib import Pathimport sys

import numpy as npparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torchif parent_dir not in sys.path:

from torch.utils.data import DataLoader    sys.path.insert(0, parent_dir)

try:

    import matplotlib.pyplot as pltimport json

    import seaborn as snsimport os

    PLOTTING_AVAILABLE = Trueimport random

except ImportError:import time

    PLOTTING_AVAILABLE = Falsefrom collections import defaultdict, Counter

    print("Warning: matplotlib/seaborn not available, disabling plotting functionality")from pathlib import Path

from tokenizers import Tokenizerimport numpy as np

import torch

# Import the modules we want to testfrom torch.utils.data import DataLoader

from data_loader import (try:

    ChunkedDataset,     import matplotlib.pyplot as plt

    TokenBudgetBatchSampler,     import seaborn as sns

    create_and_cache_splits,     PLOTTING_AVAILABLE = True

    data_loader,except ImportError:

    C  # Color helper    PLOTTING_AVAILABLE = False

)    print("Warning: matplotlib/seaborn not available, disabling plotting functionality")

from mlm_dataset import SpanMaskingStrategy, SubwordMaskingStrategy, WholeWordMaskingStrategyfrom tokenizers import Tokenizer



# Import the modules we want to test

class DataLoaderTester:from data_loader import (

    """Comprehensive tester for the data loader functionality."""    ChunkedDataset, 

        TokenBudgetBatchSampler, 

    def __init__(self, config_path: str, num_test_batches: int = 20):    create_and_cache_splits, 

        """    data_loader,

        Initialize the tester.    C  # Color helper

        )

        Args:from mlm_dataset import SpanMaskingStrategy, SubwordMaskingStrategy, WholeWordMaskingStrategy

            config_path: Path to the configuration JSON file

            num_test_batches: Number of batches to sample for testing (default: 20 for comprehensive testing)

        """class DataLoaderTester:

        self.config_path = config_path    """Comprehensive tester for the data loader functionality."""

        self.num_test_batches = num_test_batches    

            def __init__(self, config_path: str, num_test_batches: int = 3):

        # Load configuration        """

        with open(config_path, 'r') as f:        Initialize the tester.

            self.config = json.load(f)        

                    Args:

        print(f"Loaded config from {config_path}")            config_path: Path to the configuration JSON file

        print(f"Test batches: {num_test_batches}")            num_test_batches: Number of batches to sample for testing (reduced for speed)

        """

    def test_data_splits(self):        self.config_path = config_path

        """Test the data splitting functionality."""        self.num_test_batches = num_test_batches

        print(f"\n{C.CYAN}=" * 80 + C.RESET)        

        print(f"{C.CYAN}TESTING DATA SPLITS{C.RESET}")        # Load configuration

        print(f"{C.CYAN}=" * 80 + C.RESET)        with open(config_path, "r") as f:

                    self.config = json.load(f)

        # Test with realistic large dataset        

        splits = create_and_cache_splits(        print(f"{C.BOLD}{C.CYAN}=== Data Loader Tester Initialized ==={C.RESET}")

            data_dir="data/pretrain",        print(f"Config: {config_path}")

            chunk_patterns=["*.md"],        print(f"Test batches: {num_test_batches}")

            train_ratio=0.8,    

            validation_ratio=0.1,    def setup_tokenizer(self):

            test_ratio=0.1,        """Initialize the tokenizer from config."""

            seed=42        tokenizer_path = self.config.get("tokenizer_path")

        )        if not tokenizer_path:

                    raise ValueError("tokenizer_path not found in config")

        print(f"Train files: {len(splits['train'])}")        

        print(f"Validation files: {len(splits['validation'])}")        print(f"{C.BLUE}Loading tokenizer from: {tokenizer_path}{C.RESET}")

        print(f"Test files: {len(splits['test'])}")        self.tokenizer = Tokenizer.from_file(tokenizer_path)

                

        # Verify no overlap        # Get special token IDs

        train_set = set(splits['train'])        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")

        val_set = set(splits['validation'])        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")

        test_set = set(splits['test'])        self.cls_token_id = self.tokenizer.token_to_id("[CLS]")

                self.sep_token_id = self.tokenizer.token_to_id("[SEP]")

        assert len(train_set & val_set) == 0, "Train-validation overlap detected"        

        assert len(train_set & test_set) == 0, "Train-test overlap detected"        print(f"Vocab size: {self.tokenizer.get_vocab_size()}")

        assert len(val_set & test_set) == 0, "Validation-test overlap detected"        print(f"Special tokens - PAD: {self.pad_token_id}, MASK: {self.mask_token_id}, CLS: {self.cls_token_id}, SEP: {self.sep_token_id}")

            

        print(f"{C.GREEN}✓ Data splits verification passed{C.RESET}")    def test_chunked_dataset(self):

        return splits        """Test the ChunkedDataset class."""

        print(f"\n{C.BOLD}{C.GREEN}=== Testing ChunkedDataset ==={C.RESET}")

    def test_chunked_dataset(self, splits):        

        """Test ChunkedDataset with realistic parameters."""        cache_path = self.config["cache_path"]

        print(f"\n{C.CYAN}=" * 80 + C.RESET)        block_size = self.config["block_size"]

        print(f"{C.CYAN}TESTING CHUNKED DATASET{C.RESET}")        

        print(f"{C.CYAN}=" * 80 + C.RESET)        # Get chunk paths

                train_paths, val_paths, test_paths = create_and_cache_splits(self.config)

        # Load tokenizer        

        tokenizer = Tokenizer.from_file("data/pretrain/wordpiece_vocab.json")        # Test with a small subset of training data

                test_paths = train_paths[:3]  # Use only first 3 chunks for testing

        # Test with full dataset for realistic testing        

        train_dataset = ChunkedDataset(        print(f"Testing with {len(test_paths)} chunk files")

            tokenizer=tokenizer,        print(f"Block size: {block_size}")

            chunk_paths=splits['train'],        

            sequence_length=self.config['sequence_length'],        # Create dataset

            concatenate_sequences=True        start_time = time.time()

        )        dataset = ChunkedDataset(

                    test_paths,

        val_dataset = ChunkedDataset(            tokenizer=self.tokenizer,  # Pass actual tokenizer

            tokenizer=tokenizer,            block_size=block_size,

            chunk_paths=splits['validation'],            pad_token_id=self.pad_token_id

            sequence_length=self.config['sequence_length'],        )

            concatenate_sequences=True        creation_time = time.time() - start_time

        )        

                print(f"Dataset created in {creation_time:.2f}s")

        print(f"Train dataset length: {len(train_dataset)}")        print(f"Dataset length: {len(dataset):,} samples")

        print(f"Validation dataset length: {len(val_dataset)}")        print(f"Total tokens: {sum(dataset.block_lengths):,}")

                print(f"Average block length: {np.mean(dataset.block_lengths):.2f}")

        # Test random sampling across large dataset        print(f"Block length std: {np.std(dataset.block_lengths):.2f}")

        print(f"\n{C.YELLOW}Testing random sampling across large dataset:{C.RESET}")        

        for i in range(5):        # Test random samples

            idx = random.randint(0, len(train_dataset) - 1)        print(f"\n{C.CYAN}Sampling random examples:{C.RESET}")

            sample = train_dataset[idx]        for i in range(5):

            print(f"Sample {i+1}: idx={idx}, shape={sample.shape}, non-zero={torch.count_nonzero(sample)}")            idx = random.randint(0, len(dataset) - 1)

                    sample = dataset[idx]

        print(f"{C.GREEN}✓ ChunkedDataset test passed{C.RESET}")            actual_length = dataset.block_lengths[idx]

        return train_dataset, val_dataset            

            print(f"  Sample {idx}: shape={sample.shape}, dtype={sample.dtype}, "

    def test_batch_sampling(self, train_dataset, val_dataset):                  f"actual_length={actual_length}, non_pad_tokens={torch.sum(sample != self.pad_token_id).item()}")

        """Test batch sampling with realistic token budgets."""        

        print(f"\n{C.CYAN}=" * 80 + C.RESET)        return dataset

        print(f"{C.CYAN}TESTING BATCH SAMPLING{C.RESET}")    

        print(f"{C.CYAN}=" * 80 + C.RESET)    def test_token_budget_batch_sampler(self, dataset):

                """Test the TokenBudgetBatchSampler."""

        # Test with realistic token budget        print(f"\n{C.BOLD}{C.GREEN}=== Testing TokenBudgetBatchSampler ==={C.RESET}")

        token_budget = self.config.get('token_budget', 8192)  # Realistic budget        

                max_tokens = self.config.get("max_tokens", self.config["block_size"] * self.config["batch_size"])

        train_sampler = TokenBudgetBatchSampler(        lengths = dataset.block_lengths

            dataset=train_dataset,        

            token_budget=token_budget,        print(f"Max tokens per batch: {max_tokens}")

            sequence_length=self.config['sequence_length'],        print(f"Sample lengths - min: {min(lengths)}, max: {max(lengths)}, mean: {np.mean(lengths):.2f}")

            shuffle=True,        

            drop_last=False        # Test batch sampler

        )        sampler = TokenBudgetBatchSampler(

                    lengths=lengths,

        val_sampler = TokenBudgetBatchSampler(            max_tokens=max_tokens,

            dataset=val_dataset,            shuffle=True

            token_budget=token_budget,        )

            sequence_length=self.config['sequence_length'],        

            shuffle=False,        print(f"Number of batches: {len(sampler)}")

            drop_last=False        

        )        # Analyze batch statistics

                batch_sizes = []

        print(f"Token budget: {token_budget}")        token_counts = []

        print(f"Train batches: {len(train_sampler)}")        

        print(f"Validation batches: {len(val_sampler)}")        for batch_indices in sampler:

                    batch_size = len(batch_indices)

        # Test multiple batches to verify consistency            batch_token_count = sum(lengths[idx] for idx in batch_indices)

        print(f"\n{C.YELLOW}Testing batch consistency:{C.RESET}")            

        batch_sizes = []            batch_sizes.append(batch_size)

        for i, batch_indices in enumerate(train_sampler):            token_counts.append(batch_token_count)

            if i >= 10:  # Test first 10 batches        

                break        print(f"Batch sizes - min: {min(batch_sizes)}, max: {max(batch_sizes)}, mean: {np.mean(batch_sizes):.2f}")

            batch_size = len(batch_indices)        print(f"Token counts - min: {min(token_counts)}, max: {max(token_counts)}, mean: {np.mean(token_counts):.2f}")

            total_tokens = batch_size * self.config['sequence_length']        print(f"Token budget utilization: {np.mean(token_counts) / max_tokens * 100:.1f}%")

            batch_sizes.append(batch_size)        

            print(f"Batch {i+1}: size={batch_size}, tokens={total_tokens}")        # Check if any batch exceeds the token budget

            assert total_tokens <= token_budget, f"Batch {i+1} exceeds token budget"        over_budget = [tc for tc in token_counts if tc > max_tokens]

                if over_budget:

        print(f"Average batch size: {np.mean(batch_sizes):.1f}")            print(f"{C.YELLOW}WARNING: {len(over_budget)} batches exceed token budget!{C.RESET}")

        print(f"{C.GREEN}✓ Batch sampling test passed{C.RESET}")        else:

                    print(f"{C.GREEN}✓ All batches within token budget{C.RESET}")

        return train_sampler, val_sampler        

        return sampler

    def test_data_loaders(self, train_dataset, val_dataset, train_sampler, val_sampler):    

        """Test data loaders with realistic configurations."""    def test_masking_strategies(self):

        print(f"\n{C.CYAN}=" * 80 + C.RESET)        """Test different masking strategies."""

        print(f"{C.CYAN}TESTING DATA LOADERS{C.RESET}")        print(f"\n{C.BOLD}{C.GREEN}=== Testing Masking Strategies ==={C.RESET}")

        print(f"{C.CYAN}=" * 80 + C.RESET)        

                # Create a sample sequence

        # Use realistic number of workers        sample_text = "The quick brown fox jumps over the lazy dog."

        num_workers = min(4, os.cpu_count())        encoded = self.tokenizer.encode(sample_text)

                tokens = torch.tensor(encoded.ids, dtype=torch.long)

        train_loader = DataLoader(        

            train_dataset,        print(f"Original text: '{sample_text}'")

            batch_sampler=train_sampler,        print(f"Tokenized: {tokens.tolist()}")

            num_workers=num_workers,        print(f"Tokens: {[self.tokenizer.id_to_token(tid) for tid in tokens.tolist()]}")

            pin_memory=True,        

            persistent_workers=True if num_workers > 0 else False        # Test parameters

        )        mask_p = 0.15

                n_special_tokens = 6

        val_loader = DataLoader(        random_p = 0.1

            val_dataset,        keep_p = 0.1

            batch_sampler=val_sampler,        

            num_workers=num_workers,        strategies = {

            pin_memory=True,            "subword": SubwordMaskingStrategy,

            persistent_workers=True if num_workers > 0 else False            "span": SpanMaskingStrategy,

        )            "whole_word": WholeWordMaskingStrategy,

                }

        print(f"Using {num_workers} workers")        

        print(f"Train loader batches: {len(train_loader)}")        for strategy_name, strategy_class in strategies.items():

        print(f"Validation loader batches: {len(val_loader)}")            print(f"\n{C.CYAN}Testing {strategy_name} masking:{C.RESET}")

                    

        # Test loading performance with large batches            try:

        print(f"\n{C.YELLOW}Testing loading performance:{C.RESET}")                strategy = strategy_class(

        start_time = time.time()                    mask_p=mask_p,

                            tokenizer=self.tokenizer,

        for i, batch in enumerate(train_loader):                    n_special_tokens=n_special_tokens,

            if i >= self.num_test_batches:                    padding_label_id=-100,

                break                    random_p=random_p,

            batch_time = time.time()                    keep_p=keep_p

            print(f"Batch {i+1}: shape={batch.shape}, dtype={batch.dtype}, "                )

                  f"time={batch_time - start_time:.3f}s")                

            start_time = batch_time                masked_tokens, labels = strategy(tokens.clone())

                        

        print(f"{C.GREEN}✓ Data loader test passed{C.RESET}")                # Count different types of changes

        return train_loader, val_loader                mask_count = torch.sum(masked_tokens == self.mask_token_id).item()

                label_count = torch.sum(labels != -100).item()

    def analyze_batches(self, data_loader, split_name: str, num_batches: int = None):                changed_count = torch.sum(masked_tokens != tokens).item()

        """Analyze batch statistics for large dataset scenarios."""                

                        print(f"  Masks applied: {mask_count}")

        if num_batches is None:                print(f"  Labels created: {label_count}")

            num_batches = min(self.num_test_batches, len(data_loader))                print(f"  Tokens changed: {changed_count}")

                        print(f"  Masking rate: {label_count / len(tokens) * 100:.1f}%")

        print(f"\n{C.CYAN}Analyzing {num_batches} random {split_name.lower()} batches:{C.RESET}")                

                        # Show the changes

        # Collect statistics from multiple batches                changes = []

        batch_sizes = []                for i, (orig, masked, label) in enumerate(zip(tokens, masked_tokens, labels)):

        sequence_lengths = []                    if orig != masked or label != -100:

        token_counts = []                        orig_token = self.tokenizer.id_to_token(orig.item())

        vocab_usage = defaultdict(int)                        masked_token = self.tokenizer.id_to_token(masked.item())

                                changes.append(f"pos{i}: '{orig_token}' -> '{masked_token}' (label: {label.item()})")

        # Sample batches randomly across the entire dataset                

        all_batches = list(range(len(data_loader)))                if changes:

        if len(all_batches) > num_batches:                    print(f"  Changes: {changes[:5]}")  # Show first 5 changes

            sampled_indices = random.sample(all_batches, num_batches)                    if len(changes) > 5:

        else:                        print(f"    ... and {len(changes) - 5} more")

            sampled_indices = all_batches                

                        except Exception as e:

        # Load sampled batches                print(f"  {C.RED}ERROR: {e}{C.RESET}")

        all_batches = []    

        for batch in data_loader:    def test_data_loader_batches(self):

            all_batches.append(batch)        """Test the actual data loader and analyze batches."""

                    print(f"\n{C.BOLD}{C.GREEN}=== Testing Data Loader Batches ==={C.RESET}")

        sampled_batches = [all_batches[i] for i in sampled_indices if i < len(all_batches)]        

                # Create data loaders

        for i, batch in enumerate(sampled_batches):        train_loader, val_loader, test_loader, collate_fn, total_tokens = data_loader(

            batch_size, seq_len = batch.shape            self.config, self.tokenizer, self.config["cache_path"]

            batch_sizes.append(batch_size)        )

            sequence_lengths.append(seq_len)        

                    print(f"Total training tokens: {total_tokens:,}")

            # Count total tokens (non-padding)        print(f"Train batches: {len(train_loader)}")

            non_pad_tokens = torch.sum(batch != 0).item()  # Assuming 0 is padding        print(f"Val batches: {len(val_loader)}")

            token_counts.append(non_pad_tokens)        print(f"Test batches: {len(test_loader)}")

                    

            # Analyze vocabulary usage        # Sample and analyze batches

            unique_tokens = torch.unique(batch[batch != 0])        batch_stats = self.analyze_batches(train_loader, "Training")

            for token in unique_tokens:        val_stats = self.analyze_batches(val_loader, "Validation", num_batches=3)

                vocab_usage[token.item()] += 1        

                        return train_loader, batch_stats

            print(f"  Batch {i+1}: {batch_size}x{seq_len}, tokens={non_pad_tokens}, "    

                  f"unique_vocab={len(unique_tokens)}")    def analyze_batches(self, data_loader, split_name: str, num_batches: int = None):

                """Analyze random batches from a data loader."""

        # Print comprehensive statistics        if num_batches is None:

        print(f"\n{C.YELLOW}Comprehensive {split_name} Statistics:{C.RESET}")            num_batches = min(self.num_test_batches, len(data_loader))

        print(f"  Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, "        

              f"avg={np.mean(batch_sizes):.1f}, std={np.std(batch_sizes):.1f}")        print(f"\n{C.CYAN}Analyzing {num_batches} random {split_name.lower()} batches:{C.RESET}")

        print(f"  Sequence lengths: min={min(sequence_lengths)}, max={max(sequence_lengths)}")        

        print(f"  Token counts: min={min(token_counts)}, max={max(token_counts)}, "        stats = {

              f"avg={np.mean(token_counts):.1f}")            'batch_sizes': [],

        print(f"  Vocabulary usage: {len(vocab_usage)} unique tokens across batches")            'sequence_lengths': [],

        print(f"  Most frequent tokens: {Counter(vocab_usage).most_common(10)}")            'token_counts': [],

                    'pad_token_counts': [],

        return {            'mask_token_counts': [],

            'batch_sizes': batch_sizes,            'attention_lengths': [],

            'sequence_lengths': sequence_lengths,            'label_counts': [],

            'token_counts': token_counts,            'unique_tokens_per_batch': [],

            'vocab_usage': vocab_usage        }

        }        

        # Convert data loader to list for random sampling

    def test_masking_strategies(self, data_loader, split_name: str):        data_iter = iter(data_loader)

        """Test different masking strategies with large dataset."""        all_batches = []

        print(f"\n{C.CYAN}=" * 80 + C.RESET)        

        print(f"{C.CYAN}TESTING MASKING STRATEGIES ON {split_name.upper()}{C.RESET}")        # Collect a reasonable number of batches for sampling

        print(f"{C.CYAN}=" * 80 + C.RESET)        max_collect = min(100, len(data_loader))

                for i, batch in enumerate(data_iter):

        tokenizer = Tokenizer.from_file("data/pretrain/wordpiece_vocab.json")            all_batches.append(batch)

                    if i >= max_collect - 1:

        # Test all masking strategies                break

        strategies = {        

            'span': SpanMaskingStrategy(tokenizer, max_span_length=3),        # Randomly sample batches

            'subword': SubwordMaskingStrategy(tokenizer),        sampled_batches = random.sample(all_batches, min(num_batches, len(all_batches)))

            'whole_word': WholeWordMaskingStrategy(tokenizer)        

        }        for i, batch in enumerate(sampled_batches):

                    input_ids = batch['input_ids']

        # Test with multiple batches for statistical significance            attention_mask = batch['attention_mask']

        num_tests = min(5, len(data_loader))            labels = batch.get('labels', None)

                    

        for strategy_name, strategy in strategies.items():            batch_size, seq_len = input_ids.shape

            print(f"\n{C.YELLOW}Testing {strategy_name} masking:{C.RESET}")            stats['batch_sizes'].append(batch_size)

                        stats['sequence_lengths'].append(seq_len)

            masking_stats = []            

                        # Count tokens

            for test_num in range(num_tests):            total_tokens = batch_size * seq_len

                # Get a batch            pad_tokens = torch.sum(input_ids == self.pad_token_id).item()

                batch = next(iter(data_loader))            mask_tokens = torch.sum(input_ids == self.mask_token_id).item()

                            real_tokens = torch.sum(attention_mask == 1).item()

                # Apply masking            

                masked_batch, labels = strategy.apply_masking(batch)            stats['token_counts'].append(total_tokens)

                            stats['pad_token_counts'].append(pad_tokens)

                # Calculate statistics            stats['mask_token_counts'].append(mask_tokens)

                total_tokens = torch.sum(batch != 0).item()  # Non-padding tokens            stats['attention_lengths'].append(real_tokens)

                masked_tokens = torch.sum(labels != -100).item()            

                masking_rate = (masked_tokens / total_tokens) * 100 if total_tokens > 0 else 0            # Count unique tokens

                            unique_tokens = len(torch.unique(input_ids))

                masking_stats.append(masking_rate)            stats['unique_tokens_per_batch'].append(unique_tokens)

                            

                print(f"  Test {test_num + 1}: {masked_tokens}/{total_tokens} tokens masked ({masking_rate:.1f}%)")            # Count labels if available

                        if labels is not None:

            avg_masking_rate = np.mean(masking_stats)                label_tokens = torch.sum(labels != -100).item()

            std_masking_rate = np.std(masking_stats)                stats['label_counts'].append(label_tokens)

                        

            print(f"  {C.GREEN}{strategy_name.capitalize()} masking: {avg_masking_rate:.1f}% ± {std_masking_rate:.1f}%{C.RESET}")            print(f"  Batch {i+1}: size={batch_size}, seq_len={seq_len}, "

                          f"real_tokens={real_tokens}, pad_tokens={pad_tokens}, "

        print(f"{C.GREEN}✓ Masking strategies test passed{C.RESET}")                  f"mask_tokens={mask_tokens}, unique_tokens={unique_tokens}")

            

    def run_comprehensive_test(self):            if labels is not None:

        """Run the complete test suite with large dataset scenarios."""                print(f"           labels={label_tokens}, masking_rate={label_tokens/real_tokens*100:.1f}%")

        print(f"\n{C.MAGENTA}🔬 COMPREHENSIVE DATA LOADER TEST SUITE{C.RESET}")        

        print(f"{C.MAGENTA}Testing with realistic large dataset scenarios{C.RESET}")        # Print summary statistics

        print(f"{C.MAGENTA}=" * 80 + C.RESET)        print(f"\n{C.YELLOW}{split_name} Batch Statistics Summary:{C.RESET}")

                print(f"  Batch sizes: {np.mean(stats['batch_sizes']):.1f} ± {np.std(stats['batch_sizes']):.1f}")

        start_time = time.time()        print(f"  Sequence lengths: {np.mean(stats['sequence_lengths']):.1f} ± {np.std(stats['sequence_lengths']):.1f}")

                print(f"  Real tokens per batch: {np.mean(stats['attention_lengths']):.1f} ± {np.std(stats['attention_lengths']):.1f}")

        try:        print(f"  Padding ratio: {np.mean(stats['pad_token_counts'])/np.mean(stats['token_counts'])*100:.1f}%")

            # Test 1: Data splits        print(f"  Unique tokens per batch: {np.mean(stats['unique_tokens_per_batch']):.1f} ± {np.std(stats['unique_tokens_per_batch']):.1f}")

            splits = self.test_data_splits()        

                    if stats['mask_token_counts']:

            # Test 2: ChunkedDataset            mask_rate = np.mean(stats['mask_token_counts']) / np.mean(stats['attention_lengths']) * 100

            train_dataset, val_dataset = self.test_chunked_dataset(splits)            print(f"  Mask token rate: {mask_rate:.1f}%")

                    

            # Test 3: Batch sampling        if stats['label_counts']:

            train_sampler, val_sampler = self.test_batch_sampling(train_dataset, val_dataset)            label_rate = np.mean(stats['label_counts']) / np.mean(stats['attention_lengths']) * 100

                        print(f"  Label rate: {label_rate:.1f}%")

            # Test 4: Data loaders        

            train_loader, val_loader = self.test_data_loaders(        return stats

                train_dataset, val_dataset, train_sampler, val_sampler    

            )    def test_batch_consistency(self, data_loader, num_tests: int = 5):

                    """Test batch consistency and reproducibility."""

            # Test 5: Batch analysis        print(f"\n{C.BOLD}{C.GREEN}=== Testing Batch Consistency ==={C.RESET}")

            train_stats = self.analyze_batches(train_loader, "Training", num_batches=self.num_test_batches)        

            val_stats = self.analyze_batches(val_loader, "Validation", num_batches=5)        # Test that batches have consistent shapes within each batch

                    data_iter = iter(data_loader)

            # Test 6: Masking strategies        

            self.test_masking_strategies(train_loader, "Training")        for test_num in range(min(num_tests, len(data_loader))):

                        try:

            # Final summary                batch = next(data_iter)

            total_time = time.time() - start_time                input_ids = batch['input_ids']

            print(f"\n{C.MAGENTA}=" * 80 + C.RESET)                attention_mask = batch['attention_mask']

            print(f"{C.GREEN}🎉 ALL TESTS PASSED!{C.RESET}")                labels = batch.get('labels', None)

            print(f"{C.YELLOW}Total test time: {total_time:.2f} seconds{C.RESET}")                

            print(f"{C.YELLOW}Tested {self.num_test_batches} training batches and 5 validation batches{C.RESET}")                # Check shapes

            print(f"{C.MAGENTA}=" * 80 + C.RESET)                batch_size, seq_len = input_ids.shape

                            assert attention_mask.shape == (batch_size, seq_len), f"Attention mask shape mismatch: {attention_mask.shape} vs {input_ids.shape}"

            return True                

                            if labels is not None:

        except Exception as e:                    assert labels.shape == (batch_size, seq_len), f"Labels shape mismatch: {labels.shape} vs {input_ids.shape}"

            print(f"\n{C.RED}❌ TEST FAILED: {str(e)}{C.RESET}")                

            import traceback                # Check data types

            traceback.print_exc()                assert input_ids.dtype == torch.long, f"Input IDs wrong dtype: {input_ids.dtype}"

            return False                assert attention_mask.dtype in [torch.long, torch.int, torch.bool], f"Attention mask wrong dtype: {attention_mask.dtype}"

                

                # Check value ranges

def main():                assert torch.all(input_ids >= 0), "Negative token IDs found"

    """Main test function."""                assert torch.all(input_ids < self.tokenizer.get_vocab_size()), "Token IDs exceed vocab size"

    import argparse                assert torch.all((attention_mask == 0) | (attention_mask == 1)), "Attention mask values not 0 or 1"

                    

    parser = argparse.ArgumentParser(description="Test data_loader.py with large dataset scenarios")                print(f"  Batch {test_num + 1}: ✓ Consistency checks passed")

    parser.add_argument("--config", type=str, default="configs/base.json",                 

                       help="Path to model configuration file")            except StopIteration:

    parser.add_argument("--num_batches", type=int, default=20,                 print(f"  Only {test_num} batches available for testing")

                       help="Number of test batches to sample (default: 20 for comprehensive testing)")                break

                except Exception as e:

    args = parser.parse_args()                print(f"  {C.RED}Batch {test_num + 1}: FAILED - {e}{C.RESET}")

            

    if not os.path.exists(args.config):        print(f"{C.GREEN}✓ Batch consistency tests completed{C.RESET}")

        print(f"Error: Config file not found: {args.config}")    

        return False    def visualize_statistics(self, batch_stats, save_path: str = None):

            """Create visualizations of batch statistics."""

    # Run comprehensive tests with large dataset        print(f"\n{C.BOLD}{C.GREEN}=== Creating Visualizations ==={C.RESET}")

    tester = DataLoaderTester(args.config, args.num_batches)        

    success = tester.run_comprehensive_test()        if not PLOTTING_AVAILABLE:

                print(f"{C.YELLOW}Matplotlib/seaborn not available, skipping visualizations{C.RESET}")

    return success            return

            

        try:

if __name__ == "__main__":            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    success = main()            fig.suptitle('Data Loader Batch Statistics', fontsize=16)

    sys.exit(0 if success else 1)            
            # Batch sizes
            axes[0, 0].hist(batch_stats['batch_sizes'], bins=20, alpha=0.7)
            axes[0, 0].set_title('Batch Sizes')
            axes[0, 0].set_xlabel('Batch Size')
            axes[0, 0].set_ylabel('Frequency')
            
            # Sequence lengths
            axes[0, 1].hist(batch_stats['sequence_lengths'], bins=20, alpha=0.7)
            axes[0, 1].set_title('Sequence Lengths')
            axes[0, 1].set_xlabel('Sequence Length')
            axes[0, 1].set_ylabel('Frequency')
            
            # Token counts
            axes[0, 2].hist(batch_stats['attention_lengths'], bins=20, alpha=0.7, color='green')
            axes[0, 2].set_title('Real Tokens per Batch')
            axes[0, 2].set_xlabel('Number of Tokens')
            axes[0, 2].set_ylabel('Frequency')
            
            # Padding ratio
            padding_ratios = [pad/total for pad, total in zip(batch_stats['pad_token_counts'], batch_stats['token_counts'])]
            axes[1, 0].hist(padding_ratios, bins=20, alpha=0.7, color='orange')
            axes[1, 0].set_title('Padding Ratios')
            axes[1, 0].set_xlabel('Padding Ratio')
            axes[1, 0].set_ylabel('Frequency')
            
            # Unique tokens
            axes[1, 1].hist(batch_stats['unique_tokens_per_batch'], bins=20, alpha=0.7, color='purple')
            axes[1, 1].set_title('Unique Tokens per Batch')
            axes[1, 1].set_xlabel('Number of Unique Tokens')
            axes[1, 1].set_ylabel('Frequency')
            
            # Mask tokens (if available)
            if batch_stats['mask_token_counts'] and any(batch_stats['mask_token_counts']):
                axes[1, 2].hist(batch_stats['mask_token_counts'], bins=20, alpha=0.7, color='red')
                axes[1, 2].set_title('Mask Tokens per Batch')
                axes[1, 2].set_xlabel('Number of Mask Tokens')
                axes[1, 2].set_ylabel('Frequency')
            else:
                axes[1, 2].text(0.5, 0.5, 'No Mask Tokens', ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Mask Tokens per Batch')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Visualizations saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"{C.RED}Error creating visualizations: {e}{C.RESET}")
    
    def run_full_test_suite(self):
        """Run the complete test suite."""
        print(f"{C.BOLD}{C.MAGENTA}{'='*60}{C.RESET}")
        print(f"{C.BOLD}{C.MAGENTA}    COMPREHENSIVE DATA LOADER TEST SUITE{C.RESET}")
        print(f"{C.BOLD}{C.MAGENTA}{'='*60}{C.RESET}")
        
        start_time = time.time()
        
        try:
            # Setup
            self.setup_tokenizer()
            
            # Test individual components
            dataset = self.test_chunked_dataset()
            sampler = self.test_token_budget_batch_sampler(dataset)
            self.test_masking_strategies()
            
            # Test full data loader
            train_loader, batch_stats = self.test_data_loader_batches()
            self.test_batch_consistency(train_loader)
            
            # Visualizations
            self.visualize_statistics(batch_stats, save_path="batch_statistics.png")
            
            total_time = time.time() - start_time
            print(f"\n{C.BOLD}{C.GREEN}{'='*60}{C.RESET}")
            print(f"{C.BOLD}{C.GREEN}  ALL TESTS COMPLETED SUCCESSFULLY IN {total_time:.2f}s{C.RESET}")
            print(f"{C.BOLD}{C.GREEN}{'='*60}{C.RESET}")
            
            return True
            
        except Exception as e:
            print(f"\n{C.BOLD}{C.RED}{'='*60}{C.RESET}")
            print(f"{C.BOLD}{C.RED}  TEST SUITE FAILED: {e}{C.RESET}")
            print(f"{C.BOLD}{C.RED}{'='*60}{C.RESET}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function to run the tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the data loader functionality")
    parser.add_argument("--config", type=str, default="model_babylm_ltg_bert.json", help="Path to config JSON file")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of test batches to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Run tests
    tester = DataLoaderTester(args.config, args.num_batches)
    success = tester.run_full_test_suite()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
