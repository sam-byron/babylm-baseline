#!/usr/bin/env python3
"""
Comprehensive test suite for data_loader.py
Tests batch sampling, dataset statistics, and masking strategies.
"""

import json
import os
import random
import time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizers import Tokenizer

# Import the modules we want to test
from data_loader import (
    ChunkedDataset, 
    TokenBudgetBatchSampler, 
    create_and_cache_splits, 
    data_loader,
    C  # Color helper
)
from mlm_dataset import SpanMaskingStrategy, SubwordMaskingStrategy, WholeWordMaskingStrategy


class DataLoaderTester:
    """Comprehensive tester for the data loader functionality."""
    
    def __init__(self, config_path: str, num_test_batches: int = 10):
        """
        Initialize the tester.
        
        Args:
            config_path: Path to the configuration JSON file
            num_test_batches: Number of batches to sample for testing
        """
        self.config_path = config_path
        self.num_test_batches = num_test_batches
        
        # Load configuration
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        print(f"{C.BOLD}{C.CYAN}=== Data Loader Tester Initialized ==={C.RESET}")
        print(f"Config: {config_path}")
        print(f"Test batches: {num_test_batches}")
    
    def setup_tokenizer(self):
        """Initialize the tokenizer from config."""
        tokenizer_path = self.config.get("tokenizer_path")
        if not tokenizer_path:
            raise ValueError("tokenizer_path not found in config")
        
        print(f"{C.BLUE}Loading tokenizer from: {tokenizer_path}{C.RESET}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Get special token IDs
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")
        self.cls_token_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_token_id = self.tokenizer.token_to_id("[SEP]")
        
        print(f"Vocab size: {self.tokenizer.get_vocab_size()}")
        print(f"Special tokens - PAD: {self.pad_token_id}, MASK: {self.mask_token_id}, CLS: {self.cls_token_id}, SEP: {self.sep_token_id}")
    
    def test_chunked_dataset(self):
        """Test the ChunkedDataset class."""
        print(f"\n{C.BOLD}{C.GREEN}=== Testing ChunkedDataset ==={C.RESET}")
        
        cache_path = self.config["cache_path"]
        block_size = self.config["block_size"]
        
        # Get chunk paths
        train_paths, val_paths, test_paths = create_and_cache_splits(self.config)
        
        # Test with a small subset of training data
        test_paths = train_paths[:3]  # Use only first 3 chunks for testing
        
        print(f"Testing with {len(test_paths)} chunk files")
        print(f"Block size: {block_size}")
        
        # Create dataset
        start_time = time.time()
        dataset = ChunkedDataset(
            test_paths, 
            block_size=block_size, 
            pad_token_id=self.pad_token_id
        )
        creation_time = time.time() - start_time
        
        print(f"Dataset created in {creation_time:.2f}s")
        print(f"Dataset length: {len(dataset):,} samples")
        print(f"Total tokens: {sum(dataset.block_lengths):,}")
        print(f"Average block length: {np.mean(dataset.block_lengths):.2f}")
        print(f"Block length std: {np.std(dataset.block_lengths):.2f}")
        
        # Test random samples
        print(f"\n{C.CYAN}Sampling random examples:{C.RESET}")
        for i in range(5):
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            actual_length = dataset.block_lengths[idx]
            
            print(f"  Sample {idx}: shape={sample.shape}, dtype={sample.dtype}, "
                  f"actual_length={actual_length}, non_pad_tokens={torch.sum(sample != self.pad_token_id).item()}")
        
        return dataset
    
    def test_token_budget_batch_sampler(self, dataset):
        """Test the TokenBudgetBatchSampler."""
        print(f"\n{C.BOLD}{C.GREEN}=== Testing TokenBudgetBatchSampler ==={C.RESET}")
        
        max_tokens = self.config.get("max_tokens", self.config["block_size"] * self.config["batch_size"])
        lengths = dataset.block_lengths
        
        print(f"Max tokens per batch: {max_tokens}")
        print(f"Sample lengths - min: {min(lengths)}, max: {max(lengths)}, mean: {np.mean(lengths):.2f}")
        
        # Test batch sampler
        sampler = TokenBudgetBatchSampler(
            lengths=lengths,
            max_tokens=max_tokens,
            shuffle=True
        )
        
        print(f"Number of batches: {len(sampler)}")
        
        # Analyze batch statistics
        batch_sizes = []
        token_counts = []
        
        for batch_indices in sampler:
            batch_size = len(batch_indices)
            batch_token_count = sum(lengths[idx] for idx in batch_indices)
            
            batch_sizes.append(batch_size)
            token_counts.append(batch_token_count)
        
        print(f"Batch sizes - min: {min(batch_sizes)}, max: {max(batch_sizes)}, mean: {np.mean(batch_sizes):.2f}")
        print(f"Token counts - min: {min(token_counts)}, max: {max(token_counts)}, mean: {np.mean(token_counts):.2f}")
        print(f"Token budget utilization: {np.mean(token_counts) / max_tokens * 100:.1f}%")
        
        # Check if any batch exceeds the token budget
        over_budget = [tc for tc in token_counts if tc > max_tokens]
        if over_budget:
            print(f"{C.YELLOW}WARNING: {len(over_budget)} batches exceed token budget!{C.RESET}")
        else:
            print(f"{C.GREEN}✓ All batches within token budget{C.RESET}")
        
        return sampler
    
    def test_masking_strategies(self):
        """Test different masking strategies."""
        print(f"\n{C.BOLD}{C.GREEN}=== Testing Masking Strategies ==={C.RESET}")
        
        # Create a sample sequence
        sample_text = "The quick brown fox jumps over the lazy dog."
        encoded = self.tokenizer.encode(sample_text)
        tokens = torch.tensor(encoded.ids, dtype=torch.long)
        
        print(f"Original text: '{sample_text}'")
        print(f"Tokenized: {tokens.tolist()}")
        print(f"Tokens: {[self.tokenizer.id_to_token(tid) for tid in tokens.tolist()]}")
        
        # Test parameters
        mask_p = 0.15
        n_special_tokens = 6
        random_p = 0.1
        keep_p = 0.1
        
        strategies = {
            "subword": SubwordMaskingStrategy,
            "span": SpanMaskingStrategy,
            "whole_word": WholeWordMaskingStrategy,
        }
        
        for strategy_name, strategy_class in strategies.items():
            print(f"\n{C.CYAN}Testing {strategy_name} masking:{C.RESET}")
            
            try:
                strategy = strategy_class(
                    mask_p=mask_p,
                    tokenizer=self.tokenizer,
                    n_special_tokens=n_special_tokens,
                    padding_label_id=-100,
                    random_p=random_p,
                    keep_p=keep_p
                )
                
                masked_tokens, labels = strategy(tokens.clone())
                
                # Count different types of changes
                mask_count = torch.sum(masked_tokens == self.mask_token_id).item()
                label_count = torch.sum(labels != -100).item()
                changed_count = torch.sum(masked_tokens != tokens).item()
                
                print(f"  Masks applied: {mask_count}")
                print(f"  Labels created: {label_count}")
                print(f"  Tokens changed: {changed_count}")
                print(f"  Masking rate: {label_count / len(tokens) * 100:.1f}%")
                
                # Show the changes
                changes = []
                for i, (orig, masked, label) in enumerate(zip(tokens, masked_tokens, labels)):
                    if orig != masked or label != -100:
                        orig_token = self.tokenizer.id_to_token(orig.item())
                        masked_token = self.tokenizer.id_to_token(masked.item())
                        changes.append(f"pos{i}: '{orig_token}' -> '{masked_token}' (label: {label.item()})")
                
                if changes:
                    print(f"  Changes: {changes[:5]}")  # Show first 5 changes
                    if len(changes) > 5:
                        print(f"    ... and {len(changes) - 5} more")
                
            except Exception as e:
                print(f"  {C.RED}ERROR: {e}{C.RESET}")
    
    def test_data_loader_batches(self):
        """Test the actual data loader and analyze batches."""
        print(f"\n{C.BOLD}{C.GREEN}=== Testing Data Loader Batches ==={C.RESET}")
        
        # Create data loaders
        train_loader, val_loader, test_loader, collate_fn, total_tokens = data_loader(
            self.config, self.tokenizer, self.config["cache_path"]
        )
        
        print(f"Total training tokens: {total_tokens:,}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Sample and analyze batches
        batch_stats = self.analyze_batches(train_loader, "Training")
        val_stats = self.analyze_batches(val_loader, "Validation", num_batches=3)
        
        return train_loader, batch_stats
    
    def analyze_batches(self, data_loader, split_name: str, num_batches: int = None):
        """Analyze random batches from a data loader."""
        if num_batches is None:
            num_batches = min(self.num_test_batches, len(data_loader))
        
        print(f"\n{C.CYAN}Analyzing {num_batches} random {split_name.lower()} batches:{C.RESET}")
        
        stats = {
            'batch_sizes': [],
            'sequence_lengths': [],
            'token_counts': [],
            'pad_token_counts': [],
            'mask_token_counts': [],
            'attention_lengths': [],
            'label_counts': [],
            'unique_tokens_per_batch': [],
        }
        
        # Convert data loader to list for random sampling
        data_iter = iter(data_loader)
        all_batches = []
        
        # Collect a reasonable number of batches for sampling
        max_collect = min(100, len(data_loader))
        for i, batch in enumerate(data_iter):
            all_batches.append(batch)
            if i >= max_collect - 1:
                break
        
        # Randomly sample batches
        sampled_batches = random.sample(all_batches, min(num_batches, len(all_batches)))
        
        for i, batch in enumerate(sampled_batches):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch.get('labels', None)
            
            batch_size, seq_len = input_ids.shape
            stats['batch_sizes'].append(batch_size)
            stats['sequence_lengths'].append(seq_len)
            
            # Count tokens
            total_tokens = batch_size * seq_len
            pad_tokens = torch.sum(input_ids == self.pad_token_id).item()
            mask_tokens = torch.sum(input_ids == self.mask_token_id).item()
            real_tokens = torch.sum(attention_mask == 1).item()
            
            stats['token_counts'].append(total_tokens)
            stats['pad_token_counts'].append(pad_tokens)
            stats['mask_token_counts'].append(mask_tokens)
            stats['attention_lengths'].append(real_tokens)
            
            # Count unique tokens
            unique_tokens = len(torch.unique(input_ids))
            stats['unique_tokens_per_batch'].append(unique_tokens)
            
            # Count labels if available
            if labels is not None:
                label_tokens = torch.sum(labels != -100).item()
                stats['label_counts'].append(label_tokens)
            
            print(f"  Batch {i+1}: size={batch_size}, seq_len={seq_len}, "
                  f"real_tokens={real_tokens}, pad_tokens={pad_tokens}, "
                  f"mask_tokens={mask_tokens}, unique_tokens={unique_tokens}")
            
            if labels is not None:
                print(f"           labels={label_tokens}, masking_rate={label_tokens/real_tokens*100:.1f}%")
        
        # Print summary statistics
        print(f"\n{C.YELLOW}{split_name} Batch Statistics Summary:{C.RESET}")
        print(f"  Batch sizes: {np.mean(stats['batch_sizes']):.1f} ± {np.std(stats['batch_sizes']):.1f}")
        print(f"  Sequence lengths: {np.mean(stats['sequence_lengths']):.1f} ± {np.std(stats['sequence_lengths']):.1f}")
        print(f"  Real tokens per batch: {np.mean(stats['attention_lengths']):.1f} ± {np.std(stats['attention_lengths']):.1f}")
        print(f"  Padding ratio: {np.mean(stats['pad_token_counts'])/np.mean(stats['token_counts'])*100:.1f}%")
        print(f"  Unique tokens per batch: {np.mean(stats['unique_tokens_per_batch']):.1f} ± {np.std(stats['unique_tokens_per_batch']):.1f}")
        
        if stats['mask_token_counts']:
            mask_rate = np.mean(stats['mask_token_counts']) / np.mean(stats['attention_lengths']) * 100
            print(f"  Mask token rate: {mask_rate:.1f}%")
        
        if stats['label_counts']:
            label_rate = np.mean(stats['label_counts']) / np.mean(stats['attention_lengths']) * 100
            print(f"  Label rate: {label_rate:.1f}%")
        
        return stats
    
    def test_batch_consistency(self, data_loader, num_tests: int = 5):
        """Test batch consistency and reproducibility."""
        print(f"\n{C.BOLD}{C.GREEN}=== Testing Batch Consistency ==={C.RESET}")
        
        # Test that batches have consistent shapes within each batch
        data_iter = iter(data_loader)
        
        for test_num in range(min(num_tests, len(data_loader))):
            try:
                batch = next(data_iter)
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch.get('labels', None)
                
                # Check shapes
                batch_size, seq_len = input_ids.shape
                assert attention_mask.shape == (batch_size, seq_len), f"Attention mask shape mismatch: {attention_mask.shape} vs {input_ids.shape}"
                
                if labels is not None:
                    assert labels.shape == (batch_size, seq_len), f"Labels shape mismatch: {labels.shape} vs {input_ids.shape}"
                
                # Check data types
                assert input_ids.dtype == torch.long, f"Input IDs wrong dtype: {input_ids.dtype}"
                assert attention_mask.dtype in [torch.long, torch.int, torch.bool], f"Attention mask wrong dtype: {attention_mask.dtype}"
                
                # Check value ranges
                assert torch.all(input_ids >= 0), "Negative token IDs found"
                assert torch.all(input_ids < self.tokenizer.get_vocab_size()), "Token IDs exceed vocab size"
                assert torch.all((attention_mask == 0) | (attention_mask == 1)), "Attention mask values not 0 or 1"
                
                print(f"  Batch {test_num + 1}: ✓ Consistency checks passed")
                
            except StopIteration:
                print(f"  Only {test_num} batches available for testing")
                break
            except Exception as e:
                print(f"  {C.RED}Batch {test_num + 1}: FAILED - {e}{C.RESET}")
        
        print(f"{C.GREEN}✓ Batch consistency tests completed{C.RESET}")
    
    def visualize_statistics(self, batch_stats, save_path: str = None):
        """Create visualizations of batch statistics."""
        print(f"\n{C.BOLD}{C.GREEN}=== Creating Visualizations ==={C.RESET}")
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Data Loader Batch Statistics', fontsize=16)
            
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
            
        except ImportError:
            print(f"{C.YELLOW}Matplotlib not available, skipping visualizations{C.RESET}")
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
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
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
