#!/usr/bin/env python3
"""
Detailed batch analysis script for the data loader.
This provides comprehensive statistics about batches, masking patterns, and token distributions.
"""


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import json
import torch
import random
import numpy as np
from collections import Counter, defaultdict
from tokenizers import Tokenizer
from data_loader import data_loader

def analyze_masking_patterns(batch, tokenizer):
    """Analyze masking patterns in a batch."""
    input_ids = batch['input_ids']
    labels = batch.get('labels')
    
    if labels is None:
        return {}
    
    mask_token_id = tokenizer.token_to_id("[MASK]")
    pad_token_id = tokenizer.token_to_id("[PAD]")
    
    # Count different masking operations
    masked_positions = labels != -100
    
    # Count original vs masked tokens
    original_tokens = labels[masked_positions]  # Original tokens at masked positions
    current_tokens = input_ids[masked_positions]  # Current tokens at masked positions
    
    # Classify masking operations
    mask_operations = {
        'replaced_with_mask': torch.sum(current_tokens == mask_token_id).item(),
        'replaced_with_random': torch.sum((current_tokens != mask_token_id) & (current_tokens != original_tokens)).item(),
        'kept_original': torch.sum(current_tokens == original_tokens).item(),
        'total_masked': torch.sum(masked_positions).item()
    }
    
    return mask_operations

def analyze_token_frequencies(batch, tokenizer, top_k=20):
    """Analyze token frequency distribution in a batch."""
    input_ids = batch['input_ids']
    
    # Flatten and count tokens
    all_tokens = input_ids.flatten()
    token_counts = Counter(all_tokens.tolist())
    
    # Get most common tokens
    most_common = token_counts.most_common(top_k)
    
    # Convert to readable format
    readable_common = []
    for token_id, count in most_common:
        token_text = tokenizer.id_to_token(token_id)
        readable_common.append((token_text, token_id, count))
    
    return {
        'total_tokens': len(all_tokens),
        'unique_tokens': len(token_counts),
        'most_common': readable_common,
        'vocabulary_coverage': len(token_counts) / tokenizer.get_vocab_size() * 100
    }

def analyze_sequence_patterns(batch, tokenizer):
    """Analyze patterns in sequences."""
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    
    batch_size, seq_len = input_ids.shape
    
    # Analyze sequence lengths
    real_lengths = torch.sum(attention_mask, dim=1)
    
    # Find special token patterns
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    pad_token_id = tokenizer.token_to_id("[PAD]")
    
    cls_positions = []
    sep_positions = []
    
    for i in range(batch_size):
        seq = input_ids[i]
        cls_pos = torch.where(seq == cls_token_id)[0]
        sep_pos = torch.where(seq == sep_token_id)[0]
        
        cls_positions.extend(cls_pos.tolist())
        sep_positions.extend(sep_pos.tolist())
    
    return {
        'sequence_lengths': {
            'min': torch.min(real_lengths).item(),
            'max': torch.max(real_lengths).item(),
            'mean': torch.mean(real_lengths.float()).item(),
            'std': torch.std(real_lengths.float()).item()
        },
        'cls_positions': cls_positions,
        'sep_positions': sep_positions,
        'avg_cls_per_seq': len(cls_positions) / batch_size,
        'avg_sep_per_seq': len(sep_positions) / batch_size
    }

def comprehensive_batch_analysis(config_path, num_batches=10, seed=42):
    """Run comprehensive analysis on multiple batches."""
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print("🔬 Starting Comprehensive Batch Analysis")
    print("=" * 60)
    
    # Load config and create data loader
    with open(config_path, "r") as f:
        config = json.load(f)
    
    tokenizer = Tokenizer.from_file(config["tokenizer_path"])
    train_loader, val_loader, test_loader, collate_fn, total_tokens = data_loader(
        config, tokenizer, config["cache_path"]
    )
    
    print(f"📊 Dataset Overview:")
    print(f"  - Total training tokens: {total_tokens:,}")
    print(f"  - Vocabulary size: {tokenizer.get_vocab_size():,}")
    print(f"  - Train batches: {len(train_loader):,}")
    print(f"  - Block size: {config['block_size']}")
    print(f"  - Batch size: {config['batch_size']}")
    print()
    
    # Collect sample batches
    print(f"🎯 Collecting {num_batches} random batches for analysis...")
    data_iter = iter(train_loader)
    batches = []
    for i, batch in enumerate(data_iter):
        batches.append(batch)
        if i >= min(100, len(train_loader) - 1):  # Collect up to 100 batches
            break
    
    sample_batches = random.sample(batches, min(num_batches, len(batches)))
    
    # Analysis containers
    all_masking_stats = []
    all_token_stats = []
    all_sequence_stats = []
    
    print(f"📈 Analyzing {len(sample_batches)} batches...")
    print()
    
    for i, batch in enumerate(sample_batches):
        print(f"Batch {i+1}/{len(sample_batches)}:")
        
        # Basic stats
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch.get('labels')
        
        batch_size, seq_len = input_ids.shape
        real_tokens = torch.sum(attention_mask == 1).item()
        
        print(f"  📋 Basic Stats:")
        print(f"    - Shape: {input_ids.shape}")
        print(f"    - Real tokens: {real_tokens:,}")
        print(f"    - Token density: {real_tokens/(batch_size*seq_len)*100:.1f}%")
        
        # Masking analysis
        if labels is not None:
            masking_stats = analyze_masking_patterns(batch, tokenizer)
            all_masking_stats.append(masking_stats)
            
            print(f"  🎭 Masking Stats:")
            print(f"    - Total masked: {masking_stats['total_masked']:,}")
            print(f"    - Replaced with [MASK]: {masking_stats['replaced_with_mask']:,}")
            print(f"    - Replaced with random: {masking_stats['replaced_with_random']:,}")
            print(f"    - Kept original: {masking_stats['kept_original']:,}")
            print(f"    - Masking rate: {masking_stats['total_masked']/real_tokens*100:.2f}%")
        
        # Token frequency analysis
        token_stats = analyze_token_frequencies(batch, tokenizer, top_k=10)
        all_token_stats.append(token_stats)
        
        print(f"  🔤 Token Stats:")
        print(f"    - Unique tokens: {token_stats['unique_tokens']:,}")
        print(f"    - Vocab coverage: {token_stats['vocabulary_coverage']:.2f}%")
        print(f"    - Top 5 tokens: {[f'{tok}({cnt})' for tok, _, cnt in token_stats['most_common'][:5]]}")
        
        # Sequence pattern analysis
        seq_stats = analyze_sequence_patterns(batch, tokenizer)
        all_sequence_stats.append(seq_stats)
        
        print(f"  📝 Sequence Stats:")
        print(f"    - Length range: {seq_stats['sequence_lengths']['min']}-{seq_stats['sequence_lengths']['max']}")
        print(f"    - Mean length: {seq_stats['sequence_lengths']['mean']:.1f} ± {seq_stats['sequence_lengths']['std']:.1f}")
        print(f"    - Avg CLS per seq: {seq_stats['avg_cls_per_seq']:.2f}")
        print(f"    - Avg SEP per seq: {seq_stats['avg_sep_per_seq']:.2f}")
        
        print()
    
    # Aggregate statistics
    print("=" * 60)
    print("📊 AGGREGATE STATISTICS")
    print("=" * 60)
    
    if all_masking_stats:
        print("🎭 Masking Summary:")
        total_masked = sum(s['total_masked'] for s in all_masking_stats)
        total_mask_tokens = sum(s['replaced_with_mask'] for s in all_masking_stats)
        total_random = sum(s['replaced_with_random'] for s in all_masking_stats)
        total_kept = sum(s['kept_original'] for s in all_masking_stats)
        
        print(f"  - Total masked tokens across all batches: {total_masked:,}")
        print(f"  - [MASK] token usage: {total_mask_tokens/total_masked*100:.1f}%")
        print(f"  - Random token usage: {total_random/total_masked*100:.1f}%")
        print(f"  - Original kept: {total_kept/total_masked*100:.1f}%")
        print()
    
    print("🔤 Token Distribution Summary:")
    avg_unique = np.mean([s['unique_tokens'] for s in all_token_stats])
    avg_coverage = np.mean([s['vocabulary_coverage'] for s in all_token_stats])
    print(f"  - Average unique tokens per batch: {avg_unique:.0f}")
    print(f"  - Average vocabulary coverage: {avg_coverage:.2f}%")
    
    # Most frequent tokens across all batches
    all_token_counts = Counter()
    for stats in all_token_stats:
        for token, token_id, count in stats['most_common']:
            all_token_counts[token] += count
    
    print(f"  - Most frequent tokens overall:")
    for token, count in all_token_counts.most_common(10):
        print(f"    {token}: {count:,}")
    print()
    
    print("📝 Sequence Summary:")
    all_lengths = []
    for stats in all_sequence_stats:
        # We don't have individual lengths, but we have mean/std
        all_lengths.append(stats['sequence_lengths']['mean'])
    
    print(f"  - Average sequence length across batches: {np.mean(all_lengths):.1f}")
    print(f"  - Sequence length variation: {np.std(all_lengths):.1f}")
    
    avg_cls = np.mean([s['avg_cls_per_seq'] for s in all_sequence_stats])
    avg_sep = np.mean([s['avg_sep_per_seq'] for s in all_sequence_stats])
    print(f"  - Average CLS tokens per sequence: {avg_cls:.2f}")
    print(f"  - Average SEP tokens per sequence: {avg_sep:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Comprehensive analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive batch analysis")
    parser.add_argument("--config", type=str, default="model_babylm_ltg_bert.json", help="Config file")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches to analyze")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    comprehensive_batch_analysis(args.config, args.num_batches, args.seed)
