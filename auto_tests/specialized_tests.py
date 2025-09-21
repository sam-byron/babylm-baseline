#!/usr/bin/env python3
"""
Specialized test for span masking patterns and token budget efficiency.
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
from collections import defaultdict
from tokenizers import Tokenizer
from data_loader import data_loader

def analyze_span_patterns(batch, tokenizer):
    """Analyze span masking patterns in detail."""
    input_ids = batch['input_ids']
    labels = batch.get('labels')
    
    if labels is None:
        return {}
    
    batch_size, seq_len = input_ids.shape
    mask_token_id = tokenizer.token_to_id("[MASK]")
    
    span_stats = {
        'spans_per_sequence': [],
        'span_lengths': [],
        'total_spans': 0,
        'avg_span_length': 0,
        'max_span_length': 0,
        'span_positions': []
    }
    
    for seq_idx in range(batch_size):
        seq_labels = labels[seq_idx]
        seq_input = input_ids[seq_idx]
        
        # Find masked positions
        masked_positions = (seq_labels != -100).nonzero(as_tuple=False).flatten()
        
        if len(masked_positions) == 0:
            span_stats['spans_per_sequence'].append(0)
            continue
        
        # Group consecutive masked positions into spans
        spans = []
        current_span = [masked_positions[0].item()]
        
        for i in range(1, len(masked_positions)):
            pos = masked_positions[i].item()
            if pos == current_span[-1] + 1:
                current_span.append(pos)
            else:
                spans.append(current_span)
                current_span = [pos]
        
        spans.append(current_span)
        
        span_stats['spans_per_sequence'].append(len(spans))
        span_stats['total_spans'] += len(spans)
        
        for span in spans:
            span_length = len(span)
            span_stats['span_lengths'].append(span_length)
            span_stats['span_positions'].append((seq_idx, span[0], span[-1]))
            
            if span_length > span_stats['max_span_length']:
                span_stats['max_span_length'] = span_length
    
    if span_stats['span_lengths']:
        span_stats['avg_span_length'] = np.mean(span_stats['span_lengths'])
    
    return span_stats

def analyze_token_budget_efficiency(batches_sample, config):
    """Analyze how efficiently the token budget is being used."""
    max_tokens = config.get("max_tokens", config["block_size"] * config["batch_size"])
    block_size = config["block_size"]
    
    efficiency_stats = {
        'token_utilization': [],
        'batch_sizes': [],
        'wasted_tokens': [],
        'padding_efficiency': []
    }
    
    for batch in batches_sample:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        batch_size, seq_len = input_ids.shape
        total_slots = batch_size * seq_len
        real_tokens = torch.sum(attention_mask == 1).item()
        
        efficiency_stats['token_utilization'].append(real_tokens / max_tokens)
        efficiency_stats['batch_sizes'].append(batch_size)
        efficiency_stats['wasted_tokens'].append(total_slots - real_tokens)
        efficiency_stats['padding_efficiency'].append(real_tokens / total_slots)
    
    return {
        'avg_token_utilization': np.mean(efficiency_stats['token_utilization']) * 100,
        'avg_batch_size': np.mean(efficiency_stats['batch_sizes']),
        'avg_wasted_tokens': np.mean(efficiency_stats['wasted_tokens']),
        'avg_padding_efficiency': np.mean(efficiency_stats['padding_efficiency']) * 100,
        'utilization_std': np.std(efficiency_stats['token_utilization']) * 100
    }

def test_masking_consistency(batch, tokenizer, num_checks=5):
    """Test that masking follows the expected 80/10/10 rule."""
    input_ids = batch['input_ids']
    labels = batch.get('labels')
    
    if labels is None:
        return {}
    
    mask_token_id = tokenizer.token_to_id("[MASK]")
    
    # Get all masked positions
    masked_positions = labels != -100
    original_tokens = labels[masked_positions]
    current_tokens = input_ids[masked_positions]
    
    # Count the different masking strategies
    mask_tokens = torch.sum(current_tokens == mask_token_id).item()
    same_tokens = torch.sum(current_tokens == original_tokens).item()
    random_tokens = len(current_tokens) - mask_tokens - same_tokens
    
    total_masked = len(current_tokens)
    
    if total_masked == 0:
        return {'error': 'No masked tokens found'}
    
    return {
        'total_masked': total_masked,
        'mask_percentage': mask_tokens / total_masked * 100,
        'random_percentage': random_tokens / total_masked * 100,
        'keep_percentage': same_tokens / total_masked * 100,
        'expected_mask': 80.0,
        'expected_random': 10.0,
        'expected_keep': 10.0
    }

def run_specialized_tests(config_path="model_babylm_ltg_bert.json", num_batches=8):
    """Run specialized tests focusing on span masking and efficiency."""
    
    print("🔬 SPECIALIZED DATA LOADER TESTS")
    print("=" * 50)
    
    # Load config and setup
    with open(config_path, "r") as f:
        config = json.load(f)
    
    tokenizer = Tokenizer.from_file(config["tokenizer_path"])
    train_loader, _, _, _, total_tokens = data_loader(config, tokenizer, config["cache_path"])
    
    print(f"🎯 Testing with {num_batches} batches")
    print(f"📊 Dataset: {total_tokens:,} total tokens, {len(train_loader):,} batches")
    print()
    
    # Collect sample batches
    data_iter = iter(train_loader)
    sample_batches = []
    for i, batch in enumerate(data_iter):
        sample_batches.append(batch)
        if i >= num_batches - 1:
            break
    
    # Test 1: Span Masking Analysis
    print("🎭 TEST 1: SPAN MASKING PATTERNS")
    print("-" * 30)
    
    all_span_stats = []
    for i, batch in enumerate(sample_batches):
        span_stats = analyze_span_patterns(batch, tokenizer)
        all_span_stats.append(span_stats)
        
        print(f"Batch {i+1}:")
        print(f"  - Total spans: {span_stats['total_spans']}")
        print(f"  - Avg spans per sequence: {np.mean(span_stats['spans_per_sequence']):.2f}")
        if span_stats['span_lengths']:
            print(f"  - Avg span length: {span_stats['avg_span_length']:.2f}")
            print(f"  - Max span length: {span_stats['max_span_length']}")
            print(f"  - Span length distribution: {dict(zip(*np.unique(span_stats['span_lengths'], return_counts=True)))}")
    
    # Aggregate span statistics
    all_spans = []
    all_spans_per_seq = []
    for stats in all_span_stats:
        all_spans.extend(stats['span_lengths'])
        all_spans_per_seq.extend(stats['spans_per_sequence'])
    
    if all_spans:
        print(f"\n📈 OVERALL SPAN STATISTICS:")
        print(f"  - Total spans analyzed: {len(all_spans)}")
        print(f"  - Average span length: {np.mean(all_spans):.2f} ± {np.std(all_spans):.2f}")
        print(f"  - Span length range: {min(all_spans)} - {max(all_spans)}")
        print(f"  - Average spans per sequence: {np.mean(all_spans_per_seq):.2f}")
        
        # Show span length distribution
        unique_lengths, counts = np.unique(all_spans, return_counts=True)
        print(f"  - Span length distribution:")
        for length, count in zip(unique_lengths, counts):
            percentage = count / len(all_spans) * 100
            print(f"    Length {length}: {count} spans ({percentage:.1f}%)")
    print()
    
    # Test 2: Token Budget Efficiency
    print("💰 TEST 2: TOKEN BUDGET EFFICIENCY")
    print("-" * 30)
    
    efficiency_stats = analyze_token_budget_efficiency(sample_batches, config)
    
    print(f"📊 Efficiency Metrics:")
    print(f"  - Average token utilization: {efficiency_stats['avg_token_utilization']:.1f}%")
    print(f"  - Utilization consistency (std): {efficiency_stats['utilization_std']:.1f}%")
    print(f"  - Average batch size: {efficiency_stats['avg_batch_size']:.1f}")
    print(f"  - Average wasted tokens per batch: {efficiency_stats['avg_wasted_tokens']:.0f}")
    print(f"  - Padding efficiency: {efficiency_stats['avg_padding_efficiency']:.1f}%")
    print()
    
    # Test 3: Masking Strategy Compliance
    print("⚖️  TEST 3: MASKING STRATEGY COMPLIANCE")
    print("-" * 30)
    
    all_masking_stats = []
    for i, batch in enumerate(sample_batches):
        masking_stats = test_masking_consistency(batch, tokenizer)
        if 'error' not in masking_stats:
            all_masking_stats.append(masking_stats)
            
            print(f"Batch {i+1}:")
            print(f"  - Masked tokens: {masking_stats['total_masked']}")
            print(f"  - [MASK] usage: {masking_stats['mask_percentage']:.1f}% (expected: {masking_stats['expected_mask']:.1f}%)")
            print(f"  - Random usage: {masking_stats['random_percentage']:.1f}% (expected: {masking_stats['expected_random']:.1f}%)")
            print(f"  - Keep usage: {masking_stats['keep_percentage']:.1f}% (expected: {masking_stats['expected_keep']:.1f}%)")
    
    if all_masking_stats:
        print(f"\n📊 OVERALL MASKING COMPLIANCE:")
        avg_mask = np.mean([s['mask_percentage'] for s in all_masking_stats])
        avg_random = np.mean([s['random_percentage'] for s in all_masking_stats])
        avg_keep = np.mean([s['keep_percentage'] for s in all_masking_stats])
        
        print(f"  - Average [MASK] usage: {avg_mask:.1f}% (target: 80%)")
        print(f"  - Average random usage: {avg_random:.1f}% (target: 10%)")
        print(f"  - Average keep usage: {avg_keep:.1f}% (target: 10%)")
        
        # Check compliance
        mask_compliance = abs(avg_mask - 80) <= 5  # Within 5% tolerance
        random_compliance = abs(avg_random - 10) <= 5
        keep_compliance = abs(avg_keep - 10) <= 5
        
        print(f"  - Compliance check:")
        print(f"    [MASK]: {'✅' if mask_compliance else '❌'}")
        print(f"    Random: {'✅' if random_compliance else '❌'}")
        print(f"    Keep: {'✅' if keep_compliance else '❌'}")
    
    print("\n" + "=" * 50)
    print("✅ Specialized tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run specialized data loader tests")
    parser.add_argument("--config", type=str, default="model_babylm_ltg_bert.json", help="Config file")
    parser.add_argument("--num_batches", type=int, default=8, help="Number of batches to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    run_specialized_tests(args.config, args.num_batches)
