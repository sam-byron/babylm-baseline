#!/usr/bin/env python3

import torch
import json
from pathlib import Path
import glob
from data_loader import ChunkedDataset
from tokenizers import Tokenizer
from dynamic_collator import create_dynamic_collator
import statistics

def final_validation_test():
    """Final comprehensive validation of concatenation and masking logic."""
    print("🎯 FINAL VALIDATION TEST")
    print("=" * 70)
    
    # Load config and tokenizer
    with open("model_babylm_ltg_bert.json", "r") as f:
        config = json.load(f)
    
    tokenizer = Tokenizer.from_file("data/pretrain/wordpiece_vocab.json")
    
    # Test with moderate dataset size
    all_chunk_paths = sorted(glob.glob("model_babylm_bert_ltg/chunk*.pt"))
    chunk_paths = all_chunk_paths[:25]  # Test with 25 chunks
    print(f"📊 Testing with {len(chunk_paths)} chunks (out of {len(all_chunk_paths)} total)")
    
    # Create ChunkedDataset
    block_size = config["block_size"]
    pad_id = tokenizer.token_to_id("[PAD]")
    
    print(f"\n📈 Creating ChunkedDataset...")
    dataset = ChunkedDataset(chunk_paths, block_size=block_size, pad_token_id=pad_id)
    
    # === CONCATENATION VALIDATION ===
    print(f"\n🔗 CONCATENATION VALIDATION:")
    print(f"  Total blocks: {len(dataset):,}")
    print(f"  Block lengths: {len(dataset.block_lengths):,}")
    
    # Analyze block length distribution
    full_blocks = sum(1 for length in dataset.block_lengths if length == block_size)
    partial_blocks = len(dataset.block_lengths) - full_blocks
    avg_length = sum(dataset.block_lengths) / len(dataset.block_lengths) if dataset.block_lengths else 0
    
    print(f"  Full blocks (512 tokens): {full_blocks:,} ({full_blocks/len(dataset)*100:.1f}%)")
    print(f"  Partial blocks: {partial_blocks:,} ({partial_blocks/len(dataset)*100:.1f}%)")
    print(f"  Average block length: {avg_length:.1f} tokens")
    
    # Efficiency analysis
    total_real_tokens = sum(dataset.block_lengths)
    total_possible_tokens = len(dataset) * block_size
    efficiency = total_real_tokens / total_possible_tokens * 100 if total_possible_tokens > 0 else 0
    
    print(f"  Token efficiency: {efficiency:.1f}%")
    
    # Validate concatenation success
    if efficiency > 95:
        print(f"  ✅ EXCELLENT: Concatenation working perfectly (>95% efficiency)")
    else:
        print(f"  ❌ PROBLEM: Low efficiency suggests concatenation issues")
        return False
    
    # === MASKING VALIDATION ===
    print(f"\n🎭 MASKING VALIDATION:")
    
    # Get special token IDs
    special_token_ids = {0, 1, 2, 3, 4}  # UNK, CLS, SEP, PAD, MASK
    special_names = {0: '[UNK]', 1: '[CLS]', 2: '[SEP]', 3: '[PAD]', 4: '[MASK]'}
    mask_token_id = tokenizer.token_to_id("[MASK]")
    
    # Create collator
    collate_fn = create_dynamic_collator(config, tokenizer)
    print(f"  Configured mlm_probability: {collate_fn.mlm_probability}")
    
    # Test multiple samples to get statistics
    num_test_samples = 100
    test_samples = []
    special_token_counts = []
    maskable_token_counts = []
    
    print(f"  Analyzing {num_test_samples} samples for special token distribution...")
    for i in range(min(num_test_samples, len(dataset))):
        sample = dataset[i]
        test_samples.append(sample)
        
        # Count special tokens
        special_count = sum((sample == token_id).sum().item() for token_id in special_token_ids)
        special_token_counts.append(special_count)
        maskable_token_counts.append(len(sample) - special_count)
    
    # Special token statistics
    avg_special = statistics.mean(special_token_counts)
    avg_maskable = statistics.mean(maskable_token_counts)
    avg_special_percent = avg_special / block_size * 100
    
    print(f"  Average special tokens per block: {avg_special:.1f} ({avg_special_percent:.1f}%)")
    print(f"  Average maskable tokens per block: {avg_maskable:.1f}")
    
    # Test masking on multiple batches
    num_test_batches = 20
    batch_size = 8
    overall_masking_rates = []  # Rate based on total tokens (for reference)
    maskable_masking_rates = []  # Rate based on maskable tokens only (this is what we want to validate)
    masked_counts = []
    
    print(f"  Testing masking across {num_test_batches} batches...")
    
    for batch_idx in range(num_test_batches):
        start_idx = batch_idx * batch_size
        if start_idx + batch_size > len(test_samples):
            break
            
        batch_samples = test_samples[start_idx:start_idx + batch_size]
        batch = collate_fn(batch_samples)
        
        # Analyze each sample in the batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        for sample_idx in range(input_ids.shape[0]):
            sample_input_ids = input_ids[sample_idx]
            sample_attention_mask = attention_mask[sample_idx]
            original_sample = batch_samples[sample_idx]
            
            # Count ALL masked positions (not just [MASK] tokens)
            # The labels tensor contains the original token IDs for all masked positions
            # All non-masked positions have labels = -100
            masked_positions = (batch['labels'][sample_idx] != -100).sum().item()
            real_tokens = sample_attention_mask.sum().item()
            
            # Count maskable tokens in the original sample (before masking)
            maskable_tokens = 0
            for token_id in original_sample:
                if token_id.item() not in special_token_ids:
                    maskable_tokens += 1
            
            # Also count just [MASK] tokens for reference
            mask_only_count = (sample_input_ids == mask_token_id).sum().item()
            
            if real_tokens > 0 and maskable_tokens > 0:
                # Overall masking rate (for reference)
                overall_rate = masked_positions / real_tokens * 100
                overall_masking_rates.append(overall_rate)
                
                # Maskable-only masking rate (this is what we want to validate)
                maskable_rate = masked_positions / maskable_tokens * 100
                maskable_masking_rates.append(maskable_rate)
                
                masked_counts.append(masked_positions)
                
                # Debug info for first few samples
                if len(masked_counts) <= 3:
                    print(f"    Sample {len(masked_counts)}: {masked_positions} total masked ({mask_only_count} [MASK], {masked_positions-mask_only_count} random/keep)")
    
    # Masking statistics
    if maskable_masking_rates:
        # Statistics for maskable-token-only rates (primary validation metric)
        avg_maskable_rate = statistics.mean(maskable_masking_rates)
        min_maskable_rate = min(maskable_masking_rates)
        max_maskable_rate = max(maskable_masking_rates)
        std_maskable_rate = statistics.stdev(maskable_masking_rates) if len(maskable_masking_rates) > 1 else 0
        
        # Statistics for overall rates (for reference)
        avg_overall_rate = statistics.mean(overall_masking_rates)
        
        avg_masked_count = statistics.mean(masked_counts)
        expected_maskable = avg_maskable * 0.15
        
        print(f"\n📈 MASKING RESULTS:")
        print(f"  Samples tested: {len(maskable_masking_rates)}")
        print(f"\n  🎯 MASKABLE-TOKEN-ONLY RATES (Primary Metric):")
        print(f"    Average: {avg_maskable_rate:.2f}% ± {std_maskable_rate:.2f}%")
        print(f"    Range: {min_maskable_rate:.2f}% - {max_maskable_rate:.2f}%")
        print(f"    Target: 15.00% ± 1.00%")
        
        print(f"\n  📊 OVERALL RATES (Reference Only):")
        print(f"    Average: {avg_overall_rate:.2f}% (includes special tokens in denominator)")
        
        print(f"\n  📋 TOKEN COUNTS:")
        print(f"    Average tokens masked: {avg_masked_count:.1f}")
        print(f"    Expected tokens masked: {expected_maskable:.1f}")
        
        # Primary validation: maskable-token-only rate should be 15% ± 1%
        target_min = 14.0
        target_max = 16.0
        within_target = target_min <= avg_maskable_rate <= target_max
        
        if within_target:
            print(f"  ✅ PERFECT: Maskable-token masking rate within target range (14-16%)")
        else:
            print(f"  ❌ ISSUE: Maskable-token masking rate outside target range (14-16%)")
            
        # Secondary check: consistency
        if std_maskable_rate < 2.0:
            print(f"  ✅ CONSISTENT: Low standard deviation ({std_maskable_rate:.2f}%)")
        else:
            print(f"  ⚠️  WARNING: High variation in masking rates")
            
        masking_success = within_target
        
    else:
        print(f"  ❌ ERROR: No masking data collected")
        return False
    
    # === SAMPLE INSPECTION ===
    print(f"\n🔍 SAMPLE INSPECTION:")
    sample = dataset[0]
    print(f"  Sample 0 - Total tokens: {len(sample)}")
    
    # Analyze token types in this sample
    sample_special_counts = {}
    for token_id, name in special_names.items():
        count = (sample == token_id).sum().item()
        if count > 0:
            sample_special_counts[name] = count
    
    print(f"  Special tokens in sample 0:")
    total_special_sample = 0
    for name, count in sample_special_counts.items():
        print(f"    {name}: {count}")
        total_special_sample += count
    
    maskable_sample = len(sample) - total_special_sample
    print(f"  Maskable tokens in sample 0: {maskable_sample}")
    
    # Test masking this specific sample multiple times
    print(f"  Testing masking consistency on sample 0 (5 runs):")
    sample_overall_rates = []
    sample_maskable_rates = []
    
    for run in range(5):
        batch = collate_fn([sample])
        input_ids_batch = batch['input_ids'][0]
        labels_batch = batch['labels'][0]
        
        # Count all masked positions using labels
        masked_count = (labels_batch != -100).sum().item()
        mask_only_count = (input_ids_batch == mask_token_id).sum().item()
        
        overall_rate = masked_count / len(sample) * 100
        maskable_rate = masked_count / maskable_sample * 100
        
        sample_overall_rates.append(overall_rate)
        sample_maskable_rates.append(maskable_rate)
        
        print(f"    Run {run+1}: {masked_count} tokens masked ({mask_only_count} [MASK], {masked_count-mask_only_count} random/keep) = {overall_rate:.2f}% overall, {maskable_rate:.2f}% of maskable")
    
    overall_consistency = max(sample_overall_rates) - min(sample_overall_rates)
    maskable_consistency = max(sample_maskable_rates) - min(sample_maskable_rates)
    print(f"  Consistency: {overall_consistency:.2f}% overall variation, {maskable_consistency:.2f}% maskable variation")
    
    # === FINAL VERDICT ===
    print(f"\n🎉 FINAL VALIDATION RESULTS:")
    
    concat_success = efficiency > 95
    # masking_success already defined above in the masking validation section
    consistency_success = maskable_consistency < 2.0  # Less than 2% variation on maskable tokens
    
    print(f"  ✅ Concatenation: {'PASS' if concat_success else 'FAIL'} ({efficiency:.1f}% efficiency)")
    if 'masking_success' in locals():
        print(f"  ✅ Masking: {'PASS' if masking_success else 'FAIL'} ({avg_maskable_rate:.2f}% maskable-token rate)")
    else:
        print(f"  ❌ Masking: FAIL (no data collected)")
        masking_success = False
    print(f"  ✅ Consistency: {'PASS' if consistency_success else 'FAIL'} ({maskable_consistency:.2f}% variation)")
    
    overall_success = concat_success and masking_success and consistency_success
    print(f"\n🏆 OVERALL: {'SUCCESS' if overall_success else 'NEEDS ATTENTION'}")
    
    if overall_success:
        print(f"🚀 System is ready for training!")
        print(f"   Target achieved: {avg_maskable_rate:.2f}% masking rate on content tokens")
    else:
        print(f"⚠️  Some issues detected - review above results")
    
    return overall_success

if __name__ == "__main__":
    final_validation_test()
