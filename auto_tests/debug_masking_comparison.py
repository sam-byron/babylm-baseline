#!/usr/bin/env python3

import torch
import json
import time
import glob
from data_loader import ChunkedDataset
from tokenizers import Tokenizer
from dynamic_collator import create_dynamic_collator
from mlm_dataset import SpanMaskingStrategy

def debug_masking_performance():
    """Compare dynamic vs static masking performance and correctness."""
    print("🔍 DEBUGGING DYNAMIC VS STATIC MASKING")
    print("=" * 60)
    
    # Load configuration
    with open("model_babylm_ltg_bert.json", "r") as f:
        config = json.load(f)
    
    tokenizer = Tokenizer.from_file("data/pretrain/wordpiece_vocab.json")
    chunk_paths = sorted(glob.glob("model_babylm_bert_ltg/chunk*.pt"))[:5]  # Use subset for testing
    dataset = ChunkedDataset(chunk_paths, block_size=512, pad_token_id=tokenizer.token_to_id("[PAD]"))
    
    print(f"📊 Testing with {len(chunk_paths)} chunks, {len(dataset)} samples")
    
    # Special token IDs
    special_token_ids = {0, 1, 2, 3, 4}
    mask_token_id = tokenizer.token_to_id("[MASK]")
    
    # Test samples
    test_samples = [dataset[i] for i in range(min(20, len(dataset)))]
    
    # ============= TEST DYNAMIC MASKING =============
    print(f"\\n🎭 TESTING DYNAMIC MASKING")
    print("-" * 40)
    
    dynamic_collator = create_dynamic_collator(config, tokenizer)
    
    # Performance test
    start_time = time.time()
    dynamic_results = []
    
    for i, sample in enumerate(test_samples):
        # Test masking consistency and performance
        batch = dynamic_collator([sample])
        labels = batch['labels'][0]
        input_ids = batch['input_ids'][0]
        
        # Count masking statistics
        total_tokens = len(sample)
        maskable_tokens = sum(1 for token in sample.tolist() if token not in special_token_ids)
        masked_positions = (labels != -100).sum().item()
        mask_tokens = (input_ids == mask_token_id).sum().item()
        
        dynamic_results.append({
            'sample_id': i,
            'total_tokens': total_tokens,
            'maskable_tokens': maskable_tokens,
            'masked_positions': masked_positions,
            'mask_tokens': mask_tokens,
            'masking_rate': masked_positions / maskable_tokens * 100 if maskable_tokens > 0 else 0
        })
    
    dynamic_time = time.time() - start_time
    
    # ============= TEST STATIC MASKING =============
    print(f"\\n📌 TESTING STATIC MASKING")
    print("-" * 40)
    
    # Create static masking strategy
    mask_p = config.get("mask_p", 0.15)
    random_p = config.get("random_p", 0.1) 
    keep_p = config.get("keep_p", 0.1)
    n_special_tokens = 6
    
    static_strategy = SpanMaskingStrategy(
        mask_p=mask_p,
        tokenizer=tokenizer,
        n_special_tokens=n_special_tokens,
        random_p=random_p,
        keep_p=keep_p
    )
    
    # Performance test
    start_time = time.time()
    static_results = []
    
    for i, sample in enumerate(test_samples):
        # Apply static masking - SpanMaskingStrategy returns (modified_tokens, labels)
        input_ids = sample.clone()
        input_ids, labels = static_strategy(input_ids)  # Returns modified tokens and labels
        
        # Count masking statistics
        total_tokens = len(sample)
        maskable_tokens = sum(1 for token in sample.tolist() if token not in special_token_ids)
        masked_positions = (labels != -100).sum().item()
        mask_tokens = (input_ids == mask_token_id).sum().item()
        
        static_results.append({
            'sample_id': i,
            'total_tokens': total_tokens,
            'maskable_tokens': maskable_tokens,
            'masked_positions': masked_positions,
            'mask_tokens': mask_tokens,
            'masking_rate': masked_positions / maskable_tokens * 100 if maskable_tokens > 0 else 0
        })
    
    static_time = time.time() - start_time
    
    # ============= COMPARISON ANALYSIS =============
    print(f"\\n📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Timing comparison
    print(f"⏱️  TIMING:")
    print(f"  Dynamic masking: {dynamic_time:.4f}s ({dynamic_time/len(test_samples)*1000:.2f}ms/sample)")
    print(f"  Static masking:  {static_time:.4f}s ({static_time/len(test_samples)*1000:.2f}ms/sample)")
    print(f"  Speed ratio: {dynamic_time/static_time:.2f}x {'SLOWER' if dynamic_time > static_time else 'FASTER'}")
    
    # Masking rate comparison
    dynamic_rates = [r['masking_rate'] for r in dynamic_results]
    static_rates = [r['masking_rate'] for r in static_results]
    
    print(f"\\n🎯 MASKING RATES:")
    print(f"  Dynamic - Avg: {sum(dynamic_rates)/len(dynamic_rates):.2f}%, Range: {min(dynamic_rates):.1f}%-{max(dynamic_rates):.1f}%")
    print(f"  Static  - Avg: {sum(static_rates)/len(static_rates):.2f}%, Range: {min(static_rates):.1f}%-{max(static_rates):.1f}%")
    
    # Detect potential issues
    print(f"\\n🚨 ISSUE DETECTION:")
    
    # Check for extreme masking rate variations
    dynamic_std = torch.tensor(dynamic_rates).std().item()
    static_std = torch.tensor(static_rates).std().item()
    
    print(f"  Masking rate stability:")
    print(f"    Dynamic std dev: {dynamic_std:.2f}%")
    print(f"    Static std dev:  {static_std:.2f}%")
    
    if dynamic_std > static_std * 1.5:
        print(f"    ⚠️  Dynamic masking has higher variability!")
    
    # Check for performance issues
    if dynamic_time > static_time * 2:
        print(f"    ⚠️  Dynamic masking is significantly slower!")
    
    # Memory usage test
    print(f"\\n💾 MEMORY ANALYSIS:")
    
    # Test batch processing
    batch_samples = test_samples[:8]  # 8 samples
    
    # Dynamic batch
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    dynamic_batch = dynamic_collator(batch_samples)
    end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    dynamic_mem = end_mem - start_mem
    
    # Static batch (simulate)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    static_batch_inputs = []
    static_batch_labels = []
    for sample in batch_samples:
        input_ids = sample.clone()
        input_ids, labels = static_strategy(input_ids)  # Returns modified tokens and labels
        static_batch_inputs.append(input_ids)
        static_batch_labels.append(labels)
    static_batch = {
        'input_ids': torch.stack(static_batch_inputs),
        'labels': torch.stack(static_batch_labels),
        'attention_mask': torch.ones_like(torch.stack(static_batch_inputs))
    }
    end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    static_mem = end_mem - start_mem
    
    print(f"  Dynamic masking memory: {dynamic_mem} bytes")
    print(f"  Static masking memory:  {static_mem} bytes")
    if static_mem > 0:
        print(f"  Memory ratio: {dynamic_mem/static_mem:.2f}x {'MORE' if dynamic_mem > static_mem else 'LESS'}")
    else:
        print(f"  Memory comparison: Unable to measure (both near zero)")
    
    # ============= HYPOTHESIS GENERATION =============
    print(f"\\n🧠 POTENTIAL ISSUES ANALYSIS:")
    print("-" * 40)
    
    issues_found = []
    
    if dynamic_time > static_time * 1.5:
        issues_found.append("🐌 PERFORMANCE: Dynamic masking is significantly slower")
        
    if dynamic_std > static_std * 1.5:
        issues_found.append("📊 CONSISTENCY: Dynamic masking has higher rate variance")
        
    if dynamic_mem > static_mem * 1.3:
        issues_found.append("💾 MEMORY: Dynamic masking uses more memory")
    
    # Check for masking correctness
    avg_dynamic_rate = sum(dynamic_rates) / len(dynamic_rates)
    avg_static_rate = sum(static_rates) / len(static_rates)
    
    if abs(avg_dynamic_rate - 15.0) > 2.0:
        issues_found.append(f"🎯 ACCURACY: Dynamic masking rate ({avg_dynamic_rate:.1f}%) deviates from 15%")
        
    if abs(avg_static_rate - 15.0) > 2.0:
        issues_found.append(f"🎯 ACCURACY: Static masking rate ({avg_static_rate:.1f}%) deviates from 15%")
    
    if not issues_found:
        issues_found.append("✅ No obvious issues detected")
    
    for issue in issues_found:
        print(f"  {issue}")
    
    # ============= RECOMMENDATIONS =============
    print(f"\\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    
    if dynamic_time > static_time * 2:
        print("  1. 🚀 OPTIMIZATION: Profile dynamic_collator.py for performance bottlenecks")
        print("  2. 🔧 CONSIDER: Precompute some dynamic masking components")
        
    if dynamic_std > static_std * 1.5:
        print("  3. 📊 CONSISTENCY: Review adaptive masking logic for stability")
        
    print("  4. 🧪 TRAINING TEST: Run short training experiments to confirm loss behavior")
    print("  5. 🔍 PROFILE: Use torch.profiler to identify exact bottlenecks")
    
    return {
        'dynamic_time': dynamic_time,
        'static_time': static_time, 
        'dynamic_rates': dynamic_rates,
        'static_rates': static_rates,
        'issues_found': issues_found
    }

if __name__ == "__main__":
    debug_masking_performance()
