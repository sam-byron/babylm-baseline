#!/usr/bin/env python3

import torch
import json
from tokenizers import Tokenizer
from dynamic_collator import create_dynamic_collator

def test_true_dynamic_masking():
    """Test if the fixed dynamic collator produces truly different masking patterns."""
    print("🎭 TESTING TRUE DYNAMIC MASKING")
    print("=" * 50)
    
    # Load configuration
    with open("model_babylm_ltg_bert.json", "r") as f:
        config = json.load(f)
    
    tokenizer = Tokenizer.from_file("data/pretrain/wordpiece_vocab.json")
    
    # Create a test sequence with enough content tokens for meaningful masking
    test_sequence = (
        [tokenizer.token_to_id("[CLS]")] +
        list(range(100, 150)) +  # 50 content words
        [tokenizer.token_to_id("[SEP]")] +
        [tokenizer.token_to_id("[PAD]")] * 460  # padding to 512
    )
    
    test_tensor = torch.tensor(test_sequence, dtype=torch.long)
    print(f"Test sequence: {len(test_sequence)} tokens (50 content tokens)")
    
    # Create dynamic collator
    dynamic_collator = create_dynamic_collator(config, tokenizer)
    
    print(f"\\n📊 TESTING DYNAMIC MASKING ACROSS MULTIPLE CALLS:")
    print("-" * 55)
    
    results = []
    masked_positions_sets = []
    
    for trial in range(5):
        batch = dynamic_collator([test_tensor])
        labels = batch['labels'][0]
        input_ids = batch['input_ids'][0]
        
        # Find masked positions
        masked_positions = (labels != -100).nonzero().squeeze().tolist()
        if isinstance(masked_positions, int):
            masked_positions = [masked_positions]
        
        # Count different types of masking
        total_masked = len(masked_positions)
        mask_tokens = (input_ids == tokenizer.token_to_id('[MASK]')).sum().item()
        random_keep = total_masked - mask_tokens
        
        masking_rate = total_masked / 50 * 100  # 50 content tokens
        
        results.append({
            'trial': trial,
            'total_masked': total_masked,
            'mask_tokens': mask_tokens,
            'random_keep': random_keep,
            'masking_rate': masking_rate,
            'positions': set(masked_positions[:10])  # First 10 for comparison
        })
        
        masked_positions_sets.append(set(masked_positions))
        
        print(f"Trial {trial}: {total_masked} masked ({mask_tokens} [MASK], {random_keep} rand/keep) = {masking_rate:.1f}%")
        print(f"  First 10 masked positions: {sorted(list(results[-1]['positions']))}")
    
    print(f"\\n🔍 DYNAMIC MASKING ANALYSIS:")
    print("-" * 35)
    
    # Check if masking is truly dynamic
    all_masked_positions = set()
    for pos_set in masked_positions_sets:
        all_masked_positions.update(pos_set)
    
    print(f"Total unique positions masked across trials: {len(all_masked_positions)}")
    print(f"Average positions masked per trial: {sum(len(s) for s in masked_positions_sets) / len(masked_positions_sets):.1f}")
    
    # Check for overlap between trials
    overlaps = []
    for i in range(len(masked_positions_sets)):
        for j in range(i+1, len(masked_positions_sets)):
            overlap = len(masked_positions_sets[i] & masked_positions_sets[j])
            overlaps.append(overlap)
    
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
    print(f"Average position overlap between trials: {avg_overlap:.1f}")
    
    # Check masking rate consistency
    rates = [r['masking_rate'] for r in results]
    rate_std = torch.tensor(rates).std().item()
    print(f"Masking rate std dev: {rate_std:.2f}%")
    
    # Determine if truly dynamic
    is_dynamic = len(all_masked_positions) > max(len(s) for s in masked_positions_sets) * 1.5
    
    print(f"\\n🎯 VERDICT:")
    if is_dynamic:
        print(f"✅ TRUE DYNAMIC MASKING: Different positions masked each call")
        print(f"✅ CONSISTENCY: Masking rates stable ({torch.tensor(rates).mean():.1f}% ± {rate_std:.1f}%)")
    else:
        print(f"❌ NOT TRULY DYNAMIC: Same positions being masked repeatedly")
    
    return is_dynamic, rate_std

def compare_with_static():
    """Compare dynamic vs static masking performance."""
    print(f"\\n📊 COMPARING DYNAMIC VS STATIC PERFORMANCE:")
    print("=" * 50)
    
    # Load configuration
    with open("model_babylm_ltg_bert.json", "r") as f:
        config = json.load(f)
    
    tokenizer = Tokenizer.from_file("data/pretrain/wordpiece_vocab.json")
    
    # Create test data
    test_samples = []
    for i in range(10):
        sequence = (
            [tokenizer.token_to_id("[CLS]")] +
            list(range(100+i*10, 150+i*10)) +  # 50 content words
            [tokenizer.token_to_id("[SEP]")] +
            [tokenizer.token_to_id("[PAD]")] * 460
        )
        test_samples.append(torch.tensor(sequence, dtype=torch.long))
    
    # Test dynamic collator
    dynamic_collator = create_dynamic_collator(config, tokenizer)
    
    import time
    start = time.time()
    dynamic_batch = dynamic_collator(test_samples)
    dynamic_time = time.time() - start
    
    d_masked_total = (dynamic_batch['labels'] != -100).sum().item()
    
    # Test static masking
    from mlm_dataset import SpanMaskingStrategy
    
    static_strategy = SpanMaskingStrategy(
        mask_p=config.get("mask_p", 0.15),
        tokenizer=tokenizer,
        n_special_tokens=6,
        random_p=config.get("random_p", 0.1),
        keep_p=config.get("keep_p", 0.1)
    )
    
    start = time.time()
    static_inputs = []
    static_labels = []
    for sample in test_samples:
        input_ids, labels = static_strategy(sample.clone())
        static_inputs.append(input_ids)
        static_labels.append(labels)
    static_time = time.time() - start
    
    s_masked_total = sum((labels != -100).sum().item() for labels in static_labels)
    
    print(f"Performance comparison:")
    print(f"  Dynamic: {dynamic_time:.4f}s, {d_masked_total} tokens masked")
    print(f"  Static:  {static_time:.4f}s, {s_masked_total} tokens masked")
    print(f"  Speed ratio: {dynamic_time/static_time:.2f}x")
    
    if abs(d_masked_total - s_masked_total) < 50:  # Allow some variation
        print(f"✅ Masking consistency: Similar masking amounts")
    else:
        print(f"⚠️  Masking difference: {abs(d_masked_total - s_masked_total)} tokens")

if __name__ == "__main__":
    is_dynamic, rate_std = test_true_dynamic_masking()
    compare_with_static()
    
    print(f"\\n🚀 FINAL ASSESSMENT:")
    if is_dynamic and rate_std < 1.0:
        print(f"✅ Fixed dynamic collator is working correctly!")
        print(f"✅ Ready to test in training with use_dynamic_masking=true")
    else:
        print(f"❌ Dynamic collator still has issues")
        print(f"❌ Stick with use_dynamic_masking=false for now")
