#!/usr/bin/env python3
"""
Quick test for dynamic_collator.py to verify RoBERTa-style span masking.
"""

import torch
import numpy as np
import random
from collections import Counter
from tokenizers import Tokenizer
from dynamic_collator import DynamicMaskingCollator, create_dynamic_collator


def quick_dynamic_collator_test(tokenizer_path="./data/pretrain/wordpiece_vocab.json"):
    """Run a quick test of the dynamic collator."""
    print("🧪 QUICK DYNAMIC COLLATOR TEST")
    print("=" * 50)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Configuration
    config = {
        "masking_strategy": "span",
        "mask_p": 0.15,
        "random_p": 0.1,
        "keep_p": 0.1,
        "max_span_length": 10,
        "geometric_p": 0.3,
    }
    
    # Create collator
    collator = create_dynamic_collator(config, tokenizer)
    
    # Test data - create some simple sequences
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence.",
        "Natural language processing requires sophisticated methods.",
    ]
    
    # Convert to token sequences
    test_sequences = []
    cls_token_id = tokenizer.token_to_id("[CLS]") or 1
    sep_token_id = tokenizer.token_to_id("[SEP]") or 2
    for text in test_texts:
        tokens = tokenizer.encode(text).ids
        # Add CLS and SEP
        tokens = [cls_token_id] + tokens + [sep_token_id]
        test_sequences.append(tokens)
    
    print(f"📊 Test Setup:")
    print(f"   - Tokenizer vocab: {tokenizer.get_vocab_size():,}")
    print(f"   - Test sequences: {len(test_sequences)}")
    print(f"   - Expected masking rate: {config['mask_p']*100:.1f}%")
    
    # Run multiple tests to check variability
    all_results = []
    
    for test_round in range(5):
        print(f"\n🎭 Test Round {test_round + 1}:")
        
        # Apply dynamic masking
        batch_result = collator(test_sequences)
        
        input_ids = batch_result['input_ids']
        attention_mask = batch_result['attention_mask']
        labels = batch_result['labels']
        
        # Analyze results
        total_maskable = 0
        total_masked = 0
        mask_tokens = 0
        random_tokens = 0
        kept_tokens = 0
        all_spans = []
        
        for seq_idx in range(len(test_sequences)):
            seq_input = input_ids[seq_idx]
            seq_attention = attention_mask[seq_idx]
            seq_labels = labels[seq_idx]
            original_seq = torch.tensor(test_sequences[seq_idx], dtype=torch.long)
            
            # Count real tokens
            real_tokens = torch.sum(seq_attention == 1).item()
            
            # Count special tokens
            special_count = 0
            for token_id in seq_input:
                if token_id.item() in collator.special_token_ids:
                    special_count += 1
            
            maskable = real_tokens - special_count
            
            # Count masked tokens
            masked_positions = seq_labels != -100
            num_masked = torch.sum(masked_positions).item()
            
            total_maskable += maskable
            total_masked += num_masked
            
            if num_masked > 0:
                # Analyze masking types
                original_at_masked = seq_labels[masked_positions]
                current_at_masked = seq_input[masked_positions]
                
                mask_count = torch.sum(current_at_masked == collator.mask_token_id).item()
                kept_count = torch.sum(current_at_masked == original_at_masked).item()
                random_count = num_masked - mask_count - kept_count
                
                mask_tokens += mask_count
                random_tokens += random_count
                kept_tokens += kept_count
                
                # Find spans
                positions = masked_positions.nonzero().flatten().tolist()
                if positions:
                    spans = []
                    current_span = [positions[0]]
                    
                    for i in range(1, len(positions)):
                        if positions[i] == current_span[-1] + 1:
                            current_span.append(positions[i])
                        else:
                            spans.append(len(current_span))
                            current_span = [positions[i]]
                    spans.append(len(current_span))
                    all_spans.extend(spans)
            
            masking_rate = num_masked / maskable if maskable > 0 else 0
            print(f"   Seq {seq_idx + 1}: {maskable} maskable, {num_masked} masked ({masking_rate*100:.1f}%)")
        
        # Calculate overall statistics
        overall_rate = total_masked / total_maskable if total_maskable > 0 else 0
        
        if total_masked > 0:
            mask_pct = mask_tokens / total_masked * 100
            random_pct = random_tokens / total_masked * 100
            kept_pct = kept_tokens / total_masked * 100
        else:
            mask_pct = random_pct = kept_pct = 0
        
        print(f"   Overall: {total_masked}/{total_maskable} masked ({overall_rate*100:.2f}%)")
        print(f"   Operations: [MASK]={mask_pct:.1f}%, Random={random_pct:.1f}%, Kept={kept_pct:.1f}%")
        
        if all_spans:
            avg_span_len = np.mean(all_spans)
            print(f"   Spans: {len(all_spans)} total, avg length={avg_span_len:.2f}")
            span_dist = Counter(all_spans)
            top_spans = span_dist.most_common(5)
            print(f"   Top span lengths: {top_spans}")
        
        all_results.append({
            'rate': overall_rate * 100,
            'mask_pct': mask_pct,
            'random_pct': random_pct,
            'kept_pct': kept_pct,
            'spans': all_spans
        })
    
    # Summary across all rounds
    print(f"\n📊 SUMMARY ACROSS {len(all_results)} ROUNDS:")
    rates = [r['rate'] for r in all_results]
    mask_pcts = [r['mask_pct'] for r in all_results]
    random_pcts = [r['random_pct'] for r in all_results]
    kept_pcts = [r['kept_pct'] for r in all_results]
    
    print(f"   Masking rate: {np.mean(rates):.2f}% ± {np.std(rates):.2f}%")
    print(f"   [MASK] usage: {np.mean(mask_pcts):.1f}% ± {np.std(mask_pcts):.1f}%")
    print(f"   Random usage: {np.mean(random_pcts):.1f}% ± {np.std(random_pcts):.1f}%")
    print(f"   Kept usage: {np.mean(kept_pcts):.1f}% ± {np.std(kept_pcts):.1f}%")
    
    # Check if results are reasonable
    expected_rate = 15.0
    rate_ok = abs(np.mean(rates) - expected_rate) <= 2.0
    
    compliance_ok = (
        abs(np.mean(mask_pcts) - 80) <= 10 and
        abs(np.mean(random_pcts) - 10) <= 10 and
        abs(np.mean(kept_pcts) - 10) <= 10
    )
    
    variability_ok = np.std(rates) > 0.1  # Should have some variability across rounds
    
    print(f"\n✅ ASSESSMENT:")
    print(f"   Rate accuracy: {'✅ PASS' if rate_ok else '❌ FAIL'}")
    print(f"   80/10/10 compliance: {'✅ PASS' if compliance_ok else '❌ FAIL'}")
    print(f"   Dynamic variability: {'✅ PASS' if variability_ok else '❌ FAIL'}")
    
    overall_pass = rate_ok and compliance_ok and variability_ok
    print(f"   Overall: {'🎉 SUCCESS' if overall_pass else '⚠️ NEEDS ATTENTION'}")
    
    return overall_pass


def compare_strategies():
    """Compare subword vs span masking strategies."""
    print("\n🔄 STRATEGY COMPARISON")
    print("=" * 50)
    
    tokenizer = Tokenizer.from_file("./data/pretrain/wordpiece_vocab.json")
    
    # Test both strategies
    strategies = ["subword", "span"]
    
    test_text = "The quick brown fox jumps over the lazy dog and runs through the forest."
    tokens = tokenizer.encode(test_text).ids
    test_seq = [1] + tokens + [2]  # Add CLS and SEP
    
    for strategy in strategies:
        print(f"\n🎭 Testing {strategy.upper()} strategy:")
        
        config = {
            "masking_strategy": strategy,
            "mask_p": 0.15,
            "random_p": 0.1,
            "keep_p": 0.1,
        }
        
        collator = create_dynamic_collator(config, tokenizer)
        
        # Test multiple times
        for i in range(3):
            result = collator([test_seq])
            labels = result['labels'][0]
            input_ids = result['input_ids'][0]
            
            masked_positions = labels != -100
            num_masked = torch.sum(masked_positions).item()
            
            # Show masked tokens
            masked_tokens = []
            for pos in masked_positions.nonzero().flatten():
                pos_val = pos.item()
                original = tokenizer.id_to_token(labels[pos_val].item())
                current = tokenizer.id_to_token(input_ids[pos_val].item())
                masked_tokens.append(f"{original}→{current}")
            
            print(f"   Round {i+1}: {num_masked} masked - {masked_tokens[:5]}{'...' if len(masked_tokens) > 5 else ''}")


if __name__ == "__main__":
    # Set seed for some reproducibility while still showing variability
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        success = quick_dynamic_collator_test()
        compare_strategies()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
