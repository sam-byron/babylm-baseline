#!/usr/bin/env python3
"""
Test dynamic_collator.py with longer sequences to evaluate span masking and variability.
"""


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import numpy as np
import random
from collections import Counter
from tokenizers import Tokenizer
from dynamic_collator import DynamicMaskingCollator, create_dynamic_collator


def test_long_sequences(tokenizer_path="./data/pretrain/wordpiece_vocab.json"):
    """Test dynamic collator with longer sequences."""
    print("🧪 LONG SEQUENCE DYNAMIC COLLATOR TEST")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Configuration for span masking
    config = {
        "masking_strategy": "span",
        "mask_p": 0.15,
        "random_p": 0.1,
        "keep_p": 0.1,
        "max_span_length": 10,
        "geometric_p": 0.2,  # Lower p for longer spans
    }
    
    # Create collator
    collator = create_dynamic_collator(config, tokenizer)
    
    # Longer test texts
    long_texts = [
        """Natural language processing is a subfield of linguistics, computer science, and artificial intelligence 
        concerned with the interactions between computers and human language, in particular how to program computers 
        to process and analyze large amounts of natural language data. The goal is a computer capable of understanding 
        the contents of documents, including the contextual nuances of the language within them.""",
        
        """Machine learning is a method of data analysis that automates analytical model building. It is a branch of 
        artificial intelligence based on the idea that systems can learn from data, identify patterns and make 
        decisions with minimal human intervention. Machine learning algorithms build a mathematical model based on 
        training data in order to make predictions or decisions without being explicitly programmed to do so.""",
        
        """Deep learning is part of a broader family of machine learning methods based on artificial neural networks 
        with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning 
        architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional 
        neural networks have been applied to fields including computer vision, speech recognition, natural language processing."""
    ]
    
    # Convert to token sequences
    test_sequences = []
    cls_token_id = tokenizer.token_to_id("[CLS]") or 1
    sep_token_id = tokenizer.token_to_id("[SEP]") or 2
    for text in long_texts:
        tokens = tokenizer.encode(text).ids
        # Add CLS and SEP
        tokens = [cls_token_id] + tokens + [sep_token_id]
        test_sequences.append(tokens)
    
    print(f"📊 Test Setup:")
    print(f"   - Tokenizer vocab: {tokenizer.get_vocab_size():,}")
    print(f"   - Test sequences: {len(test_sequences)}")
    seq_lengths = [len(seq) for seq in test_sequences]
    print(f"   - Sequence lengths: {seq_lengths} (avg: {np.mean(seq_lengths):.1f})")
    print(f"   - Expected masking rate: {config['mask_p']*100:.1f}%")
    print(f"   - Geometric p: {config['geometric_p']} (lower = longer spans)")
    
    # Run multiple tests to check variability
    all_results = []
    all_span_lengths = []
    
    for test_round in range(10):  # More rounds for better statistics
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
        round_spans = []
        
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
                    round_spans.extend(spans)
            
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
        
        if round_spans:
            avg_span_len = np.mean(round_spans)
            max_span_len = max(round_spans)
            print(f"   Spans: {len(round_spans)} total, avg={avg_span_len:.2f}, max={max_span_len}")
            
            # Show span distribution
            span_dist = Counter(round_spans)
            top_spans = span_dist.most_common(5)
            print(f"   Span distribution: {top_spans}")
            
            all_span_lengths.extend(round_spans)
        
        all_results.append({
            'rate': overall_rate * 100,
            'mask_pct': mask_pct,
            'random_pct': random_pct,
            'kept_pct': kept_pct,
            'spans': round_spans,
            'total_masked': total_masked,
            'total_maskable': total_maskable
        })
    
    # Comprehensive summary
    print(f"\n📊 COMPREHENSIVE SUMMARY ({len(all_results)} ROUNDS):")
    print("=" * 60)
    
    rates = [r['rate'] for r in all_results]
    mask_pcts = [r['mask_pct'] for r in all_results]
    random_pcts = [r['random_pct'] for r in all_results]
    kept_pcts = [r['kept_pct'] for r in all_results]
    total_masked_counts = [r['total_masked'] for r in all_results]
    
    print(f"📈 Masking Statistics:")
    print(f"   Rate: {np.mean(rates):.2f}% ± {np.std(rates):.2f}% (range: {min(rates):.2f}%-{max(rates):.2f}%)")
    print(f"   Total tokens masked per round: {np.mean(total_masked_counts):.1f} ± {np.std(total_masked_counts):.1f}")
    
    print(f"\n🎭 Token Operations:")
    print(f"   [MASK] usage: {np.mean(mask_pcts):.1f}% ± {np.std(mask_pcts):.1f}%")
    print(f"   Random usage: {np.mean(random_pcts):.1f}% ± {np.std(random_pcts):.1f}%")
    print(f"   Kept usage: {np.mean(kept_pcts):.1f}% ± {np.std(kept_pcts):.1f}%")
    
    print(f"\n📏 Span Analysis:")
    if all_span_lengths:
        print(f"   Total spans: {len(all_span_lengths)}")
        print(f"   Average span length: {np.mean(all_span_lengths):.2f}")
        print(f"   Span length range: {min(all_span_lengths)}-{max(all_span_lengths)}")
        
        span_dist = Counter(all_span_lengths)
        print(f"   Span distribution: {span_dist.most_common(10)}")
        
        # Calculate percentage of different span types
        single_spans = span_dist[1] if 1 in span_dist else 0
        multi_spans = len(all_span_lengths) - single_spans
        print(f"   Single-token spans: {single_spans} ({single_spans/len(all_span_lengths)*100:.1f}%)")
        print(f"   Multi-token spans: {multi_spans} ({multi_spans/len(all_span_lengths)*100:.1f}%)")
        
        if multi_spans > 0:
            long_spans = sum(count for length, count in span_dist.items() if length >= 3)
            print(f"   Long spans (≥3 tokens): {long_spans} ({long_spans/len(all_span_lengths)*100:.1f}%)")
    
    # Assessment
    print(f"\n✅ ASSESSMENT:")
    expected_rate = 15.0
    rate_ok = abs(np.mean(rates) - expected_rate) <= 10.0  # More lenient tolerance
    rate_variability = np.std(rates) > 0.5  # Should have reasonable variability
    
    compliance_ok = (
        abs(np.mean(mask_pcts) - 80) <= 15 and  # Allow more tolerance for small samples
        abs(np.mean(random_pcts) - 10) <= 10 and
        abs(np.mean(kept_pcts) - 10) <= 10
    )
    
    span_diversity = len(set(all_span_lengths)) > 1 if all_span_lengths else False
    has_long_spans = max(all_span_lengths) > 1 if all_span_lengths else False
    
    print(f"   Rate accuracy: {'✅ PASS' if rate_ok else '❌ FAIL'} ({np.mean(rates):.2f}% vs {expected_rate}%)")
    print(f"   Rate variability: {'✅ PASS' if rate_variability else '❌ FAIL'} (σ={np.std(rates):.2f}%)")
    print(f"   80/10/10 compliance: {'✅ PASS' if compliance_ok else '❌ FAIL'}")
    print(f"   Span diversity: {'✅ PASS' if span_diversity else '❌ FAIL'}")
    print(f"   Multi-token spans: {'✅ PASS' if has_long_spans else '❌ FAIL'}")
    
    overall_pass = rate_ok and rate_variability and compliance_ok and span_diversity and has_long_spans
    print(f"   Overall: {'🎉 SUCCESS' if overall_pass else '⚠️ NEEDS ATTENTION'}")
    
    return overall_pass, {
        'rates': rates,
        'span_lengths': all_span_lengths,
        'mask_compliance': (np.mean(mask_pcts), np.mean(random_pcts), np.mean(kept_pcts))
    }


if __name__ == "__main__":
    # Set seed for reproducibility while allowing variability
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        success, stats = test_long_sequences()
        
        print(f"\n🎯 FINAL VERDICT:")
        if success:
            print("🎉 Dynamic collator is working excellently with proper span masking!")
        else:
            print("⚠️ Dynamic collator needs some tuning but core functionality works.")
        
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
