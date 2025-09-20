#!/usr/bin/env python3
"""
Test dynamic collator with real chunk data from model_babylm_bert_ltg folder.
"""

import torch
import numpy as np
import random
import os
from collections import Counter
from tokenizers import Tokenizer
from dynamic_collator import create_dynamic_collator


def test_with_real_chunks():
    """Test dynamic collator with 5 random chunks from the training data."""
    print("🧪 REAL CHUNK DATA DYNAMIC COLLATOR TEST")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file("./data/pretrain/wordpiece_vocab.json")
    
    # Configuration
    config = {
        "masking_strategy": "span",
        "mask_p": 0.15,
        "random_p": 0.1,
        "keep_p": 0.1,
        "max_span_length": 10,
        "geometric_p": 0.2,
    }
    
    # Create collator
    collator = create_dynamic_collator(config, tokenizer)
    
    # Get all chunk files
    chunk_dir = "./model_babylm_bert_ltg"
    chunk_files = [f for f in os.listdir(chunk_dir) if f.startswith("chunk") and f.endswith(".pt")]
    
    if len(chunk_files) < 5:
        print(f"❌ Not enough chunk files found. Found {len(chunk_files)}, need at least 5.")
        return False
    
    # Randomly select 5 chunks
    selected_chunks = random.sample(chunk_files, 5)
    print(f"📊 Selected chunks: {selected_chunks}")
    
    all_results = []
    all_span_lengths = []
    
    for chunk_idx, chunk_file in enumerate(selected_chunks):
        print(f"\n🗂️  Loading {chunk_file}...")
        
        try:
            # Load chunk data
            chunk_path = os.path.join(chunk_dir, chunk_file)
            chunk_data = torch.load(chunk_path, map_location='cpu')
            
            # Extract sequences (assuming chunk_data contains token sequences)
            if isinstance(chunk_data, dict):
                if 'input_ids' in chunk_data:
                    sequences = chunk_data['input_ids']
                elif 'tokens' in chunk_data:
                    sequences = chunk_data['tokens']
                else:
                    sequences = list(chunk_data.values())[0]  # Take first available data
            elif isinstance(chunk_data, (list, tuple)):
                sequences = chunk_data
            else:
                sequences = chunk_data
            
            # Convert to list format if needed
            if torch.is_tensor(sequences):
                if sequences.dim() == 1:
                    # Single sequence
                    sequences = [sequences.tolist()]
                else:
                    # Batch of sequences
                    sequences = [seq.tolist() for seq in sequences]
            
            # Take first few sequences (limit to avoid overwhelming output)
            test_sequences = sequences[:3] if len(sequences) > 3 else sequences
            
            print(f"   Found {len(sequences)} sequences, testing with {len(test_sequences)}")
            
            # Analyze sequence lengths
            seq_lengths = [len(seq) for seq in test_sequences]
            print(f"   Sequence lengths: {seq_lengths}")
            
            # Test dynamic masking on these sequences
            for round_num in range(5):
                print(f"\n   🎭 Round {round_num + 1} with {chunk_file}:")
                
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
                    
                    # Count real tokens
                    real_tokens = torch.sum(seq_attention == 1).item()
                    
                    # Count special tokens based on original sequence
                    original_seq = test_sequences[seq_idx]
                    special_count = sum(1 for t in original_seq if t in collator.special_token_ids)
                    
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
                    print(f"     Seq {seq_idx + 1}: {maskable} maskable, {num_masked} masked ({masking_rate*100:.1f}%)")
                
                # Calculate overall statistics
                overall_rate = total_masked / total_maskable if total_maskable > 0 else 0
                
                if total_masked > 0:
                    mask_pct = mask_tokens / total_masked * 100
                    random_pct = random_tokens / total_masked * 100
                    kept_pct = kept_tokens / total_masked * 100
                else:
                    mask_pct = random_pct = kept_pct = 0
                
                print(f"     Overall: {total_masked}/{total_maskable} masked ({overall_rate*100:.2f}%)")
                print(f"     Operations: [MASK]={mask_pct:.1f}%, Random={random_pct:.1f}%, Kept={kept_pct:.1f}%")
                
                if round_spans:
                    avg_span_len = np.mean(round_spans)
                    print(f"     Spans: {len(round_spans)} total, avg length={avg_span_len:.2f}")
                    all_span_lengths.extend(round_spans)
                
                all_results.append({
                    'chunk': chunk_file,
                    'round': round_num + 1,
                    'rate': overall_rate * 100,
                    'mask_pct': mask_pct,
                    'random_pct': random_pct,
                    'kept_pct': kept_pct,
                    'spans': round_spans,
                    'total_masked': total_masked,
                    'total_maskable': total_maskable
                })
        
        except Exception as e:
            print(f"     ❌ Error loading {chunk_file}: {e}")
            continue
    
    # Comprehensive summary across all chunks and rounds
    if all_results:
        print(f"\n📊 COMPREHENSIVE SUMMARY ({len(all_results)} tests across {len(selected_chunks)} chunks):")
        print("=" * 60)
        
        rates = [r['rate'] for r in all_results]
        mask_pcts = [r['mask_pct'] for r in all_results]
        random_pcts = [r['random_pct'] for r in all_results]
        kept_pcts = [r['kept_pct'] for r in all_results]
        total_masked_counts = [r['total_masked'] for r in all_results]
        
        print(f"📈 Masking Statistics:")
        print(f"   Rate: {np.mean(rates):.2f}% ± {np.std(rates):.2f}% (range: {min(rates):.2f}%-{max(rates):.2f}%)")
        print(f"   Total tokens masked per test: {np.mean(total_masked_counts):.1f} ± {np.std(total_masked_counts):.1f}")
        
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
            print(f"   Top span lengths: {span_dist.most_common(10)}")
            
            # Calculate percentage of different span types
            single_spans = span_dist[1] if 1 in span_dist else 0
            multi_spans = len(all_span_lengths) - single_spans
            print(f"   Single-token spans: {single_spans} ({single_spans/len(all_span_lengths)*100:.1f}%)")
            print(f"   Multi-token spans: {multi_spans} ({multi_spans/len(all_span_lengths)*100:.1f}%)")
            
            if multi_spans > 0:
                long_spans = sum(count for length, count in span_dist.items() if length >= 3)
                print(f"   Long spans (≥3 tokens): {long_spans} ({long_spans/len(all_span_lengths)*100:.1f}%)")
        
        # Assessment with real data
        print(f"\n✅ REAL DATA ASSESSMENT:")
        expected_rate = 15.0
        rate_ok = abs(np.mean(rates) - expected_rate) <= 3.0  # Allow 3% tolerance for real data
        rate_variability = np.std(rates) > 0.5  # Should have reasonable variability
        
        compliance_ok = (
            abs(np.mean(mask_pcts) - 80) <= 15 and  # Allow tolerance for real data
            abs(np.mean(random_pcts) - 10) <= 10 and
            abs(np.mean(kept_pcts) - 10) <= 10
        )
        
        span_diversity = len(set(all_span_lengths)) > 3 if all_span_lengths else False
        has_long_spans = max(all_span_lengths) > 2 if all_span_lengths else False
        
        print(f"   Rate accuracy: {'✅ PASS' if rate_ok else '❌ FAIL'} ({np.mean(rates):.2f}% vs {expected_rate}%)")
        print(f"   Rate variability: {'✅ PASS' if rate_variability else '❌ FAIL'} (σ={np.std(rates):.2f}%)")
        print(f"   80/10/10 compliance: {'✅ PASS' if compliance_ok else '❌ FAIL'}")
        print(f"   Span diversity: {'✅ PASS' if span_diversity else '❌ FAIL'}")
        print(f"   Multi-token spans: {'✅ PASS' if has_long_spans else '❌ FAIL'}")
        
        overall_pass = rate_ok and rate_variability and compliance_ok and span_diversity and has_long_spans
        print(f"   Overall: {'🎉 SUCCESS' if overall_pass else '⚠️ NEEDS ATTENTION'}")
        
        return overall_pass
    else:
        print("❌ No results to analyze!")
        return False


if __name__ == "__main__":
    # Set seed for reproducibility while allowing variability
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        success = test_with_real_chunks()
        
        print(f"\n🎯 FINAL VERDICT WITH REAL DATA:")
        if success:
            print("🎉 Dynamic collator is working excellently with real training data!")
        else:
            print("⚠️ Dynamic collator needs adjustments for real training data.")
        
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
