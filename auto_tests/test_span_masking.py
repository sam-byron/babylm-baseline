#!/usr/bin/env python3
"""
Comprehensive test suite for SpanMaskingStrategy class.
Tests masking rate, span patterns, and compliance with 80/10/10 rule.
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
from collections import Counter, defaultdict
from tokenizers import Tokenizer
from mlm_dataset import SpanMaskingStrategy

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class SpanMaskingTester:
    """Comprehensive tester for SpanMaskingStrategy."""
    
    def __init__(self, tokenizer_path: str, mask_p: float = 0.15, random_p: float = 0.1, keep_p: float = 0.1):
        """Initialize the tester with tokenizer and masking parameters."""
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.mask_p = mask_p
        self.random_p = random_p
        self.keep_p = keep_p
        self.n_special_tokens = 6  # [PAD], [UNK], [CLS], [SEP], [MASK], [unused0]
        
        # Initialize the masking strategy
        self.masking_strategy = SpanMaskingStrategy(
            mask_p=self.mask_p,
            tokenizer=self.tokenizer,
            n_special_tokens=self.n_special_tokens,
            padding_label_id=-100,
            random_p=self.random_p,
            keep_p=self.keep_p
        )
        
        # Special token IDs
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.mask_id = self.tokenizer.token_to_id("[MASK]")
        self.cls_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_id = self.tokenizer.token_to_id("[SEP]")
        self.unk_id = self.tokenizer.token_to_id("[UNK]")
        
        print(f"🧪 SpanMaskingStrategy Tester Initialized")
        print(f"   - Vocab size: {self.tokenizer.get_vocab_size()}")
        print(f"   - Mask probability: {self.mask_p}")
        print(f"   - Random probability: {self.random_p}")
        print(f"   - Keep probability: {self.keep_p}")
        print(f"   - Special tokens: PAD={self.pad_id}, MASK={self.mask_id}, CLS={self.cls_id}, SEP={self.sep_id}")
        print()
    
    def create_test_sequences(self, num_sequences: int = 1000, seq_length: int = 512):
        """Create test sequences with known patterns."""
        sequences = []
        
        for i in range(num_sequences):
            # Create a sequence with [CLS] at start, [SEP] at end, and random tokens in between
            tokens = []
            
            # Add [CLS] token
            tokens.append(self.cls_id)
            
            # Add random content tokens (avoiding special tokens)
            content_length = seq_length - 2  # Reserve space for [CLS] and [SEP]
            for _ in range(content_length):
                # Generate random token ID that's not a special token
                token_id = random.randint(self.n_special_tokens, self.tokenizer.get_vocab_size() - 1)
                tokens.append(token_id)
            
            # Add [SEP] token
            tokens.append(self.sep_id)
            
            sequences.append(torch.tensor(tokens, dtype=torch.long))
        
        return sequences
    
    def analyze_single_sequence(self, tokens):
        """Analyze masking results for a single sequence."""
        original_tokens = tokens.clone()
        masked_tokens, labels = self.masking_strategy(tokens.clone())
        
        # Count different types of tokens
        total_tokens = len(tokens)
        special_tokens = torch.sum(original_tokens < self.n_special_tokens).item()
        maskable_tokens = total_tokens - special_tokens
        
        # Find masked positions (where labels != -100)
        masked_positions = labels != -100
        num_masked = torch.sum(masked_positions).item()
        
        if num_masked == 0:
            return {
                'total_tokens': total_tokens,
                'special_tokens': special_tokens,
                'maskable_tokens': maskable_tokens,
                'num_masked': 0,
                'masking_rate': 0.0,
                'mask_tokens': 0,
                'random_tokens': 0,
                'kept_tokens': 0,
                'spans': []
            }
        
        # Get original and current tokens at masked positions
        original_at_masked = labels[masked_positions]
        current_at_masked = masked_tokens[masked_positions]
        
        # Count different masking operations
        mask_tokens = torch.sum(current_at_masked == self.mask_id).item()
        kept_tokens = torch.sum(current_at_masked == original_at_masked).item()
        random_tokens = num_masked - mask_tokens - kept_tokens
        
        # Find spans (consecutive masked positions)
        spans = self.find_spans(masked_positions)
        
        return {
            'total_tokens': total_tokens,
            'special_tokens': special_tokens,
            'maskable_tokens': maskable_tokens,
            'num_masked': num_masked,
            'masking_rate': num_masked / maskable_tokens * 100,
            'mask_tokens': mask_tokens,
            'random_tokens': random_tokens,
            'kept_tokens': kept_tokens,
            'spans': spans,
            'original_tokens': original_tokens,
            'masked_tokens': masked_tokens,
            'labels': labels
        }
    
    def find_spans(self, masked_positions):
        """Find consecutive spans of masked positions."""
        spans = []
        positions = masked_positions.nonzero().flatten().tolist()
        
        if not positions:
            return spans
        
        current_span = [positions[0]]
        
        for i in range(1, len(positions)):
            if positions[i] == current_span[-1] + 1:
                current_span.append(positions[i])
            else:
                spans.append(current_span)
                current_span = [positions[i]]
        
        spans.append(current_span)
        return spans
    
    def test_masking_rate(self, num_sequences: int = 1000, seq_length: int = 512):
        """Test if the masking rate is approximately 15%."""
        print(f"🎯 TEST 1: MASKING RATE VERIFICATION")
        print(f"Testing {num_sequences} sequences of length {seq_length}")
        print("-" * 50)
        
        sequences = self.create_test_sequences(num_sequences, seq_length)
        results = []
        
        for i, seq in enumerate(sequences):
            result = self.analyze_single_sequence(seq)
            results.append(result)
            
            if i < 5:  # Show details for first 5 sequences
                print(f"Seq {i+1}: {result['maskable_tokens']} maskable tokens, "
                      f"{result['num_masked']} masked ({result['masking_rate']:.2f}%)")
        
        # Aggregate statistics
        total_maskable = sum(r['maskable_tokens'] for r in results)
        total_masked = sum(r['num_masked'] for r in results)
        overall_rate = total_masked / total_maskable * 100
        
        individual_rates = [r['masking_rate'] for r in results if r['maskable_tokens'] > 0]
        
        print(f"\n📊 MASKING RATE RESULTS:")
        print(f"   - Expected rate: {self.mask_p * 100:.1f}%")
        print(f"   - Overall rate: {overall_rate:.2f}%")
        print(f"   - Mean individual rate: {np.mean(individual_rates):.2f}% ± {np.std(individual_rates):.2f}%")
        print(f"   - Rate range: {np.min(individual_rates):.2f}% - {np.max(individual_rates):.2f}%")
        print(f"   - Total tokens: {total_maskable:,} maskable, {total_masked:,} masked")
        
        # Check if within acceptable range (±2%)
        expected_rate = self.mask_p * 100
        tolerance = 2.0
        
        if abs(overall_rate - expected_rate) <= tolerance:
            print(f"   ✅ PASS: Rate within ±{tolerance}% of expected")
        else:
            print(f"   ❌ FAIL: Rate outside ±{tolerance}% of expected")
        
        return results, overall_rate
    
    def test_masking_compliance(self, results):
        """Test 80/10/10 masking compliance."""
        print(f"\n⚖️  TEST 2: MASKING COMPLIANCE (80/10/10 RULE)")
        print("-" * 50)
        
        total_mask_tokens = sum(r['mask_tokens'] for r in results)
        total_random_tokens = sum(r['random_tokens'] for r in results)
        total_kept_tokens = sum(r['kept_tokens'] for r in results)
        total_masked = total_mask_tokens + total_random_tokens + total_kept_tokens
        
        if total_masked == 0:
            print("❌ No masked tokens found!")
            return
        
        mask_percent = total_mask_tokens / total_masked * 100
        random_percent = total_random_tokens / total_masked * 100
        kept_percent = total_kept_tokens / total_masked * 100
        
        expected_mask = (1.0 - self.random_p - self.keep_p) * 100
        expected_random = self.random_p * 100
        expected_keep = self.keep_p * 100
        
        print(f"📊 COMPLIANCE RESULTS:")
        print(f"   - [MASK] tokens: {mask_percent:.1f}% (expected: {expected_mask:.1f}%)")
        print(f"   - Random tokens: {random_percent:.1f}% (expected: {expected_random:.1f}%)")
        print(f"   - Kept tokens: {kept_percent:.1f}% (expected: {expected_keep:.1f}%)")
        
        # Check compliance (±5% tolerance)
        tolerance = 5.0
        mask_ok = abs(mask_percent - expected_mask) <= tolerance
        random_ok = abs(random_percent - expected_random) <= tolerance
        keep_ok = abs(kept_percent - expected_keep) <= tolerance
        
        print(f"   - Compliance check (±{tolerance}%):")
        print(f"     [MASK]: {'✅' if mask_ok else '❌'}")
        print(f"     Random: {'✅' if random_ok else '❌'}")
        print(f"     Keep: {'✅' if keep_ok else '❌'}")
        
        return mask_ok and random_ok and keep_ok
    
    def test_span_patterns(self, results):
        """Test span masking patterns."""
        print(f"\n🎭 TEST 3: SPAN PATTERN ANALYSIS")
        print("-" * 50)
        
        all_spans = []
        for result in results:
            all_spans.extend(result['spans'])
        
        if not all_spans:
            print("❌ No spans found!")
            return
        
        # Calculate span statistics
        span_lengths = [len(span) for span in all_spans]
        total_sequences = len(results)
        sequences_with_spans = sum(1 for r in results if r['spans'])
        
        print(f"📊 SPAN STATISTICS:")
        print(f"   - Total spans: {len(all_spans)}")
        print(f"   - Sequences with spans: {sequences_with_spans}/{total_sequences} ({sequences_with_spans/total_sequences*100:.1f}%)")
        print(f"   - Average spans per sequence: {len(all_spans)/total_sequences:.2f}")
        print(f"   - Span length range: {min(span_lengths)} - {max(span_lengths)}")
        print(f"   - Average span length: {np.mean(span_lengths):.2f} ± {np.std(span_lengths):.2f}")
        
        # Show span length distribution
        length_counts = Counter(span_lengths)
        print(f"   - Span length distribution:")
        for length in sorted(length_counts.keys())[:10]:  # Show first 10
            count = length_counts[length]
            percent = count / len(all_spans) * 100
            print(f"     Length {length}: {count} spans ({percent:.1f}%)")
        
        return span_lengths
    
    def test_special_token_preservation(self, results):
        """Test that special tokens are never masked."""
        print(f"\n🛡️  TEST 4: SPECIAL TOKEN PRESERVATION")
        print("-" * 50)
        
        violations = 0
        total_special = 0
        
        for result in results:
            original = result['original_tokens']
            labels = result['labels']
            
            # Find special tokens in original
            special_mask = original < self.n_special_tokens
            total_special += torch.sum(special_mask).item()
            
            # Check if any special tokens were labeled for masking
            special_masked = labels[special_mask] != -100
            violations += torch.sum(special_masked).item()
        
        print(f"📊 SPECIAL TOKEN RESULTS:")
        print(f"   - Total special tokens: {total_special:,}")
        print(f"   - Special tokens masked: {violations}")
        print(f"   - Preservation rate: {(total_special-violations)/total_special*100:.2f}%")
        
        if violations == 0:
            print(f"   ✅ PASS: All special tokens preserved")
        else:
            print(f"   ❌ FAIL: {violations} special tokens were masked")
        
        return violations == 0
    
    def visualize_results(self, results, span_lengths, overall_rate):
        """Create visualizations of the test results."""
        if not HAS_MATPLOTLIB:
            print(f"\n⚠️  Matplotlib not available, skipping visualizations")
            return
            
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'SpanMaskingStrategy Test Results (mask_p={self.mask_p})', fontsize=14)
            
            # 1. Masking rate distribution
            rates = [r['masking_rate'] for r in results if r['maskable_tokens'] > 0]
            axes[0, 0].hist(rates, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(self.mask_p * 100, color='red', linestyle='--', 
                              label=f'Expected: {self.mask_p*100:.1f}%')
            axes[0, 0].axvline(overall_rate, color='green', linestyle='-', 
                              label=f'Actual: {overall_rate:.2f}%')
            axes[0, 0].set_xlabel('Masking Rate (%)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Masking Rate Distribution')
            axes[0, 0].legend()
            
            # 2. Span length distribution
            axes[0, 1].hist(span_lengths, bins=range(1, max(span_lengths)+2), alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Span Length')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Span Length Distribution')
            
            # 3. Number of spans per sequence
            spans_per_seq = [len(r['spans']) for r in results]
            axes[1, 0].hist(spans_per_seq, bins=range(max(spans_per_seq)+2), alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Number of Spans per Sequence')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Spans per Sequence Distribution')
            
            # 4. Masking operation distribution
            total_mask = sum(r['mask_tokens'] for r in results)
            total_random = sum(r['random_tokens'] for r in results)
            total_kept = sum(r['kept_tokens'] for r in results)
            
            operations = ['[MASK]', 'Random', 'Kept']
            counts = [total_mask, total_random, total_kept]
            colors = ['red', 'orange', 'green']
            
            axes[1, 1].bar(operations, counts, color=colors, alpha=0.7, edgecolor='black')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Masking Operations Distribution')
            
            # Add percentages on bars
            total = sum(counts)
            for i, (op, count) in enumerate(zip(operations, counts)):
                percent = count / total * 100
                axes[1, 1].text(i, count + total*0.01, f'{percent:.1f}%', 
                               ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('span_masking_test_results.png', dpi=150, bbox_inches='tight')
            print(f"\n📊 Visualizations saved to 'span_masking_test_results.png'")
            plt.show()
            
        except ImportError:
            print(f"\n⚠️  Matplotlib not available, skipping visualizations")
    
    def run_comprehensive_test(self, num_sequences: int = 1000, seq_length: int = 512):
        """Run all tests and provide a comprehensive report."""
        print(f"🔬 COMPREHENSIVE SPANMASKINGSTRATEGY TEST")
        print(f"=" * 60)
        print(f"Testing {num_sequences} sequences of length {seq_length}")
        print(f"Expected masking rate: {self.mask_p * 100:.1f}%")
        print()
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        
        # Run tests
        results, overall_rate = self.test_masking_rate(num_sequences, seq_length)
        compliance_ok = self.test_masking_compliance(results)
        span_lengths = self.test_span_patterns(results)
        preservation_ok = self.test_special_token_preservation(results)
        
        # Create visualizations
        self.visualize_results(results, span_lengths, overall_rate)
        
        # Final summary
        print(f"\n" + "="*60)
        print(f"📋 FINAL SUMMARY")
        print(f"="*60)
        
        rate_ok = abs(overall_rate - self.mask_p * 100) <= 2.0
        
        print(f"✅ Masking Rate Test: {'PASS' if rate_ok else 'FAIL'}")
        print(f"✅ Compliance Test: {'PASS' if compliance_ok else 'FAIL'}")
        print(f"✅ Special Token Preservation: {'PASS' if preservation_ok else 'FAIL'}")
        
        overall_pass = rate_ok and compliance_ok and preservation_ok
        print(f"\n🎯 OVERALL RESULT: {'✅ PASS' if overall_pass else '❌ FAIL'}")
        
        if overall_pass:
            print(f"🎉 SpanMaskingStrategy is working correctly!")
        else:
            print(f"⚠️  SpanMaskingStrategy has issues that need to be addressed.")
        
        return overall_pass


def main():
    """Main function to run the comprehensive test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SpanMaskingStrategy")
    parser.add_argument("--tokenizer", type=str, default="data/pretrain/wordpiece_vocab.json", 
                       help="Path to tokenizer file")
    parser.add_argument("--sequences", type=int, default=1000, 
                       help="Number of test sequences")
    parser.add_argument("--length", type=int, default=512, 
                       help="Sequence length")
    parser.add_argument("--mask_p", type=float, default=0.15, 
                       help="Masking probability")
    parser.add_argument("--random_p", type=float, default=0.1, 
                       help="Random replacement probability")
    parser.add_argument("--keep_p", type=float, default=0.1, 
                       help="Keep original probability")
    
    args = parser.parse_args()
    
    try:
        # Initialize tester
        tester = SpanMaskingTester(
            tokenizer_path=args.tokenizer,
            mask_p=args.mask_p,
            random_p=args.random_p,
            keep_p=args.keep_p
        )
        
        # Run comprehensive test
        success = tester.run_comprehensive_test(
            num_sequences=args.sequences,
            seq_length=args.length
        )
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
