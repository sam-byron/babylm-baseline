#!/usr/bin/env python3
"""
Comprehensive test suite for dynamic_collator.py
Tests RoBERTa-style dynamic masking with detailed statistics and span analysis.
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
from dynamic_collator import DynamicMaskingCollator, create_dynamic_collator

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class DynamicCollatorTester:
    """Comprehensive tester for DynamicMaskingCollator."""
    
    def __init__(self, tokenizer_path: str, config: dict):
        """Initialize the tester with tokenizer and configuration."""
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.config = config
        
        # Create the dynamic collator
        self.collator = create_dynamic_collator(config, self.tokenizer)
        
        # Special token IDs
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.mask_id = self.tokenizer.token_to_id("[MASK]")
        self.cls_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_id = self.tokenizer.token_to_id("[SEP]")
        self.unk_id = self.tokenizer.token_to_id("[UNK]")
        
        print(f"🧪 DynamicMaskingCollator Tester Initialized")
        print(f"   - Vocab size: {self.tokenizer.get_vocab_size()}")
        print(f"   - Masking strategy: {config.get('masking_strategy', 'subword')}")
        print(f"   - Mask probability: {config.get('mask_p', 0.15)}")
        print(f"   - Random probability: {config.get('random_p', 0.1)}")
        print(f"   - Keep probability: {config.get('keep_p', 0.1)}")
        print()
    
    def create_test_sequences(self, num_sequences: int = 1000, seq_length: int = 512):
        """Create test sequences with realistic patterns."""
        sequences = []
        
        # Sample texts for more realistic testing
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming artificial intelligence research.",
            "Natural language processing requires sophisticated tokenization methods.",
            "Deep neural networks learn complex patterns from large datasets.",
            "Transformers have revolutionized the field of natural language understanding.",
            "BERT and RoBERTa are powerful pre-trained language models.",
            "Attention mechanisms allow models to focus on relevant information.",
            "Large language models demonstrate emergent capabilities at scale.",
        ]
        
        for i in range(num_sequences):
            # Choose a random base text
            base_text = random.choice(sample_texts)
            
            # Encode and potentially modify
            tokens = self.tokenizer.encode(base_text).ids
            
            # Add CLS and SEP tokens
            tokens = [self.cls_id] + tokens + [self.sep_id]
            
            # Truncate or pad to desired length
            if len(tokens) > seq_length:
                tokens = tokens[:seq_length-1] + [self.sep_id]
            elif len(tokens) < seq_length:
                # Add some random content tokens
                content_length = seq_length - len(tokens)
                for _ in range(content_length):
                    token_id = random.randint(6, self.tokenizer.get_vocab_size() - 1)
                    tokens.insert(-1, token_id)  # Insert before final SEP
            
            sequences.append(tokens)
        
        return sequences
    
    def analyze_batch_results(self, batch_result, original_sequences):
        """Analyze the results of dynamic masking on a batch."""
        input_ids = batch_result['input_ids']
        attention_mask = batch_result['attention_mask']
        labels = batch_result['labels']
        
        batch_size, seq_len = input_ids.shape
        
        results = {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'total_tokens': 0,
            'maskable_tokens': 0,
            'masked_tokens': 0,
            'mask_token_count': 0,
            'random_token_count': 0,
            'kept_token_count': 0,
            'spans': [],
            'masking_rates': [],
            'sequence_stats': []
        }
        
        for seq_idx in range(batch_size):
            seq_input = input_ids[seq_idx]
            seq_attention = attention_mask[seq_idx]
            seq_labels = labels[seq_idx]
            original_seq = torch.tensor(original_sequences[seq_idx], dtype=torch.long)
            
            # Count different token types
            real_tokens = torch.sum(seq_attention == 1).item()
            masked_positions = seq_labels != -100
            num_masked = torch.sum(masked_positions).item()
            
            # Count special tokens
            special_tokens = 0
            for token_id in seq_input:
                if token_id.item() in self.collator.special_token_ids:
                    special_tokens += 1
            
            maskable = real_tokens - special_tokens
            masking_rate = num_masked / maskable if maskable > 0 else 0
            
            results['total_tokens'] += real_tokens
            results['maskable_tokens'] += maskable
            results['masked_tokens'] += num_masked
            results['masking_rates'].append(masking_rate * 100)
            
            if num_masked > 0:
                # Analyze masking operations
                original_at_masked = seq_labels[masked_positions]
                current_at_masked = seq_input[masked_positions]
                
                mask_tokens = torch.sum(current_at_masked == self.mask_id).item()
                kept_tokens = torch.sum(current_at_masked == original_at_masked).item()
                random_tokens = num_masked - mask_tokens - kept_tokens
                
                results['mask_token_count'] += mask_tokens
                results['random_token_count'] += random_tokens
                results['kept_token_count'] += kept_tokens
                
                # Find spans
                spans = self.find_spans(masked_positions)
                results['spans'].extend(spans)
            
            # Store individual sequence stats
            seq_stats = {
                'seq_idx': seq_idx,
                'real_tokens': real_tokens,
                'maskable_tokens': maskable,
                'masked_tokens': num_masked,
                'masking_rate': masking_rate * 100,
                'spans': self.find_spans(masked_positions) if num_masked > 0 else []
            }
            results['sequence_stats'].append(seq_stats)
        
        return results
    
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
    
    def test_masking_rate(self, num_batches: int = 50, batch_size: int = 24, seq_length: int = 512):
        """Test masking rates across multiple batches."""
        print(f"🎯 TEST 1: MASKING RATE VERIFICATION")
        print(f"Testing {num_batches} batches of {batch_size} sequences (length {seq_length})")
        print("-" * 60)
        
        all_results = []
        total_maskable = 0
        total_masked = 0
        
        for batch_idx in range(num_batches):
            # Create test batch
            sequences = self.create_test_sequences(batch_size, seq_length)
            
            # Apply dynamic masking
            batch_result = self.collator(sequences)
            
            # Analyze results
            results = self.analyze_batch_results(batch_result, sequences)
            all_results.append(results)
            
            total_maskable += results['maskable_tokens']
            total_masked += results['masked_tokens']
            
            if batch_idx < 3:  # Show details for first 3 batches
                avg_rate = np.mean(results['masking_rates'])
                print(f"Batch {batch_idx+1}: {results['maskable_tokens']} maskable, "
                      f"{results['masked_tokens']} masked (avg: {avg_rate:.2f}%)")
        
        overall_rate = total_masked / total_maskable * 100 if total_maskable > 0 else 0
        all_rates = [rate for result in all_results for rate in result['masking_rates']]
        
        expected_rate = self.config.get('mask_p', 0.15) * 100
        
        print(f"\n📊 MASKING RATE RESULTS:")
        print(f"   - Expected rate: {expected_rate:.1f}%")
        print(f"   - Overall rate: {overall_rate:.2f}%")
        print(f"   - Mean rate: {np.mean(all_rates):.2f}% ± {np.std(all_rates):.2f}%")
        print(f"   - Rate range: {np.min(all_rates):.2f}% - {np.max(all_rates):.2f}%")
        print(f"   - Total: {total_maskable:,} maskable, {total_masked:,} masked")
        
        # Check compliance
        tolerance = 2.0
        rate_ok = abs(overall_rate - expected_rate) <= tolerance
        print(f"   {'✅ PASS' if rate_ok else '❌ FAIL'}: Rate within ±{tolerance}% of expected")
        
        return all_results, rate_ok
    
    def test_masking_compliance(self, results):
        """Test 80/10/10 masking compliance."""
        print(f"\n⚖️  TEST 2: MASKING COMPLIANCE (80/10/10 RULE)")
        print("-" * 60)
        
        total_mask = sum(r['mask_token_count'] for r in results)
        total_random = sum(r['random_token_count'] for r in results)
        total_kept = sum(r['kept_token_count'] for r in results)
        total_masked = total_mask + total_random + total_kept
        
        if total_masked == 0:
            print("❌ No masked tokens found!")
            return False
        
        mask_percent = total_mask / total_masked * 100
        random_percent = total_random / total_masked * 100
        kept_percent = total_kept / total_masked * 100
        
        expected_mask = (1.0 - self.config.get('random_p', 0.1) - self.config.get('keep_p', 0.1)) * 100
        expected_random = self.config.get('random_p', 0.1) * 100
        expected_keep = self.config.get('keep_p', 0.1) * 100
        
        print(f"📊 COMPLIANCE RESULTS:")
        print(f"   - [MASK] tokens: {mask_percent:.1f}% (expected: {expected_mask:.1f}%)")
        print(f"   - Random tokens: {random_percent:.1f}% (expected: {expected_random:.1f}%)")
        print(f"   - Kept tokens: {kept_percent:.1f}% (expected: {expected_keep:.1f}%)")
        print(f"   - Total masked tokens: {total_masked:,}")
        
        # Check compliance
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
        """Analyze span masking patterns."""
        print(f"\n🎭 TEST 3: SPAN PATTERN ANALYSIS")
        print("-" * 60)
        
        all_spans = []
        for result in results:
            all_spans.extend(result['spans'])
        
        if not all_spans:
            print("❌ No spans found!")
            return []
        
        span_lengths = [len(span) for span in all_spans]
        total_sequences = sum(len(r['sequence_stats']) for r in results)
        sequences_with_spans = sum(1 for r in results for seq in r['sequence_stats'] if seq['spans'])
        
        print(f"📊 SPAN STATISTICS:")
        print(f"   - Total spans: {len(all_spans):,}")
        print(f"   - Sequences with spans: {sequences_with_spans:,}/{total_sequences:,} ({sequences_with_spans/total_sequences*100:.1f}%)")
        print(f"   - Average spans per sequence: {len(all_spans)/total_sequences:.2f}")
        print(f"   - Span length range: {min(span_lengths)}-{max(span_lengths)}")
        print(f"   - Average span length: {np.mean(span_lengths):.2f} ± {np.std(span_lengths):.2f}")
        
        # Span length distribution
        length_counts = Counter(span_lengths)
        print(f"   - Span length distribution:")
        for length in sorted(length_counts.keys())[:15]:  # Show first 15 lengths
            count = length_counts[length]
            percent = count / len(all_spans) * 100
            print(f"     Length {length}: {count:,} spans ({percent:.1f}%)")
        
        return span_lengths
    
    def test_dynamic_behavior(self, num_epochs: int = 5, batch_size: int = 8, seq_length: int = 256):
        """Test that masking patterns change across epochs (dynamic behavior)."""
        print(f"\n🔄 TEST 4: DYNAMIC MASKING BEHAVIOR")
        print(f"Testing consistency across {num_epochs} epochs")
        print("-" * 60)
        
        # Create a fixed test batch
        test_sequences = self.create_test_sequences(batch_size, seq_length)
        
        epoch_results = []
        
        for epoch in range(num_epochs):
            # Set different random seed for each epoch to simulate training
            torch.manual_seed(epoch)
            random.seed(epoch)
            np.random.seed(epoch)
            
            batch_result = self.collator(test_sequences)
            results = self.analyze_batch_results(batch_result, test_sequences)
            epoch_results.append(results)
            
            avg_rate = np.mean(results['masking_rates'])
            print(f"Epoch {epoch+1}: {results['masked_tokens']} masked tokens (avg rate: {avg_rate:.2f}%)")
        
        # Analyze variability
        epoch_rates = [np.mean(r['masking_rates']) for r in epoch_results]
        epoch_masked_counts = [r['masked_tokens'] for r in epoch_results]
        
        print(f"\n📊 DYNAMIC BEHAVIOR ANALYSIS:")
        print(f"   - Rate variability: {np.mean(epoch_rates):.2f}% ± {np.std(epoch_rates):.2f}%")
        print(f"   - Masked count variability: {np.mean(epoch_masked_counts):.1f} ± {np.std(epoch_masked_counts):.1f}")
        print(f"   - Rate range: {np.min(epoch_rates):.2f}% - {np.max(epoch_rates):.2f}%")
        
        # Check if masking is actually different across epochs
        variability_ok = np.std(epoch_rates) > 0.1  # Should have some variability
        print(f"   {'✅ PASS' if variability_ok else '❌ FAIL'}: Masking patterns vary across epochs")
        
        return epoch_results, variability_ok
    
    def test_special_token_preservation(self, results):
        """Verify special tokens are never masked."""
        print(f"\n🛡️  TEST 5: SPECIAL TOKEN PRESERVATION")
        print("-" * 60)
        
        # This is harder to test directly since we'd need to compare with original sequences
        # But we can check if any special tokens appear in labels (which would be wrong)
        
        violations = 0
        total_special = 0
        
        # We'll need to create a test specifically for this
        test_sequences = self.create_test_sequences(100, 128)
        
        for seq in test_sequences:
            batch_result = self.collator([seq])
            labels = batch_result['labels'][0]
            input_ids = batch_result['input_ids'][0]
            
            for pos, (token_id, label) in enumerate(zip(input_ids, labels)):
                if token_id.item() in self.collator.special_token_ids:
                    total_special += 1
                    if label.item() != -100:  # Special token was marked for masking
                        violations += 1
        
        print(f"📊 SPECIAL TOKEN RESULTS:")
        print(f"   - Total special tokens: {total_special:,}")
        print(f"   - Special tokens masked: {violations}")
        print(f"   - Preservation rate: {(total_special-violations)/total_special*100:.2f}%")
        
        preservation_ok = violations == 0
        print(f"   {'✅ PASS' if preservation_ok else '❌ FAIL'}: All special tokens preserved")
        
        return preservation_ok
    
    def visualize_results(self, results, span_lengths):
        """Create visualizations of test results."""
        if not HAS_MATPLOTLIB:
            print(f"\n⚠️  Matplotlib not available, skipping visualizations")
            return
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'DynamicMaskingCollator Test Results', fontsize=14)
            
            # 1. Masking rate distribution
            all_rates = [rate for result in results for rate in result['masking_rates']]
            axes[0, 0].hist(all_rates, bins=30, alpha=0.7, edgecolor='black')
            expected_rate = self.config.get('mask_p', 0.15) * 100
            axes[0, 0].axvline(expected_rate, color='red', linestyle='--', label=f'Expected: {expected_rate:.1f}%')
            axes[0, 0].set_xlabel('Masking Rate (%)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Masking Rate Distribution')
            axes[0, 0].legend()
            
            # 2. Span length distribution
            if span_lengths:
                axes[0, 1].hist(span_lengths, bins=range(1, max(span_lengths)+2), alpha=0.7, edgecolor='black')
                axes[0, 1].set_xlabel('Span Length')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Span Length Distribution')
            
            # 3. Spans per sequence
            spans_per_seq = []
            for result in results:
                for seq_stat in result['sequence_stats']:
                    spans_per_seq.append(len(seq_stat['spans']))
            
            if spans_per_seq:
                axes[0, 2].hist(spans_per_seq, bins=range(max(spans_per_seq)+2), alpha=0.7, edgecolor='black')
                axes[0, 2].set_xlabel('Spans per Sequence')
                axes[0, 2].set_ylabel('Frequency')
                axes[0, 2].set_title('Spans per Sequence')
            
            # 4. Masking operations distribution
            total_mask = sum(r['mask_token_count'] for r in results)
            total_random = sum(r['random_token_count'] for r in results)
            total_kept = sum(r['kept_token_count'] for r in results)
            
            operations = ['[MASK]', 'Random', 'Kept']
            counts = [total_mask, total_random, total_kept]
            colors = ['red', 'orange', 'green']
            
            axes[1, 0].bar(operations, counts, color=colors, alpha=0.7, edgecolor='black')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Masking Operations')
            
            # Add percentages
            total = sum(counts)
            for i, (op, count) in enumerate(zip(operations, counts)):
                if total > 0:
                    percent = count / total * 100
                    axes[1, 0].text(i, count + total*0.01, f'{percent:.1f}%', ha='center', va='bottom')
            
            # 5. Batch-wise masking rates
            batch_rates = [np.mean(r['masking_rates']) for r in results]
            axes[1, 1].plot(batch_rates, 'o-', alpha=0.7)
            axes[1, 1].set_xlabel('Batch Index')
            axes[1, 1].set_ylabel('Average Masking Rate (%)')
            axes[1, 1].set_title('Masking Rate per Batch')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Sequence length vs masking rate scatter
            seq_lengths = []
            seq_rates = []
            for result in results:
                for seq_stat in result['sequence_stats']:
                    if seq_stat['maskable_tokens'] > 0:
                        seq_lengths.append(seq_stat['maskable_tokens'])
                        seq_rates.append(seq_stat['masking_rate'])
            
            if seq_lengths and seq_rates:
                axes[1, 2].scatter(seq_lengths, seq_rates, alpha=0.5, s=10)
                axes[1, 2].set_xlabel('Maskable Tokens per Sequence')
                axes[1, 2].set_ylabel('Masking Rate (%)')
                axes[1, 2].set_title('Sequence Length vs Masking Rate')
                axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('dynamic_collator_test_results.png', dpi=150, bbox_inches='tight')
            print(f"\n📊 Visualizations saved to 'dynamic_collator_test_results.png'")
            plt.show()
            
        except Exception as e:
            print(f"\n⚠️  Error creating visualizations: {e}")
    
    def run_comprehensive_test(self, num_batches: int = 50, batch_size: int = 24, seq_length: int = 512):
        """Run all tests and provide comprehensive analysis."""
        print(f"🔬 COMPREHENSIVE DYNAMIC COLLATOR TEST")
        print(f"=" * 80)
        print(f"Configuration:")
        print(f"  - Batches: {num_batches}, Batch size: {batch_size}, Sequence length: {seq_length}")
        print(f"  - Masking strategy: {self.config.get('masking_strategy', 'subword')}")
        print(f"  - Expected masking rate: {self.config.get('mask_p', 0.15) * 100:.1f}%")
        print()
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        
        # Run tests
        results, rate_ok = self.test_masking_rate(num_batches, batch_size, seq_length)
        compliance_ok = self.test_masking_compliance(results)
        span_lengths = self.test_span_patterns(results)
        epoch_results, variability_ok = self.test_dynamic_behavior()
        preservation_ok = self.test_special_token_preservation(results)
        
        # Visualizations
        self.visualize_results(results, span_lengths)
        
        # Final summary
        print(f"\n" + "="*80)
        print(f"📋 FINAL SUMMARY")
        print(f"="*80)
        
        tests_passed = 0
        total_tests = 5
        
        print(f"✅ Masking Rate Test: {'PASS' if rate_ok else 'FAIL'}")
        if rate_ok: tests_passed += 1
        
        print(f"✅ Compliance Test: {'PASS' if compliance_ok else 'FAIL'}")
        if compliance_ok: tests_passed += 1
        
        print(f"✅ Span Pattern Test: {'PASS' if span_lengths else 'FAIL'}")
        if span_lengths: tests_passed += 1
        
        print(f"✅ Dynamic Behavior Test: {'PASS' if variability_ok else 'FAIL'}")
        if variability_ok: tests_passed += 1
        
        print(f"✅ Special Token Preservation: {'PASS' if preservation_ok else 'FAIL'}")
        if preservation_ok: tests_passed += 1
        
        overall_pass = tests_passed == total_tests
        print(f"\n🎯 OVERALL RESULT: {'✅ PASS' if overall_pass else '❌ FAIL'} ({tests_passed}/{total_tests} tests passed)")
        
        if overall_pass:
            print(f"🎉 DynamicMaskingCollator is working correctly!")
        else:
            print(f"⚠️  DynamicMaskingCollator has issues that need attention.")
        
        return overall_pass


def main():
    """Main function to run comprehensive dynamic collator tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DynamicMaskingCollator")
    parser.add_argument("--tokenizer", type=str, default="data/pretrain/wordpiece_vocab.json", help="Path to tokenizer file")
    parser.add_argument("--batches", type=int, default=50, help="Number of test batches")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--strategy", type=str, default="span", choices=["span", "subword"], help="Masking strategy")
    parser.add_argument("--mask_p", type=float, default=0.15, help="Masking probability")
    
    args = parser.parse_args()
    
    # Configuration for dynamic collator
    config = {
        "masking_strategy": args.strategy,
        "mask_p": args.mask_p,
        "random_p": 0.1,
        "keep_p": 0.1,
        "max_span_length": 10,
        "geometric_p": 0.3,
    }
    
    try:
        # Initialize tester
        tester = DynamicCollatorTester(args.tokenizer, config)
        
        # Run comprehensive test
        success = tester.run_comprehensive_test(
            num_batches=args.batches,
            batch_size=args.batch_size,
            seq_length=args.seq_length
        )
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
