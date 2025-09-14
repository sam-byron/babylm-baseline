#!/usr/bin/env python3

import torch
import json
import glob
from data_loader import ChunkedDataset
from tokenizers import Tokenizer
from dynamic_collator import create_dynamic_collator

def test_adaptive_masking():
    """Test the new adaptive masking logic across different sequence lengths."""
    print("🎯 TESTING ADAPTIVE MASKING BY SEQUENCE LENGTH")
    print("=" * 60)
    
    # Setup
    with open("model_babylm_ltg_bert.json", "r") as f:
        config = json.load(f)
    
    tokenizer = Tokenizer.from_file("data/pretrain/wordpiece_vocab.json")
    chunk_paths = sorted(glob.glob("model_babylm_bert_ltg/chunk*.pt"))[:10]
    dataset = ChunkedDataset(chunk_paths, block_size=512, pad_token_id=tokenizer.token_to_id("[PAD]"))
    
    # Create collator
    collate_fn = create_dynamic_collator(config, tokenizer)
    
    # Special token IDs
    special_token_ids = {0, 1, 2, 3, 4}
    mask_token_id = tokenizer.token_to_id("[MASK]")
    
    # Test multiple samples to find different sequence length patterns
    results_by_length = {}
    
    print("📊 Testing 50 samples...")
    for i in range(min(50, len(dataset))):
        sample = dataset[i]
        
        # Count content tokens between [CLS] and [SEP] boundaries
        sample_list = sample.tolist()
        cls_positions = [j for j, token in enumerate(sample_list) if token == 1]  # [CLS] = ID 1
        sep_positions = [j for j, token in enumerate(sample_list) if token == 2]  # [SEP] = ID 2
        
        # Analyze each sentence segment
        for cls_idx, cls_pos in enumerate(cls_positions):
            # Find corresponding SEP
            matching_seps = [sep for sep in sep_positions if sep > cls_pos]
            if not matching_seps:
                continue
            
            sep_pos = matching_seps[0]
            segment = sample_list[cls_pos:sep_pos+1]
            
            # Count maskable tokens in this segment
            maskable_in_segment = sum(1 for token in segment if token not in special_token_ids)
            
            if maskable_in_segment == 0:
                continue
            
            # Test masking on just this segment (simulate the issue)
            segment_tensor = torch.tensor([segment], dtype=torch.long)
            attention_mask = torch.ones_like(segment_tensor)
            
            batch = collate_fn([segment_tensor[0]])
            labels = batch['labels'][0]
            input_ids = batch['input_ids'][0]
            
            # Count actual masking in this segment
            masked_in_segment = (labels[:len(segment)] != -100).sum().item()
            mask_only_in_segment = (input_ids[:len(segment)] == mask_token_id).sum().item()
            
            rate = masked_in_segment / maskable_in_segment * 100 if maskable_in_segment > 0 else 0
            
            # Group by maskable token count ranges
            if maskable_in_segment < 5:
                length_category = "Very Short (1-4)"
            elif maskable_in_segment < 10:
                length_category = "Short (5-9)"
            elif maskable_in_segment < 20:
                length_category = "Medium (10-19)"
            else:
                length_category = "Long (20+)"
            
            if length_category not in results_by_length:
                results_by_length[length_category] = []
            
            results_by_length[length_category].append({
                'maskable': maskable_in_segment,
                'masked': masked_in_segment,
                'rate': rate,
                'segment': segment[:10]  # First 10 tokens for debugging
            })
    
    # Analyze results by category
    print(f"\\n📈 ADAPTIVE MASKING RESULTS BY SEQUENCE LENGTH:")
    print(f"{'Category':<20} {'Count':<8} {'Avg Rate':<12} {'Range':<20} {'Expected'}")
    print("-" * 80)
    
    for category in ["Very Short (1-4)", "Short (5-9)", "Medium (10-19)", "Long (20+)"]:
        if category not in results_by_length:
            continue
            
        data = results_by_length[category]
        rates = [item['rate'] for item in data]
        
        if rates:
            avg_rate = sum(rates) / len(rates)
            min_rate = min(rates)
            max_rate = max(rates)
            
            # Expected rates based on our adaptive logic
            if "Very Short" in category:
                expected = "~10%"
            elif "Short" in category:
                expected = "~12%"
            else:
                expected = "~15%"
            
            print(f"{category:<20} {len(data):<8} {avg_rate:<12.1f}% {min_rate:.1f}%-{max_rate:.1f}%{'':<8} {expected}")
            
            # Show examples of problematic cases
            problematic = [item for item in data if item['rate'] > 25]  # Over 25% masking
            if problematic:
                print(f"  ⚠️  {len(problematic)} cases with >25% masking:")
                for case in problematic[:3]:  # Show first 3 examples
                    print(f"    {case['maskable']} maskable, {case['masked']} masked ({case['rate']:.1f}%)")
    
    print(f"\\n🎯 SUMMARY:")
    total_segments = sum(len(data) for data in results_by_length.values())
    print(f"  Total segments analyzed: {total_segments}")
    
    # Check if adaptive masking is working
    short_segments = results_by_length.get("Very Short (1-4)", []) + results_by_length.get("Short (5-9)", [])
    if short_segments:
        short_rates = [item['rate'] for item in short_segments]
        avg_short_rate = sum(short_rates) / len(short_rates)
        over_masking = sum(1 for rate in short_rates if rate > 20)
        
        print(f"  Short sequences (1-9 tokens): {avg_short_rate:.1f}% avg masking")
        print(f"  Over-masking cases (>20%): {over_masking}/{len(short_segments)}")
        
        if avg_short_rate < 18 and over_masking < len(short_segments) * 0.1:
            print(f"  ✅ Adaptive masking working well!")
        else:
            print(f"  ⚠️  Still some over-masking issues")

if __name__ == "__main__":
    test_adaptive_masking()
