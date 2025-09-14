#!/usr/bin/env python3

import torch
import json
import glob
from data_loader import ChunkedDataset
from tokenizers import Tokenizer
from dynamic_collator import create_dynamic_collator

def debug_dynamic_masking_correctness():
    """Debug the actual correctness of dynamic masking implementation."""
    print("🔍 DEBUGGING DYNAMIC MASKING CORRECTNESS")
    print("=" * 60)
    
    # Load configuration
    with open("model_babylm_ltg_bert.json", "r") as f:
        config = json.load(f)
    
    tokenizer = Tokenizer.from_file("data/pretrain/wordpiece_vocab.json")
    chunk_paths = sorted(glob.glob("model_babylm_bert_ltg/chunk*.pt"))[:2]
    dataset = ChunkedDataset(chunk_paths, block_size=512, pad_token_id=tokenizer.token_to_id("[PAD]"))
    
    # Create dynamic collator
    dynamic_collator = create_dynamic_collator(config, tokenizer)
    
    # Special tokens
    special_token_ids = {0, 1, 2, 3, 4}
    mask_token_id = tokenizer.token_to_id("[MASK]")
    
    # Test same sample multiple times (simulating multiple epochs)
    test_sample = dataset[0]
    
    print(f"📊 Testing dynamic masking consistency...")
    print(f"Sample shape: {test_sample.shape}")
    
    # Count content tokens
    content_tokens = sum(1 for token in test_sample.tolist() if token not in special_token_ids)
    print(f"Content tokens: {content_tokens}")
    
    print(f"\\n🎭 DYNAMIC MASKING ACROSS 'EPOCHS':")
    print("-" * 50)
    
    epoch_results = []
    for epoch in range(5):
        batch = dynamic_collator([test_sample])
        labels = batch['labels'][0]
        input_ids = batch['input_ids'][0]
        attention_mask = batch['attention_mask'][0]
        
        # Analyze masking pattern
        masked_positions = (labels != -100)
        total_masked = masked_positions.sum().item()
        mask_tokens = (input_ids == mask_token_id).sum().item()
        random_keep = total_masked - mask_tokens
        
        masking_rate = total_masked / content_tokens * 100
        
        # Check for issues
        masked_special = sum(1 for i, token in enumerate(test_sample.tolist()) 
                           if token in special_token_ids and masked_positions[i])
        
        epoch_results.append({
            'epoch': epoch,
            'total_masked': total_masked,
            'mask_tokens': mask_tokens,
            'random_keep': random_keep,
            'masking_rate': masking_rate,
            'masked_special': masked_special,
            'masked_positions': masked_positions.tolist()[:20]  # First 20 for comparison
        })
        
        print(f"Epoch {epoch}: {total_masked} masked ({mask_tokens} [MASK], {random_keep} rand/keep) = {masking_rate:.1f}% | Special masked: {masked_special}")
    
    print(f"\\n🔍 CONSISTENCY ANALYSIS:")
    print("-" * 30)
    
    # Check if different positions are being masked
    all_positions = set()
    for result in epoch_results:
        positions = [i for i, val in enumerate(result['masked_positions']) if val]
        all_positions.update(positions[:10])  # Just check first 10
    
    print(f"Unique masked positions (first 20 tokens): {sorted(all_positions)}")
    
    # Check masking rate consistency
    rates = [r['masking_rate'] for r in epoch_results]
    print(f"Masking rates: {[f'{r:.1f}%' for r in rates]}")
    print(f"Rate std dev: {torch.tensor(rates).std().item():.3f}%")
    
    # Check for potential issues
    issues = []
    
    # Issue 1: Masking special tokens
    special_masked = sum(r['masked_special'] for r in epoch_results)
    if special_masked > 0:
        issues.append(f"❌ SPECIAL TOKEN MASKING: {special_masked} special tokens masked across epochs")
    
    # Issue 2: Inconsistent masking rates
    rate_std = torch.tensor(rates).std().item()
    if rate_std > 1.0:
        issues.append(f"❌ INCONSISTENT RATES: High std dev ({rate_std:.2f}%) in masking rates")
    
    # Issue 3: Too low/high masking
    avg_rate = sum(rates) / len(rates)
    if abs(avg_rate - 15.0) > 2.0:
        issues.append(f"❌ WRONG RATE: Average masking rate ({avg_rate:.1f}%) far from 15%")
    
    # Issue 4: Not actually dynamic (same positions always masked)
    if len(all_positions) < 5:  # Should have variety in first 20 positions
        issues.append(f"❌ NOT DYNAMIC: Only {len(all_positions)} unique positions masked")
    
    print(f"\\n🚨 ISSUES DETECTED:")
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"  ✅ No obvious correctness issues detected")
    
    # Now test the REAL issue: gradient/loss computation
    print(f"\\n🔥 LOSS COMPUTATION TEST:")
    print("-" * 30)
    
    # Simulate what happens in training
    from torch.nn import CrossEntropyLoss
    
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    
    # Test with dummy model predictions
    vocab_size = config["vocab_size"]
    batch_size = 1
    seq_len = test_sample.shape[0]
    
    for epoch in range(3):
        batch = dynamic_collator([test_sample])
        labels = batch['labels']
        
        # Simulate model predictions (random logits)
        predictions = torch.randn(batch_size, seq_len, vocab_size)
        
        # Compute loss
        loss = loss_fn(predictions.view(-1, vocab_size), labels.view(-1))
        
        # Count valid targets for loss
        valid_targets = (labels != -100).sum().item()
        
        print(f"Epoch {epoch}: Loss={loss:.4f}, Valid targets={valid_targets}")
        
        # Check if valid targets are consistent
        if epoch == 0:
            first_valid = valid_targets
        elif abs(valid_targets - first_valid) > 2:
            issues.append(f"❌ INCONSISTENT TARGETS: Valid targets vary significantly ({first_valid} vs {valid_targets})")
    
    return issues

def debug_adaptive_masking_logic():
    """Specifically debug the adaptive masking logic we added."""
    print(f"\\n🎯 DEBUGGING ADAPTIVE MASKING LOGIC")
    print("=" * 50)
    
    # Test with different sequence lengths
    with open("model_babylm_ltg_bert.json", "r") as f:
        config = json.load(f)
    
    tokenizer = Tokenizer.from_file("data/pretrain/wordpiece_vocab.json")
    dynamic_collator = create_dynamic_collator(config, tokenizer)
    
    # Create test sequences of different lengths
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    
    test_sequences = [
        # Very short: 2 content tokens
        torch.tensor([cls_id, 100, 101, sep_id] + [3] * 508),
        
        # Short: 7 content tokens  
        torch.tensor([cls_id, 100, 101, 102, 103, 104, 105, 106, sep_id] + [3] * 503),
        
        # Medium: 15 content tokens
        torch.tensor([cls_id] + list(range(100, 115)) + [sep_id] + [3] * 496),
        
        # Long: 50 content tokens
        torch.tensor([cls_id] + list(range(100, 150)) + [sep_id] + [3] * 461),
    ]
    
    special_token_ids = {0, 1, 2, 3, 4}
    
    for i, seq in enumerate(test_sequences):
        content_count = sum(1 for token in seq.tolist() if token not in special_token_ids)
        
        print(f"\\nSequence {i+1}: {content_count} content tokens")
        
        # Test masking 3 times
        masking_rates = []
        for trial in range(3):
            batch = dynamic_collator([seq])
            labels = batch['labels'][0]
            
            masked = (labels != -100).sum().item()
            rate = masked / content_count * 100 if content_count > 0 else 0
            masking_rates.append(rate)
        
        avg_rate = sum(masking_rates) / len(masking_rates)
        print(f"  Masking rates: {[f'{r:.1f}%' for r in masking_rates]} (avg: {avg_rate:.1f}%)")
        
        # Check if adaptive logic is working
        if content_count < 5:
            expected = "~0% (too short)"
        elif content_count < 10:
            expected = "~10% (conservative)"
        elif content_count < 20:
            expected = "~12% (moderate)"  
        else:
            expected = "~15% (standard)"
        
        print(f"  Expected: {expected}")
        
        if avg_rate > 25 and content_count < 10:
            print(f"  ⚠️  OVER-MASKING detected!")

if __name__ == "__main__":
    issues = debug_dynamic_masking_correctness()
    debug_adaptive_masking_logic()
    
    print(f"\\n🎯 SUMMARY:")
    if issues:
        print(f"❌ {len(issues)} issues found - these could cause training problems!")
    else:
        print(f"✅ Dynamic masking appears to be working correctly")
