#!/usr/bin/env python3

import torch
import json
import glob
from data_loader import ChunkedDataset, data_loader
from tokenizers import Tokenizer

def compare_collator_outputs():
    """Compare the actual outputs from dynamic vs static collators."""
    print("🔍 COMPARING DYNAMIC VS STATIC COLLATOR OUTPUTS")
    print("=" * 60)
    
    # Load configuration  
    with open("model_babylm_ltg_bert.json", "r") as f:
        config = json.load(f)
    
    tokenizer = Tokenizer.from_file("data/pretrain/wordpiece_vocab.json")
    chunk_paths = sorted(glob.glob("model_babylm_bert_ltg/chunk*.pt"))[:2]
    dataset = ChunkedDataset(chunk_paths, block_size=512, pad_token_id=tokenizer.token_to_id("[PAD]"))
    
    # Get a test sample
    test_samples = [dataset[i] for i in range(3)]
    
    print(f"📊 Testing with {len(test_samples)} samples")
    print(f"Sample shapes: {[s.shape for s in test_samples]}")
    
    # ============= TEST DYNAMIC COLLATOR =============
    print(f"\\n🎭 DYNAMIC COLLATOR OUTPUT:")
    print("-" * 40)
    
    config_dynamic = config.copy()
    config_dynamic["use_dynamic_masking"] = True
    
    train_loader, val_loader, test_loader, dynamic_collator, total_tokens = data_loader(config_dynamic, tokenizer, config["cache_path"])
    
    dynamic_batch = dynamic_collator(test_samples)
    
    print(f"Dynamic batch keys: {list(dynamic_batch.keys())}")
    print(f"Dynamic input_ids shape: {dynamic_batch['input_ids'].shape}")
    print(f"Dynamic labels shape: {dynamic_batch['labels'].shape}")
    print(f"Dynamic attention_mask shape: {dynamic_batch['attention_mask'].shape}")
    
    # Check content
    d_input = dynamic_batch['input_ids']
    d_labels = dynamic_batch['labels']  
    d_attention = dynamic_batch['attention_mask']
    
    print(f"Dynamic masked positions (sample 0): {(d_labels[0] != -100).sum().item()}")
    print(f"Dynamic [MASK] tokens (sample 0): {(d_input[0] == tokenizer.token_to_id('[MASK]')).sum().item()}")
    print(f"Dynamic attention sum (sample 0): {d_attention[0].sum().item()}")
    
    # ============= TEST STATIC COLLATOR =============
    print(f"\\n📌 STATIC COLLATOR OUTPUT:")
    print("-" * 40)
    
    config_static = config.copy()
    config_static["use_dynamic_masking"] = False
    
    train_loader_static, val_loader_static, test_loader_static, static_collator, total_tokens_static = data_loader(config_static, tokenizer, config["cache_path"])
    
    static_batch = static_collator(test_samples)
    
    print(f"Static batch keys: {list(static_batch.keys())}")
    print(f"Static input_ids shape: {static_batch['input_ids'].shape}")
    print(f"Static labels shape: {static_batch['labels'].shape}")
    print(f"Static attention_mask shape: {static_batch['attention_mask'].shape}")
    
    # Check content
    s_input = static_batch['input_ids']
    s_labels = static_batch['labels']
    s_attention = static_batch['attention_mask']
    
    print(f"Static masked positions (sample 0): {(s_labels[0] != -100).sum().item()}")
    print(f"Static [MASK] tokens (sample 0): {(s_input[0] == tokenizer.token_to_id('[MASK]')).sum().item()}")
    print(f"Static attention sum (sample 0): {s_attention[0].sum().item()}")
    
    # ============= DETAILED COMPARISON =============
    print(f"\\n🔍 DETAILED COMPARISON:")
    print("=" * 40)
    
    # Compare shapes
    shape_match = (d_input.shape == s_input.shape and 
                   d_labels.shape == s_labels.shape and 
                   d_attention.shape == s_attention.shape)
    print(f"Shapes match: {shape_match}")
    
    # Compare attention masks (should be identical)
    attention_identical = torch.equal(d_attention, s_attention)
    print(f"Attention masks identical: {attention_identical}")
    
    if not attention_identical:
        print(f"  ⚠️  CRITICAL: Attention masks differ!")
        diff_positions = (d_attention != s_attention).nonzero()[:5]  # First 5 differences
        print(f"  First few differences: {diff_positions.tolist()}")
        
        # Show specific differences
        for pos in diff_positions[:3]:
            batch, seq = pos[0].item(), pos[1].item()
            print(f"    Position [{batch}, {seq}]: Dynamic={d_attention[batch, seq].item()}, Static={s_attention[batch, seq].item()}")
    
    # Compare masking patterns
    print(f"\\n🎭 MASKING PATTERN ANALYSIS:")
    for i in range(min(3, len(test_samples))):
        d_masked = (d_labels[i] != -100).sum().item()
        s_masked = (s_labels[i] != -100).sum().item()
        
        d_mask_tokens = (d_input[i] == tokenizer.token_to_id('[MASK]')).sum().item()
        s_mask_tokens = (s_input[i] == tokenizer.token_to_id('[MASK]')).sum().item()
        
        print(f"Sample {i}:")
        print(f"  Dynamic: {d_masked} masked ({d_mask_tokens} [MASK])")
        print(f"  Static:  {s_masked} masked ({s_mask_tokens} [MASK])")
        print(f"  Difference: {abs(d_masked - s_masked)} positions")
    
    # Check for potential issues
    issues = []
    
    if not shape_match:
        issues.append("❌ SHAPE MISMATCH: Batch shapes differ between dynamic and static")
    
    if not attention_identical:
        issues.append("❌ ATTENTION MISMATCH: Attention masks differ (padding issue?)")
    
    # Check for extreme differences in masking
    total_d_masked = sum((d_labels[i] != -100).sum().item() for i in range(len(test_samples)))
    total_s_masked = sum((s_labels[i] != -100).sum().item() for i in range(len(test_samples)))
    
    if abs(total_d_masked - total_s_masked) > len(test_samples) * 10:  # >10 tokens difference per sample
        issues.append(f"❌ MASKING DIFFERENCE: Large masking difference ({total_d_masked} vs {total_s_masked})")
    
    print(f"\\n🚨 ISSUES FOUND:")
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"  ✅ No major differences detected")
    
    # ============= LOSS SIMULATION =============
    print(f"\\n🔥 LOSS SIMULATION:")
    print("-" * 25)
    
    from torch.nn import CrossEntropyLoss
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    
    # Simulate random model predictions
    vocab_size = config["vocab_size"]
    batch_size, seq_len = d_input.shape
    
    # Same predictions for both
    predictions = torch.randn(batch_size, seq_len, vocab_size)
    
    d_loss = loss_fn(predictions.view(-1, vocab_size), d_labels.view(-1))
    s_loss = loss_fn(predictions.view(-1, vocab_size), s_labels.view(-1))
    
    print(f"Dynamic loss: {d_loss.item():.4f}")
    print(f"Static loss:  {s_loss.item():.4f}")
    print(f"Loss difference: {abs(d_loss.item() - s_loss.item()):.4f}")
    
    if abs(d_loss.item() - s_loss.item()) > 0.1:
        issues.append("❌ LOSS DIFFERENCE: Significant loss computation difference")
    
    return issues

if __name__ == "__main__":
    issues = compare_collator_outputs()
    
    print(f"\\n🎯 FINAL DIAGNOSIS:")
    if issues:
        print(f"❌ Found {len(issues)} potential issues:")
        for issue in issues:
            print(f"  {issue}")
        print(f"\\n💡 The loss plateau is likely caused by one of these differences!")
    else:
        print(f"✅ No significant differences found between dynamic and static collators")
        print(f"💭 The issue may be elsewhere - check training loop, optimizer, or data loading")
