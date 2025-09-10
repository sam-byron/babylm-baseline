"""
Dynamic Masking Demonstration

This script demonstrates the difference between static and dynamic masking,
showing how RoBERTa-style dynamic masking multiplies the training signal.
"""

import torch
import json
from tokenizer import Tokenizer
from dynamic_collator import create_dynamic_collator
from data_loader import data_loader


def compare_masking_strategies():
    """
    Compare static vs dynamic masking to show the training signal multiplication.
    """
    print("🎭 Dynamic Masking vs Static Masking Comparison")
    print("=" * 60)
    
    # Load configuration and tokenizer
    with open("model_babylm_ltg_bert.json", "r") as f:
        config = json.load(f)
    
    tokenizer = Tokenizer.from_file(config.get("tokenizer_path"))
    
    # Sample text for demonstration
    sample_text = "The quick brown fox jumps over the lazy dog and runs through the forest."
    sample_tokens = tokenizer.encode(sample_text).ids
    
    print(f"Original text: {sample_text}")
    print(f"Tokenized: {tokenizer.decode(sample_tokens)}")
    print(f"Number of tokens: {len(sample_tokens)}")
    print()
    
    # Create batch with the same sequence repeated
    batch = [sample_tokens] * 3
    
    print("🔧 Static Masking (Traditional Approach)")
    print("-" * 40)
    
    # Test static masking (current approach)
    config_static = config.copy()
    config_static["use_dynamic_masking"] = False
    
    # For static masking, we would normally pre-compute masks
    # Let's simulate this by running the same masking multiple times
    static_results = []
    
    # Import static masking strategy
    from mlm_dataset import SpanMaskingStrategy, SubwordMaskingStrategy
    
    if config["masking_strategy"] == "span":
        masking_strategy = SpanMaskingStrategy(
            mask_p=config["mask_p"],
            tokenizer=tokenizer,
            n_special_tokens=6,
            padding_label_id=-100,
            random_p=config["random_p"],
            keep_p=config["keep_p"]
        )
    else:
        masking_strategy = SubwordMaskingStrategy(
            mask_p=config["mask_p"],
            tokenizer=tokenizer,
            n_special_tokens=6,
            padding_label_id=-100,
            random_p=config["random_p"],
            keep_p=config["keep_p"]
        )
    
    # Simulate static masking over multiple epochs
    torch.manual_seed(42)  # Fixed seed for static masking
    for epoch in range(3):
        input_ids = torch.tensor(sample_tokens)
        masked_input, labels = masking_strategy(input_ids)
        
        masked_text = tokenizer.decode(masked_input.tolist())
        masked_positions = (labels != -100).sum().item()
        total_tokens = len(sample_tokens)
        
        print(f"  Epoch {epoch + 1}: {masked_text}")
        print(f"    Masked: {masked_positions}/{total_tokens} ({masked_positions/total_tokens:.1%})")
        
        static_results.append({
            'masked_positions': masked_positions,
            'masked_input': masked_input.clone(),
            'labels': labels.clone()
        })
    
    # Check if static masking is truly static (same masks)
    static_identical = True
    for i in range(1, len(static_results)):
        if not torch.equal(static_results[0]['labels'], static_results[i]['labels']):
            static_identical = False
            break
    
    print(f"    → Static masking produces {'identical' if static_identical else 'different'} masks across epochs")
    print()
    
    print("🎭 Dynamic Masking (RoBERTa-style)")
    print("-" * 40)
    
    # Test dynamic masking
    config_dynamic = config.copy()
    config_dynamic["use_dynamic_masking"] = True
    
    dynamic_collator = create_dynamic_collator(config_dynamic, tokenizer)
    
    dynamic_results = []
    
    for epoch in range(3):
        # Each call to the collator produces different masks
        result = dynamic_collator(batch)
        
        # Show results for the first sequence in the batch
        input_ids = result['input_ids'][0]
        labels = result['labels'][0]
        attention_mask = result['attention_mask'][0]
        
        # Only consider non-padding tokens
        valid_length = attention_mask.sum().item()
        input_ids_valid = input_ids[:valid_length]
        labels_valid = labels[:valid_length]
        
        masked_text = tokenizer.decode(input_ids_valid.tolist())
        masked_positions = (labels_valid != -100).sum().item()
        
        print(f"  Epoch {epoch + 1}: {masked_text}")
        print(f"    Masked: {masked_positions}/{valid_length} ({masked_positions/valid_length:.1%})")
        
        dynamic_results.append({
            'masked_positions': masked_positions,
            'masked_input': input_ids_valid.clone(),
            'labels': labels_valid.clone()
        })
    
    # Check if dynamic masking produces different masks
    dynamic_different = False
    for i in range(1, len(dynamic_results)):
        if not torch.equal(dynamic_results[0]['labels'], dynamic_results[i]['labels']):
            dynamic_different = True
            break
    
    print(f"    → Dynamic masking produces {'different' if dynamic_different else 'identical'} masks across epochs")
    print()
    
    print("📊 Training Signal Analysis")
    print("-" * 40)
    
    # Calculate unique masked positions across epochs
    static_unique_positions = set()
    dynamic_unique_positions = set()
    
    for result in static_results:
        masked_pos = (result['labels'] != -100).nonzero(as_tuple=True)[0].tolist()
        static_unique_positions.update(masked_pos)
    
    for result in dynamic_results:
        masked_pos = (result['labels'] != -100).nonzero(as_tuple=True)[0].tolist()
        dynamic_unique_positions.update(masked_pos)
    
    print(f"Static masking:")
    print(f"  - Unique positions masked across 3 epochs: {len(static_unique_positions)}")
    print(f"  - Training signal multiplier: {len(static_unique_positions) / len(sample_tokens):.2f}x")
    
    print(f"Dynamic masking:")
    print(f"  - Unique positions masked across 3 epochs: {len(dynamic_unique_positions)}")
    print(f"  - Training signal multiplier: {len(dynamic_unique_positions) / len(sample_tokens):.2f}x")
    
    improvement = len(dynamic_unique_positions) / max(len(static_unique_positions), 1)
    print(f"  - Improvement over static: {improvement:.2f}x more diverse training signal")
    print()
    
    print("💡 Key Benefits of Dynamic Masking:")
    print("   ✅ Different masking patterns each epoch")
    print("   ✅ More diverse training signal from same data")
    print("   ✅ Better utilization of limited training data")
    print("   ✅ Reduced overfitting to specific masking patterns")
    print("   ✅ Implements RoBERTa's key innovation")
    print()
    
    return {
        'static_unique_positions': len(static_unique_positions),
        'dynamic_unique_positions': len(dynamic_unique_positions),
        'improvement_factor': improvement
    }


def demonstrate_epoch_variation():
    """
    Show how dynamic masking creates different training examples across epochs.
    """
    print("🔄 Dynamic Masking: Epoch-to-Epoch Variation")
    print("=" * 60)
    
    # Load configuration and tokenizer
    with open("model_babylm_ltg_bert.json", "r") as f:
        config = json.load(f)
    
    tokenizer = Tokenizer.from_file(config.get("tokenizer_path"))
    
    # Use a longer, more complex sentence
    sample_text = "Machine learning models benefit from diverse training data and robust optimization techniques."
    sample_tokens = tokenizer.encode(sample_text).ids
    
    print(f"Text: {sample_text}")
    print(f"Tokens: {len(sample_tokens)}")
    print()
    
    # Create dynamic collator
    config["use_dynamic_masking"] = True
    dynamic_collator = create_dynamic_collator(config, tokenizer)
    
    # Show variation across 5 epochs
    all_masked_positions = []
    
    for epoch in range(5):
        result = dynamic_collator([sample_tokens])
        
        input_ids = result['input_ids'][0]
        labels = result['labels'][0]
        attention_mask = result['attention_mask'][0]
        
        # Get valid tokens
        valid_length = attention_mask.sum().item()
        input_ids_valid = input_ids[:valid_length]
        labels_valid = labels[:valid_length]
        
        # Find masked positions
        masked_positions = (labels_valid != -100).nonzero(as_tuple=True)[0].tolist()
        all_masked_positions.append(set(masked_positions))
        
        # Create visualization
        visualization = []
        for i, (token_id, label) in enumerate(zip(input_ids_valid, labels_valid)):
            token = tokenizer.decode([token_id.item()])
            if label.item() != -100:
                if token_id.item() == tokenizer.token_to_id("[MASK]"):
                    visualization.append(f"[MASK]")
                else:
                    visualization.append(f"<{token}>")  # Random or kept token
            else:
                visualization.append(token)
        
        masked_text = "".join(visualization)
        print(f"Epoch {epoch + 1}: {masked_text}")
        print(f"    Masked positions: {sorted(masked_positions)}")
        print(f"    Count: {len(masked_positions)}/{valid_length} ({len(masked_positions)/valid_length:.1%})")
        print()
    
    # Calculate overlap between epochs
    print("📈 Masking Pattern Diversity:")
    print("-" * 40)
    
    total_unique_positions = set()
    for positions in all_masked_positions:
        total_unique_positions.update(positions)
    
    print(f"Total unique positions masked: {len(total_unique_positions)} out of {len(sample_tokens)} tokens")
    print(f"Coverage: {len(total_unique_positions)/len(sample_tokens):.1%} of all tokens seen during training")
    
    # Calculate pairwise overlaps
    overlaps = []
    for i in range(len(all_masked_positions)):
        for j in range(i + 1, len(all_masked_positions)):
            overlap = len(all_masked_positions[i].intersection(all_masked_positions[j]))
            total_in_both = len(all_masked_positions[i].union(all_masked_positions[j]))
            overlap_ratio = overlap / total_in_both if total_in_both > 0 else 0
            overlaps.append(overlap_ratio)
    
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
    print(f"Average overlap between epochs: {avg_overlap:.1%}")
    print(f"Diversity score: {1 - avg_overlap:.1%} (higher is better)")
    print()


if __name__ == "__main__":
    print("🚀 Dynamic Masking Demonstration for LTG BERT")
    print("=" * 80)
    print()
    
    # Run comparison
    results = compare_masking_strategies()
    
    print()
    
    # Show epoch variation
    demonstrate_epoch_variation()
    
    print("🎯 Recommendation:")
    if results['improvement_factor'] > 1.2:
        print("   ✅ Enable dynamic masking for your training!")
        print("   📝 Set 'use_dynamic_masking': true in model_babylm_ltg_bert.json")
        print("   🎭 This will provide RoBERTa-style training signal multiplication")
    else:
        print("   ⚖️  Both approaches show similar coverage for this example")
        print("   📝 Dynamic masking benefits increase with longer training")
    
    print()
    print("💾 To enable dynamic masking in your training:")
    print('   Edit model_babylm_ltg_bert.json and set "use_dynamic_masking": true')
    print("   The training pipeline will automatically use the dynamic collator")
    print()
    print("✅ Demo completed!")
