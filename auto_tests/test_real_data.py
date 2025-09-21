#!/usr/bin/env python3


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import json
from data_loader import data_loader
from ltg_bert import LtgBertForMaskedLM
from ltg_bert_config import LtgBertConfig
from tokenizers import Tokenizer

def test_actual_dataloader():
    """Test with actual data from your training pipeline"""
    
    # Load config
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config = json.load(f)
    
    print("🔍 Testing Actual Data Pipeline")
    print("=" * 50)
    
    # Create model
    bert_config = LtgBertConfig(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        max_position_embeddings=config.get('max_position_embeddings', config.get('block_size', 512)),
        layer_norm_eps=config["layer_norm_eps"],
        attention_probs_dropout_prob=config["attention_probs_dropout_prob"],
        hidden_dropout_prob=config["hidden_dropout_prob"]
    )
    
    model = LtgBertForMaskedLM(bert_config)
    model.train()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Get actual data loaders
    try:
        # Load tokenizer
        tokenizer = Tokenizer.from_file(config["tokenizer_path"])
        
        # Get data loader 
        train_loader, val_loader, test_loader, collate_fn, total_tokens = data_loader(config, tokenizer, config["cache_path"])
        print(f"✅ Data loader created successfully")
        print(f"   Train loader length: {len(train_loader)}")
        print(f"   Total training tokens: {total_tokens:,}")
        
        # Test first few batches
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 5:  # Test only first 5 batches
                break
                
            print(f"\n🔬 Testing Batch {batch_idx + 1}")
            print("-" * 30)
            
            try:
                # Unpack batch based on your data loader format
                if isinstance(batch, dict):
                    input_ids = batch.get('input_ids')
                    attention_mask = batch.get('attention_mask') 
                    labels = batch.get('labels')
                    
                    if input_ids is None or labels is None:
                        print(f"❌ Missing required keys in batch: {batch.keys()}")
                        continue
                        
                    if attention_mask is None:
                        attention_mask = torch.ones_like(input_ids)
                        
                elif isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        input_ids, labels = batch
                        attention_mask = torch.ones_like(input_ids)
                    elif len(batch) == 3:
                        input_ids, attention_mask, labels = batch
                    else:
                        print(f"❌ Unexpected batch format: {len(batch)} elements")
                        continue
                else:
                    print(f"❌ Unexpected batch type: {type(batch)}")
                    continue
                
                print(f"  Input IDs shape: {input_ids.shape}")
                print(f"  Attention mask shape: {attention_mask.shape}")
                print(f"  Labels shape: {labels.shape}")
                
                # Check for problematic values
                if torch.isnan(input_ids).any():
                    print(f"  ❌ NaN values in input_ids!")
                    continue
                if torch.isinf(input_ids).any():
                    print(f"  ❌ Inf values in input_ids!")
                    continue
                if torch.isnan(labels).any():
                    print(f"  ❌ NaN values in labels!")
                    continue
                if torch.isinf(labels).any():
                    print(f"  ❌ Inf values in labels!")
                    continue
                
                # Check value ranges
                if input_ids.min() < 0 or input_ids.max() >= config["vocab_size"]:
                    print(f"  ⚠️  Input IDs out of range: {input_ids.min()} to {input_ids.max()}")
                
                # Count masked tokens
                masked_tokens = (labels != -100).sum()
                total_tokens = labels.numel()
                mask_ratio = masked_tokens.float() / total_tokens
                
                print(f"  Masked tokens: {masked_tokens} / {total_tokens} ({mask_ratio:.1%})")
                
                if masked_tokens == 0:
                    print(f"  ⚠️  No masked tokens in this batch!")
                    continue
                
                # Test forward pass
                optimizer.zero_grad()
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                if torch.isnan(loss):
                    print(f"  ❌ NaN loss!")
                    continue
                if torch.isinf(loss):
                    print(f"  ❌ Inf loss!")
                    continue
                
                print(f"  Loss: {loss.item():.4f}")
                
                # Test backward pass
                loss.backward()
                
                # Calculate gradient norm
                total_norm = 0.0
                max_grad = 0.0
                nan_grads = 0
                inf_grads = 0
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            nan_grads += 1
                            print(f"    ❌ NaN gradients in {name}")
                        if torch.isinf(param.grad).any():
                            inf_grads += 1
                            print(f"    ❌ Inf gradients in {name}")
                        
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        max_grad = max(max_grad, param.grad.data.abs().max().item())
                
                total_norm = total_norm ** 0.5
                
                print(f"  Gradient norm: {total_norm:.2f}")
                print(f"  Max gradient: {max_grad:.2e}")
                
                if total_norm > 1000.0:
                    print(f"  🚨 HIGH gradient norm!")
                    
                    # Investigate which parameters have high gradients
                    print(f"    Parameter-wise gradient norms:")
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2).item()
                            if param_norm > 100.0:
                                print(f"      {name}: {param_norm:.2f}")
                
                elif total_norm > 100.0:
                    print(f"  ⚠️  Moderate gradient norm")
                else:
                    print(f"  ✅ Good gradient norm")
                    
            except Exception as e:
                print(f"  ❌ Error processing batch {batch_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    test_actual_dataloader()
