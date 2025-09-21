#!/usr/bin/env python3


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import json
from ltg_bert import LtgBertForMaskedLM
from ltg_bert_config import LtgBertConfig

def test_model_gradients():
    """Test if the model produces reasonable gradients"""
    
    # Load config
    with open('model_babylm_ltg_bert.json', 'r') as f:
        config = json.load(f)
    
    print("🔬 Testing Model Gradient Behavior")
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
    
    # Create a very simple batch
    batch_size = 2
    seq_len = 64  # Short sequence
    
    # Create input where only a few tokens are masked
    input_ids = torch.randint(1, config["vocab_size"], (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)  # Add attention mask
    labels = input_ids.clone()
    
    # Mask only 3 tokens per sequence (very conservative)
    mask_indices = torch.randint(0, seq_len, (batch_size, 3))
    for b in range(batch_size):
        for i in mask_indices[b]:
            labels[b, i] = input_ids[b, i]  # Keep original for loss
            input_ids[b, i] = 103  # [MASK] token
    
    # Set non-masked tokens to -100 in labels
    for b in range(batch_size):
        for i in range(seq_len):
            if i not in mask_indices[b]:
                labels[b, i] = -100
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Masked tokens: {(labels != -100).sum()} out of {batch_size * seq_len}")
    
    # Test different learning rates
    learning_rates = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    
    for lr in learning_rates:
        print(f"\n📊 Testing LR: {lr:.1e}")
        print("-" * 30)
        
        # Fresh model copy
        model_copy = LtgBertForMaskedLM(bert_config)
        model_copy.train()
        
        optimizer = torch.optim.AdamW(model_copy.parameters(), lr=lr, eps=1e-8)
        
        # Test 3 gradient steps
        for step in range(3):
            optimizer.zero_grad()
            
            try:
                outputs = model_copy(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  Step {step}: ❌ Loss is {loss.item()}")
                    break
                
                loss.backward()
                
                # Calculate gradient norm
                total_norm = 0.0
                param_count = 0
                max_grad = 0.0
                
                for name, param in model_copy.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        max_grad = max(max_grad, param.grad.data.abs().max().item())
                        param_count += 1
                
                total_norm = total_norm ** 0.5
                
                print(f"  Step {step}: Loss={loss.item():.4f} | GradNorm={total_norm:.2f} | MaxGrad={max_grad:.2e}")
                
                if total_norm > 100.0:
                    print(f"    ⚠️  HIGH gradient norm!")
                if max_grad > 1.0:
                    print(f"    ⚠️  HIGH max gradient!")
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model_copy.parameters(), 0.1)
                optimizer.step()
                
                if total_norm > 10000.0:
                    print(f"    🚨 EXPLODING gradients - stopping this LR")
                    break
                    
            except Exception as e:
                print(f"  Step {step}: ❌ Error: {e}")
                break
    
    # Test with different layer norm epsilon
    print(f"\n🧪 Testing Different LayerNorm Epsilon Values")
    print("-" * 50)
    
    eps_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    
    for eps in eps_values:
        print(f"\nTesting LayerNorm eps: {eps:.1e}")
        
        # Create config with different epsilon
        test_config = LtgBertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=3,  # Smaller for testing
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=config["intermediate_size"],
            max_position_embeddings=config.get('max_position_embeddings', config.get('block_size', 512)),
            layer_norm_eps=eps,  # Different epsilon
            attention_probs_dropout_prob=0.0,  # No dropout for testing
            hidden_dropout_prob=0.0
        )
        
        test_model = LtgBertForMaskedLM(test_config)
        test_model.train()
        
        optimizer = torch.optim.AdamW(test_model.parameters(), lr=1e-5, eps=1e-8)
        optimizer.zero_grad()
        
        try:
            outputs = test_model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            total_norm = torch.nn.utils.clip_grad_norm_(test_model.parameters(), float('inf'))
            
            print(f"  Loss={loss.item():.4f} | GradNorm={total_norm:.2f}")
            
            if total_norm > 1000.0:
                print(f"  🚨 PROBLEMATIC epsilon value!")
            elif total_norm < 10.0:
                print(f"  ✅ Good epsilon value")
            else:
                print(f"  ⚠️  Moderate gradient norm")
                
        except Exception as e:
            print(f"  ❌ Error with eps={eps}: {e}")

if __name__ == "__main__":
    test_model_gradients()
