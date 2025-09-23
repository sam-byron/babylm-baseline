#!/usr/bin/env python3
"""
Simple BLiMP Test Script

A focused test script to validate BLiMP benchmark computation with your BERT model.
This script tests the core functionality without extensive mocking.

Usage:
    python simple_blimp_test.py [--checkpoint_path PATH]
"""

import argparse
import json
import os
import torch
from typing import Dict, Any

# Try to import required modules
try:
    from ltg_bert import LtgBertForMaskedLM
    from ltg_bert_config import LtgBertConfig
    from tokenizers import Tokenizer
    HAS_MODEL = True
except ImportError as e:
    print(f"Warning: Model modules not available: {e}")
    HAS_MODEL = False

try:
    import lm_eval
    from lm_eval import evaluator, tasks
    from lm_eval.api.model import LM
    HAS_LM_EVAL = True
except ImportError:
    print("Warning: lm_eval not available")
    HAS_LM_EVAL = False

# Color helper
class C:
    RESET = "\033[0m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"


class SimpleBertLM(LM):
    """Simple wrapper for BERT masked language models for lm_eval."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        print(f"SimpleBertLM using device: {self.device}")
        
    def loglikelihood(self, requests):
        """Calculate log-likelihood for requests."""
        results = []
        
        for request in requests:
            try:
                # Simple tokenization - just split on spaces and map to IDs
                tokens = request.text.split()
                
                # Convert tokens to IDs
                if hasattr(self.tokenizer, '__call__'):
                    # HuggingFace-style tokenizer
                    inputs = self.tokenizer(request.text, return_tensors="pt")
                    input_ids = inputs["input_ids"].to(self.device)
                else:
                    # Manual tokenization with our tokenizer
                    token_ids = []
                    for token in tokens:
                        token_id = self.tokenizer.token_to_id(token)
                        if token_id is not None:
                            token_ids.append(token_id)
                        else:
                            token_ids.append(self.tokenizer.token_to_id("[UNK]"))
                    
                    input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
                
                # Get model prediction
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, labels=input_ids)
                    
                # Calculate log-likelihood from loss
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    # Use the model's calculated loss (negative log-likelihood per token)
                    log_likelihood = -outputs.loss.item() * input_ids.size(1)
                else:
                    # Fallback: calculate from logits
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    target_log_probs = torch.gather(log_probs, -1, input_ids.unsqueeze(-1))
                    log_likelihood = target_log_probs.sum().item()
                
                results.append((log_likelihood, False))
                
            except Exception as e:
                print(f"{C.YELLOW}Warning: Failed to process '{request.text[:50]}...': {e}{C.RESET}")
                results.append((float('-inf'), False))
                
        return results
    
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Rolling likelihood not supported")
    
    def generate_until(self, requests):
        raise NotImplementedError("Generation not supported for masked LM")


def load_model_and_tokenizer(checkpoint_path: str):
    """Load model and tokenizer from checkpoint."""
    print(f"{C.CYAN}Loading model from {checkpoint_path}...{C.RESET}")
    
    # Load config
    config = LtgBertConfig.from_pretrained(checkpoint_path)
    print(f"✓ Loaded config: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")
    
    # Load model
    model = LtgBertForMaskedLM(config)
    
    # Load weights
    weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
    if os.path.exists(weights_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(weights_path, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"{C.YELLOW}Warning: Missing keys: {len(missing)}{C.RESET}")
        if unexpected:
            print(f"{C.YELLOW}Warning: Unexpected keys: {len(unexpected)}{C.RESET}")
    else:
        print(f"{C.RED}Error: Weights file not found: {weights_path}{C.RESET}")
        return None, None
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded and moved to {device}")
    
    # Load tokenizer
    tokenizer_paths = [
        os.path.join(os.path.dirname(checkpoint_path), "..", "data", "pretrain", "wordpiece_vocab.json"),
        "/home/sam-byron/engineering/ML/playground/babylm/bnc/bl-bnc-custom-ltg-bert/data/pretrain/wordpiece_vocab.json",
        "./data/pretrain/wordpiece_vocab.json"
    ]
    
    tokenizer = None
    for tok_path in tokenizer_paths:
        if os.path.exists(tok_path):
            try:
                tokenizer = Tokenizer.from_file(tok_path)
                print(f"✓ Loaded tokenizer from {tok_path}")
                break
            except Exception as e:
                print(f"{C.YELLOW}Failed to load tokenizer from {tok_path}: {e}{C.RESET}")
    
    if tokenizer is None:
        print(f"{C.RED}Error: Could not load tokenizer from any path{C.RESET}")
        return None, None
    
    return model, tokenizer


def test_model_basic(model, tokenizer):
    """Basic test of model functionality."""
    print(f"\n{C.CYAN}Testing basic model functionality...{C.RESET}")
    
    try:
        # Test tokenization
        test_text = "the cat is big"
        device = next(model.parameters()).device
        
        if hasattr(tokenizer, '__call__'):
            inputs = tokenizer(test_text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            tokens = test_text.split()
            token_ids = [tokenizer.token_to_id(token) or tokenizer.token_to_id("[UNK]") for token in tokens]
            inputs = {"input_ids": torch.tensor([token_ids], device=device)}
        
        print(f"✓ Tokenized '{test_text}' -> {inputs['input_ids'].shape} on {device}")
        
        # Test model forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            
        if hasattr(outputs, 'logits'):
            print(f"✓ Model forward pass successful, logits shape: {outputs.logits.shape}")
        else:
            print(f"✓ Model forward pass successful, output shape: {outputs[0].shape}")
            
        return True
        
    except Exception as e:
        print(f"{C.RED}✗ Basic model test failed: {e}{C.RESET}")
        return False


def run_blimp_sample(model, tokenizer):
    """Run BLiMP evaluation on a small sample."""
    if not HAS_LM_EVAL:
        print(f"{C.YELLOW}Skipping BLiMP test - lm_eval not available{C.RESET}")
        return None
        
    print(f"\n{C.CYAN}Running BLiMP sample evaluation...{C.RESET}")
    
    try:
        # Create LM wrapper
        lm = SimpleBertLM(model, tokenizer)
        
        # Load BLiMP tasks
        blimp_tasks = tasks.get_task_dict("blimp")
        print(f"✓ Loaded {len(blimp_tasks)} BLiMP tasks")
        
        # Run on 25% subset for testing (at least 5 tasks)
        subset_size = max(5, len(blimp_tasks) // 4)
        task_names = list(blimp_tasks.keys())[:subset_size]
        sample_tasks = {name: blimp_tasks[name] for name in task_names}
        
        print(f"Running evaluation on {len(task_names)}/{len(blimp_tasks)} tasks (25% subset): {task_names[:3]}...")
        
        # Run evaluation
        results = evaluator.evaluate(lm, sample_tasks, num_fewshot=0)
        
        print(f"{C.GREEN}✓ BLiMP sample evaluation completed{C.RESET}")
        
        # Print results summary
        if "results" in results:
            print("\nSample Results:")
            for task_name, task_results in results["results"].items():
                acc = task_results.get("acc", task_results.get("accuracy", "N/A"))
                print(f"  {task_name}: {acc}")
        
        return results
        
    except Exception as e:
        print(f"{C.RED}✗ BLiMP sample evaluation failed: {e}{C.RESET}")
        import traceback
        traceback.print_exc()
        return None


def run_comprehensive_blimp(model, tokenizer):
    """Run comprehensive BLiMP evaluation (25% subset for efficiency)."""
    if not HAS_LM_EVAL:
        print(f"{C.YELLOW}Skipping comprehensive BLiMP - lm_eval not available{C.RESET}")
        return None
        
    print(f"\n{C.CYAN}Running comprehensive BLiMP evaluation (25% subset)...{C.RESET}")
    print(f"{C.YELLOW}This may take a few minutes...{C.RESET}")
    
    try:
        # Create LM wrapper
        lm = SimpleBertLM(model, tokenizer)
        
        # Load all BLiMP tasks
        blimp_tasks = tasks.get_task_dict("blimp")
        
        # Use 25% subset for faster evaluation (or at least 10 tasks for full test)
        subset_size = max(10, len(blimp_tasks) // 4)
        task_names = list(blimp_tasks.keys())[:subset_size]
        selected_tasks = {name: blimp_tasks[name] for name in task_names}
        
        print(f"✓ Running evaluation on {len(selected_tasks)}/{len(blimp_tasks)} BLiMP tasks (25% subset)")
        
        # Run evaluation on subset
        results = evaluator.evaluate(lm, selected_tasks, num_fewshot=0)
        
        print(f"{C.GREEN}✓ Comprehensive BLiMP evaluation completed{C.RESET}")
        
        return results
        
    except Exception as e:
        print(f"{C.RED}✗ Comprehensive BLiMP evaluation failed: {e}{C.RESET}")
        import traceback
        traceback.print_exc()
        return None


def save_results(results: Dict[str, Any], checkpoint_path: str):
    """Save results to JSON file."""
    if not results:
        return
        
    try:
        output_path = os.path.join(checkpoint_path, "blimp_test_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"{C.GREEN}✓ Results saved to {output_path}{C.RESET}")
        
    except Exception as e:
        print(f"{C.YELLOW}Warning: Could not save results: {e}{C.RESET}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple BLiMP benchmark test")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--sample_only", action="store_true",
                       help="Run only sample evaluation (faster)")
    
    args = parser.parse_args()
    
    if not HAS_MODEL:
        print(f"{C.RED}Error: Required model modules not available{C.RESET}")
        return 1
    
    if not os.path.exists(args.checkpoint_path):
        print(f"{C.RED}Error: Checkpoint path does not exist: {args.checkpoint_path}{C.RESET}")
        return 1
    
    print(f"{C.BOLD}{C.CYAN}🧪 Simple BLiMP Test{C.RESET}")
    print(f"{C.CYAN}==================={C.RESET}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.checkpoint_path)
    if model is None or tokenizer is None:
        return 1
    
    # Basic functionality test
    if not test_model_basic(model, tokenizer):
        return 1
    
    # BLiMP evaluation
    if args.sample_only:
        results = run_blimp_sample(model, tokenizer)
    else:
        results = run_blimp_sample(model, tokenizer)  # Always run sample first
        if results is not None:
            user_input = input(f"\n{C.CYAN}Sample test successful. Run comprehensive evaluation (25% of tasks)? (y/N): {C.RESET}")
            if user_input.lower() == 'y':
                results = run_comprehensive_blimp(model, tokenizer)
    
    # Save results
    if results:
        save_results(results, args.checkpoint_path)
    
    print(f"\n{C.BOLD}{C.GREEN}🎉 Test completed!{C.RESET}")
    return 0


if __name__ == "__main__":
    exit(main())