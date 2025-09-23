import torch
import json
import os

def generate_special_tokens_map(tokenizer, checkpoint_path, accelerator=None, C=None):
    """
    Auto-generate special_tokens_map.json for AutoTokenizer compatibility.
    
    Args:
        tokenizer: The tokenizer to extract special tokens from
        checkpoint_path: Directory where to save the special_tokens_map.json file
        accelerator: Optional accelerator instance for distributed training
        C: Optional color class for formatted output
    
    Returns:
        dict: The generated special tokens mapping
    """
    if C is None:
        # Fallback color class if none provided
        class C:
            RESET = "\033[0m"
            CYAN = "\033[36m"
    
    # Only generate on main process if using accelerator
    if accelerator is not None and not accelerator.is_main_process:
        return {}
    
    special_tokens_map = {}
    
    # Extract special tokens from the tokenizer
    vocab = tokenizer.get_vocab()
    
    # Common BERT special token patterns - adjust based on your tokenizer
    special_token_patterns = {
        'cls_token': ['[CLS]', '<cls>', '<CLS>'],
        'sep_token': ['[SEP]', '<sep>', '<SEP>'],
        'pad_token': ['[PAD]', '<pad>', '<PAD>'],
        'unk_token': ['[UNK]', '<unk>', '<UNK>'],
        'mask_token': ['[MASK]', '<mask>', '<MASK>'],
        'bos_token': ['[BOS]', '<bos>', '<BOS>', '[CLS]'],  # Often CLS serves as BOS
        'eos_token': ['[EOS]', '<eos>', '<EOS>', '[SEP]']   # Often SEP serves as EOS
    }
    
    # Find special tokens in vocabulary
    for token_type, patterns in special_token_patterns.items():
        for pattern in patterns:
            if pattern in vocab:
                special_tokens_map[token_type] = pattern
                break
    
    # Ensure directory exists
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save special_tokens_map.json
    special_tokens_path = os.path.join(checkpoint_path, "special_tokens_map.json")
    with open(special_tokens_path, "w") as f:
        json.dump(special_tokens_map, f, indent=2)
    
    print(f"{C.CYAN}Auto-generated special_tokens_map.json with: {special_tokens_map}{C.RESET}")
    
    return special_tokens_map

# ===== Simple ANSI color helper =====
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"

# Training utilities to simplify the main training loop

def log_first_batch_info(accelerator, batch, tokenizer, loss, C):
    """Log detailed information about the first batch for debugging."""
    accelerator.print(f"{C.GREEN}Starting first batch processing... Batch keys: {list(batch.keys())}{C.RESET}")
    accelerator.print(f"{C.CYAN}Batch shapes - input_ids: {batch['input_ids'].shape}, attention_mask: {batch['attention_mask'].shape}, labels: {batch['labels'].shape}{C.RESET}")
    accelerator.print(f"{C.GREEN}Model forward pass completed, loss: {loss}{C.RESET}")
    
    # Debug: Check how many tokens are being masked
    masked_tokens = (batch["labels"] != -100).sum().item()
    total_tokens = batch["labels"].numel()
    masking_ratio = masked_tokens / total_tokens
    accelerator.print(f"{C.CYAN}Masked tokens: {masked_tokens}/{total_tokens} ({masking_ratio:.1%}){C.RESET}")
    
    # Check label distribution
    unique_labels = torch.unique(batch["labels"][batch["labels"] != -100])
    accelerator.print(f"{C.CYAN}Unique masked labels: {len(unique_labels)} (vocab size: {tokenizer.get_vocab_size()}){C.RESET}")
    
    # Sample some of the actual tokens being masked
    sample_labels = batch["labels"][batch["labels"] != -100][:20]  # First 20 masked tokens
    sample_tokens = [tokenizer.id_to_token(int(label)) for label in sample_labels if int(label) < tokenizer.get_vocab_size()]
    accelerator.print(f"{C.CYAN}Sample masked tokens: {sample_tokens[:10]}{C.RESET}")
    
    # Check the full input to see vocabulary diversity
    all_input_tokens = torch.unique(batch["input_ids"])
    accelerator.print(f"{C.CYAN}Unique tokens in input_ids: {len(all_input_tokens)}/{tokenizer.get_vocab_size()}{C.RESET}")
    
    # Sample some input tokens to see what we're working with
    sample_input_ids = batch["input_ids"][0]
    num_masked_in_sample = (sample_input_ids == tokenizer.token_to_id("[MASK]")).sum().item()
    accelerator.print(f"{C.GREEN}Sample input has {num_masked_in_sample}/{len(sample_input_ids)} masked tokens ({num_masked_in_sample/len(sample_input_ids):.1%}){C.RESET}")
    
    num_unk_in_sample = (sample_input_ids == tokenizer.token_to_id("[UNK]")).sum().item()
    accelerator.print(f"{C.GREEN}Sample input has {num_unk_in_sample}/{len(sample_input_ids)} unknown tokens ({num_unk_in_sample/len(sample_input_ids):.1%}){C.RESET}")


def forward_pass(model, batch):
    """Perform model forward pass with proper attention mask conversion."""
    # Convert attention mask to bool (0 -> True for padding)
    batch["attention_mask"] = (batch["attention_mask"] == 0)
    
    prediction = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return prediction.loss


def gradient_step(accelerator, model, optimizer, scheduler, monitor, config, step, loss, is_main, C):
    """Handle gradient clipping, monitoring, and optimizer step."""
    if not accelerator.sync_gradients:
        return float(loss.detach())
    
    # Calculate gradient norm for monitoring
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    # Get current learning rate
    current_lr = scheduler.get_last_lr()[0]
    
    # Update monitor with current metrics
    monitor.update(float(loss.detach()), total_norm, current_lr, step)
    
    # Check for training issues
    spike_detected, spike_msg = monitor.check_loss_spike()
    oscillation_detected, osc_msg = monitor.check_oscillation()
    explosion_detected, exp_msg = monitor.check_gradient_explosion(config.get("max_grad_norm", 1.0))
    
    # Print warnings for issues
    if spike_detected and is_main:
        print(f"{C.YELLOW}⚠️  {spike_msg}{C.RESET}")
    if oscillation_detected and is_main:
        print(f"{C.YELLOW}⚠️  {osc_msg}{C.RESET}")
    if explosion_detected and is_main:
        print(f"{C.RED}🚨 {exp_msg}{C.RESET}")
    
    # Clip gradients
    accelerator.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
    
    # Check for NaN gradients
    has_nan = any(
        param.grad is not None and torch.isnan(param.grad).any()
        for param in model.parameters()
    )
    
    if has_nan:
        if is_main:
            print(f"{C.RED}Skipping optimizer step due to NaN gradients{C.RESET}")
        optimizer.zero_grad()
        return float(loss.detach())
    
    # Optimizer step
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    # Return gathered loss
    return accelerator.gather(loss).mean()


def update_progress_bars(batch_bar, loss_bar, monitor, loss_value, avg_loss, step, log_steps, 
                        processed_batches, epoch_start, total_batches, is_main):
    """Update progress bars with current metrics."""
    if not is_main or step % log_steps != 0:
        return
    
    batch_bar.update(log_steps)
    loss_bar.update(1)
    
    # Add monitoring stats to loss bar
    monitor_stats = monitor.get_stats()
    loss_bar.set_postfix({
        "loss": f"{loss_value:.4f}",
        "avg_loss": f"{avg_loss:.4f}",
        "monitor": f"S:{monitor.spike_count} O:{monitor.oscillation_count} E:{monitor.explosion_count}"
    })
    
    # Print detailed monitor stats every 50 steps
    if step % (log_steps * 50) == 0:
        print(f"[Monitor] {monitor_stats}")
    
    # Update batch bar with ETA
    import time
    import datetime
    elapsed = time.time() - epoch_start
    batches_per_sec = processed_batches / elapsed if elapsed > 0 else 0
    remaining = max(total_batches - processed_batches, 0)
    eta_sec = remaining / batches_per_sec if batches_per_sec > 0 else 0
    eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
    batch_bar.set_postfix({
        "batch/s": f"{batches_per_sec:.0f}",
        "ETA": eta_str,
    })

def save_model_official_way(model, model_config, tokenizer, checkpoint_path):
    """
    Save model the official HuggingFace way for compatibility with lm_eval and other tools.
    """
    print(f"{C.CYAN}Saving model the official HuggingFace way to {checkpoint_path}{C.RESET}")
    
    # Ensure directory exists
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Register for auto class - this is the key part!
    model_config.register_for_auto_class()
    model.register_for_auto_class()  # No arguments needed for models
    
    # Save model and config using HF methods
    # Use safe_serialization=False to handle shared tensors
    model.save_pretrained(checkpoint_path, safe_serialization=False)
    model_config.save_pretrained(checkpoint_path)
    
    # Generate special_tokens_map.json for AutoTokenizer compatibility
    generate_special_tokens_map(tokenizer, checkpoint_path, C=C)
    
    print(f"{C.BOLD}{C.GREEN}Model saved the official HuggingFace way! Ready for lm_eval with trust_remote_code=True{C.RESET}")

def save_checkpoint(accelerator, model, tokenizer, checkpoint_path, step, save_steps, is_main, C, epoch=None):
    """Save full checkpoint at reduced intervals - includes optimizer state for complete recovery."""
    if step % save_steps != 0 or step == 0 or not is_main:
        return
    
    try:
        print(f"{C.CYAN}[Checkpoint] Starting full save at step {step} (includes optimizer state)...{C.RESET}")
        
        # Full save with optimizer state for complete recovery
        # This is slower but ensures no learning rate/momentum loss on crash recovery
        accelerator.save_state(output_dir=checkpoint_path, safe_serialization=False)
        accelerator.save_model(model, checkpoint_path)
        
        # Also save HuggingFace compatible version for evaluation
        unwrapped = accelerator.unwrap_model(model)
        save_model_official_way(unwrapped, unwrapped.config, tokenizer, checkpoint_path)
        
        # Save training state with current step within the epoch
        if epoch is not None:
            save_training_state(checkpoint_path, epoch, step, is_main)
        
        print(f"{C.GREEN}[Checkpoint] Full checkpoint saved at step {step} - complete recovery guaranteed{C.RESET}")
        
    except Exception as e:
        print(f"{C.YELLOW}[Checkpoint] Warning: failed to save at step {step}: {e}{C.RESET}")
        import traceback
        traceback.print_exc()


def save_training_state(checkpoint_path, epoch, step=None, is_main=True):
    """Save current training state (epoch, step) to resume later."""
    if not is_main:
        return
    
    try:
        training_state = {
            "epoch": epoch,
            "step": step if step is not None else 0,
            "completed_epochs": epoch  # For clarity
        }
        
        state_file = os.path.join(checkpoint_path, "training_state.json")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        with open(state_file, 'w') as f:
            json.dump(training_state, f, indent=2)
            
    except Exception as e:
        print(f"Warning: failed to save training state: {e}")


def load_training_state(checkpoint_path):
    """Load saved training state (epoch, step) to resume from."""
    try:
        state_file = os.path.join(checkpoint_path, "training_state.json")
        
        if not os.path.exists(state_file):
            return 0, 0  # Default: start from epoch 0, step 0
            
        with open(state_file, 'r') as f:
            training_state = json.load(f)
            
        epoch = training_state.get("epoch", 0)
        step = training_state.get("step", 0)
        
        return epoch, step
        
    except Exception as e:
        print(f"Warning: failed to load training state: {e}")
        return 0, 0  # Default: start from epoch 0, step 0


def save_epoch_checkpoint(accelerator, model, tokenizer, checkpoint_path, epoch, is_main, C):
    """Save comprehensive checkpoint at the end of an epoch - includes optimizer state."""
    if not is_main:
        return
    
    try:
        print(f"{C.CYAN}[End-of-Epoch] Saving checkpoint after epoch {epoch+1}...{C.RESET}")
        
        # Always do full save at epoch boundaries for safety
        accelerator.save_state(output_dir=checkpoint_path, safe_serialization=False)
        accelerator.save_model(model, checkpoint_path)
        
        # Also save the official HuggingFace way for evaluation compatibility
        unwrapped = accelerator.unwrap_model(model)
        save_model_official_way(unwrapped, unwrapped.config, tokenizer, checkpoint_path)
        
        # Save that we completed this epoch and reset step to 0 for next epoch
        save_training_state(checkpoint_path, epoch + 1, step=0, is_main=is_main)
        
        print(f"{C.GREEN}[End-of-Epoch] Full checkpoint saved after completing epoch {epoch+1}{C.RESET}")
        
    except Exception as e:
        print(f"{C.YELLOW}[End-of-Epoch] Warning: failed to save checkpoint: {e}{C.RESET}")
        import traceback
        traceback.print_exc()


def save_final_checkpoint(accelerator, model, tokenizer, checkpoint_path, is_main, C):
    """Save final comprehensive checkpoint at the end of training."""
    if not is_main:
        return
    
    try:
        print(f"{C.CYAN}[Final] Saving final comprehensive checkpoint...{C.RESET}")
        
        # Full save with everything needed for inference and evaluation
        accelerator.save_state(output_dir=checkpoint_path, safe_serialization=False)
        accelerator.save_model(model, checkpoint_path)
        
        # Save the official HuggingFace way for evaluation compatibility
        unwrapped = accelerator.unwrap_model(model)
        save_model_official_way(unwrapped, unwrapped.config, tokenizer, checkpoint_path)
        
        print(f"{C.GREEN}[Final] Comprehensive checkpoint saved - ready for evaluation!{C.RESET}")
        
    except Exception as e:
        print(f"{C.YELLOW}[Final] Warning: failed to save final checkpoint: {e}{C.RESET}")
        import traceback
        traceback.print_exc()