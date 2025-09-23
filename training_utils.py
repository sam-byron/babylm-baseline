import torch

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


def save_checkpoint(accelerator, model, tokenizer, checkpoint_path, step, save_steps, is_main, C):
    """Save model checkpoint at specified intervals."""
    if step % save_steps != 0 or step == 0 or not is_main:
        return
    
    try:
        print(f"{C.CYAN}[Checkpoint] Starting save at step {step}...{C.RESET}")
        accelerator.save_state(output_dir=checkpoint_path, safe_serialization=False)
        accelerator.save_model(model, checkpoint_path)
        
        unwrapped = accelerator.unwrap_model(model)
        from transformer_trainer import save_model_official_way
        save_model_official_way(unwrapped, unwrapped.config, tokenizer, checkpoint_path)
        print(f"{C.GREEN}[Checkpoint] Saved model state and optimizer at step {step}{C.RESET}")
    except Exception as e:
        print(f"{C.YELLOW}[Checkpoint] Warning: failed to save at step {step}: {e}{C.RESET}")
        import traceback
        traceback.print_exc()