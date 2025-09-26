import os
import torch
import json
import math
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, get_scheduler
from transformers import BertForMaskedLM, BertConfig as TransformersBertConfig, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.local_sgd import LocalSGD
# from iter_data_loader import iter_data_loader
from data_loader import data_loader
import argparse
import time
import torch.distributed as dist
import traceback
from datetime import timedelta
# import torch.distributed as dist
from config import BertConfig

from ltg_bert import LtgBertForMaskedLM
from ltg_bert_config import LtgBertConfig
from save_config import save_ltg_bert_config

from tokenizer import Tokenizer
import shutil

from lamb import Lamb

import torch.nn.functional as F

# # Set CUDA_HOME to avoid DeepSpeed compilation issues
# import os
# os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.9')
# # Disable DeepSpeed compilation since we don't have nvcc
# os.environ.setdefault('DS_BUILD_OPS', '0')
# os.environ.setdefault('DS_SKIP_CUDA_CHECK', '1')

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

class EmptyDataset(Dataset):
    def __len__(self): 
        return 0
    def __getitem__(self, idx): 
        raise IndexError

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
    
    print(f"{C.BOLD}{C.GREEN}Model saved the official HuggingFace way! Ready for lm_eval with trust_remote_code=True{C.RESET}")

# @torch.compile
def build_model(checkpoint_path, config_file):
    # Use the proper config saving function from save_config.py
    save_ltg_bert_config("./configs/base.json", checkpoint_path)
    
    # Load the saved config
    transformers_config = LtgBertConfig.from_pretrained(checkpoint_path)
    print(f"{C.CYAN}Loaded and saved LtgBertConfig to {checkpoint_path}{C.RESET}")
    
    model = LtgBertForMaskedLM(transformers_config)

    # Copy source files with proper naming convention
    model_type = transformers_config.model_type
    # copy ltg_bert.py and ltg_bert_config.py to the checkpoint directory
    shutil.copy2("ltg_bert.py", checkpoint_path)
    shutil.copy2("ltg_bert_config.py", checkpoint_path)
    # copy config_file to the checkpoint directory using same name
    shutil.copy2(config_file, checkpoint_path)

    print(f"{C.GREEN}Model initialized and saved to {checkpoint_path}{C.RESET}")

    return model

import time
import datetime

class TrainingMonitor:
    """Monitor training stability and detect issues"""
    def __init__(self, window_size=20, spike_threshold=0.5, oscillation_threshold=0.1):
        self.window_size = window_size
        self.spike_threshold = spike_threshold  # Relative increase that counts as a spike
        self.oscillation_threshold = oscillation_threshold  # StdDev threshold for oscillation detection
        
        self.loss_history = []
        self.grad_norm_history = []
        self.lr_history = []
        
        self.spike_count = 0
        self.oscillation_count = 0
        self.explosion_count = 0
        
    def update(self, loss, grad_norm, lr, step):
        """Update monitoring with new values"""
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        self.lr_history.append(lr)
        
        # Keep only recent history
        if len(self.loss_history) > self.window_size * 2:
            self.loss_history = self.loss_history[-self.window_size:]
            self.grad_norm_history = self.grad_norm_history[-self.window_size:]
            self.lr_history = self.lr_history[-self.window_size:]
    
    def check_loss_spike(self):
        """Detect sudden loss increases"""
        if len(self.loss_history) < 3:
            return False, ""
            
        recent_loss = self.loss_history[-1]
        prev_loss = self.loss_history[-2]
        
        if prev_loss > 0 and (recent_loss - prev_loss) / prev_loss > self.spike_threshold:
            self.spike_count += 1
            return True, f"Loss spike detected: {prev_loss:.4f} → {recent_loss:.4f} (+{((recent_loss-prev_loss)/prev_loss)*100:.1f}%)"
        return False, ""
    
    def check_oscillation(self):
        """Detect loss oscillation in recent window"""
        if len(self.loss_history) < self.window_size:
            return False, ""
            
        recent_losses = self.loss_history[-self.window_size:]
        mean_loss = sum(recent_losses) / len(recent_losses)
        variance = sum((l - mean_loss) ** 2 for l in recent_losses) / len(recent_losses)
        std_dev = variance ** 0.5
        
        if mean_loss > 0 and (std_dev / mean_loss) > self.oscillation_threshold:
            self.oscillation_count += 1
            return True, f"Loss oscillation detected: std/mean = {(std_dev/mean_loss)*100:.1f}% (threshold: {self.oscillation_threshold*100:.1f}%)"
        return False, ""
    
    def check_gradient_explosion(self, max_grad_norm=1.0):
        """Detect gradient explosion"""
        if len(self.grad_norm_history) < 2:
            return False, ""
            
        recent_grad = self.grad_norm_history[-1]
        if recent_grad >= max_grad_norm * 0.98:  # Only trigger very close to actual clipping
            self.explosion_count += 1
            return True, f"Gradient norm near explosion: {recent_grad:.4f} (max: {max_grad_norm})"
        return False, ""
    
    def get_stats(self):
        """Get monitoring statistics"""
        if not self.loss_history:
            return "No data yet"
            
        recent_loss = self.loss_history[-1]
        recent_grad = self.grad_norm_history[-1] if self.grad_norm_history else 0
        recent_lr = self.lr_history[-1] if self.lr_history else 0
        
        return (f"Recent: loss={recent_loss:.4f}, grad_norm={recent_grad:.4f}, lr={recent_lr:.2e} | "
                f"Issues: spikes={self.spike_count}, oscillations={self.oscillation_count}, explosions={self.explosion_count}")

def train_loop(
    accelerator,
    model,
    tokenizer,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    config,
    checkpoint_path,
    start_epoch,
):
    # Initialize training monitor with config values
    monitoring_config = config.get("monitoring", {})
    monitor = TrainingMonitor(
        window_size=monitoring_config.get("window_size", 20),
        spike_threshold=monitoring_config.get("spike_threshold", 0.3),
        oscillation_threshold=monitoring_config.get("oscillation_threshold", 0.15)
    )
    
    # save tokenizer
    # Save tokenizer manually since our custom tokenizer doesn't have save_pretrained
    os.makedirs(checkpoint_path, exist_ok=True)  # Ensure directory exists
    tokenizer_dest = os.path.join(checkpoint_path, "tokenizer.json")
    tokenizer.save(tokenizer_dest)
    # Use accelerator.print to avoid N prints across ranks
    accelerator.print(f"{C.BOLD}{C.CYAN}Starting training loop from epoch {start_epoch} with config: {config}{C.RESET}")
    num_epochs = config["num_epochs"]
    if start_epoch >= num_epochs:
        accelerator.print(f"{C.GREEN}Training already completed. Exiting.{C.RESET}")
        return

    # 1) Compute number of steps and total tokens for ETA
    steps_per_epoch = len(train_loader)
    save_steps = max(1, steps_per_epoch // 5)
    print(f"{C.BLUE}Total steps per epoch: {steps_per_epoch}, save every {save_steps} steps{C.RESET}")
    # val_log_steps = max(1, steps_per_epoch // 200)
    # print(f"{C.BLUE}Validation every {val_log_steps} steps{C.RESET}")

    for epoch in range(start_epoch, num_epochs):
        print(f"{C.BOLD}{C.MAGENTA}Starting epoch {epoch+1}/{num_epochs}{C.RESET}")
        model.train()
        total_loss = 0.0
        processed_batches = 0
        epoch_start = time.time()
        log_steps = 1
    
        # DEBUG
        # train_loader = val_loader

        # Only the main process draws the bars
        is_main = accelerator.is_main_process
        total_batches = len(train_loader)
        # Token‐based progress bar
        batch_bar = tqdm(
            total=total_batches,
            unit="batches",
            unit_scale=True,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=False,
            position=0,
            disable=not is_main,
        )
        # Loss bar (steps)
        loss_bar = tqdm(
            total=steps_per_epoch,
            desc="Loss",
            leave=False,
            position=1,
            bar_format="{l_bar}{bar}| {postfix}",
            disable=not is_main,
        )

        for step, batch in enumerate(train_loader):
            if step == 0:
                accelerator.print(f"{C.GREEN}Starting first batch processing... Batch keys: {list(batch.keys())}{C.RESET}")
                accelerator.print(f"{C.CYAN}Batch shapes - input_ids: {batch['input_ids'].shape}, attention_mask: {batch['attention_mask'].shape}, labels: {batch['labels'].shape}{C.RESET}")
            
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    if step == 0:
                        accelerator.print(f"{C.YELLOW}About to run model forward pass...{C.RESET}")
                    # print(f"[Epoch {epoch+1}] Processing batch {step+1}/{total_batches} (size={len(batch)})")
                    # convert batch["attention_mask"] to bool
                    batch["attention_mask"] = (batch["attention_mask"] == 0)
                    target_ids = batch["labels"].t()
                    prediction = model(
                        batch["input_ids"].t(),
                        batch["attention_mask"],
                        target_ids,
                    )
                    target_ids = target_ids.flatten()
                    target_ids = target_ids[target_ids != -100]
                    loss = F.cross_entropy(prediction[0], target_ids)
                    if step == 0:
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
                        
                        # Print shape of batch
                        accelerator.print(f"{C.CYAN}Batch input_ids shape: {batch['input_ids'].shape}{C.RESET}")
                        # Sample some input tokens to see what we're working with
                        sample_input_ids = batch["input_ids"][0] # First 50 tokens from batch
                        # Print proportion of masked tokens in sample_input_ids
                        num_masked_in_sample = (sample_input_ids == tokenizer.token_to_id("[MASK]")).sum().item()
                        accelerator.print(f"{C.GREEN}Sample input has {num_masked_in_sample}/{len(sample_input_ids)} masked tokens ({num_masked_in_sample/len(sample_input_ids):.1%}){C.RESET}")
                        # Print number of unknown tokens in sample_input_ids
                        num_unk_in_sample = (sample_input_ids == tokenizer.token_to_id("[UNK]")).sum().item()
                        # Print labels for sample_input_ids
                        # sample_labels = batch["labels"][0]  # Labels for the first sequence in the batch
                        # accelerator.print(f"{C.CYAN}Sample labels: {sample_labels}{C.RESET}")
                        accelerator.print(f"{C.GREEN}Sample input has {num_unk_in_sample}/{len(sample_input_ids)} unknown tokens ({num_unk_in_sample/len(sample_input_ids):.1%}){C.RESET}")
                        # sample_input_tokens = [tokenizer.id_to_token(int(token_id)) for token_id in sample_input_ids if int(token_id) < tokenizer.get_vocab_size()]
                        sample_input_tokens = [
                            tokenizer.id_to_token(int(token_id)) if int(token_id) < tokenizer.get_vocab_size()
                            else "[UNK]"
                            for token_id in sample_input_ids
                        ]
                        accelerator.print(f"{C.CYAN}Sample input tokens: {sample_input_tokens}{C.RESET}")
                    # loss = outputs.loss
                    # Ensure all ranks participate in backward/all-reduce: replace non-finite with 0
                    if not torch.isfinite(loss):
                        if is_main:
                            print(f"{C.YELLOW}Warning: non-finite loss at step {step} in epoch {epoch+1}. Replacing with 0.{C.RESET}")
                        loss = torch.zeros_like(loss)
                
                if step == 0:
                    accelerator.print(f"{C.YELLOW}About to call accelerator.backward()...{C.RESET}")
                accelerator.backward(loss)
                if step == 0:
                    accelerator.print(f"{C.GREEN}Backward pass completed{C.RESET}")
                
                if accelerator.sync_gradients:
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
                    
                    if step == 0:
                        accelerator.print(f"{C.YELLOW}Syncing gradients (skipping clipping for debugging)...{C.RESET}")

                    if (step + 1) % config.get("grad_accum", 8) == 0:
                        accelerator.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
                # Only step optimizer/scheduler when we're at an accumulation boundary
                if accelerator.sync_gradients:
                    if step == 0:
                        accelerator.print(f"{C.YELLOW}Stepping optimizer and scheduler...{C.RESET}")
                    try:
                        if step == 0:
                            accelerator.print(f"{C.CYAN}About to step optimizer...{C.RESET}")
                            # Check for any NaN parameters before optimizer step
                            has_nan = False
                            for name, param in model.named_parameters():
                                if param.grad is not None and torch.isnan(param.grad).any():
                                    accelerator.print(f"{C.RED}NaN gradient found in {name}{C.RESET}")
                                    has_nan = True
                                    break
                            if has_nan:
                                accelerator.print(f"{C.RED}Skipping optimizer step due to NaN gradients{C.RESET}")
                                optimizer.zero_grad()
                                continue
                        optimizer.step()
                        if step == 0:
                            accelerator.print(f"{C.CYAN}Optimizer step completed, about to step scheduler...{C.RESET}")
                        scheduler.step()
                        if step == 0:
                            accelerator.print(f"{C.CYAN}Scheduler step completed, about to zero gradients...{C.RESET}")
                        optimizer.zero_grad()
                        if step == 0:
                            accelerator.print(f"{C.GREEN}Optimizer step completed{C.RESET}")
                    except Exception as e:
                        if step == 0:
                            accelerator.print(f"{C.RED}Optimizer/scheduler step failed: {e}{C.RESET}")
                        raise e
                
                # Gather and average loss across GPUs
                if accelerator.sync_gradients:
                    loss_value = accelerator.gather(loss).mean()
                else:
                    loss_value = loss.detach()
                # Use local loss value only; avoid cross-rank collectives in the hot path
                # loss_value = float(loss.detach())
                total_loss += loss_value

                if step % log_steps == 0:
                    avg_loss = total_loss / (step + 1)

                    processed_batches += log_steps
                    if is_main:
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
                            accelerator.print(f"{C.CYAN}[Monitor] {monitor_stats}{C.RESET}")
                        
                        elapsed = time.time() - epoch_start
                        batches_per_sec = processed_batches / elapsed if elapsed > 0 else 0
                        remaining = max(total_batches - processed_batches, 0)
                        eta_sec = remaining / batches_per_sec if batches_per_sec > 0 else 0
                        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                        batch_bar.set_postfix({
                            "batch/s": f"{batches_per_sec:.0f}",
                            "ETA": eta_str,
                        })
                if step % save_steps == 0 and step > 0:
                    # Save the model state and optimizer only on the main process
                    if is_main:
                        try:
                            print(f"{C.CYAN}[Checkpoint] Starting save at step {step}...{C.RESET}")
                            print(f"{C.CYAN}[Checkpoint] accelerator.save_state to {checkpoint_path}{C.RESET}")
                            accelerator.save_state(output_dir=checkpoint_path, safe_serialization=False)
                            print(f"{C.CYAN}[Checkpoint] accelerator.save_model to {checkpoint_path}{C.RESET}")
                            accelerator.save_model(model, checkpoint_path)
                            unwrapped = accelerator.unwrap_model(model)
                            print(f"{C.CYAN}[Checkpoint] Saving model the official HuggingFace way...{C.RESET}")
                            # Use the official saving method
                            save_model_official_way(unwrapped, unwrapped.config, tokenizer, checkpoint_path)
                            print(f"{C.GREEN}[Checkpoint] Saved model state and optimizer at step {step}{C.RESET}")
                        except Exception as e:
                            print(f"{C.YELLOW}[Checkpoint] Warning: failed to save at step {step}: {e}{C.RESET}")
                            import traceback
                            traceback.print_exc()
        
        batch_bar.close()
        loss_bar.close()

        # End‑of‑epoch checkpoint (main-only, with barriers)
        # accelerator.wait_for_everyone()
        # if accelerator.is_main_process:
        try:
            accelerator.save_state(output_dir=checkpoint_path, safe_serialization=False)
            unwrapped = accelerator.unwrap_model(model)
            # Save the model the official HuggingFace way
            save_model_official_way(unwrapped, unwrapped.config, tokenizer, checkpoint_path)
            print(f"{C.GREEN}[Checkpoint] Saved model state and optimizer{C.RESET}")
        except Exception as e:
            print(f"{C.YELLOW}[Checkpoint] Warning: failed to save: {e}{C.RESET}")
        
        # Run validation only on main process with unprepared loader
        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            seen = 0
            # randomize the validation loader to sample different batches each time
            # shuffled_val_loader = torch.utils.data.RandomSampler(val_loader)
            for val_batch in tqdm(val_loader, desc="Validating (sample)", disable=not is_main):
                val_batch["attention_mask"] = (val_batch["attention_mask"] == 0)
                val_target_ids = val_batch["labels"].t()
                val_outputs = model(
                    val_batch["input_ids"].t(),
                    val_batch["attention_mask"],
                    val_target_ids,
                )
                val_target_ids = val_target_ids.flatten()
                val_target_ids = val_target_ids[val_target_ids != -100]
                val_loss = F.cross_entropy(val_outputs[0], val_target_ids)
                total_val_loss += float(val_loss.detach())
                seen += 1
            if seen > 0:
                print(f"{C.CYAN}[End-epoch] step {step}: avg_val_loss={total_val_loss/seen:.4f} over {seen} batches{C.RESET}")
                
        # Print epoch monitoring summary
        if is_main:
            epoch_monitor_stats = monitor.get_stats()
            print(f"{C.BOLD}{C.GREEN}[Epoch {epoch+1} Summary] {epoch_monitor_stats}{C.RESET}")
            if monitor.spike_count > 0 or monitor.oscillation_count > 0 or monitor.explosion_count > 0:
                print(f"{C.YELLOW}⚠️  Training stability issues detected this epoch. Consider reducing learning rate if issues persist.{C.RESET}")
                
        model.train()
    
        # Wait for main process to finish validation before continuing
        # accelerator.wait_for_everyone()
        # accelerator.wait_for_everyone()


def main():
    print(torch._dynamo.list_backends())
    
    # 2) pin this process to the correct CUDA device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # grab LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT from torchrun/accelerate launch
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))

    parser = argparse.ArgumentParser(description="Chatbot Training Script with Accelerate")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config_path, "r") as config_file:
        config = json.load(config_file)

    # Add gradient accumulation and mixed precision - more conservative for custom models
    accelerator = Accelerator(
        mixed_precision="bf16",  # Use bf16 for better stability and performance
        # mixed_precision="fp16",  # Use fp16 instead of bf16 for better compatibility
        gradient_accumulation_steps=config["grad_accum"],
        dataloader_config=DataLoaderConfiguration(
            even_batches=True,
            split_batches=False,  # Don't split batches across devices
        ),
        # Add more conservative settings for custom models
        # dispatch_batches=False,  # Disable batch dispatching
    )

    checkpoint_path = os.path.join(config["cache_path"], "checkpoint")
    tokenizer = Tokenizer.from_file(config.get("tokenizer_path"))

    print(f"{C.CYAN}Creating data loaders...{C.RESET}")
    train_loader, val_loader, test_loader, collate_fn, total_tokens_train = data_loader(config, tokenizer, config["cache_path"])
    print(f"{C.GREEN}Data loaders created successfully{C.RESET}")
    # Build the GPT-2 model from scratch based on our config
    model = build_model(checkpoint_path, args.config_path)
    
    # # Enable gradient checkpointing for memory efficiency but comes at a speed penalty of 15-20%s
    # if hasattr(model, 'gradient_checkpointing_enable'):
    #     model.gradient_checkpointing_enable()
    #     print(f"{C.BLUE}Enabled gradient checkpointing for memory efficiency{C.RESET}")

    # if accelerator.is_main_process:
    #     print(f"{C.BLUE}LR base={base_lr:.2e} scaled={scaled_lr:.2e} (effective_bs={effective_bs}){C.RESET}")

    # --- Effective batch & conservative late-stage scaling ---
    base_lr    = float(config.get("learning_rate", 1e-4))    # sensible default for BERT-base-ish late training
    grad_accum = int(config.get("grad_accum", 1))
    world_size = getattr(accelerator, "num_processes", int(os.environ.get("WORLD_SIZE", 1)))
    effective_bs = int(config["batch_size"]) * max(1, grad_accum) * max(1, world_size)

    # Sub-linear scaling (AdamW-friendly). Disable by setting alpha=0.0
    alpha = float(config.get("lr_scale_alpha", 0.5))  # 0.5–0.7 work well; or 0.0 to disable scaling
    scale = (effective_bs / 256.0) ** alpha
    peak_cap = float(config.get("peak_lr_cap", 1e-4))  # late-stage cap; avoids going to 3e-4
    scaled_lr = max(1e-5, min(base_lr * scale, peak_cap))

    if accelerator.is_main_process:
        print(f"{C.BLUE}LR base={base_lr:.2e} scaled={scaled_lr:.2e} "
            f"(effective_bs={effective_bs}, alpha={alpha}){C.RESET}")

    # --- Build AdamW param groups (no decay for LN & bias; optional: embeddings) ---
    no_decay_keys = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay_params, nodecay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in no_decay_keys):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    weight_decay = float(config.get("weight_decay", 0.01))
    # Use a *separate* optimizer eps; don't reuse layer_norm_eps
    optim_eps = float(config.get("optimizer_eps", 1e-8))
    betas = (0.9, 0.999)

    # Prefer fused AdamW if available (PyTorch 2+)
    use_fused = bool(config.get("use_fused_adamw", True))
    adamw_cls = torch.optim.AdamW
    if use_fused and hasattr(torch.optim, "adamw") and "fused" in adamw_cls.__init__.__code__.co_varnames:
        pass  # torch.optim.AdamW supports fused=True in recent PyTorch
    param_groups = [
        {"params": decay_params,   "weight_decay": weight_decay, "lr": scaled_lr},
        {"params": nodecay_params, "weight_decay": 0.0,          "lr": scaled_lr},
    ]
    optimizer = adamw_cls(param_groups, lr=scaled_lr, betas=betas, eps=optim_eps,
                        fused=use_fused if "fused" in adamw_cls.__init__.__code__.co_varnames else False)

    # --- Cosine schedule with warmup as a fraction of *optimizer updates* ---
    # Make sure you compute these with the *true* number of batches and grad_accum
    num_train_batches = len(train_loader)
    updates_per_epoch = math.ceil(num_train_batches / max(1, grad_accum))
    already_done_updates = int(config.get("global_update_step", 0))  # set from checkpoint if resuming
    total_updates = int(config.get("num_epochs", 30)) * updates_per_epoch - already_done_updates

    warmup_prop = float(config.get("warmup_steps_proportion", 0.06))
    warmup_updates = max(1, int(warmup_prop * total_updates))

    min_lr = float(config.get("min_lr", 1e-5))
    floor_ratio = min_lr / float(scaled_lr)

    def lr_lambda(step):
        # step is the *optimizer* update index (after grad_accum)
        if step < warmup_updates:
            return (step + 1) / warmup_updates
        t = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        # cosine from 1.0 -> floor_ratio
        return floor_ratio + 0.5 * (1.0 - floor_ratio) * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Gradient clipping: engage earlier to stop "near explosion" warnings ---
    max_grad_norm = float(config.get("max_grad_norm", 1.0))  # ↓ from 1.5
    
    if config.get("optimizer", "adamw") == "lamb":
        print(f"{C.BLUE}Using LAMB optimizer with layer-wise weight decay{C.RESET}")
        low_decay_params = list(model.embedding.named_parameters()) + list(model.transformer.named_parameters())
        high_decay_params = list(model.classifier.named_parameters())
        no_decay = ['bias', 'layer_norm', 'embedding']
        decay_params = [(n, p) for n, p in low_decay_params if not any(nd in n for nd in no_decay) and "word_embedding" not in n]
        no_decay_params = [(n, p) for n, p in low_decay_params + high_decay_params if any(nd in n for nd in no_decay) and "word_embedding" not in n]
        high_decay_params = [(n, p) for n, p in high_decay_params if not any(nd in n for nd in no_decay)]
        optimizer_grouped_parameters = [
        {'params': [p for _, p in decay_params], 'weight_decay': config.get("weight_decay", 0.01)},
        {'params': [p for _, p in high_decay_params], 'weight_decay': config.get("head_weight_decay", 0.01)},
        {'params': [p for _, p in no_decay_params], 'weight_decay': 0.0}
        ]
        optimizer = Lamb(
            optimizer_grouped_parameters,
            config.get("learning_rate", 1e-2),
            betas=(0.9, 0.98),
            eps=1e-8,
        )
   
    # Prepare model, optimizer, and train/test loaders. Leave val_loader unprepared for main-only eval.
    model, optimizer, train_loader, test_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader, val_loader
    )
    # Keep val_loader unprepared to avoid NCCL timeouts during validation

    if accelerator.is_main_process:
        # —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        # Print model architecture, parameter counts, and full size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        total_mb = total_bytes / (1024 ** 2)
        print(f"{C.BOLD}{C.CYAN}===== Model Summary ====={C.RESET}", flush=True)
        # Colorize the full multi-line repr of the model
        print(f"{C.DIM}{C.RED}{model}{C.RESET}", flush=True)
        print(f"{C.CYAN}Total parameters:     {total_params:,}{C.RESET}", flush=True)
        print(f"{C.CYAN}Trainable parameters: {trainable_params:,}{C.RESET}", flush=True)
        print(f"{C.CYAN}Approx. model size:   {total_mb:.2f} MB{C.RESET}", flush=True)
        print(f"{C.BOLD}{C.CYAN}=========================={C.RESET}", flush=True)
        # —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    
    # Scheduler with auto warmup fraction if not provided
    # total_steps = config["num_epochs"] * len(train_loader)
    # warmup_cfg = int(config.get("warmup_steps_proportion", 0))
    # # warmup_steps = warmup_cfg if warmup_cfg > 0 else max(1, int(0.06 * total_steps))
    # if accelerator.is_main_process and warmup_cfg <= 0:
    #     print(f"{C.BLUE}Using warmup_steps_proportion={warmup_cfg}% of {total_steps} steps{C.RESET}")

    # scheduler = get_scheduler(
    #     "cosine",
    #     optimizer=optimizer,
    #     num_warmup_steps=warmup_cfg*total_steps,
    #     num_training_steps=total_steps,
    # )

    # Ensure the scheduler is part of the checkpoint state
    accelerator.register_for_checkpointing(scheduler)

    if checkpoint_path and os.path.isdir(checkpoint_path) and os.listdir(checkpoint_path):
        try:
            accelerator.print(f"{C.CYAN}Loading checkpoint from {checkpoint_path}{C.RESET}")
            accelerator.load_state(checkpoint_path)
            accelerator.print(f"{C.GREEN}Checkpoint load completed.{C.RESET}")
        except Exception as e:
            accelerator.print(
                f"{C.YELLOW}Warning: could not fully load accelerator state from {checkpoint_path}: {e}. "
                f"Will continue; optimizer/scheduler may start fresh.{C.RESET}"
            )
            # Fallback: try to restore model weights only if saved via save_pretrained
            try:
                unwrapped = accelerator.unwrap_model(model)
                weights_path = os.path.join(checkpoint_path, "pytorch_model.bin")
                if os.path.isfile(weights_path):
                    state_dict = torch.load(weights_path, map_location="cpu")
                    missing, unexpected = unwrapped.load_state_dict(state_dict, strict=False)
                    accelerator.print(
                        f"{C.YELLOW}Loaded model weights only (missing={len(missing)}, unexpected={len(unexpected)}).{C.RESET}"
                    )
            except Exception as ee:
                accelerator.print(f"{C.RED}Model-only weight restore failed: {ee}{C.RESET}")
    else:
        accelerator.print(f"{C.YELLOW}No checkpoint found at {checkpoint_path}, starting from scratch.{C.RESET}")

    start_epoch = 0
    # train_loader = test_loader
    train_loop(accelerator, model, tokenizer, train_loader, val_loader, optimizer, scheduler, config, checkpoint_path, start_epoch)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise