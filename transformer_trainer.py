import os
import torch
import json
import math
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer, get_scheduler
from transformers import BertForMaskedLM, BertConfig, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.local_sgd import LocalSGD
# from iter_data_loader import iter_data_loader
from data_loader import data_loader
from evaluation import evaluate_perplexity, create_test_subset
import argparse
import time
import torch.distributed as dist
import traceback
from datetime import timedelta
# import torch.distributed as dist
from config import BertConfig

from model import BertForMaskedLM

from tokenizer import Tokenizer

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

# @torch.compile
def build_model(config):
    # cfg_path = os.path.join(os.path.dirname("./configs"), "base.json")
    # if os.path.exists(cfg_path):
    bert_config = BertConfig.from_json_file("./configs/base.json")
    print(f"{C.CYAN}Loaded BERT config from {"./configs/base.json"}{C.RESET}")
    # config = BertConfig(cfg_path)

    model = BertForMaskedLM(bert_config)

    return model

import time
import datetime

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
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    # print(f"[Epoch {epoch+1}] Processing batch {step+1}/{total_batches} (size={len(batch)})")
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss
                    # Ensure all ranks participate in backward/all-reduce: replace non-finite with 0
                    if not torch.isfinite(loss):
                        if is_main:
                            print(f"{C.YELLOW}Warning: non-finite loss at step {step} in epoch {epoch+1}. Replacing with 0.{C.RESET}")
                        loss = torch.zeros_like(loss)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config["max_grad_norm"]) 
                # Only step optimizer/scheduler when we're at an accumulation boundary
                if accelerator.sync_gradients:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Gather and average loss across GPUs
                loss_value = accelerator.gather(loss).mean()
                # Use local loss value only; avoid cross-rank collectives in the hot path
                # loss_value = float(loss.detach())
                total_loss += loss_value

                if step % log_steps == 0:
                    avg_loss = total_loss / (step + 1)

                    processed_batches += log_steps
                    if is_main:
                        batch_bar.update(log_steps)
                        loss_bar.update(1)
                        loss_bar.set_postfix({
                            "loss": f"{loss_value:.4f}",
                            "avg_loss": f"{avg_loss:.4f}",
                        })
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
                    accelerator.wait_for_everyone()
                    if is_main:
                        try:
                            accelerator.save_state(output_dir=checkpoint_path, safe_serialization=False)
                            unwrapped = accelerator.unwrap_model(model)
                            unwrapped.save_pretrained(checkpoint_path)
                            tokenizer.save_pretrained(checkpoint_path)
                            print(f"{C.GREEN}[Checkpoint] Saved model state and optimizer at step {step}{C.RESET}")
                        except Exception as e:
                            print(f"{C.YELLOW}[Checkpoint] Warning: failed to save at step {step}: {e}{C.RESET}")
                    accelerator.wait_for_everyone()
        
        batch_bar.close()
        loss_bar.close()

        # End‑of‑epoch checkpoint (main-only, with barriers)
        # accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            try:
                accelerator.save_state(output_dir=checkpoint_path, safe_serialization=False)
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                print(f"{C.GREEN}[Checkpoint] Saved model state and optimizer{C.RESET}")
            except Exception as e:
                print(f"{C.YELLOW}[Checkpoint] Warning: failed to save: {e}{C.RESET}")
            model.eval()
            with torch.no_grad():
                total_val_loss = 0.0
                # max_batches = int(config.get("val_sample_batches", 128))
                max_batches = len(val_loader)
                seen = 0
                # randomize the validation loader to sample different batches each time
                # shuffled_val_loader = torch.utils.data.RandomSampler(val_loader)
                for val_batch in tqdm(val_loader, desc="Validating (sample)", disable=not is_main):
                    val_outputs = model(
                        input_ids=val_batch["input_ids"],
                        attention_mask=val_batch["attention_mask"],
                        labels=val_batch["labels"],
                    )
                    total_val_loss += float(val_outputs.loss.detach())
                    seen += 1
                    if seen >= max_batches:
                        break
            if seen > 0:
                print(f"{C.CYAN}[End-epoch] step {step}: avg_val_loss={total_val_loss/seen:.4f} over {seen} batches{C.RESET}")
            model.train()
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

    # Add gradient accumulation and mixed precision
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=config["grad_accum"],
        dataloader_config=DataLoaderConfiguration(even_batches=True),
    )

    checkpoint_path = os.path.join(config["cache_path"], "checkpoint")
    tokenizer = Tokenizer.from_file(config.get("tokenizer_path"))

    train_loader, val_loader, test_loader, collate_fn, total_tokens_train = data_loader(config, tokenizer, config["cache_path"])
    # Build the GPT-2 model from scratch based on our config
    model = build_model(config)

    # Optimizer with LR scaling and weight-decay fix
    base_lr = float(config.get("learning_rate", 5e-5))
    grad_accum = int(config.get("grad_accum", 1))
    world_size = accelerator.num_processes if hasattr(accelerator, "num_processes") else int(os.environ.get("WORLD_SIZE", 1))
    effective_bs = int(config["batch_size"]) * max(1, grad_accum) * max(1, world_size)
    scale = effective_bs / 256.0
    scaled_lr = max(1e-5, min(base_lr * scale, 3e-4))

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    decay_params = []
    nodecay_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            nodecay_params.append(p)
        else:
            decay_params.append(p)
    param_groups = [
        {"params": decay_params, "weight_decay": float(config.get("weight_decay", 0.01)), "lr": scaled_lr},
        {"params": nodecay_params, "weight_decay": 0.0, "lr": scaled_lr},
    ]

    if accelerator.is_main_process:
        print(f"{C.BLUE}LR base={base_lr:.2e} scaled={scaled_lr:.2e} (effective_bs={effective_bs}){C.RESET}")

    optimizer = AdamW(param_groups, lr=scaled_lr, betas=(0.9, 0.999), eps=1e-8)
   
    # Prepare model, optimizer, and train/test loaders. Leave val_loader unprepared (CPU) for main-only eval.
    model, optimizer, train_loader, test_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader, val_loader
    )

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
    total_steps = config["num_epochs"] * len(train_loader)
    warmup_cfg = int(config.get("warmup_steps", 0))
    warmup_steps = warmup_cfg if warmup_cfg > 0 else max(1, int(0.06 * total_steps))
    if accelerator.is_main_process and warmup_cfg <= 0:
        print(f"{C.BLUE}Using warmup_steps={warmup_steps} (~6% of {total_steps} steps){C.RESET}")

    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Ensure the scheduler is part of the checkpoint state
    # accelerator.register_for_checkpointing(scheduler)

    # # Robust checkpoint loading with a broadcasted decision to avoid divergence
    # has_ckpt_local = 1 if (checkpoint_path and os.path.isdir(checkpoint_path) and os.listdir(checkpoint_path)) else 0
    # # Derive a consistent decision across ranks using an all-gather via Accelerate
    # local_flag = torch.tensor([has_ckpt_local], device=accelerator.device, dtype=torch.int64)
    # if hasattr(accelerator, "num_processes") and accelerator.num_processes > 1:
    #     all_flags = accelerator.gather(local_flag)
    #     has_ckpt = bool(int(all_flags.max().item()))
    # else:
    #     has_ckpt = bool(int(local_flag.item()))

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

    train_loop(accelerator, model, tokenizer, train_loader, val_loader, optimizer, scheduler, config, checkpoint_path, start_epoch)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise