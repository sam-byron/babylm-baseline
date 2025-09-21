# coding=utf-8

import argparse
import torch
import torch.nn.functional as F
import gzip
import pickle

from tokenizers import Tokenizer

from config import BertConfig
from ltg_bert import LtgBertForMaskedLM as Bert

import wandb

from accelerate import Accelerator

from ltg_bert import LtgBertForMaskedLM
from ltg_bert_config import LtgBertConfig
from save_config import save_ltg_bert_config

import os

import numpy as np

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

accelerator = Accelerator()


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_path", default="/home/sam-byron/engineering/ML/playground/babylm/bnc/bl-bnc-custom-ltg-bert/data/extrinsic/blimp.pkl.gz", type=str, help="Path to BLiMP.")
    parser.add_argument("--checkpoint_path", default="/home/sam-byron/engineering/ML/playground/babylm/bnc/bl-bnc-custom-ltg-bert/model_vault/bert_ltg_bl_128_run_epo_156_90", type=str, help="The initial checkpoint to start training from.")

    # Other parameters
    parser.add_argument("--vocab_path", default="/home/sam-byron/engineering/ML/playground/babylm/bnc/bl-bnc-custom-ltg-bert/data/pretrain/wordpiece_vocab.json", type=str, help="The vocabulary the BERT model will train on.")

    args = parser.parse_args()

    return args


def setup_training(args):
    assert torch.cuda.is_available()

    device = torch.device("cuda")
    # checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint = args.checkpoint_path
    # checkpoint = os.path.join(args.checkpoint_path, "pytorch_model.bin")
    
    # accelerator.load_state(checkpoint)

    return device, args, checkpoint


def prepare_model(checkpoint_path):
    
    # Load the saved config
    transformers_config = LtgBertConfig.from_pretrained(checkpoint_path)
    print(f"{C.CYAN}Loaded and saved LtgBertConfig to {checkpoint_path}{C.RESET}")
    
    model = LtgBertForMaskedLM(transformers_config)

    print(f"{C.GREEN}Model initialized from {checkpoint_path}{C.RESET}")

    return model


def is_right(good, bad, model, tokenizer, device):
    mask_index = tokenizer.token_to_id("[MASK]")
    pad_index = tokenizer.token_to_id("[PAD]")
    cls_index = torch.tensor([tokenizer.token_to_id("[CLS]")], dtype=torch.long)
    sep_index = torch.tensor([tokenizer.token_to_id("[SEP]")], dtype=torch.long)

    # Fix: Convert strings to token IDs first, then to tensors
    good_tokens = good.split(" ")
    bad_tokens = bad.split(" ")
    
    # Convert token strings to token IDs
    unk_token_id = tokenizer.token_to_id("[UNK]")
    good_ids = []
    for token in good_tokens:
        token_id = tokenizer.token_to_id(token)
        good_ids.append(token_id if token_id is not None else unk_token_id)

    bad_ids = []
    for token in bad_tokens:
        token_id = tokenizer.token_to_id(token)
        bad_ids.append(token_id if token_id is not None else unk_token_id)
        
    # Convert to tensors
    good = torch.tensor(good_ids, dtype=torch.long)
    bad = torch.tensor(bad_ids, dtype=torch.long)
    labels = torch.cat([good, bad]).unsqueeze(-1).to(device)

    def prepare(tokens, padding: int):
        tokens = torch.cat([cls_index, tokens, sep_index, torch.full((padding,), fill_value=pad_index)]).to(device)
        tokens = tokens.repeat(tokens.size(0) - 2 - padding, 1)
        mask = torch.eye(tokens.size(1), device=device).bool()[1:-(1 + padding), :]
        input_ids = tokens.masked_fill(mask, value=mask_index)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        attention_mask[:, attention_mask.size(-1) - padding:] = True
        return input_ids, attention_mask

    good_input_ids, good_attention_mask = prepare(good, max(0, len(bad) - len(good)))
    bad_input_ids, bad_attention_mask = prepare(bad, max(0, len(good) - len(bad)))

    # logits = model(
    #     torch.cat([good_input_ids, bad_input_ids], dim=0).t(),
    #     torch.cat([good_attention_mask, bad_attention_mask], dim=0)
    # ).transpose(0, 1)

    logits = model(
        torch.cat([good_input_ids, bad_input_ids], dim=0),
        torch.cat([good_attention_mask, bad_attention_mask], dim=0)
    ).logits

    indices = torch.cat([torch.arange(1, 1 + len(good), device=device), torch.arange(1, 1 + len(bad), device=device)])
    indices = indices.view(-1, 1, 1).expand(-1, -1, logits.size(-1))
    logits = torch.gather(logits, dim=1, index=indices).squeeze(1)
    log_p = F.log_softmax(logits, dim=-1)

    log_p = log_p.gather(index=labels, dim=-1).squeeze(-1)

    return log_p[:len(good)].sum() > log_p[len(good):].sum()


@torch.no_grad()
def evaluate(model, tokenizer, pairs, device):
    correct = 0
    # for pair in pairs:
    good, bad = pairs["sentence_good"], pairs["sentence_bad"]

    if is_right(good, bad, model, tokenizer, device):
        correct += 1

    return correct / len(pairs) * 100.0


@torch.no_grad()
def evaluate_all(model, tokenizer, blimp, device):
    total_accuracy, total = 0.0, 0
    for group_key, group in blimp.items():
        total_group_accuracy = 0.0
        for subgroup in group:
            accuracy = evaluate(model, tokenizer, subgroup, device)
            total_group_accuracy += accuracy
            total_accuracy += accuracy
            total += 1

        #     wandb.log(
        #         {
        #             f"blimp_detailed/{group_key}_{subgroup_key}": accuracy
        #         },
        #         step=global_step,
        #         commit=False
        #     )
        #     print(f"{group_key} / {subgroup_key}: {accuracy} %", flush=True)

        # wandb.log(
        #     {
        #         f"blimp/{group_key}": total_group_accuracy / len(group)
        #     },
        #     step=global_step,
        #     commit=False
        # )

    # wandb.run.summary["BLiMP/accuracy"] = total_accuracy / total
    print(f"BLiMP accuracy: {total_accuracy / total} %", flush=True)


if __name__ == "__main__":
    args = parse_arguments()

    device, args, checkpoint = setup_training(args)
    # if wandb.run is None:
    #     wandb.init(
    #         name=checkpoint["args"].run_name,
    #         id=checkpoint["args"].wandb_id,
    #         project="bert-bnc",
    #         entity="ltg",
    #         resume="auto",
    #         mode="offline",
    #         allow_val_change=True,
    #     )

    tokenizer = Tokenizer.from_file(args.vocab_path)
    # Print checkpoint info
    print(f"{C.CYAN}Resuming from checkpoint at {checkpoint}{C.RESET}")
    model = prepare_model(checkpoint)

    model = accelerator.prepare(model)
    unwrapped = accelerator.unwrap_model(model)
    weights_path = os.path.join(checkpoint, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu")
    unwrapped.load_state_dict(state_dict, strict=False)

    unwrapped.eval()

    with gzip.open(args.input_path, "rb") as f:
        blimp = pickle.load(f)

    evaluate_all(unwrapped, tokenizer, blimp, device)
