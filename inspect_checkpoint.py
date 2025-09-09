#!/usr/bin/env python3
"""
Debug script to check what's in the model checkpoint
"""

import torch
from safetensors.torch import load_file
import os

def inspect_checkpoint():
    checkpoint_path = "model_babylm_bert_ltg/checkpoint"
    
    # Load safetensors file
    model_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(model_path):
        print("Loading from model.safetensors")
        state_dict = load_file(model_path)
    else:
        model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        print("Loading from pytorch_model.bin")
        state_dict = torch.load(model_path, map_location='cpu')
    
    print(f"Keys in state_dict ({len(state_dict)} total):")
    for key in sorted(state_dict.keys()):
        print(f"  {key}: {state_dict[key].shape}")

if __name__ == "__main__":
    inspect_checkpoint()
