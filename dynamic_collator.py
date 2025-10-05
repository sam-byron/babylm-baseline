"""
dynamic_collator.py — Dynamic MLM masking collator reusing existing strategies

Overview
    Implements a RoBERTa-style dynamic masking collate function that delegates
    the masking decisions to the proven strategies in mlm_dataset.py
    (SpanMaskingStrategy, SubwordMaskingStrategy, WholeWordMaskingStrategy).

Usage
    from dynamic_collator import create_dynamic_collator
    collate_fn = create_dynamic_collator(config, tokenizer)

Key args
    - mask_p, random_p, keep_p
    - masking_strategy: "span" | "subword" | "whole_word"
    - max_span_length (when masking_strategy == "span")

Innovations & efficiency
    - Reuses existing, well-tested masking strategies to ensure consistent behavior
        between static and dynamic pipelines.
    - Builds an explicit protected_ids list for structural tokens ([CLS]/[SEP]/[DOC]/...),
        preventing accidental masking of control tokens even when special IDs are not contiguous.
    - Pads on-the-fly and constructs attention masks in one pass.
"""
import torch
import random
import numpy as np
from typing import Dict, List, Union, Any, Optional
from mlm_dataset import SpanMaskingStrategy, SubwordMaskingStrategy, WholeWordMaskingStrategy

class DynamicMaskingCollator:
    """
    RoBERTa-style dynamic masking data collator that reuses existing masking strategies.
    
    Instead of reimplementing masking logic, this collator leverages the proven
    masking strategies from mlm_dataset.py and applies them dynamically during training.
    """
    
    def __init__(
        self,
        tokenizer,
        mlm_probability: float = 0.15,
        random_probability: float = 0.1,
        keep_probability: float = 0.1,
        masking_strategy: str = "span",
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
        max_span_length: int = 10,
        seq_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.random_probability = random_probability
        self.keep_probability = keep_probability
        self.masking_strategy = masking_strategy
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.max_span_length = max_span_length
        self.seq_length = seq_length
        
        # Get special token IDs
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.mask_token_id = tokenizer.token_to_id("[MASK]")
        self.cls_token_id = tokenizer.token_to_id("[CLS]")
        self.sep_token_id = tokenizer.token_to_id("[SEP]")
        
        # Create special_token_ids property for convenience
        self.special_token_ids = set([
            self.pad_token_id, self.mask_token_id,
            self.cls_token_id, self.sep_token_id
        ])
        
        # Initialize the masking strategy - REUSE existing implementation!
        # Build explicit protected_ids for structural tokens; pass through to strategies.
        structural_tokens = [
            "[PAD]","[UNK]","[CLS]","[SEP]","[MASK]",
            "[PAR]","[TAB]","[DOC]","[EOD]","[SPK]"
        ]
        protected_ids_set = set()
        for tok in structural_tokens:
            tid = tokenizer.token_to_id(tok)
            if isinstance(tid, int) and tid >= 0:
                protected_ids_set.add(int(tid))
        special_ids = sorted(protected_ids_set)
        # Back-compat heuristic for n_special_tokens expected by strategies
        if special_ids and special_ids == list(range(len(special_ids))):
            n_special_tokens = len(special_ids)
        else:
            n_special_tokens = max(6, len(special_ids))
            if special_ids:
                print(f"[DynamicCollator] Warning: special token IDs non-contiguous. Using n_special_tokens={n_special_tokens} with explicit protected_ids of size {len(protected_ids_set)}")
        
        masking_strategies = {
            "span": SpanMaskingStrategy,
            "subword": SubwordMaskingStrategy, 
            "whole_word": WholeWordMaskingStrategy,
        }
        
        if masking_strategy not in masking_strategies:
            raise ValueError(f"Unknown masking strategy: {masking_strategy}. Choose from {list(masking_strategies.keys())}")
        
        masking_strategy_class = masking_strategies[masking_strategy]
        if masking_strategy == "span":
            self.masking_strategy_instance = masking_strategy_class(
                mask_p=mlm_probability,
                tokenizer=tokenizer,
                n_special_tokens=n_special_tokens,
                padding_label_id=-100,
                random_p=random_probability,
                keep_p=keep_probability,
                max_span_length=max_span_length,
                protected_ids=sorted(protected_ids_set),
                # seq_length=self.seq_length,
            )
        else:   
            self.masking_strategy_instance = masking_strategy_class(
                mask_p=mlm_probability,
                tokenizer=tokenizer,
                n_special_tokens=n_special_tokens,
                padding_label_id=-100,
                random_p=random_probability,
                keep_p=keep_probability,
                protected_ids=sorted(protected_ids_set),
                # seq_length=self.seq_length,
            )
        print(f"[DynamicCollator] 🎭 Using strategy={self.masking_strategy} n_special_tokens={n_special_tokens} protected={len(protected_ids_set)} span_max={max_span_length if masking_strategy=='span' else 'N/A'}")

    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Convert examples to tensors if needed
        if isinstance(examples[0], dict):
            examples = [example["input_ids"] for example in examples]
        
        # Ensure all examples are tensors
        input_ids = []
        for ex in examples:
            if isinstance(ex, torch.Tensor):
                input_ids.append(ex.clone())
            else:
                input_ids.append(torch.tensor(ex, dtype=torch.long))
        
        # Pad sequences to the same length
        max_len = max(len(seq) for seq in input_ids)
        padded_input_ids = []
        attention_masks = []
        all_labels = []
        
        for seq in input_ids:
            # Pad sequence
            padding_length = max_len - len(seq)
            if padding_length > 0:
                padded_seq = torch.cat([seq, torch.full((padding_length,), self.pad_token_id, dtype=torch.long)])
            else:
                padded_seq = seq
            
            # Apply DYNAMIC masking using the existing strategy
            # Each call to the strategy produces different masking patterns due to randomness
            masked_tokens, labels = self.masking_strategy_instance(padded_seq)
            
            padded_input_ids.append(masked_tokens)
            all_labels.append(labels)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            if padding_length > 0:
                attention_mask = torch.cat([torch.ones(len(seq), dtype=torch.long), 
                                          torch.zeros(padding_length, dtype=torch.long)])
            else:
                attention_mask = torch.ones(len(seq), dtype=torch.long)
            attention_masks.append(attention_mask)
        
        # Stack into batch tensors
        input_ids_batch = torch.stack(padded_input_ids)
        attention_mask_batch = torch.stack(attention_masks)
        labels_batch = torch.stack(all_labels)
        
        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
        }

def create_dynamic_collator(config, tokenizer):
    """Factory function to create a dynamic masking collator.

    Args:
        config: Training/runtime config containing mask_p/random_p/keep_p and strategy
        tokenizer: Tokenizer with token_to_id implemented

    Returns:
        Callable suitable as PyTorch DataLoader collate_fn returning a batch dict
        with input_ids, attention_mask, labels (with -100 on unmasked positions).
    """
    masking_strategy = config.get("masking_strategy", "span")
    mlm_probability = config.get("mask_p", 0.15)
    random_probability = config.get("random_p", 0.1)
    keep_probability = config.get("keep_p", 0.1)
    seq_length = config.get("block_size", 512)

    if config.get("masking_strategy") == "span":
        max_span_length = config.get("max_span_length", 10)
        print(f"Using max_span_length={max_span_length} for span masking")
        return DynamicMaskingCollator(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
            random_probability=random_probability,
            keep_probability=keep_probability,
            masking_strategy=masking_strategy,
            max_span_length=max_span_length,
            seq_length=seq_length,
        )
    else:
        return DynamicMaskingCollator(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
            random_probability=random_probability,
            keep_probability=keep_probability,
            masking_strategy=masking_strategy,
            seq_length=seq_length,
        )
