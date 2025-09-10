"""
RoBERTa-style Dynamic Masking Data Collator

This implements dynamic masking where tokens are masked differently on each epoch,
multiplying the training signal without requiring additional data.

Key features:
- On-the-fly masking during training (not pre-computed)
- Different masking patterns for each epoch
- Configurable masking probabilities and strategies
- Compatible with span masking and subword masking
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Union, Any, Optional
import random
import math


class DynamicMaskingCollator:
    """
    RoBERTa-style dynamic masking data collator.
    
    Unlike static masking where tokens are masked once during preprocessing,
    this collator applies masking dynamically during training, creating
    different masking patterns for each epoch.
    """
    
    def __init__(
        self,
        tokenizer,
        mlm_probability: float = 0.15,
        random_probability: float = 0.1,
        keep_probability: float = 0.1,
        masking_strategy: str = "subword",  # "subword" or "span"
        max_span_length: int = 10,
        geometric_p: float = 0.3,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        """
        Initialize the dynamic masking collator.
        
        Args:
            tokenizer: The tokenizer to use for special tokens
            mlm_probability: Probability of masking tokens (default 15%)
            random_probability: Probability of replacing with random token (default 10% of masked)
            keep_probability: Probability of keeping original token (default 10% of masked)
            masking_strategy: "subword" for token-level or "span" for span-level masking
            max_span_length: Maximum span length for span masking
            geometric_p: Geometric distribution parameter for span lengths
            pad_to_multiple_of: Pad sequences to multiple of this value
            return_tensors: Format of returned tensors
        """
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.random_probability = random_probability
        self.keep_probability = keep_probability
        self.masking_strategy = masking_strategy
        self.max_span_length = max_span_length
        self.geometric_p = geometric_p
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        
        # Get special token IDs
        self.pad_token_id = tokenizer.token_to_id("[PAD]")
        self.mask_token_id = tokenizer.token_to_id("[MASK]")
        self.cls_token_id = tokenizer.token_to_id("[CLS]")
        self.sep_token_id = tokenizer.token_to_id("[SEP]")
        self.unk_token_id = tokenizer.token_to_id("[UNK]")
        
        # Special tokens that should never be masked
        self.special_token_ids = {
            self.pad_token_id,
            self.cls_token_id, 
            self.sep_token_id,
            self.unk_token_id
        }
        
        self.vocab_size = tokenizer.get_vocab_size()
        
        print(f"🎭 Dynamic Masking Collator initialized:")
        print(f"   Strategy: {masking_strategy}")
        print(f"   MLM probability: {mlm_probability}")
        print(f"   Random/Keep probabilities: {random_probability}/{keep_probability}")
        print(f"   Vocab size: {self.vocab_size}")
    
    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Apply dynamic masking to a batch of examples.
        
        Args:
            examples: List of examples, each can be:
                - List of token IDs
                - torch.Tensor of token IDs  
                - Dict with 'input_ids' key
        
        Returns:
            Dict with 'input_ids', 'attention_mask', and 'labels' tensors
        """
        # Handle different input formats
        if isinstance(examples[0], dict):
            # Extract input_ids from dict format
            batch_input_ids = [torch.tensor(ex['input_ids'], dtype=torch.long) if not isinstance(ex['input_ids'], torch.Tensor) 
                              else ex['input_ids'].long() for ex in examples]
        else:
            # Handle raw token sequences
            batch_input_ids = [torch.tensor(ex, dtype=torch.long) if not isinstance(ex, torch.Tensor) 
                              else ex.long() for ex in examples]
        
        # Pad sequences to same length
        batch_input_ids = self._pad_sequences(batch_input_ids)
        
        # Create attention mask
        attention_mask = (batch_input_ids != self.pad_token_id).long()
        
        # Apply dynamic masking
        if self.masking_strategy == "span":
            input_ids, labels = self._mask_tokens_span(batch_input_ids, attention_mask)
        else:
            input_ids, labels = self._mask_tokens_subword(batch_input_ids, attention_mask)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _pad_sequences(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """Pad sequences to the same length."""
        max_length = max(len(seq) for seq in sequences)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of is not None:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        batch_size = len(sequences)
        padded = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long)
        
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        
        return padded
    
    def _mask_tokens_subword(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply subword-level dynamic masking (original BERT/RoBERTa style).
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Tuple of (masked_input_ids, labels)
        """
        batch_size, seq_len = input_ids.shape
        labels = input_ids.clone()
        
        # Create probability matrix for masking
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        
        # Don't mask special tokens or padding
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                token_id = input_ids[batch_idx, seq_idx].item()
                if token_id in self.special_token_ids or attention_mask[batch_idx, seq_idx] == 0:
                    probability_matrix[batch_idx, seq_idx] = 0.0
        
        # Sample which tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Set non-masked tokens to -100 in labels (ignored in loss)
        labels[~masked_indices] = -100
        
        # Apply masking strategy:
        # 80% of time: replace with [MASK]
        # 10% of time: replace with random token
        # 10% of time: keep original token
        
        mask_prob = 1.0 - self.random_probability - self.keep_probability  # 0.8
        
        # 80% of masked tokens → [MASK]
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, mask_prob)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id
        
        # 10% of masked tokens → random token
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, self.random_probability / (self.random_probability + self.keep_probability))
        ).bool() & masked_indices & ~indices_replaced
        
        # Generate random tokens (avoiding special tokens)
        random_tokens = torch.randint(
            low=5,  # Skip special tokens [PAD], [UNK], [CLS], [SEP], [MASK]
            high=self.vocab_size,
            size=(indices_random.sum().item(),),
            dtype=torch.long
        )
        input_ids[indices_random] = random_tokens
        
        # Remaining 10% keep original token (already done, no change needed)
        
        return input_ids, labels
    
    def _mask_tokens_span(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply span-level dynamic masking.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Tuple of (masked_input_ids, labels)
        """
        batch_size, seq_len = input_ids.shape
        input_ids = input_ids.clone()
        labels = torch.full_like(input_ids, -100)
        
        for batch_idx in range(batch_size):
            # Get valid positions (non-special, non-padding tokens)
            valid_positions = []
            for seq_idx in range(seq_len):
                token_id = input_ids[batch_idx, seq_idx].item()
                if (token_id not in self.special_token_ids and 
                    attention_mask[batch_idx, seq_idx] == 1):
                    valid_positions.append(seq_idx)
            
            if not valid_positions:
                continue
            
            # Calculate target number of tokens to mask
            num_to_mask = int(len(valid_positions) * self.mlm_probability)
            
            masked_count = 0
            attempts = 0
            max_attempts = num_to_mask * 3  # Prevent infinite loops
            
            while masked_count < num_to_mask and attempts < max_attempts:
                attempts += 1
                
                # Sample span length from geometric distribution
                # Simple geometric sampling using random.random()
                span_length = 1
                while random.random() < (1 - self.geometric_p) and span_length < self.max_span_length:
                    span_length += 1
                
                # Sample starting position
                if len(valid_positions) < span_length:
                    span_length = len(valid_positions)
                
                start_idx = random.choice(valid_positions[:len(valid_positions) - span_length + 1])
                
                # Check if span overlaps with already masked tokens
                span_positions = list(range(start_idx, min(start_idx + span_length, seq_len)))
                
                # Only mask if positions are valid and not already masked
                valid_span = all(
                    pos in valid_positions and labels[batch_idx, pos] == -100
                    for pos in span_positions
                )
                
                if valid_span:
                    # Apply masking to this span
                    for pos in span_positions:
                        if pos < seq_len and masked_count < num_to_mask:
                            labels[batch_idx, pos] = input_ids[batch_idx, pos]
                            
                            # Decide how to mask this token
                            rand_val = random.random()
                            if rand_val < 1.0 - self.random_probability - self.keep_probability:
                                # 80%: [MASK] token
                                input_ids[batch_idx, pos] = self.mask_token_id
                            elif rand_val < 1.0 - self.keep_probability:
                                # 10%: random token
                                input_ids[batch_idx, pos] = random.randint(5, self.vocab_size - 1)
                            # else: 10%: keep original (no change)
                            
                            masked_count += 1
        
        return input_ids, labels


def create_dynamic_collator(config: dict, tokenizer) -> DynamicMaskingCollator:
    """
    Create a dynamic masking collator based on configuration.
    
    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer instance
    
    Returns:
        DynamicMaskingCollator instance
    """
    masking_strategy = config.get("masking_strategy", "subword")
    mlm_probability = config.get("mask_p", 0.15)
    random_probability = config.get("random_p", 0.1) 
    keep_probability = config.get("keep_p", 0.1)
    
    return DynamicMaskingCollator(
        tokenizer=tokenizer,
        mlm_probability=mlm_probability,
        random_probability=random_probability,
        keep_probability=keep_probability,
        masking_strategy=masking_strategy,
        max_span_length=config.get("max_span_length", 10),
        geometric_p=config.get("geometric_p", 0.3),
        pad_to_multiple_of=config.get("pad_to_multiple_of", None),
    )


# Example usage and testing
if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from tokenizer import Tokenizer
    
    # Test the dynamic collator
    print("🧪 Testing Dynamic Masking Collator")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file("./data/pretrain/wordpiece_vocab.json")
    
    # Create test config
    test_config = {
        "masking_strategy": "subword",
        "mask_p": 0.15,
        "random_p": 0.1,
        "keep_p": 0.1,
    }
    
    # Create collator
    collator = create_dynamic_collator(test_config, tokenizer)
    
    # Test with sample sequences
    sample_text = "The quick brown fox jumps over the lazy dog."
    sample_tokens = tokenizer.encode(sample_text).ids
    
    # Create a batch
    batch = [sample_tokens, sample_tokens[:8], sample_tokens[2:10]]
    
    print(f"\nOriginal sequences:")
    for i, seq in enumerate(batch):
        decoded = tokenizer.decode(seq)
        print(f"  {i}: {decoded}")
    
    # Apply dynamic masking multiple times to show different patterns
    for epoch in range(3):
        print(f"\n🎭 Epoch {epoch + 1} - Dynamic Masking:")
        result = collator(batch)
        
        print(f"Input shape: {result['input_ids'].shape}")
        print(f"Attention mask shape: {result['attention_mask'].shape}")
        print(f"Labels shape: {result['labels'].shape}")
        
        # Show masked sequences
        for i in range(result['input_ids'].shape[0]):
            input_ids = result['input_ids'][i]
            labels = result['labels'][i]
            
            # Decode the masked sequence
            masked_sequence = []
            original_sequence = []
            
            for j, (token_id, label) in enumerate(zip(input_ids, labels)):
                if result['attention_mask'][i][j] == 0:
                    break
                    
                token = tokenizer.decode([token_id.item()])
                masked_sequence.append(token)
                
                if label.item() != -100:
                    original_token = tokenizer.decode([label.item()])
                    original_sequence.append(f"[{original_token}]")
                else:
                    original_sequence.append(token)
            
            masked_text = "".join(masked_sequence)
            original_text = "".join(original_sequence)
            
            print(f"  Seq {i}: {masked_text}")
            print(f"    Orig: {original_text}")
            
            # Count masked tokens
            masked_count = (labels != -100).sum().item()
            total_count = (result['attention_mask'][i] == 1).sum().item()
            mask_ratio = masked_count / total_count if total_count > 0 else 0
            print(f"    Masked: {masked_count}/{total_count} ({mask_ratio:.1%})")
    
    print("\n✅ Dynamic masking test completed!")
