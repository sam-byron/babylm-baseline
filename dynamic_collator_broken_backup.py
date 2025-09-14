import torch
import random
import numpy as np
from typing import Dict, List, Union, Any, Optional

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
        masking_strategy: str = "subword",
        max_span_length: int = 10,
        geometric_p: float = 0.3,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.random_probability = random_probability
        self.keep_probability = keep_probability
        self.masking_strategy = masking_strategy
        self.max_span_length = max_span_length
        self.geometric_p = geometric_p
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        
        # Get special token IDs more comprehensively
        self.special_token_ids = set()
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        for token in special_tokens:
            token_id = tokenizer.token_to_id(token)
            if token_id is not None:
                self.special_token_ids.add(token_id)
        
        # Also get common special token IDs by value (in case tokenizer uses different names)
        common_special_ids = {0, 1, 2, 3, 4}  # Common special token IDs
        self.special_token_ids.update(common_special_ids)
        
        self.mask_token_id = tokenizer.token_to_id("[MASK]")
        if self.mask_token_id is None:
            self.mask_token_id = 4  # Common default
        self.vocab_size = tokenizer.get_vocab_size()
        
        # FIX 1: Initialize random number generator with a different seed each time
        self.rng = random.Random()
        self.np_rng = np.random.RandomState()
        
    def _reseed_random_generators(self):
        """FIX 2: Reseed generators for each batch to ensure variability"""
        # Use a combination of time, process ID, and random state for better seeding
        import time
        import os
        base_seed = int(time.time() * 1000000) % (2**31 - 1)
        # Add some additional randomness
        pid_component = os.getpid() % 1000
        random_component = random.randint(0, 999)
        
        seed = (base_seed + pid_component + random_component) % (2**31 - 1)
        
        self.rng.seed(seed)
        self.np_rng.seed((seed + 1) % (2**31 - 1))  # Slightly different seed for numpy
        
    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # FIX 3: Reseed at the start of each call
        self._reseed_random_generators()
        
        # Convert examples to tensors if needed
        if isinstance(examples[0], dict):
            input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) if not isinstance(ex["input_ids"], torch.Tensor) 
                        else ex["input_ids"] for ex in examples]
        else:
            input_ids = [torch.tensor(ex, dtype=torch.long) if not isinstance(ex, torch.Tensor) 
                        else ex for ex in examples]
        
        # Stack sequences (assuming they are already padded by ChunkedDataset)
        input_ids = torch.stack(input_ids)
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        if hasattr(self.tokenizer, 'token_to_id'):
            pad_token_id = self.tokenizer.token_to_id("[PAD]")
            if pad_token_id is not None:
                attention_mask[input_ids == pad_token_id] = 0
        
        # Apply masking based on strategy
        if self.masking_strategy == "span":
            masked_input_ids, labels = self._mask_tokens_span(input_ids, attention_mask)
        else:
            masked_input_ids, labels = self._mask_tokens_subword(input_ids, attention_mask)
        
        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _pad_sequences(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        max_len = max(len(seq) for seq in sequences)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        pad_token_id = self.tokenizer.token_to_id("[PAD]")
        if pad_token_id is None:
            pad_token_id = 0
        
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_len:
                padding = torch.full((max_len - len(seq),), pad_token_id, dtype=torch.long)
                padded_seq = torch.cat([seq, padding])
            else:
                padded_seq = seq[:max_len]
            padded_sequences.append(padded_seq)
        
        return torch.stack(padded_sequences)
    
    def _sample_geometric_span_length(self) -> int:
        """FIX 4: Improved geometric sampling with proper randomization"""
        # Use numpy's geometric distribution for better randomness
        span_length = self.np_rng.geometric(self.geometric_p)
        return min(span_length, self.max_span_length)
    
    def _mask_tokens_span(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply span-level dynamic masking with improved randomness.
        """
        batch_size, seq_len = input_ids.shape
        input_ids = input_ids.clone()
        labels = torch.full_like(input_ids, -100)
        
        for batch_idx in range(batch_size):
            # Get valid positions based on ORIGINAL tokens (before any masking)
            original_input = input_ids[batch_idx].clone()  # Store original for consistent counting
            valid_positions = []
            for seq_idx in range(seq_len):
                token_id = original_input[seq_idx].item()
                if (token_id not in self.special_token_ids and 
                    attention_mask[batch_idx, seq_idx] == 1):
                    valid_positions.append(seq_idx)
            
            if not valid_positions:
                continue
            
            # Calculate target number of tokens to mask with sequence-boundary awareness
            exact_target = len(valid_positions) * self.mlm_probability
            
            # TRUE ROBERTA DYNAMIC MASKING: Use probabilistic approach
            # Instead of deterministic num_to_mask, use probability per token
            
            # Apply adaptive probability based on sequence content density
            if len(valid_positions) < 5:
                # For very short sequences, skip masking to prevent over-masking
                mask_probability = 0.0
            elif len(valid_positions) < 10:
                # For short sequences, use conservative probability
                mask_probability = 0.10
            elif len(valid_positions) < 20:
                # For medium sequences, use moderate probability
                mask_probability = 0.12
            else:
                # For longer sequences, use standard probability
                mask_probability = self.mlm_probability
            
            # Create probability matrix for valid positions only
            mask_probabilities = torch.zeros(seq_len)
            for pos in valid_positions:
                mask_probabilities[pos] = mask_probability
            
            # STOCHASTIC SAMPLING: This creates variability across epochs!
            masked_indices = torch.bernoulli(mask_probabilities).bool()
            
            # Convert to positions list for compatibility with existing code
            masked_positions = set(torch.where(masked_indices)[0].tolist())
            
            # Set labels for all masked positions
            for pos in masked_positions:
                labels[batch_idx, pos] = input_ids[batch_idx, pos]
                attempts += 1
                
                # Get remaining valid positions
                remaining_valid = [pos for pos in valid_positions if pos not in masked_positions]
                if not remaining_valid:
                    break
                
                # Calculate how many more tokens we need
                remaining_to_mask = num_to_mask - len(masked_positions)
                if remaining_to_mask <= 0:
                    break
                
                # If we're running out of attempts or need many tokens, be more aggressive
                if attempts > max_attempts * 0.7 or remaining_to_mask >= len(remaining_valid) * 0.8:
                    # Just pick the remaining tokens directly
                    positions_to_add = remaining_valid[:remaining_to_mask]
                    for pos in positions_to_add:
                        masked_positions.add(pos)
                        labels[batch_idx, pos] = input_ids[batch_idx, pos]
                    break
                
                # For efficiency, if we need few tokens, just pick them individually
                if remaining_to_mask <= 2:
                    # Just pick remaining positions directly
                    positions_to_add = remaining_valid[:remaining_to_mask]
                    for pos in positions_to_add:
                        masked_positions.add(pos)
                        labels[batch_idx, pos] = input_ids[batch_idx, pos]
                    break
                
                # Sample span length
                span_length = self._sample_geometric_span_length()
                span_length = min(span_length, remaining_to_mask)
                
                # Pick random start position
                start_pos = self.rng.choice(remaining_valid)
                
                # Create span
                span_positions = []
                for i in range(span_length):
                    pos = start_pos + i
                    if (pos in valid_positions and 
                        pos not in masked_positions and
                        len(span_positions) < remaining_to_mask):
                        span_positions.append(pos)
                    else:
                        break
                
                # Add span positions
                for pos in span_positions:
                    masked_positions.add(pos)
                    labels[batch_idx, pos] = input_ids[batch_idx, pos]
            
            # Apply controlled 80/10/10 replacement strategy
            if len(masked_positions) > 0:
                # Convert to list for easier manipulation
                masked_pos_list = list(masked_positions)
                
                # Calculate exact counts for 80/10/10 distribution
                total_masked = len(masked_pos_list)
                num_mask_tokens = round(total_masked * 0.8)  # 80%
                num_random_tokens = round(total_masked * 0.1)  # 10%
                num_keep_tokens = total_masked - num_mask_tokens - num_random_tokens  # remaining 10%
                
                # Shuffle positions for random assignment
                self.rng.shuffle(masked_pos_list)
                
                # Apply replacements deterministically based on calculated counts
                for i, pos in enumerate(masked_pos_list):
                    if i < num_mask_tokens:
                        # 80% -> [MASK]
                        input_ids[batch_idx, pos] = self.mask_token_id
                    elif i < num_mask_tokens + num_random_tokens:
                        # 10% -> random token
                        available_tokens = list(range(max(self.special_token_ids) + 1, self.vocab_size))
                        if available_tokens:
                            random_token = self.rng.choice(available_tokens)
                            input_ids[batch_idx, pos] = random_token
                    # else: remaining 10% -> keep original (no change needed)
        
        return input_ids, labels
    
    def _mask_tokens_subword(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply subword-level masking with improved randomness."""
        batch_size, seq_len = input_ids.shape
        input_ids = input_ids.clone()
        labels = torch.full_like(input_ids, -100)
        
        for batch_idx in range(batch_size):
            # Get maskable positions
            maskable_positions = []
            for seq_idx in range(seq_len):
                token_id = input_ids[batch_idx, seq_idx].item()
                if (token_id not in self.special_token_ids and 
                    attention_mask[batch_idx, seq_idx] == 1):
                    maskable_positions.append(seq_idx)
            
            if not maskable_positions:
                continue
            
            # Calculate target number of tokens to mask with sequence-boundary awareness
            exact_target = len(maskable_positions) * self.mlm_probability
            
            # Apply adaptive masking based on sequence content density
            if len(maskable_positions) < 5:
                # For very short sequences (1-4 tokens), only mask if it won't exceed reasonable rate
                # Only mask 1 token if we have 5+ content tokens (20% max), otherwise skip
                num_to_mask = 1 if len(maskable_positions) >= 5 else 0
            elif len(maskable_positions) < 10:
                # For short sequences (5-9 tokens), use conservative 10% rate
                conservative_target = len(maskable_positions) * 0.10
                num_to_mask = max(1, round(conservative_target))
            elif len(maskable_positions) < 20:
                # For medium sequences, use 12% masking rate
                moderate_target = len(maskable_positions) * 0.12
                num_to_mask = round(moderate_target)
            else:
                # For longer sequences, use standard 15% masking rate
                num_to_mask = round(exact_target)
            
            # Ensure we don't mask more than available positions
            num_to_mask = min(num_to_mask, len(maskable_positions))
            
            # Skip sequences that are too short for reasonable masking
            # (Adaptive logic above already handles this)
            
            # Use random sampling without replacement
            positions_to_mask = self.rng.sample(maskable_positions, 
                                              min(num_to_mask, len(maskable_positions)))
            
            # Set labels for all masked positions
            for pos in positions_to_mask:
                labels[batch_idx, pos] = input_ids[batch_idx, pos]
            
            # Apply controlled 80/10/10 replacement strategy
            if positions_to_mask:
                total_masked = len(positions_to_mask)
                num_mask_tokens = round(total_masked * 0.8)  # 80%
                num_random_tokens = round(total_masked * 0.1)  # 10%
                # num_keep_tokens = remaining 10%
                
                # Shuffle positions for random assignment
                self.rng.shuffle(positions_to_mask)
                
                # Apply replacements deterministically based on calculated counts
                for i, pos in enumerate(positions_to_mask):
                    if i < num_mask_tokens:
                        # 80% -> [MASK]
                        input_ids[batch_idx, pos] = self.mask_token_id
                    elif i < num_mask_tokens + num_random_tokens:
                        # 10% -> random token
                        available_tokens = list(range(max(self.special_token_ids) + 1, self.vocab_size))
                        if available_tokens:
                            input_ids[batch_idx, pos] = self.rng.choice(available_tokens)
                    # else: remaining 10% -> keep original (no change needed)
        
        return input_ids, labels

def create_dynamic_collator(config: dict, tokenizer) -> DynamicMaskingCollator:
    """Create a dynamic masking collator based on configuration."""
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