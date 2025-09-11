"""
Sentence-aware MLM dataset that preserves syntactic structure for training.

This dataset class replaces the document-level processing in mlm_dataset.py
with proper sentence-aware masking that maintains syntactic boundaries.
"""

import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
import random
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SentenceAwareMLMDataset(Dataset):
    """
    MLM Dataset that processes tokenized sentences with preserved boundaries.
    """
    
    def __init__(
        self,
        tokenized_data_dir: str,
        tokenizer,
        mlm_probability: float = 0.15,
        max_length: int = 512,
        respect_sentence_boundaries: bool = True
    ):
        """
        Initialize the sentence-aware MLM dataset.
        
        Args:
            tokenized_data_dir: Directory containing tokenized sentence chunks
            tokenizer: The tokenizer used for encoding
            mlm_probability: Probability of masking tokens
            max_length: Maximum sequence length
            respect_sentence_boundaries: Whether to avoid masking across sentence boundaries
        """
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.respect_sentence_boundaries = respect_sentence_boundaries
        
        # Load metadata
        metadata_path = Path(tokenized_data_dir) / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load all chunks
        self.chunks = []
        data_dir = Path(tokenized_data_dir)
        
        for i in range(self.metadata['num_chunks']):
            chunk_file = data_dir / f"chunk{i}.pt"
            if chunk_file.exists():
                chunk_data = torch.load(chunk_file)
                self.chunks.append(chunk_data)
                logger.info(f"Loaded chunk {i} with {len(chunk_data['input_ids'])} sequences")
        
        # Calculate total sequences
        self.total_sequences = sum(len(chunk['input_ids']) for chunk in self.chunks)
        logger.info(f"Loaded {self.total_sequences} sequences from {len(self.chunks)} chunks")
        
        # Get special token IDs
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        
        # Vocabulary info for masking
        self.vocab_size = len(self.tokenizer)
    
    def __len__(self) -> int:
        return self.total_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training example with MLM masking.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        # Find which chunk contains this index
        chunk_idx = 0
        current_idx = idx
        
        for i, chunk in enumerate(self.chunks):
            chunk_size = len(chunk['input_ids'])
            if current_idx < chunk_size:
                chunk_idx = i
                break
            current_idx -= chunk_size
        
        # Get the sequence from the appropriate chunk
        chunk = self.chunks[chunk_idx]
        input_ids = chunk['input_ids'][current_idx].clone()
        attention_mask = chunk['attention_mask'][current_idx].clone()
        
        # Apply MLM masking
        input_ids, labels = self._apply_mlm_masking(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _apply_mlm_masking(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MLM masking to input_ids while respecting sentence boundaries.
        
        Args:
            input_ids: Original token IDs
            
        Returns:
            Tuple of (masked_input_ids, labels)
        """
        # Create labels (start with -100 for all positions)
        labels = torch.full_like(input_ids, -100)
        
        # Find positions that can be masked
        maskable_positions = self._get_maskable_positions(input_ids)
        
        if len(maskable_positions) == 0:
            return input_ids, labels
        
        # Determine how many tokens to mask
        num_to_mask = max(1, int(len(maskable_positions) * self.mlm_probability))
        
        # Randomly select positions to mask
        masked_positions = random.sample(maskable_positions, min(num_to_mask, len(maskable_positions)))
        
        # Apply masking strategy (80% [MASK], 10% random, 10% unchanged)
        for pos in masked_positions:
            labels[pos] = input_ids[pos].clone()  # Store original token for loss
            
            rand = random.random()
            if rand < 0.8:
                # 80% of the time, replace with [MASK]
                input_ids[pos] = self.mask_token_id
            elif rand < 0.9:
                # 10% of the time, replace with random token
                input_ids[pos] = random.randint(0, self.vocab_size - 1)
            # 10% of the time, keep unchanged
        
        return input_ids, labels
    
    def _get_maskable_positions(self, input_ids: torch.Tensor) -> List[int]:
        """
        Get positions that can be masked, respecting sentence boundaries if enabled.
        
        Args:
            input_ids: Token IDs
            
        Returns:
            List of maskable positions
        """
        maskable_positions = []
        
        for i, token_id in enumerate(input_ids):
            # Skip special tokens
            if token_id in [self.cls_token_id, self.sep_token_id, self.pad_token_id]:
                continue
            
            # Skip padding
            if token_id == self.pad_token_id:
                continue
            
            # If respecting sentence boundaries, check if we're near sentence boundaries
            if self.respect_sentence_boundaries:
                # Don't mask tokens immediately adjacent to [CLS] or [SEP]
                if i > 0 and input_ids[i-1] in [self.cls_token_id, self.sep_token_id]:
                    continue
                if i < len(input_ids) - 1 and input_ids[i+1] in [self.cls_token_id, self.sep_token_id]:
                    continue
            
            maskable_positions.append(i)
        
        return maskable_positions
    
    def get_sequence_info(self, idx: int) -> Dict:
        """
        Get information about a specific sequence for debugging.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Dictionary with sequence information
        """
        # Find chunk and local index
        chunk_idx = 0
        current_idx = idx
        
        for i, chunk in enumerate(self.chunks):
            chunk_size = len(chunk['input_ids'])
            if current_idx < chunk_size:
                chunk_idx = i
                break
            current_idx -= chunk_size
        
        chunk = self.chunks[chunk_idx]
        input_ids = chunk['input_ids'][current_idx]
        
        # Count special tokens
        cls_count = (input_ids == self.cls_token_id).sum().item()
        sep_count = (input_ids == self.sep_token_id).sum().item()
        pad_count = (input_ids == self.pad_token_id).sum().item()
        
        # Decode text
        text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        
        return {
            'sequence_length': len(input_ids),
            'cls_tokens': cls_count,
            'sep_tokens': sep_count,
            'pad_tokens': pad_count,
            'non_pad_length': len(input_ids) - pad_count,
            'text_preview': text[:200] + "..." if len(text) > 200 else text,
            'chunk_idx': chunk_idx,
            'local_idx': current_idx
        }

def create_sentence_aware_dataloader(
    tokenized_data_dir: str,
    tokenizer,
    batch_size: int = 8,
    mlm_probability: float = 0.15,
    shuffle: bool = True,
    num_workers: int = 0
):
    """
    Create a DataLoader for sentence-aware MLM training.
    
    Args:
        tokenized_data_dir: Directory containing tokenized data
        tokenizer: The tokenizer
        batch_size: Batch size for training
        mlm_probability: Probability of masking tokens
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes
        
    Returns:
        DataLoader for training
    """
    from torch.utils.data import DataLoader
    
    dataset = SentenceAwareMLMDataset(
        tokenized_data_dir=tokenized_data_dir,
        tokenizer=tokenizer,
        mlm_probability=mlm_probability
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created DataLoader with {len(dataset)} sequences, batch_size={batch_size}")
    
    return dataloader

if __name__ == "__main__":
    # Test the dataset
    import sys
    from transformers import AutoTokenizer
    
    if len(sys.argv) != 3:
        print("Usage: python sentence_aware_mlm_dataset.py <tokenized_data_dir> <tokenizer_path>")
        sys.exit(1)
    
    tokenized_data_dir = sys.argv[1]
    tokenizer_path = sys.argv[2]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Create dataset
    dataset = SentenceAwareMLMDataset(tokenized_data_dir, tokenizer)
    
    # Test a few samples
    print(f"Dataset size: {len(dataset)}")
    
    for i in range(min(3, len(dataset))):
        info = dataset.get_sequence_info(i)
        print(f"\nSequence {i}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Get actual training sample
        sample = dataset[i]
        print(f"  Sample shapes: input_ids={sample['input_ids'].shape}, labels={sample['labels'].shape}")
        print(f"  Masked positions: {(sample['labels'] != -100).sum().item()}")
