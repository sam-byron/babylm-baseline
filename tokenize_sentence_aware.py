"""
Sentence-aware tokenization script that preserves sentence boundaries for MLM training.

This script replaces the document-level chunking in utils_mp.py with proper
sentence-level tokenization including [CLS] and [SEP] tokens.
"""

import torch
from transformers import AutoTokenizer
from datasets import load_from_disk
from pathlib import Path
import logging
from typing import List, Dict, Any
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentenceAwareTokenizer:
    """Tokenizer that processes sentences with proper boundaries."""
    
    def __init__(self, tokenizer_path: str, max_length: int = 512):
        """
        Initialize the sentence-aware tokenizer.
        
        Args:
            tokenizer_path: Path to the custom tokenizer
            max_length: Maximum sequence length for training
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        
        # Ensure we have special tokens
        if self.tokenizer.cls_token is None:
            self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
        if self.tokenizer.sep_token is None:
            self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({'mask_token': '[MASK]'})
            
        logger.info(f"Tokenizer loaded with vocab size: {len(self.tokenizer)}")
        logger.info(f"Special tokens: CLS={self.tokenizer.cls_token}, SEP={self.tokenizer.sep_token}")
    
    def tokenize_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Tokenize a single sentence with proper boundaries.
        
        Args:
            sentence: Input sentence text
            
        Returns:
            Dictionary with tokenized data
        """
        # Tokenize with special tokens
        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,  # Adds [CLS] and [SEP]
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'sentence_length': len(encoding['input_ids'].squeeze()),
            'original_text': sentence
        }
    
    def create_training_sequences(self, sentences: List[str], pack_sequences: bool = True) -> List[Dict[str, Any]]:
        """
        Create training sequences from sentences.
        
        Args:
            sentences: List of sentence strings
            pack_sequences: Whether to pack multiple short sentences into longer sequences
            
        Returns:
            List of training sequences
        """
        training_sequences = []
        
        if not pack_sequences:
            # Process each sentence individually
            for i, sentence in enumerate(sentences):
                if i % 1000 == 0:
                    logger.info(f"Processing sentence {i}/{len(sentences)}")
                
                tokenized = self.tokenize_sentence(sentence)
                training_sequences.append(tokenized)
        else:
            # Pack multiple sentences into longer sequences (more efficient)
            current_sequence = []
            current_length = 0
            
            for i, sentence in enumerate(sentences):
                if i % 1000 == 0:
                    logger.info(f"Processing sentence {i}/{len(sentences)}")
                
                # Tokenize sentence without special tokens first to check length
                temp_encoding = self.tokenizer(sentence, add_special_tokens=False)
                sentence_length = len(temp_encoding['input_ids'])
                
                # If adding this sentence would exceed max_length, save current sequence
                if current_length + sentence_length + 3 > self.max_length and current_sequence:  # +3 for CLS, SEP, SEP
                    # Create sequence from accumulated sentences
                    combined_text = ' '.join(current_sequence)
                    tokenized = self.tokenize_sentence(combined_text)
                    training_sequences.append(tokenized)
                    
                    # Start new sequence
                    current_sequence = [sentence]
                    current_length = sentence_length
                else:
                    # Add sentence to current sequence
                    current_sequence.append(sentence)
                    current_length += sentence_length
            
            # Add final sequence if not empty
            if current_sequence:
                combined_text = ' '.join(current_sequence)
                tokenized = self.tokenize_sentence(combined_text)
                training_sequences.append(tokenized)
        
        return training_sequences
    
    def save_tokenized_data(self, sequences: List[Dict[str, Any]], output_dir: str) -> None:
        """
        Save tokenized sequences to disk in chunks.
        
        Args:
            sequences: List of tokenized sequences
            output_dir: Directory to save chunks
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save in chunks of 1000 sequences
        chunk_size = 1000
        num_chunks = (len(sequences) + chunk_size - 1) // chunk_size
        
        logger.info(f"Saving {len(sequences)} sequences in {num_chunks} chunks")
        
        for i in range(0, len(sequences), chunk_size):
            chunk_sequences = sequences[i:i + chunk_size]
            
            # Prepare chunk data
            chunk_data = {
                'input_ids': torch.stack([seq['input_ids'] for seq in chunk_sequences]),
                'attention_mask': torch.stack([seq['attention_mask'] for seq in chunk_sequences]),
                'lengths': [seq['sentence_length'] for seq in chunk_sequences],
                'texts': [seq['original_text'] for seq in chunk_sequences]
            }
            
            # Save chunk
            chunk_file = output_path / f"chunk{i // chunk_size}.pt"
            torch.save(chunk_data, chunk_file)
            logger.info(f"Saved chunk {i // chunk_size} with {len(chunk_sequences)} sequences")
        
        # Save metadata
        metadata = {
            'num_sequences': len(sequences),
            'num_chunks': num_chunks,
            'chunk_size': chunk_size,
            'max_length': self.max_length,
            'vocab_size': len(self.tokenizer),
            'avg_length': sum(seq['sentence_length'] for seq in sequences) / len(sequences)
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Tokenization complete. Metadata saved.")
        logger.info(f"Average sequence length: {metadata['avg_length']:.1f} tokens")

def main():
    """Main function to run sentence-aware tokenization."""
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python tokenize_sentence_aware.py <sentence_dataset_path> <tokenizer_path> <output_dir>")
        print("Example: python tokenize_sentence_aware.py sentence_dataset/ ./ tokenized_sentences/")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    tokenizer_path = sys.argv[2]
    output_dir = sys.argv[3]
    
    logger.info("Starting sentence-aware tokenization")
    
    # Load sentence dataset
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    sentences = dataset['text']
    logger.info(f"Loaded {len(sentences)} sentences")
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = SentenceAwareTokenizer(tokenizer_path)
    
    # Tokenize sentences
    logger.info("Creating training sequences...")
    sequences = tokenizer.create_training_sequences(sentences, pack_sequences=True)
    
    # Save tokenized data
    logger.info(f"Saving tokenized data to {output_dir}")
    tokenizer.save_tokenized_data(sequences, output_dir)
    
    logger.info("Sentence-aware tokenization complete!")

if __name__ == "__main__":
    main()
