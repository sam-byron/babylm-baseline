"""
Fixed data preparation script that preserves sentence boundaries for proper syntax learning.

This script replaces the document-level processing in prepare_data.py with sentence-level
processing to enable the model to learn syntactic patterns properly.
"""

import re
from pathlib import Path
from datasets import Dataset
from typing import List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_sentences_from_markdown(text: str) -> List[str]:
    """
    Extract individual sentences from BNC markdown format, preserving dialogue structure.
    
    Args:
        text: Raw markdown text from BNC conversion
        
    Returns:
        List of clean sentences ready for tokenization
    """
    sentences = []
    
    # Split by lines to handle speaker attribution
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):  # Skip empty lines and headers
            continue
            
        # Handle speaker attribution format: "Speaker: 'dialogue'"
        if ':' in line and "'" in line:
            # Extract the actual speech content
            parts = line.split(':', 1)
            if len(parts) == 2:
                speech_content = parts[1].strip()
                
                # Remove outer quotes if present
                if speech_content.startswith("'") and speech_content.endswith("'"):
                    speech_content = speech_content[1:-1]
                elif speech_content.startswith('"') and speech_content.endswith('"'):
                    speech_content = speech_content[1:-1]
                
                # Split on sentence-ending punctuation while preserving the punctuation
                sentence_parts = re.split(r'([.!?]+)', speech_content)
                
                current_sentence = ""
                for i, part in enumerate(sentence_parts):
                    if part.strip():
                        current_sentence += part
                        # If this part ends with punctuation, complete the sentence
                        if re.match(r'[.!?]+', part):
                            if current_sentence.strip():
                                sentences.append(current_sentence.strip())
                            current_sentence = ""
                
                # Add any remaining content as a sentence
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
        else:
            # Handle non-dialogue text - split on sentence boundaries
            sentence_parts = re.split(r'([.!?]+)', line)
            current_sentence = ""
            for i, part in enumerate(sentence_parts):
                if part.strip():
                    current_sentence += part
                    if re.match(r'[.!?]+', part):
                        if current_sentence.strip():
                            sentences.append(current_sentence.strip())
                        current_sentence = ""
            
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
    
    # Filter out very short sentences and clean up
    cleaned_sentences = []
    for sentence in sentences:
        # Remove special tokens and clean up
        cleaned = re.sub(r'\[UNK\]', '', sentence)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Keep sentences with at least 3 words
        if len(cleaned.split()) >= 3:
            cleaned_sentences.append(cleaned)
    
    return cleaned_sentences

def load_md_files_to_sentences(directory: str) -> List[str]:
    """
    Load all markdown files and extract sentences from them.
    
    Args:
        directory: Path to directory containing .md files
        
    Returns:
        List of sentences from all files
    """
    data_dir = Path(directory)
    all_sentences = []
    
    md_files = list(data_dir.glob("*.md"))
    logger.info(f"Found {len(md_files)} markdown files")
    
    for md_file in md_files:
        logger.info(f"Processing {md_file.name}")
        
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            sentences = extract_sentences_from_markdown(content)
            all_sentences.extend(sentences)
            logger.info(f"  Extracted {len(sentences)} sentences")
            
        except Exception as e:
            logger.error(f"Error processing {md_file}: {e}")
            continue
    
    logger.info(f"Total sentences extracted: {len(all_sentences)}")
    return all_sentences

def create_sentence_dataset(sentences: List[str]) -> Dataset:
    """
    Create a HuggingFace dataset from sentences.
    
    Args:
        sentences: List of individual sentences
        
    Returns:
        Dataset object ready for tokenization
    """
    return Dataset.from_dict({"text": sentences})

def prepare_sentence_aware_data(input_dir: str, output_path: str) -> None:
    """
    Main function to prepare sentence-aware training data.
    
    Args:
        input_dir: Directory containing BNC markdown files
        output_path: Path to save the sentence dataset
    """
    logger.info("Starting sentence-aware data preparation")
    
    # Extract sentences from all markdown files
    sentences = load_md_files_to_sentences(input_dir)
    
    if not sentences:
        raise ValueError("No sentences extracted from markdown files")
    
    # Create dataset
    dataset = create_sentence_dataset(sentences)
    
    # Save dataset
    dataset.save_to_disk(output_path)
    logger.info(f"Sentence dataset saved to {output_path}")
    
    # Print statistics
    avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
    logger.info(f"Dataset statistics:")
    logger.info(f"  Total sentences: {len(sentences)}")
    logger.info(f"  Average sentence length: {avg_length:.1f} words")
    
    # Sample sentences for verification
    logger.info("Sample sentences:")
    for i, sentence in enumerate(sentences[:5]):
        logger.info(f"  {i+1}: {sentence}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python prepare_data_sentence_aware.py <input_dir> <output_path>")
        print("Example: python prepare_data_sentence_aware.py bnc_converted/ sentence_dataset/")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_path = sys.argv[2]
    
    prepare_sentence_aware_data(input_dir, output_path)
