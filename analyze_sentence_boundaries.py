#!/usr/bin/env python3
"""
Script to analyze sentence boundary preservation in the BNC tokenization pipeline.
This script will help diagnose why the model is failing on syntax-dependent BLiMP tasks.
"""

import torch
import re
from transformers import AutoTokenizer
import glob
import os

def analyze_sentence_boundaries():
    """Analyze how sentence boundaries are handled in the current pipeline."""
    
    print("=" * 60)
    print("SENTENCE BOUNDARY ANALYSIS FOR BNC CUSTOM LTG-BERT")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('model_babylm_bert_ltg/checkpoint', trust_remote_code=True)
    
    # 1. Examine raw data structure
    print("\n1. RAW DATA STRUCTURE:")
    print("-" * 30)
    raw_file = "./data/pretrain/shard/train_5.md"
    with open(raw_file, 'r') as f:
        raw_content = f.read()
    
    # Count sentences in raw data
    sentences_with_speakers = [line for line in raw_content.split('\n') 
                              if ':' in line and "'" in line and len(line.strip()) > 10]
    
    print(f"Sample raw sentences (first 5):")
    for i, sent in enumerate(sentences_with_speakers[:5]):
        print(f"  {i+1}: {sent[:80]}...")
    
    print(f"\nTotal sentences with speakers in sample file: {len(sentences_with_speakers)}")
    
    # 2. Examine tokenized data structure  
    print("\n2. TOKENIZED DATA STRUCTURE:")
    print("-" * 30)
    chunk = torch.load('model_babylm_bert_ltg/chunk0.pt')
    
    print(f"Number of sequences in chunk: {len(chunk)}")
    print(f"Sequence lengths: {[len(seq) for seq in chunk[:5]]}")
    
    # 3. Analyze sentence boundaries in tokenized sequences
    print("\n3. SENTENCE BOUNDARY PRESERVATION:")
    print("-" * 30)
    
    for i, seq in enumerate(chunk[:3]):
        decoded = tokenizer.decode(seq, skip_special_tokens=False)
        
        # Count potential sentence endings
        sentence_markers = decoded.count("'") // 2  # Each sentence wrapped in quotes
        periods = decoded.count('.')
        questions = decoded.count('?')
        exclamations = decoded.count('!')
        
        print(f"\nSequence {i+1}:")
        print(f"  Length: {len(seq)} tokens")
        print(f"  Quoted speech segments: {sentence_markers}")
        print(f"  Period endings: {periods}")
        print(f"  Question endings: {questions}")
        print(f"  Exclamation endings: {exclamations}")
        print(f"  Total potential sentences: {sentence_markers + periods + questions + exclamations}")
        
        # Show sample
        sample = decoded[:300].replace('\n', ' ')
        print(f"  Sample: {sample}...")
        
        # Check for sentence boundary tokens
        special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id]
        boundary_count = sum(1 for token in seq if token in special_tokens)
        print(f"  Sentence boundary tokens ([CLS]/[SEP]): {boundary_count}")
    
    # 4. Compare with ideal sentence-level processing
    print("\n4. IDEAL VS ACTUAL PROCESSING:")
    print("-" * 30)
    
    # Extract some sentences manually
    sample_sentences = []
    for line in sentences_with_speakers[:10]:
        if ':' in line and "'" in line:
            # Extract sentence from "Speaker: 'sentence'" format
            match = re.search(r"'([^']+)'", line)
            if match:
                sentence = match.group(1).strip()
                if len(sentence.split()) > 3:  # Filter very short sentences
                    sample_sentences.append(sentence)
    
    print("Sample extracted sentences:")
    for i, sent in enumerate(sample_sentences[:5]):
        print(f"  {i+1}: {sent}")
    
    # Show how these should be tokenized
    print(f"\nIdeal tokenization (first sentence):")
    if sample_sentences:
        ideal_tokens = tokenizer.encode(sample_sentences[0], add_special_tokens=True)
        print(f"  Tokens: {ideal_tokens}")
        print(f"  Length: {len(ideal_tokens)}")
        print(f"  Decoded: {tokenizer.decode(ideal_tokens)}")
    
    # 5. Calculate the problem magnitude
    print("\n5. PROBLEM MAGNITUDE:")
    print("-" * 30)
    
    total_tokens = sum(len(seq) for seq in chunk)
    total_sequences = len(chunk)
    avg_sequence_length = total_tokens / total_sequences
    
    print(f"Current average sequence length: {avg_sequence_length:.1f} tokens")
    print(f"Ideal sentence length: 15-30 tokens")
    print(f"Current sequences are {avg_sequence_length/20:.1f}x longer than ideal sentences")
    
    # Estimate how many sentences are concatenated
    estimated_sentences = sum(decoded.count("'") // 2 for seq in chunk 
                             for decoded in [tokenizer.decode(seq, skip_special_tokens=False)])
    print(f"Estimated sentences concatenated into {total_sequences} sequences: {estimated_sentences}")
    print(f"Average sentences per sequence: {estimated_sentences/total_sequences:.1f}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    os.chdir("/home/sam-byron/engineering/ML/playground/babylm/bnc/bl-bnc-custom-ltg-bert")
    analyze_sentence_boundaries()
