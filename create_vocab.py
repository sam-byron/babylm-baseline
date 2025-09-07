import os
import json
import pickle
import gzip
import numpy as np
import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from tqdm import tqdm

def create_directories(data_folder):
    """Create necessary directories"""
    Path(f"{data_folder}/shard").mkdir(parents=True, exist_ok=True)
    Path(f"{data_folder}/tokenized").mkdir(parents=True, exist_ok=True)
    print(f"Created directories in {data_folder}")

def shard_data(source_file, shard_folder, n_train_shards, n_valid_shards):
    """Split data into train/valid shards"""
    print(f"Sharding data from {source_file}...")
    
    if not os.path.exists(source_file):
        print(f"Error: Source file {source_file} not found!")
        return False
    
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by documents (assuming documents are separated by double newlines or document markers)
    documents = []
    if '# ' in content:  # Markdown-style documents
        doc_parts = content.split('\n# ')
        for i, part in enumerate(doc_parts):
            if i == 0:
                documents.append(part)
            else:
                documents.append('# ' + part)
    else:
        # Simple paragraph-based splitting
        documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
    
    print(f"Found {len(documents)} documents")
    
    # Simple split: 90% train, 10% valid
    split_idx = int(0.9 * len(documents))
    train_docs = documents[:split_idx]
    valid_docs = documents[split_idx:]
    
    print(f"Train documents: {len(train_docs)}, Valid documents: {len(valid_docs)}")
    
    # Split train into shards
    train_shard_size = len(train_docs) // n_train_shards
    for i in range(n_train_shards):
        start_idx = i * train_shard_size
        end_idx = (i + 1) * train_shard_size if i < n_train_shards - 1 else len(train_docs)
        
        with open(f"{shard_folder}/train_{i}.md", 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(train_docs[start_idx:end_idx]))
        print(f"Created train shard {i} with {end_idx - start_idx} documents")
    
    # Split valid into shards
    valid_shard_size = len(valid_docs) // n_valid_shards
    for i in range(n_valid_shards):
        start_idx = i * valid_shard_size
        end_idx = (i + 1) * valid_shard_size if i < n_valid_shards - 1 else len(valid_docs)
        
        with open(f"{shard_folder}/valid_{i}.md", 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(valid_docs[start_idx:end_idx]))
        print(f"Created valid shard {i} with {end_idx - start_idx} documents")
    
    return True

def create_wordpiece_vocab(train_file, vocab_file, vocab_size=2**14):
    """Create WordPiece vocabulary"""
    print(f"Creating vocabulary from {train_file}...")
    
    if not os.path.exists(train_file):
        print(f"Error: Training file {train_file} not found!")
        return False
    
    # Initialize tokenizer
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    
    # Special tokens
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2
    )
    
    # Train tokenizer
    print("Training tokenizer...")
    tokenizer.train([train_file], trainer)
    
    # Save tokenizer
    tokenizer.save(vocab_file)
    print(f"Vocabulary saved to {vocab_file}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    return True

def parse_documents(file_path):
    """Parse file into documents and sentences"""
    documents = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by documents
    if '# ' in content:
        # Markdown-style documents
        doc_parts = content.split('\n# ')
        for i, part in enumerate(doc_parts):
            if i == 0:
                doc_content = part
            else:
                doc_content = '# ' + part
            
            # Split into sentences (simple approach)
            sentences = [s.strip() for s in doc_content.split('\n') if s.strip() and not s.strip().startswith('#')]
            if sentences:
                documents.append(sentences)
    else:
        # Simple paragraph-based approach
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        for paragraph in paragraphs:
            sentences = [s.strip() for s in paragraph.split('\n') if s.strip()]
            if sentences:
                documents.append(sentences)
    
    return documents

def tokenize_documents(input_file, vocab_file, output_file):
    """Tokenize documents and save as pickle"""
    print(f"Tokenizing {input_file}...")
    
    if not os.path.exists(vocab_file):
        print(f"Error: Vocabulary file {vocab_file} not found!")
        return False
    
    tokenizer = Tokenizer.from_file(vocab_file)
    documents = parse_documents(input_file)
    
    print(f"Found {len(documents)} documents to tokenize")
    
    tokenized_docs = []
    for doc in tqdm(documents, desc="Processing documents"):
        tokenized_doc = []
        for sentence in doc:
            # Tokenize sentence
            encoding = tokenizer.encode(sentence)
            tokens = encoding.ids
            
            # Add to document if not empty
            if tokens:
                tokenized_doc.append(np.array(tokens, dtype=np.int32))
        
        if tokenized_doc:  # Only add non-empty documents
            tokenized_docs.append(tokenized_doc)
    
    # Save as compressed pickle
    with gzip.open(output_file, 'wb') as f:
        pickle.dump(tokenized_docs, f)
    
    print(f"Saved {len(tokenized_docs)} documents to {output_file}")
    return True

def main():
    # Configuration
    N_TRAIN_SHARDS = 8
    N_VALID_SHARDS = 1
    DATA_FOLDER = "./data/pretrain"
    SOURCE_FOLDER = f"{DATA_FOLDER}/bnc"
    SHARD_FOLDER = f"{DATA_FOLDER}/shard"
    TOKENIZED_FOLDER = f"{DATA_FOLDER}/tokenized"
    TRAIN_FILE = f"{SOURCE_FOLDER}/train.md"
    VOCAB_FILE = f"{DATA_FOLDER}/wordpiece_vocab.json"
    
    print("=== BNC BERT Preprocessing ===")
    print(f"Source file: {TRAIN_FILE}")
    print(f"Output folder: {DATA_FOLDER}")
    print(f"Train shards: {N_TRAIN_SHARDS}")
    print(f"Valid shards: {N_VALID_SHARDS}")
    
    # Create directories
    create_directories(DATA_FOLDER)
    
    # Check if source file exists
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: Source file {TRAIN_FILE} not found!")
        print("Please ensure your BNC data is in the correct location.")
        print("Expected structure:")
        print(f"  {SOURCE_FOLDER}/")
        print(f"    train.md")
        return 1
    
    # Step 1: Shard the data
    print("\n=== Step 1: Sharding Data ===")
    if not shard_data(TRAIN_FILE, SHARD_FOLDER, N_TRAIN_SHARDS, N_VALID_SHARDS):
        return 1
    
    # Step 2: Create vocabulary
    print("\n=== Step 2: Creating Vocabulary ===")
    if not create_wordpiece_vocab(TRAIN_FILE, VOCAB_FILE):
        return 1
    
    # Step 3: Tokenize training shards
    print("\n=== Step 3: Tokenizing Training Shards ===")
    for i in range(N_TRAIN_SHARDS):
        input_file = f"{SHARD_FOLDER}/train_{i}.md"
        output_file = f"{TOKENIZED_FOLDER}/train_{i}.pickle.gz"
        if not tokenize_documents(input_file, VOCAB_FILE, output_file):
            return 1
    
    # Step 4: Tokenize validation shards
    print("\n=== Step 4: Tokenizing Validation Shards ===")
    for i in range(N_VALID_SHARDS):
        input_file = f"{SHARD_FOLDER}/valid_{i}.md"
        output_file = f"{TOKENIZED_FOLDER}/valid_{i}.pickle.gz"
        if not tokenize_documents(input_file, VOCAB_FILE, output_file):
            return 1
    
    print("\n=== Preprocessing Complete! ===")
    print(f"Tokenized files created in: {TOKENIZED_FOLDER}")
    print(f"Vocabulary file: {VOCAB_FILE}")
    
    # Verify output
    print("\nOutput verification:")
    for i in range(N_TRAIN_SHARDS):
        output_file = f"{TOKENIZED_FOLDER}/train_{i}.pickle.gz"
        if os.path.exists(output_file):
            print(f"  ✓ {output_file} ({os.path.getsize(output_file)} bytes)")
        else:
            print(f"  ✗ {output_file} missing")
    
    for i in range(N_VALID_SHARDS):
        output_file = f"{TOKENIZED_FOLDER}/valid_{i}.pickle.gz"
        if os.path.exists(output_file):
            print(f"  ✓ {output_file} ({os.path.getsize(output_file)} bytes)")
        else:
            print(f"  ✗ {output_file} missing")
    
    return 0

if __name__ == "__main__":
    exit(main())