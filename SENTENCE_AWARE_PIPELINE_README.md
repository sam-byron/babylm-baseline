# Sentence-Aware Training Pipeline for LTG-BERT

This directory contains the **fixed data preprocessing pipeline** that resolves the critical sentence boundary destruction issue identified in the original BNC Custom LTG-BERT training.

## Problem Summary

The original pipeline destroyed sentence boundaries during data preprocessing, concatenating entire documents (3K-25K tokens) instead of processing individual sentences (15-30 tokens). This prevented the model from learning proper syntactic patterns, explaining the significant performance gap on BLiMP Filtered tasks (59.6% vs expected 70%+).

## Solution: Sentence-Aware Processing

The new pipeline preserves sentence boundaries throughout the entire data processing chain, enabling proper syntax learning.

## Pipeline Components

### 1. Sentence Extraction (`prepare_data_sentence_aware.py`)
- **Purpose**: Extract individual sentences from BNC markdown files
- **Input**: BNC converted markdown files (from `convert_bnc.py`)
- **Output**: Dataset of individual sentences
- **Key Features**:
  - Preserves dialogue structure with speaker attribution
  - Proper sentence segmentation on punctuation
  - Filters out very short fragments
  - Maintains original sentence meaning

### 2. Sentence-Aware Tokenization (`tokenize_sentence_aware.py`)
- **Purpose**: Tokenize sentences with proper boundary markers
- **Input**: Sentence dataset from step 1
- **Output**: Tokenized sequences with [CLS]/[SEP] tokens
- **Key Features**:
  - Adds proper [CLS] and [SEP] tokens for sentence boundaries
  - Packs multiple short sentences into training sequences efficiently
  - Maintains 15-30 token ideal sentence length
  - Preserves syntactic structure for learning

### 3. Sentence-Aware MLM Dataset (`sentence_aware_mlm_dataset.py`)
- **Purpose**: MLM training dataset that respects sentence boundaries
- **Input**: Tokenized sentences from step 2
- **Output**: Training batches with proper masking
- **Key Features**:
  - Masks tokens while respecting sentence boundaries
  - Avoids masking across [CLS]/[SEP] boundaries
  - Proper MLM masking strategy (80% [MASK], 10% random, 10% unchanged)
  - Efficient batch loading from chunked data

### 4. Complete Training Script (`train_sentence_aware.py`)
- **Purpose**: Orchestrates the entire sentence-aware training pipeline
- **Input**: BNC converted files, tokenizer, base model
- **Output**: Retrained model with proper syntax learning
- **Key Features**:
  - Runs all pipeline steps automatically
  - Handles intermediate data management
  - Configurable training parameters
  - Saves comprehensive training information

### 5. Validation Tools (`validate_sentence_boundaries.py`)
- **Purpose**: Validates that sentence boundaries are properly preserved
- **Input**: Original vs sentence-aware tokenized data
- **Output**: Comprehensive comparison report
- **Key Features**:
  - Quantifies the improvement in data structure
  - Compares sequence lengths and boundary tokens
  - Generates detailed validation report
  - Proves the fix is working correctly

## Usage Instructions

### Quick Start: Complete Pipeline

Run the entire sentence-aware training pipeline:

```bash
python train_sentence_aware.py bnc_converted/ ./ model_babylm_bert_ltg/ model_babylm_bert_ltg_fixed/
```

**Arguments**:
- `bnc_converted/`: Directory with BNC markdown files
- `./`: Path to tokenizer files
- `model_babylm_bert_ltg/`: Original model directory
- `model_babylm_bert_ltg_fixed/`: Output directory for retrained model

### Step-by-Step Execution

If you prefer to run each step individually:

#### Step 1: Extract Sentences
```bash
python prepare_data_sentence_aware.py bnc_converted/ sentence_dataset/
```

#### Step 2: Tokenize with Boundaries
```bash
python tokenize_sentence_aware.py sentence_dataset/ ./ tokenized_sentences/
```

#### Step 3: Train Model
```bash
python train_sentence_aware.py bnc_converted/ ./ model_babylm_bert_ltg/ model_babylm_bert_ltg_fixed/
```

### Validation

Validate that the fix is working:

```bash
python validate_sentence_boundaries.py model_babylm_bert_ltg/chunk0.pt sentence_pipeline_data/tokenized_sentences/chunk0.pt ./
```

This will generate a detailed report showing the before/after comparison.

## Expected Improvements

### Data Structure
- **Before**: 26,749 average tokens per sequence (entire documents)
- **After**: 15-30 tokens per sequence (proper sentences)
- **Improvement**: 1,337x more appropriate sequence length

### Sentence Boundaries
- **Before**: 0 [CLS]/[SEP] tokens in dataset
- **After**: Proper boundary tokens for every sentence
- **Result**: Model can learn syntactic patterns

### BLiMP Performance
- **Before**: 59.6% on BLiMP Filtered (syntax tasks)
- **Expected After**: 70%+ (10-15% improvement)
- **Reason**: Model can now learn proper syntax from sentence structure

## File Dependencies

```
BNC XML files
    ↓ (convert_bnc.py - already working)
BNC Markdown files
    ↓ (prepare_data_sentence_aware.py - NEW)
Sentence Dataset
    ↓ (tokenize_sentence_aware.py - NEW)
Tokenized Sentences
    ↓ (sentence_aware_mlm_dataset.py - NEW)
Training Batches
    ↓ (train_sentence_aware.py - NEW)
Retrained Model
```

## Technical Details

### Sentence Extraction
- Handles BNC dialogue format: `[UNK]: 'Well now James, what can we do for you?'`
- Splits on sentence-ending punctuation (`.`, `!`, `?`)
- Preserves speaker attribution and context
- Filters sentences with <3 words to remove fragments

### Tokenization Strategy
- Adds [CLS] at sequence start, [SEP] at sequence end
- Packs multiple short sentences into max_length sequences
- Maintains sentence boundaries within packed sequences
- Uses efficient chunked storage for large datasets

### MLM Masking
- Respects sentence boundaries (no masking across [CLS]/[SEP])
- Standard BERT masking probabilities (15% of tokens)
- Avoids masking special tokens
- Maintains syntactic coherence within sentences

### Training Configuration
- 3 epochs (adjustable)
- Batch size 8 with gradient accumulation
- Learning rate 5e-5
- FP16 precision if GPU available
- Saves checkpoints every 1000 steps

## Requirements

The pipeline requires the same dependencies as the original project:
- `transformers`
- `torch`
- `datasets`
- `accelerate`

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're in the correct conda environment with all dependencies
2. **Memory issues**: Reduce batch size in training arguments
3. **Path errors**: Use absolute paths for all directory arguments
4. **Tokenizer mismatch**: Ensure tokenizer path contains the custom LTG tokenizer files

### Verification Steps

1. Check sentence extraction worked:
   ```bash
   python -c "from datasets import load_from_disk; ds = load_from_disk('sentence_dataset'); print(f'Sentences: {len(ds)}, Sample: {ds[0]}')"
   ```

2. Verify tokenization preserved boundaries:
   ```bash
   python -c "import torch; chunk = torch.load('tokenized_sentences/chunk0.pt'); print(f'Sequences: {len(chunk[\"input_ids\"])}, Avg length: {chunk[\"input_ids\"].shape[1]}')"
   ```

3. Confirm training data structure:
   ```bash
   python sentence_aware_mlm_dataset.py tokenized_sentences/ ./
   ```

## Next Steps

After retraining with the sentence-aware pipeline:

1. **Evaluate on BLiMP**: Test the retrained model on BLiMP Filtered tasks
2. **Compare Performance**: Document the improvement in syntax-dependent tasks
3. **Analyze Results**: Verify that specific syntactic phenomena (NPI licensing, argument structure) show improvement
4. **Document Success**: Update the main project with the successful fix

## Support

If you encounter issues with the sentence-aware pipeline:

1. Check that all input paths exist and contain expected data
2. Verify conda environment has all required packages
3. Review the validation report to confirm sentence boundaries are preserved
4. Check training logs for any model-specific errors

The pipeline has been designed to be robust and provide clear error messages for common issues.
