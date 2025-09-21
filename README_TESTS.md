# BLiMP Evaluation Testing Suite

This directory contains comprehensive tests for the BLiMP (Benchmark of Linguistic Minimal Pairs) evaluation script used to assess the linguistic competence of BERT-like language models.

## Overview

The BLiMP evaluation tests a model's ability to distinguish between grammatically correct and incorrect sentence pairs across various linguistic phenomena. This test suite validates the correctness of the evaluation pipeline.

## Files

- **`test_blimp.py`** - Main test suite with comprehensive test cases
- **`run_tests.py`** - Test runner with detailed reporting
- **`generate_mock_data.py`** - Generates mock data for testing
- **`README_TESTS.md`** - This file

## Test Categories

### 1. TestBlimpTokenization
Tests the tokenization and token ID conversion logic:
- Basic token-to-ID conversion
- Unknown token handling ([UNK] fallback)
- Special token mapping ([MASK], [PAD], [CLS], [SEP])

### 2. TestBlimpPrepareFunction  
Tests the `prepare` function within `is_right`:
- Tensor shape validation
- Attention mask creation
- Sequence padding logic
- Special token insertion ([CLS], [SEP])

### 3. TestBlimpModelInteraction
Tests interaction with the language model:
- Model forward pass validation
- Log probability calculations
- Logit-to-probability conversion
- Softmax operations

### 4. TestBlimpEvaluation
Tests the core evaluation functions:
- `is_right` function with biased models
- Single pair evaluation (`evaluate`)
- Multi-group evaluation (`evaluate_all`)
- Accuracy calculations

### 5. TestBlimpDataHandling
Tests data loading and structure validation:
- BLiMP data format validation
- Pickle file loading/saving
- Data structure integrity

### 6. TestBlimpArgumentParsing
Tests command-line argument parsing:
- Default argument validation
- Custom argument handling
- Path validation

### 7. TestBlimpModelSetup
Tests model and device setup:
- CUDA availability checks
- Model configuration loading
- Device assignment

### 8. TestBlimpIntegration
Integration tests for the complete pipeline:
- End-to-end evaluation workflow
- Edge case handling
- Error condition testing

### 9. TestBlimpPerformance
Tests performance and efficiency:
- Memory leak detection
- Resource usage validation

## Running Tests

### Quick Start
```bash
# Run all tests
python run_tests.py

# Run quick smoke test
python run_tests.py --quick

# Run specific test class
python run_tests.py --class TestBlimpTokenization
```

### Detailed Test Execution

#### 1. Generate Mock Data (First Time Only)
```bash
python generate_mock_data.py --output-dir test_data
```

#### 2. Run Full Test Suite
```bash
python run_tests.py
```

#### 3. Run Individual Test Categories
```bash
# Tokenization tests
python run_tests.py --class TestBlimpTokenization

# Model interaction tests  
python run_tests.py --class TestBlimpModelInteraction

# Integration tests
python run_tests.py --class TestBlimpIntegration
```

### Using Standard unittest
```bash
# Run all tests with standard unittest
python -m unittest test_blimp -v

# Run specific test class
python -m unittest test_blimp.TestBlimpEvaluation -v

# Run specific test method
python -m unittest test_blimp.TestBlimpTokenization.test_token_conversion_basic -v
```

## Test Data

The test suite uses mock data to avoid dependencies on large model files:

### Mock BLiMP Data Structure
```python
{
    "anaphor_agreement": [
        {
            "sentence_good": "The woman likes herself",
            "sentence_bad": "The woman likes himself" 
        }
    ],
    "subject_verb_agreement": [
        {
            "sentence_good": "The cat runs quickly",
            "sentence_bad": "The cat run quickly"
        }
    ]
}
```

### Mock Tokenizer Vocabulary
Common tokens with realistic IDs:
- Special tokens: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`
- Common words: "the", "cat", "runs", etc.
- Linguistic phenomena tokens: pronouns, reflexives, quantifiers

## Key Test Scenarios

### 1. Grammaticality Preference
Tests that models correctly prefer grammatical over ungrammatical sentences:
```python
# Should return True (prefers grammatical)
is_right("the cat runs", "the cat run", good_model, tokenizer, device)
```

### 2. Token Handling
Tests proper handling of unknown and special tokens:
```python
# Unknown tokens should map to [UNK]
tokenizer.token_to_id("unknown_word")  # Returns None
# Should be handled gracefully in evaluation
```

### 3. Attention Masking
Tests that attention masks correctly exclude padding:
```python
attention_mask = [True, True, True, False, False]  # Last 2 are padding
# Model should ignore padded positions
```

### 4. Batch Processing
Tests that batched evaluation works correctly:
```python
# Good and bad sentences processed together
combined_input = torch.cat([good_input_ids, bad_input_ids], dim=0)
```

## Expected Test Results

### Successful Test Run Output
```
==================================================
RUNNING BLIMP EVALUATION TESTS
==================================================

==================== TestBlimpTokenization ====================
✅ TestBlimpTokenization passed all tests

==================== TestBlimpModelInteraction ====================  
✅ TestBlimpModelInteraction passed all tests

==================== TestBlimpEvaluation ====================
✅ TestBlimpEvaluation passed all tests

...

Overall: 47 tests, 0 failures, 0 errors
```

### Failure Indicators
- `❌ FAIL` - Test assertions failed
- `🔥 ERROR` - Exception during test execution  
- Memory usage growth > 2x during performance tests

## Mock vs Real Data

### Mock Data Benefits
- No dependency on large model files
- Predictable, controlled test conditions
- Fast test execution
- Isolated testing of individual components

### Testing with Real Data
To test with real BLiMP data and models:

1. Update file paths in test arguments
2. Ensure CUDA availability for GPU tests  
3. Allow longer test execution times
4. Monitor memory usage on large models

## Debugging Failed Tests

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'ltg_bert'
   ```
   - Ensure all dependencies are installed
   - Check PYTHONPATH includes project directory

2. **Shape Mismatches**
   ```
   RuntimeError: The expanded size of the tensor (20) must match the existing size (12)
   ```
   - Usually indicates attention mask/input dimension mismatch
   - Check `prepare` function tensor operations

3. **Device Errors**
   ```
   RuntimeError: Expected all tensors to be on the same device
   ```
   - Ensure all tensors are moved to the same device
   - Check device consistency in mock setup

### Debug Mode
Add debug prints to tests:
```python
def test_with_debug(self):
    print(f"Input shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    # ... test logic
```

## Contributing New Tests

### Adding Test Cases
1. Identify the component/function to test
2. Create mock setup if needed
3. Write test with clear assertions
4. Add edge case testing
5. Update documentation

### Test Naming Convention
- `test_<functionality>_<scenario>` 
- Example: `test_tokenization_unknown_tokens`

### Mock Creation Guidelines
- Use realistic but simplified data
- Ensure deterministic behavior
- Include edge cases in mock data
- Document mock behavior

## Performance Benchmarks

Expected test execution times:
- Quick smoke test: < 5 seconds
- Full test suite: < 30 seconds  
- Individual test class: < 10 seconds

Memory usage should remain stable across test runs.

## Continuous Integration

For CI/CD integration:
```bash
# Add to your CI pipeline
python run_tests.py --quick  # Fast validation
python run_tests.py          # Full test suite
```

Exit codes:
- `0` - All tests passed
- `1` - Tests failed or errors occurred