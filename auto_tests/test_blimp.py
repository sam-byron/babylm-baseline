#!/usr/bin/env python3
# coding=utf-8


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import unittest
import torch
import torch.nn.functional as F
import tempfile
import os
import gzip
import pickle
from unittest.mock import Mock, MagicMock, patch
import numpy as np

# Import the modules we're testing
from blimp import (
    is_right, 
    evaluate, 
    evaluate_all, 
    prepare_model, 
    setup_training,
    parse_arguments
)
from tokenizers import Tokenizer
from ltg_bert import LtgBertForMaskedLM
from ltg_bert_config import LtgBertConfig


class TestBlimpTokenization(unittest.TestCase):
    """Test tokenization and token ID conversion logic."""
    
    def setUp(self):
        """Set up mock tokenizer for testing."""
        self.mock_tokenizer = Mock()
        # Setup common token mappings
        self.token_map = {
            "[MASK]": 103,
            "[PAD]": 0,
            "[CLS]": 101,
            "[SEP]": 102,
            "[UNK]": 100,
            "the": 1996,
            "cat": 4937,
            "dog": 3899,
            "runs": 3216,
            "quickly": 5221,
            "slowly": 3254,
            "unknown_token": None  # This should return None
        }
        
        def mock_token_to_id(token):
            return self.token_map.get(token, None)
            
        self.mock_tokenizer.token_to_id = mock_token_to_id
        self.device = torch.device('cpu')
        
    def test_token_conversion_basic(self):
        """Test basic token to ID conversion."""
        good_sentence = "the cat runs quickly"
        bad_sentence = "the dog runs slowly"
        
        # Mock model for testing
        mock_model = Mock()
        mock_model.return_value = Mock()
        mock_model.return_value.logits = torch.randn(10, 5, 30522)  # Arbitrary logits
        
        # This should not raise an exception
        try:
            result = is_right(good_sentence, bad_sentence, mock_model, self.mock_tokenizer, self.device)
            self.assertIsInstance(result, bool)
        except Exception as e:
            self.fail(f"Token conversion failed: {e}")
            
    def test_unknown_token_handling(self):
        """Test handling of unknown tokens."""
        good_sentence = "the unknown_token runs"
        bad_sentence = "the cat runs"
        
        mock_model = Mock()
        mock_model.return_value = Mock()
        mock_model.return_value.logits = torch.randn(8, 4, 30522)
        
        # Should handle unknown tokens by converting to [UNK]
        try:
            result = is_right(good_sentence, bad_sentence, mock_model, self.mock_tokenizer, self.device)
            self.assertIsInstance(result, bool)
        except Exception as e:
            self.fail(f"Unknown token handling failed: {e}")


class TestBlimpPrepareFunction(unittest.TestCase):
    """Test the prepare function within is_right."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        
    def test_prepare_function_shapes(self):
        """Test that prepare function returns correct tensor shapes."""
        # Create mock tokens
        tokens = torch.tensor([1996, 4937, 3216], dtype=torch.long)  # "the cat runs"
        cls_index = torch.tensor([101], dtype=torch.long)  # [CLS]
        sep_index = torch.tensor([102], dtype=torch.long)  # [SEP]
        pad_index = 0  # [PAD]
        mask_index = 103  # [MASK]
        
        # Simulate the prepare function logic
        padding = 2
        full_tokens = torch.cat([cls_index, tokens, sep_index, torch.full((padding,), fill_value=pad_index)])
        
        # Check basic concatenation
        expected_length = 1 + len(tokens) + 1 + padding  # CLS + tokens + SEP + padding
        self.assertEqual(len(full_tokens), expected_length)
        
        # Check token values
        self.assertEqual(full_tokens[0].item(), 101)  # CLS
        self.assertEqual(full_tokens[-3].item(), 102)  # SEP (before padding)
        self.assertEqual(full_tokens[-1].item(), 0)   # Last padding token
        
    def test_attention_mask_creation(self):
        """Test attention mask creation logic."""
        # Create a sample input_ids tensor
        input_ids = torch.tensor([
            [101, 1996, 4937, 102, 0, 0],  # [CLS] the cat [SEP] [PAD] [PAD]
            [101, 1996, 3899, 102, 0, 0],  # [CLS] the dog [SEP] [PAD] [PAD]
        ])
        
        # Create attention mask (True for positions to mask out)
        padding = 2
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        attention_mask[:, attention_mask.size(-1) - padding:] = True
        
        # Check that only padding positions are masked
        expected_mask = torch.tensor([
            [False, False, False, False, True, True],
            [False, False, False, False, True, True]
        ], dtype=torch.bool)
        
        self.assertTrue(torch.equal(attention_mask, expected_mask))


class TestBlimpModelInteraction(unittest.TestCase):
    """Test interaction with the language model."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        
    def create_mock_model(self, batch_size, seq_len, vocab_size=30522):
        """Create a mock model that returns realistic logits."""
        mock_model = Mock()
        mock_output = Mock()
        
        # Create logits with higher probability for certain tokens
        logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1
        
        # Make some tokens more likely for the "good" examples
        if batch_size > 1:
            # Boost probability for first half (good examples)
            logits[:batch_size//2, :, 1996] += 2.0  # "the"
            logits[:batch_size//2, :, 4937] += 1.5  # "cat"
            
        mock_output.logits = logits
        mock_model.return_value = mock_output
        return mock_model
        
    def test_model_forward_pass(self):
        """Test that model forward pass works correctly."""
        batch_size, seq_len = 6, 8
        mock_model = self.create_mock_model(batch_size, seq_len)
        
        # Create mock inputs
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        # Call model
        output = mock_model(input_ids, attention_mask)
        
        # Check output shape
        self.assertEqual(output.logits.shape, (batch_size, seq_len, 30522))
        
    def test_log_probability_calculation(self):
        """Test log probability calculation from logits."""
        # Create sample logits
        logits = torch.tensor([
            [1.0, 2.0, 0.5],  # Position 0
            [0.5, 1.5, 2.0],  # Position 1
        ])
        
        # Calculate log probabilities
        log_p = F.log_softmax(logits, dim=-1)
        
        # Check that probabilities sum to 1 (in linear space)
        probs = torch.exp(log_p)
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(2)))
        
        # Check that higher logits lead to higher log probabilities
        self.assertGreater(log_p[0, 1].item(), log_p[0, 0].item())  # 2.0 > 1.0
        self.assertGreater(log_p[1, 2].item(), log_p[1, 1].item())  # 2.0 > 1.5


class TestBlimpEvaluation(unittest.TestCase):
    """Test the evaluation functions."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.mock_tokenizer = Mock()
        
        # Setup tokenizer
        token_map = {
            "[MASK]": 103, "[PAD]": 0, "[CLS]": 101, "[SEP]": 102, "[UNK]": 100,
            "the": 1996, "cat": 4937, "dog": 3899, "runs": 3216, "run": 3216
        }
        self.mock_tokenizer.token_to_id = lambda token: token_map.get(token, None)
        
    def create_biased_model(self, prefer_good=True):
        """Create a model that prefers grammatical or ungrammatical sentences."""
        mock_model = Mock()
        
        def mock_forward(input_ids, attention_mask):
            batch_size, seq_len = input_ids.shape
            logits = torch.randn(batch_size, seq_len, 30522) * 0.1
            
            # If prefer_good, give higher logits to first half of batch (good examples)
            # Otherwise, give higher logits to second half (bad examples)
            if prefer_good:
                logits[:batch_size//2] += 1.0
            else:
                logits[batch_size//2:] += 1.0
                
            output = Mock()
            output.logits = logits
            return output
            
        mock_model.side_effect = mock_forward
        return mock_model
        
    def test_is_right_with_good_model(self):
        """Test is_right with a model that prefers grammatical sentences."""
        good_sentence = "the cat runs"
        bad_sentence = "the cat run"  # Grammatically incorrect
        
        model = self.create_biased_model(prefer_good=True)
        
        result = is_right(good_sentence, bad_sentence, model, self.mock_tokenizer, self.device)
        self.assertTrue(result, "Model should prefer grammatical sentence")
        
    def test_is_right_with_bad_model(self):
        """Test is_right with a model that prefers ungrammatical sentences."""
        good_sentence = "the cat runs"
        bad_sentence = "the cat run"
        
        model = self.create_biased_model(prefer_good=False)
        
        result = is_right(good_sentence, bad_sentence, model, self.mock_tokenizer, self.device)
        self.assertFalse(result, "Poorly trained model should prefer ungrammatical sentence")
        
    def test_evaluate_single_pair(self):
        """Test evaluation on a single sentence pair."""
        pairs = {
            "sentence_good": "the cat runs",
            "sentence_bad": "the cat run"
        }
        
        model = self.create_biased_model(prefer_good=True)
        
        # Mock the pairs to have length 1 for the accuracy calculation
        with patch('builtins.len', return_value=1):
            accuracy = evaluate(model, self.mock_tokenizer, pairs, self.device)
            
        self.assertEqual(accuracy, 100.0, "Perfect model should get 100% accuracy")
        
    def test_evaluate_all_groups(self):
        """Test evaluation across multiple BLiMP groups."""
        # Create mock BLiMP data structure
        blimp_data = {
            "group1": [
                {"sentence_good": "the cat runs", "sentence_bad": "the cat run"}
            ],
            "group2": [
                {"sentence_good": "the dog runs", "sentence_bad": "the dog run"}
            ]
        }
        
        model = self.create_biased_model(prefer_good=True)
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            evaluate_all(model, self.mock_tokenizer, blimp_data, self.device)
            
        # Check that evaluation completed (print was called)
        mock_print.assert_called()
        call_args = str(mock_print.call_args_list[-1])
        self.assertIn("BLiMP accuracy", call_args)


class TestBlimpDataHandling(unittest.TestCase):
    """Test data loading and handling."""
    
    def test_blimp_data_structure(self):
        """Test expected BLiMP data structure."""
        # Create sample BLiMP data
        sample_data = {
            "anaphor_agreement": [
                {
                    "sentence_good": "The woman likes herself",
                    "sentence_bad": "The woman likes himself"
                }
            ],
            "determiner_noun_agreement": [
                {
                    "sentence_good": "This book is good",
                    "sentence_bad": "These book is good"
                }
            ]
        }
        
        # Test data structure
        self.assertIsInstance(sample_data, dict)
        for group_name, group_data in sample_data.items():
            self.assertIsInstance(group_data, list)
            for pair in group_data:
                self.assertIn("sentence_good", pair)
                self.assertIn("sentence_bad", pair)
                self.assertIsInstance(pair["sentence_good"], str)
                self.assertIsInstance(pair["sentence_bad"], str)
                
    def test_pickle_data_loading(self):
        """Test loading BLiMP data from pickle file."""
        sample_data = {
            "test_group": [
                {
                    "sentence_good": "The cat runs",
                    "sentence_bad": "The cat run"
                }
            ]
        }
        
        # Create temporary pickle file
        with tempfile.NamedTemporaryFile(suffix='.pkl.gz', delete=False) as f:
            with gzip.open(f.name, 'wb') as gz_f:
                pickle.dump(sample_data, gz_f)
            temp_path = f.name
            
        try:
            # Load data
            with gzip.open(temp_path, 'rb') as f:
                loaded_data = pickle.load(f)
                
            self.assertEqual(loaded_data, sample_data)
            
        finally:
            os.unlink(temp_path)


class TestBlimpArgumentParsing(unittest.TestCase):
    """Test command line argument parsing."""
    
    def test_parse_arguments_defaults(self):
        """Test parsing with default arguments."""
        with patch('sys.argv', ['blimp.py']):
            args = parse_arguments()
            
        self.assertIn('blimp.pkl.gz', args.input_path)
        self.assertIn('wordpiece_vocab.json', args.vocab_path)
        self.assertTrue(len(args.checkpoint_path) > 0)
        
    def test_parse_arguments_custom(self):
        """Test parsing with custom arguments."""
        test_args = [
            'blimp.py',
            '--input_path', '/custom/path/blimp.pkl.gz',
            '--checkpoint_path', '/custom/checkpoint',
            '--vocab_path', '/custom/vocab.json'
        ]
        
        with patch('sys.argv', test_args):
            args = parse_arguments()
            
        self.assertEqual(args.input_path, '/custom/path/blimp.pkl.gz')
        self.assertEqual(args.checkpoint_path, '/custom/checkpoint')
        self.assertEqual(args.vocab_path, '/custom/vocab.json')


class TestBlimpModelSetup(unittest.TestCase):
    """Test model setup and configuration."""
    
    @patch('torch.cuda.is_available')
    def test_setup_training_cuda_available(self, mock_cuda):
        """Test setup when CUDA is available."""
        mock_cuda.return_value = True
        
        mock_args = Mock()
        mock_args.checkpoint_path = "/path/to/checkpoint"
        
        device, args, checkpoint = setup_training(mock_args)
        
        self.assertEqual(device.type, 'cuda')
        self.assertEqual(checkpoint, "/path/to/checkpoint")
        
    @patch('torch.cuda.is_available')
    def test_setup_training_cuda_unavailable(self, mock_cuda):
        """Test setup when CUDA is not available."""
        mock_cuda.return_value = False
        
        mock_args = Mock()
        mock_args.checkpoint_path = "/path/to/checkpoint"
        
        with self.assertRaises(AssertionError):
            setup_training(mock_args)


class TestBlimpIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        self.device = torch.device('cpu')
        
    def create_full_mock_setup(self):
        """Create a complete mock setup for integration testing."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        token_map = {
            "[MASK]": 103, "[PAD]": 0, "[CLS]": 101, "[SEP]": 102, "[UNK]": 100,
            "the": 1996, "cat": 4937, "dog": 3899, "runs": 3216, "run": 3216,
            "quickly": 5221, "slowly": 3254
        }
        mock_tokenizer.token_to_id = lambda token: token_map.get(token, None)
        
        # Mock model
        mock_model = Mock()
        def mock_forward(input_ids, attention_mask):
            batch_size, seq_len = input_ids.shape
            # Create logits that prefer the first half of the batch (good examples)
            logits = torch.randn(batch_size, seq_len, 30522) * 0.5
            logits[:batch_size//2] += 2.0  # Boost good examples
            
            output = Mock()
            output.logits = logits
            return output
            
        mock_model.side_effect = mock_forward
        
        # Mock BLiMP data
        blimp_data = {
            "test_group": [
                {"sentence_good": "the cat runs quickly", "sentence_bad": "the cat run quickly"},
                {"sentence_good": "the dog runs slowly", "sentence_bad": "the dog run slowly"}
            ]
        }
        
        return mock_tokenizer, mock_model, blimp_data
        
    def test_full_evaluation_pipeline(self):
        """Test the complete evaluation pipeline."""
        tokenizer, model, blimp_data = self.create_full_mock_setup()
        
        # Test individual is_right calls
        result1 = is_right("the cat runs", "the cat run", model, tokenizer, self.device)
        self.assertTrue(result1, "Should prefer grammatical sentence")
        
        # Test evaluate function
        pairs = {"sentence_good": "the cat runs", "sentence_bad": "the cat run"}
        with patch('builtins.len', return_value=1):
            accuracy = evaluate(model, tokenizer, pairs, self.device)
        self.assertEqual(accuracy, 100.0)
        
        # Test evaluate_all function
        with patch('builtins.print') as mock_print:
            evaluate_all(model, tokenizer, blimp_data, self.device)
        
        # Check that results were printed
        mock_print.assert_called()
        
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        tokenizer, model, _ = self.create_full_mock_setup()
        
        # Test with empty strings
        with self.assertRaises(Exception):
            is_right("", "", model, tokenizer, self.device)
            
        # Test with very long sentences
        long_sentence = " ".join(["the", "cat"] * 100)
        try:
            result = is_right(long_sentence, "the cat", model, tokenizer, self.device)
            self.assertIsInstance(result, bool)
        except Exception as e:
            # Long sentences might cause memory issues, which is expected
            self.assertIn("memory", str(e).lower(), f"Unexpected error: {e}")


class TestBlimpPerformance(unittest.TestCase):
    """Test performance and efficiency."""
    
    def test_memory_efficiency(self):
        """Test that evaluation doesn't cause memory leaks."""
        import gc
        
        # Get initial memory state
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Run a small evaluation
        tokenizer = Mock()
        tokenizer.token_to_id = lambda token: {"the": 1, "cat": 2, "runs": 3, "[MASK]": 4, "[PAD]": 0, "[CLS]": 5, "[SEP]": 6, "[UNK]": 7}.get(token, None)
        
        model = Mock()
        model.return_value = Mock()
        model.return_value.logits = torch.randn(4, 6, 100)
        
        # Run multiple evaluations
        for _ in range(10):
            is_right("the cat runs", "the cat", model, tokenizer, torch.device('cpu'))
            
        # Check memory
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Allow some growth but not excessive
        growth_ratio = final_objects / initial_objects
        self.assertLess(growth_ratio, 2.0, "Memory usage grew too much during evaluation")


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"SUMMARY: Ran {result.testsRun} tests")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.splitlines()[-1]}")
            
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.splitlines()[-1]}")
            
    print(f"{'='*50}")
    
    # Exit with proper code
    exit(0 if result.wasSuccessful() else 1)