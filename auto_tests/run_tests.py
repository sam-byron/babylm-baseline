#!/usr/bin/env python3
"""
Simple test runner for BLiMP evaluation tests.
Run with: python run_tests.py
"""


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import sys
import os
import unittest
from io import StringIO
import traceback

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

def run_specific_test_class(test_class_name):
    """Run a specific test class."""
    try:
        import test_blimp
        suite = unittest.TestLoader().loadTestsFromName(f'test_blimp.{test_class_name}')
        runner = unittest.TextTestRunner(verbosity=2, buffer=True)
        return runner.run(suite)
    except Exception as e:
        print(f"Error running {test_class_name}: {e}")
        return None

def run_all_tests():
    """Run all tests with detailed reporting."""
    print("="*60)
    print("RUNNING BLIMP EVALUATION TESTS")
    print("="*60)
    
    # Import test module
    try:
        import test_blimp
    except ImportError as e:
        print(f"Failed to import test module: {e}")
        return False
    
    # Get all test classes
    test_classes = [
        'TestBlimpTokenization',
        'TestBlimpPrepareFunction', 
        'TestBlimpModelInteraction',
        'TestBlimpEvaluation',
        'TestBlimpDataHandling',
        'TestBlimpArgumentParsing',
        'TestBlimpModelSetup',
        'TestBlimpIntegration',
        'TestBlimpPerformance'
    ]
    
    all_results = []
    
    for test_class in test_classes:
        print(f"\n{'='*20} {test_class} {'='*20}")
        result = run_specific_test_class(test_class)
        if result:
            all_results.append((test_class, result))
            if not result.wasSuccessful():
                print(f"❌ {test_class} had failures or errors")
            else:
                print(f"✅ {test_class} passed all tests")
        else:
            print(f"🔥 {test_class} failed to run")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_class, result in all_results:
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        
        status = "✅ PASS" if result.wasSuccessful() else "❌ FAIL"
        print(f"{status} {test_class}: {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
    
    print(f"\nOverall: {total_tests} tests, {total_failures} failures, {total_errors} errors")
    
    if total_failures > 0 or total_errors > 0:
        print("\n🔍 DETAILED FAILURE REPORT:")
        for test_class, result in all_results:
            if result.failures or result.errors:
                print(f"\n{test_class}:")
                for test, error in result.failures:
                    print(f"  FAIL {test}")
                    print(f"       {error.strip().split(chr(10))[-1]}")
                for test, error in result.errors:
                    print(f"  ERROR {test}")
                    print(f"        {error.strip().split(chr(10))[-1]}")
    
    return total_failures == 0 and total_errors == 0

def quick_smoke_test():
    """Run a quick smoke test to verify basic functionality."""
    print("Running quick smoke test...")
    
    try:
        # Test imports
        from tokenizers import Tokenizer
        from ltg_bert import LtgBertForMaskedLM
        from ltg_bert_config import LtgBertConfig
        import torch
        print("✅ All imports successful")
        
        # Test basic torch functionality
        device = torch.device('cpu')
        tensor = torch.randn(2, 3)
        assert tensor.shape == (2, 3)
        print("✅ PyTorch working correctly")
        
        # Test basic blimp function imports
        from blimp import parse_arguments, setup_training
        print("✅ BLiMP functions importable")
        
        return True
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run BLiMP evaluation tests")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test only")
    parser.add_argument("--class", dest="test_class", help="Run specific test class")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.quick:
        success = quick_smoke_test()
        sys.exit(0 if success else 1)
    elif args.test_class:
        result = run_specific_test_class(args.test_class)
        sys.exit(0 if result and result.wasSuccessful() else 1)
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)