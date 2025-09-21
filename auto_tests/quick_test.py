#!/usr/bin/env python3
"""
Quick test runner for the simplified BLIMP tests.
Usage: python quick_test.py
"""


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import sys
import os
import subprocess

# Add conda activation to the environment
def run_with_conda():
    """Run the tests directly without conda activation (already in environment)."""
    
    # Since we're already in the torch-black environment, just run the test directly
    try:
        result = subprocess.run([sys.executable, "auto_tests/test_blimp_simple.py"], 
                              capture_output=True, text=True, cwd="/home/sam-byron/engineering/ML/playground/babylm/bnc/bl-bnc-custom-ltg-bert")
        
        print("=== TEST OUTPUT ===")
        print(result.stdout)
        
        if result.stderr:
            print("=== ERRORS ===")
            print(result.stderr)
        
        print(f"=== EXIT CODE: {result.returncode} ===")
        return result.returncode == 0
    
    except Exception as e:
        print(f"Failed to run tests: {e}")
        return False

def run_smoke_test():
    """Run just a basic smoke test."""
    try:
        # Test basic imports
        print("Testing basic imports...")
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        # Test that we can import from the current directory
        sys.path.insert(0, os.path.dirname(__file__))
        
        try:
            from ltg_bert import LtgBertForMaskedLM
            print("✅ LtgBert import successful")
        except ImportError as e:
            print(f"⚠️  LtgBert import failed: {e}")
        
        try:
            from blimp import is_right, parse_arguments
            print("✅ BLIMP functions import successful")
        except ImportError as e:
            print(f"⚠️  BLIMP imports failed: {e}")
        
        # Test basic tensor operations
        x = torch.randn(3, 4)
        y = torch.softmax(x, dim=-1)
        print(f"✅ Basic tensor operations work: {y.shape}")
        
        return True
    
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        success = run_smoke_test()
    else:
        success = run_with_conda()
    
    sys.exit(0 if success else 1)