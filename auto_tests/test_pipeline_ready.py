#!/usr/bin/env python3
"""
Quick test to verify the complete pipeline is ready to run
"""


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import os
import sys

def check_file_exists(filepath, description):
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (NOT FOUND)")
        return False

def check_python_imports():
    """Test critical imports"""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not found")
        return False
    
    try:
        import datasets
        print(f"✅ Datasets available")
    except ImportError:
        print("❌ Datasets not found")
        return False
        
    return True

def main():
    print("🔍 Testing Complete Pipeline Readiness")
    print("="*50)
    
    all_good = True
    
    # Check core files
    files_to_check = [
        ("run_complete_pipeline.py", "Main pipeline script"),
        ("transformer_trainer.py", "Pretraining script"),
        ("finetune_classification.py", "Fine-tuning script"),
        ("accelerate_config.yaml", "Accelerate config"),
        ("model.py", "Model definition"),
        ("config.py", "Config definition"),
        ("tokenizer.py", "Tokenizer"),
    ]
    
    for filepath, desc in files_to_check:
        if not check_file_exists(filepath, desc):
            all_good = False
    
    print("\n🐍 Testing Python Dependencies")
    print("-" * 30)
    if not check_python_imports():
        all_good = False
    
    # Check data availability
    print("\n📁 Checking Data Availability")
    print("-" * 30)
    data_files = [
        ("data/pretrain/train.md", "Training data"),
        ("data/pretrain/wordpiece_vocab.json", "Vocabulary"),
    ]
    
    for filepath, desc in data_files:
        check_file_exists(filepath, desc)  # Not critical for pipeline test
    
    print("\n" + "="*50)
    if all_good:
        print("🎉 PIPELINE IS READY TO RUN!")
        print("📝 Next steps:")
        print("   1. python run_complete_pipeline.py")
        print("   2. Wait for pretraining + fine-tuning + evaluation")
        print("   3. Check results/ directory for lm_eval outputs")
    else:
        print("❌ PIPELINE NOT READY - Fix missing files/dependencies")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
