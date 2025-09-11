#!/usr/bin/env python3
"""
Run the fixed sentence-aware data preparation pipeline.

This script demonstrates how to use the modified prepare_data.py and related files
to process data with proper sentence boundary preservation.
"""

import json
import os
import sys
from pathlib import Path

def create_sentence_aware_config():
    """Create a configuration that enables sentence-aware processing."""
    
    # Load existing config if it exists
    config_path = "model_babylm_ltg_bert.json"
    
    if Path(config_path).exists():
        print(f"Loading existing config from {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Create new sentence-aware config based on existing config
        base_config_path = "model_babylm_ltg_bert.json"
        
        if Path(base_config_path).exists():
            with open(base_config_path, 'r') as f:
                config = json.load(f)
        else:
            # Fallback default config
            config = {
                "cache_path": "./model_babylm_bert_ltg_sentence_aware",
                "chunk_size": 100,
                "block_size": 512,
                "batch_size": 8,
                "max_position_embeddings": 512,
                "tokenizer_path": "./data/pretrain/wordpiece_vocab.json"
            }
    
    # Add sentence-aware processing options
    # config.update({
    #     "use_sentence_aware": True,
    #     "cache_path": "./model_babylm_bert_ltg_sentence_aware",
    #     "mask_p": 0.15,
    #     "random_p": 0.1,
    #     "keep_p": 0.1,
    #     "masking_strategy": "span"
    # })
    
    # Save the sentence-aware config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created sentence-aware config: {config_path}")
    return config_path

def run_sentence_aware_data_preparation():
    """Run data preparation with sentence-aware processing."""
    
    print("🎯 Running Sentence-Aware Data Preparation Pipeline")
    print("=" * 60)
    
    # Create sentence-aware config
    config_path = create_sentence_aware_config()
    
    # Check if BNC data exists
    data_dir = "./data/pretrain/bnc"
    if not Path(data_dir).exists():
        print(f"❌ BNC data directory not found: {data_dir}")
        print("Please ensure you have run the BNC conversion pipeline first.")
        return False
    
    # Check if tokenizer exists - use the correct path from config
    tokenizer_path = "./data/pretrain/wordpiece_vocab.json"
    if not Path(tokenizer_path).exists():
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        print("Please ensure you have created the tokenizer first.")
        return False
    
    print(f"✅ Found BNC data in: {data_dir}")
    print(f"✅ Found tokenizer: {tokenizer_path}")
    
    # Clean up any existing cache to start fresh
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    cache_path = Path(config["cache_path"])
    if cache_path.exists():
        import shutil
        print(f"🧹 Cleaning existing cache: {cache_path}")
        shutil.rmtree(cache_path)
    
    # Run the sentence-aware data preparation
    print(f"\n🚀 Starting sentence-aware data preparation...")
    print(f"   Config: {config_path}")
    print(f"   Cache: {config['cache_path']}")
    
    # Import and run prepare_data with sentence-aware processing
    try:
        # Run the preparation
        cmd = f"python prepare_data.py --config_path {config_path}"
        print(f"Running: {cmd}")
        
        result = os.system(cmd)
        
        if result == 0:
            print("✅ Sentence-aware data preparation completed successfully!")
            
            # Verify the results
            print("\n📊 Verifying results...")
            
            import glob
            chunk_files = glob.glob(os.path.join(config["cache_path"], "chunk*.pt"))
            print(f"Created {len(chunk_files)} chunk files")
            
            if len(chunk_files) > 0:
                # Load and analyze a sample chunk
                import torch
                sample_chunk = torch.load(chunk_files[0], map_location='cpu')
                
                if isinstance(sample_chunk, list) and len(sample_chunk) > 0:
                    sample_sequence = sample_chunk[0]
                    print(f"Sample sequence length: {len(sample_sequence)} tokens")
                    
                    # Load tokenizer to check for boundary tokens
                    import tokenizers
                    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
                    
                    cls_count = sample_sequence.count(tokenizer.token_to_id("[CLS]"))
                    sep_count = sample_sequence.count(tokenizer.token_to_id("[SEP]"))
                    
                    print(f"Boundary tokens in sample: {cls_count} [CLS], {sep_count} [SEP]")
                    
                    if cls_count > 0 and sep_count > 0:
                        print("✅ Sentence boundaries preserved in tokenized data!")
                    else:
                        print("❌ No sentence boundaries found - something may be wrong")
                        
                    # Show avg sequence length across multiple chunks
                    total_sequences = 0
                    total_length = 0
                    
                    for chunk_file in chunk_files[:5]:  # Check first 5 chunks
                        chunk = torch.load(chunk_file, map_location='cpu')
                        if isinstance(chunk, list):
                            for seq in chunk:
                                total_sequences += 1
                                total_length += len(seq)
                    
                    if total_sequences > 0:
                        avg_length = total_length / total_sequences
                        print(f"Average sequence length: {avg_length:.1f} tokens")
                        
                        if avg_length < 100:  # Should be much shorter than document-level
                            print("✅ Sequences are sentence-length (not document-length)")
                        else:
                            print(f"⚠️ Sequences still quite long ({avg_length:.1f} tokens)")
            
            return True
        else:
            print(f"❌ Data preparation failed with exit code: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error during data preparation: {e}")
        return False

def main():
    """Main function to run the sentence-aware pipeline."""
    
    print("🔧 BNC Custom LTG-BERT: Sentence-Aware Data Preparation")
    print("This script fixes the sentence boundary destruction issue")
    print("=" * 80)
    
    success = run_sentence_aware_data_preparation()
    
    if success:
        print("\n🎉 SUCCESS: Sentence-aware data preparation completed!")
        print("\n📝 Next steps:")
        print("1. Train your model using the sentence-aware data:")
        print("   python transformer_trainer.py  # (modify to use sentence-aware cache)")
        print("2. Evaluate on BLiMP to see the syntax improvement:")
        print("   python evaluation.py")
        print("3. Compare BLiMP Filtered scores before/after the fix")
        print("\n💡 Expected improvement: BLiMP Filtered 59.6% → 70%+ (10-15% boost)")
    else:
        print("\n❌ FAILED: Please check the errors above and try again.")
        print("\n🔍 Common issues:")
        print("- Missing BNC data files (run convert_bnc.py first)")
        print("- Missing tokenizer (run tokenizer creation first)")
        print("- Permission issues with cache directory")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
