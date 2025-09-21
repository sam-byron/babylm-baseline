#!/usr/bin/env python3
"""
Test that our save_model_official_way function in transformer_trainer.py is working correctly.
"""


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import sys
import os
sys.path.insert(0, os.getcwd())

from ltg_bert_config import LtgBertConfig
from ltg_bert import LtgBertForMaskedLM
from transformer_trainer import save_model_official_way
from transformers import AutoConfig, AutoModelForMaskedLM
import tempfile
import shutil

def test_official_saving():
    """Test that our official saving method works"""
    print("🧪 Testing official saving method from transformer_trainer.py")
    
    # Create a small test model
    config = LtgBertConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=128
    )
    
    model = LtgBertForMaskedLM(config)
    
    # Test saving
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Saving test model to {temp_dir}")
        
        # Use the function from transformer_trainer.py
        save_model_official_way(model, config, None, temp_dir)
        
        # Check that required files exist
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "modeling_ltg_bert.py",
            "configuration_ltg_bert.py"
        ]
        
        missing_files = []
        for file in required_files:
            file_path = os.path.join(temp_dir, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
        
        print("✅ All required files created")
        
        # Test loading with AutoModel
        try:
            print("🔄 Testing AutoModel loading...")
            loaded_config = AutoConfig.from_pretrained(temp_dir, trust_remote_code=True)
            loaded_model = AutoModelForMaskedLM.from_pretrained(temp_dir, trust_remote_code=True)
            
            print(f"✅ Loaded config: {type(loaded_config).__name__}")
            print(f"✅ Loaded model: {type(loaded_model).__name__}")
            
            # Test forward pass
            import torch
            input_ids = torch.randint(0, config.vocab_size, (1, 10))
            attention_mask = torch.ones(1, 10)
            
            with torch.no_grad():
                outputs = loaded_model(input_ids=input_ids, attention_mask=attention_mask)
            
            print(f"✅ Forward pass works! Output shape: {outputs.logits.shape}")
            return True
            
        except Exception as e:
            print(f"❌ AutoModel loading failed: {e}")
            return False

if __name__ == "__main__":
    success = test_official_saving()
    if success:
        print("\n🎉 Official saving method is working correctly!")
        print("✅ transformer_trainer.py will save models compatible with lm_eval")
    else:
        print("\n❌ Official saving method has issues")
        print("⚠️ Need to fix transformer_trainer.py before running pipeline")
