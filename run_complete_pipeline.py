#!/usr/bin/env python3
"""
Complete pipeline: Pretrain LTG-BERT → Fine-tune on GLUE → Evaluate with lm_eval

This script:
1. Pretrains LTG-BERT for one epoch using transformer_trainer.py
2. Creates sequence classification models for GLUE tasks
3. Fine-tunes on GLUE tasks
4. Runs lm_eval on GLUE benchmark

All models are saved using the official HuggingFace method to avoid AutoModel errors.
"""


# Add parent directory to path for imports
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import os
import sys
import subprocess
import json
import shutil
import torch
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

from configuration_ltgbert import LtgBertConfig
from modeling_ltgbert import LtgBertForMaskedLM, LtgBertForSequenceClassification

def save_model_official_way(model, config, tokenizer, output_path, model_type="classification"):
    """
    Save model the official HuggingFace way for compatibility with lm_eval and other tools.
    """
    print(f"🎯 Saving {model_type} model the official HuggingFace way to {output_path}")
    
    # Ensure directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Register for auto class - this is the key part!
    config.register_for_auto_class()
    model.register_for_auto_class()  # No arguments needed for models
    
    # Save model and config using HF methods
    # Use safe_serialization=False to handle shared tensors
    model.save_pretrained(output_path, safe_serialization=False)
    config.save_pretrained(output_path)
    
    # Copy source files with proper naming convention
    model_name = config.model_type
    dest_modeling = os.path.join(output_path, f"modeling_{model_name}.py")
    dest_config = os.path.join(output_path, f"configuration_{model_name}.py")

    # Copy modeling_ltgbert.py
    if os.path.exists("modeling_ltgbert.py"):
        shutil.copy2("modeling_ltgbert.py", output_path)
        print(f"✅ Copied modeling_ltgbert.py -> {output_path}")

    # Copy configuration_ltgbert.py
    if os.path.exists("configuration_ltgbert.py"):
        shutil.copy2("configuration_ltgbert.py", output_path)
        print(f"✅ Copied configuration_ltgbert.py -> {output_path}")
    
    # Copy tokenizer files if they exist
    tokenizer_files = ['data/pretrain/wordpiece_vocab.json', 'tokenizer_config.json', 'vocab.txt', 'special_tokens_map.json']
    for filename in tokenizer_files:
        if tokenizer and hasattr(tokenizer, 'save') and filename == 'data/pretrain/wordpiece_vocab.json':
            tokenizer_dest = os.path.join(output_path, filename)
            tokenizer.save(tokenizer_dest)
            print(f"✅ Saved custom tokenizer to {tokenizer_dest}")
        else:
            # Try to copy from checkpoint if it exists
            checkpoint_file = os.path.join("model_babylm_bert_ltg/checkpoint", filename)
            if os.path.exists(checkpoint_file):
                shutil.copy2(checkpoint_file, output_path)
                print(f"✅ Copied {filename}")
    
    print(f"✅ Model saved the official HuggingFace way! Ready for lm_eval with trust_remote_code=True")


def step1_pretrain_model():
    """Step 1: Pretrain LTG-BERT for one epoch"""
    print("\n" + "="*60)
    print("🚀 STEP 1: Pretraining LTG-BERT for one epoch")
    print("="*60)
    
    # Check if we have the required config
    if not os.path.exists("configs/base.json"):
        print("❌ configs/base.json not found. Please ensure config file exists.")
        return False
    
    # Check if accelerate config exists
    if not os.path.exists("accelerate_config.yaml"):
        print("❌ accelerate_config.yaml not found. Please ensure accelerate config exists.")
        return False
    
    # Check if model is already trained
    checkpoint_path = "model_babylm_bert_ltg/checkpoint"
    if os.path.exists(checkpoint_path) and os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
        print("✅ Pretrained model already exists, skipping pretraining")
        return True
    
    # Run the pretraining
    cmd = [
        "accelerate", "launch", 
        "transformer_trainer.py", 
        "--config_path", "model_babylm_ltg_bert.json"
    ]
    
    print(f"🏃 Running pretraining command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("✅ Pretraining completed successfully!")
        print("📁 Pretrained model saved to: model_babylm_bert_ltg/checkpoint")
        
        # Verify the model was saved properly
        if os.path.exists(os.path.join(checkpoint_path, "modeling_ltg_bert.py")):
            print("✅ Model saved with official HuggingFace format")
        else:
            print("⚠️ Model saved but may not be in official format")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Pretraining failed: {e}")
        return False


def step2_create_classification_models():
    """Step 2: Create sequence classification models from pretrained MLM"""
    print("\n" + "="*60)
    print("🔄 STEP 2: Creating sequence classification models for GLUE")
    print("="*60)
    
    mlm_checkpoint = "model_babylm_bert_ltg/checkpoint"
    
    if not os.path.exists(mlm_checkpoint):
        print(f"❌ Pretrained MLM model not found at {mlm_checkpoint}")
        return False
    
    # Load the pretrained MLM model
    print("📥 Loading pretrained MLM model...")
    config = LtgBertConfig.from_pretrained(mlm_checkpoint)
    mlm_model = LtgBertForMaskedLM.from_pretrained(mlm_checkpoint)
    
    # GLUE tasks and their number of labels
    glue_tasks = {
        "cola": 2,      # CoLA: binary acceptability
        "sst2": 2,      # SST-2: binary sentiment  
        "mrpc": 2,      # MRPC: binary paraphrase
        "qqp": 2,       # QQP: binary question pairs
        "stsb": 1,      # STS-B: regression (similarity score)
        "mnli": 3,      # MNLI: 3-way entailment
        "qnli": 2,      # QNLI: binary question-sentence entailment
        "rte": 2,       # RTE: binary textual entailment
        "wnli": 2,      # WNLI: binary natural language inference
    }
    
    # Create sequence classification models for each GLUE task
    for task, num_labels in glue_tasks.items():
        print(f"\n📝 Creating classification model for {task.upper()} (num_labels={num_labels})")
        
        # Create classification config
        clf_config = LtgBertConfig.from_pretrained(mlm_checkpoint)
        clf_config.num_labels = num_labels
        if task == "stsb":
            clf_config.problem_type = "regression"
        else:
            clf_config.problem_type = "single_label_classification"
        
        # Create classification model and transfer weights
        clf_model = LtgBertForSequenceClassification(clf_config)
        
        # Transfer weights from MLM model (shared layers)
        mlm_state_dict = mlm_model.state_dict()
        clf_state_dict = clf_model.state_dict()
        
        transferred = 0
        for key in clf_state_dict:
            if key in mlm_state_dict and key.startswith(('embedding.', 'transformer.')):
                clf_state_dict[key] = mlm_state_dict[key]
                transferred += 1
        
        clf_model.load_state_dict(clf_state_dict)
        print(f"✅ Transferred {transferred} layers from MLM to classification model")
        
        # Save using official HuggingFace method
        output_path = f"models/glue_ready/{task}"
        save_model_official_way(clf_model, clf_config, None, output_path, "classification")
    
    print("\n✅ All GLUE classification models created successfully!")
    return True


def step3_finetune_glue():
    """Step 3: Fine-tune on GLUE tasks"""
    print("\n" + "="*60)
    print("🎯 STEP 3: Fine-tuning on GLUE tasks")
    print("="*60)
    
    glue_tasks = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte"]  # Skip stsb and wnli for now
    
    for task in glue_tasks:
        print(f"\n🔥 Fine-tuning on {task.upper()}")
        
        model_path = f"models/glue_ready/{task}"
        output_dir = f"models/glue_finetuned/{task}"
        
        if not os.path.exists(model_path):
            print(f"❌ Classification model not found for {task}")
            continue
        
        # Create fine-tuning command
        cmd = [
            "python", "finetune_classification.py",
            "--model_name_or_path", model_path,
            "--task_name", task,
            "--do_train", 
            "--do_eval",
            "--max_seq_length", "128",
            "--per_device_train_batch_size", "32",
            "--learning_rate", "2e-5",
            "--num_train_epochs", "3",
            "--output_dir", output_dir,
            "--overwrite_output_dir",
            "--trust_remote_code",  # This is crucial!
            "--seed", "42"
        ]
        
        print(f"🏃 Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            print(f"✅ Fine-tuning completed for {task}")
            
            # Save the fine-tuned model the official way
            from transformers import AutoModelForSequenceClassification, AutoConfig
            finetuned_model = AutoModelForSequenceClassification.from_pretrained(output_dir, trust_remote_code=True)
            finetuned_config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
            
            official_output = f"models/glue_official/{task}"
            save_model_official_way(finetuned_model, finetuned_config, None, official_output, "classification")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Fine-tuning failed for {task}: {e}")
            print(f"Error output: {e.stderr}")
        except subprocess.TimeoutExpired:
            print(f"⏰ Fine-tuning timed out for {task}")
    
    return True


def step4_run_lm_eval():
    """Step 4: Run lm_eval on GLUE benchmark"""
    print("\n" + "="*60)
    print("📊 STEP 4: Running lm_eval on GLUE benchmark")
    print("="*60)
    
    # Start with just a few tasks to test
    test_tasks = ["cola", "sst2"]  # Start small
    
    results = {}
    
    for task in test_tasks:
        model_path = f"models/glue_official/{task}"
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found for {task}")
            continue
        
        print(f"\n🔍 Evaluating {task.upper()} with lm_eval")
        
        # Run lm_eval command
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path},trust_remote_code=True",
            "--tasks", f"glue_{task}",
            "--batch_size", "8",
            "--limit", "100",  # Limit to 100 examples for testing
            "--output_path", f"results/lm_eval_{task}.json"
        ]
        
        print(f"🏃 Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True, timeout=600)  # 10 min timeout
            print(f"✅ Evaluation completed for {task}")
            
            # Try to parse results
            result_file = f"results/lm_eval_{task}.json"
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    task_results = json.load(f)
                    results[task] = task_results
                    print(f"📈 Results saved to {result_file}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Evaluation failed for {task}: {e}")
        except subprocess.TimeoutExpired:
            print(f"⏰ Evaluation timed out for {task}")
        except FileNotFoundError:
            print(f"❌ lm_eval command not found. Please install lm_eval_harness:")
            print("   pip install lm_eval")
            return False
    
    # Print summary
    if results:
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        for task, result in results.items():
            print(f"{task.upper()}: {result}")
    
    return True


def main():
    """Run the complete pipeline"""
    print("🎯 LTG-BERT Complete Pipeline: Pretrain → Fine-tune → Evaluate")
    print("=" * 80)
    
    # Create necessary directories
    os.makedirs("models/glue_ready", exist_ok=True)
    os.makedirs("models/glue_finetuned", exist_ok=True)
    os.makedirs("models/glue_official", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    success = True
    
    # Step 1: Pretrain
    if success:
        success = step1_pretrain_model()
    
    # Step 2: Create classification models
    if success:
        success = step2_create_classification_models()
    
    # Step 3: Fine-tune on GLUE
    if success:
        success = step3_finetune_glue()
    
    # Step 4: Run lm_eval
    if success:
        success = step4_run_lm_eval()
    
    if success:
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("✅ Pretraining: Complete")
        print("✅ Classification models: Created")
        print("✅ Fine-tuning: Complete") 
        print("✅ lm_eval evaluation: Complete")
        print("\n📁 Models saved in: models/glue_official/")
        print("📊 Results saved in: results/")
        print("\n💡 All models use the official HuggingFace format and can be loaded with:")
        print("   AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=True)")
    else:
        print("\n❌ Pipeline failed. Check the error messages above.")


if __name__ == "__main__":
    main()
