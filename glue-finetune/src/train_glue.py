import torch
import sys
import os
import argparse
from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from accelerate import Accelerator
from accelerate.utils import set_seed
from typing import Optional

try:
    from safetensors.torch import load_file as load_safetensors
except Exception:
    load_safetensors = None

# Fix NumPy 2.0+ compatibility with datasets library
# Following NumPy 2.0 migration guide: replace np.array(..., copy=False) with np.asarray(...)
_original_array = np.array
def _numpy_array_copy_false_fix(*args, **kwargs):
    if kwargs.get('copy') is False:
        kwargs.pop('copy')
        return np.asarray(*args, **kwargs)
    return _original_array(*args, **kwargs)
np.array = _numpy_array_copy_false_fix

# Disable wandb to prevent prompting
os.environ["WANDB_DISABLED"] = "true"

# Fix NumPy 2.0+ compatibility issue with datasets library
# os.environ["NPY_PROMOTION_STATE"] = "legacy"

# Add the checkpoint directory to Python path to import custom modules
checkpoint_path = "/home/sam-byron/engineering/ML/playground/babylm/bnc/bl-bnc-custom-ltg-bert/model_babylm_bert_ltg/checkpoint"
sys.path.insert(0, checkpoint_path)

# Import custom LTG-BERT classes from the checkpoint folder
from modeling_ltgbert import LtgBertForSequenceClassification, LtgBertForMaskedLM
from configuration_ltgbert import LtgBertConfig

# GLUE task configurations
GLUE_TASKS = {
    'boolq': {
        'num_labels': 2,
        'text_cols': ['question', 'passage'],
        'is_regression': False,
        'metric': 'accuracy'
    },
    'cola': {
        'num_labels': 2,
        'text_cols': ['sentence'],
        'is_regression': False,
        'metric': 'matthews_correlation'
    },
    'mnli': {
        'num_labels': 3,
        'text_cols': ['premise', 'hypothesis'],
        'is_regression': False,
        'metric': 'accuracy'
    },
    'mnli-mm': {  # MNLI mismatched uses same config as MNLI
        'num_labels': 3,
        'text_cols': ['premise', 'hypothesis'],
        'is_regression': False,
        'metric': 'accuracy',
        'dataset_name': 'mnli',
        'split_map': {'validation': 'validation_mismatched'}
    },
    'mrpc': {
        'num_labels': 2,
        'text_cols': ['sentence1', 'sentence2'],
        'is_regression': False,
        'metric': 'f1'
    },
    'multirc': {
        'num_labels': 2,
        'text_cols': ['paragraph', 'question', 'answer'],
        'is_regression': False,
        'metric': 'f1'
    },
    'qnli': {
        'num_labels': 2,
        'text_cols': ['question', 'sentence'],
        'is_regression': False,
        'metric': 'accuracy'
    },
    'qqp': {
        'num_labels': 2,
        'text_cols': ['question1', 'question2'],
        'is_regression': False,
        'metric': 'f1'
    },
    'rte': {
        'num_labels': 2,
        'text_cols': ['sentence1', 'sentence2'],
        'is_regression': False,
        'metric': 'accuracy'
    },
    'sst2': {
        'num_labels': 2,
        'text_cols': ['sentence'],
        'is_regression': False,
        'metric': 'accuracy'
    },
    'wsc': {
        'num_labels': 2,
        'text_cols': ['text', 'span1_text', 'span2_text'],
        'is_regression': False,
        'metric': 'accuracy'
    }
}

def load_glue_dataset(task_name):
    """Load the appropriate GLUE dataset and normalize validation split"""
    task_config = GLUE_TASKS[task_name]
    
    if task_name == 'boolq':
        dataset = load_dataset("boolq")
        # Map 'answer' to 'label' for consistency
        dataset = dataset.map(lambda x: {'label': int(x['answer'])})
    elif task_name == 'multirc':
        dataset = load_dataset("super_glue", "multirc")
    elif task_name == 'wsc':
        dataset = load_dataset("super_glue", "wsc")
    elif task_name == 'mnli-mm':
        dataset = load_dataset("glue", "mnli")
        # Normalize to 'validation' for downstream code
        dataset['validation'] = dataset['validation_mismatched']
    elif task_name == 'mnli':
        dataset = load_dataset("glue", "mnli")
        # Normalize to 'validation' for downstream code
        dataset['validation'] = dataset['validation_matched']
    else:
        dataset = load_dataset("glue", task_config.get('dataset_name', task_name))
    
    # Generic fallback: if 'validation' is missing, prefer matched then mismatched
    if 'validation' not in dataset:
        if 'validation_matched' in dataset:
            dataset['validation'] = dataset['validation_matched']
        elif 'validation_mismatched' in dataset:
            dataset['validation'] = dataset['validation_mismatched']
    
    return dataset

def tokenize_function_factory(task_name, tokenizer):
    """Factory function to create task-specific tokenization functions"""
    task_config = GLUE_TASKS[task_name]
    text_cols = task_config['text_cols']
    
    def tokenize_function(examples):
        if task_name == 'cola':
            return tokenizer(examples['sentence'], truncation=True, padding=True, max_length=512)
        elif task_name == 'sst2':
            return tokenizer(examples['sentence'], truncation=True, padding=True, max_length=512)
        elif task_name in ['mrpc', 'rte']:
            return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding=True, max_length=512)
        elif task_name in ['mnli', 'mnli-mm']:
            return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding=True, max_length=512)
        elif task_name == 'qnli':
            return tokenizer(examples['question'], examples['sentence'], truncation=True, padding=True, max_length=512)
        elif task_name == 'qqp':
            return tokenizer(examples['question1'], examples['question2'], truncation=True, padding=True, max_length=512)
        elif task_name == 'boolq':
            return tokenizer(examples['question'], examples['passage'], truncation=True, padding=True, max_length=512)
        elif task_name == 'multirc':
            # Combine paragraph, question, and answer
            texts = [f"{p} {q}" for p, q in zip(examples['paragraph'], examples['question'])]
            return tokenizer(texts, examples['answer'], truncation=True, padding=True, max_length=512)
        elif task_name == 'wsc':
            # Combine text with span information
            texts = [f"{text} {span1} {span2}" for text, span1, span2 in 
                    zip(examples['text'], examples['span1_text'], examples['span2_text'])]
            return tokenizer(texts, truncation=True, padding=True, max_length=512)
        else:
            # Default case - use first two text columns if available
            if len(text_cols) >= 2:
                return tokenizer(examples[text_cols[0]], examples[text_cols[1]], truncation=True, padding=True, max_length=512)
            else:
                return tokenizer(examples[text_cols[0]], truncation=True, padding=True, max_length=512)
    
    return tokenize_function

def compute_metrics_factory(task_name):
    """Factory function to create task-specific metric computation functions"""
    task_config = GLUE_TASKS[task_name]
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        if task_config['is_regression']:
            # For regression tasks (like STS-B, but none in our current list)
            predictions = predictions.squeeze()
            return {"pearson": pearsonr(predictions, labels)[0], 
                   "spearmanr": spearmanr(predictions, labels)[0]}
        else:
            # For classification tasks
            predictions = np.argmax(predictions, axis=1)
            
            if task_config['metric'] == 'accuracy':
                return {"accuracy": accuracy_score(labels, predictions)}
            elif task_config['metric'] == 'f1':
                return {"f1": f1_score(labels, predictions, average='binary' if task_config['num_labels'] == 2 else 'macro')}
            elif task_config['metric'] == 'matthews_correlation':
                return {"matthews_correlation": matthews_corrcoef(labels, predictions)}
            else:
                # Default to accuracy
                return {"accuracy": accuracy_score(labels, predictions)}
    
    return compute_metrics

def _maybe_load_clean_state_dict(checkpoint_dir: str):
    """If the checkpoint was saved from a wrapped/compiled model (keys start with '_orig_mod.'),
    load and return a cleaned state dict with the prefix removed. Returns None otherwise.
    """
    # Prefer safetensors if present
    st_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.isfile(st_path) and load_safetensors is not None:
        try:
            sd = load_safetensors(st_path)
        except Exception:
            sd = None
    else:
        sd = None

    # Fall back to PyTorch bin
    if sd is None:
        pt_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.isfile(pt_path):
            try:
                sd = torch.load(pt_path, map_location="cpu")
            except Exception:
                sd = None

    if not sd:
        return None

    # Detect problematic prefix
    has_orig_mod = any(k.startswith("_orig_mod.") for k in sd.keys())
    if not has_orig_mod:
        return None

    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        # Also handle common wrappers
        if nk.startswith("module."):
            nk = nk[len("module."):]
        cleaned[nk] = v
    return cleaned

def _has_meta_params(model: torch.nn.Module) -> bool:
    for p in model.parameters(recurse=True):
        if getattr(p, "device", None) is not None and p.device.type == "meta":
            return True
    for b in model.buffers(recurse=True):
        if getattr(b, "device", None) is not None and b.device.type == "meta":
            return True
    return False

def load_ltgbert_seqcls(checkpoint_dir: str, config: "LtgBertConfig"):
    """Robustly load LtgBertForSequenceClassification from a checkpoint directory.
    Handles checkpoints saved under wrappers (e.g., torch.compile) by cleaning state dict prefixes.
    Ensures no parameter remains on meta device to satisfy Trainer move_to_device.
    """
    # First try the standard Transformers loading path but force real allocation
    try:
        model = LtgBertForSequenceClassification.from_pretrained(
            checkpoint_dir,
            config=config,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,  # avoid meta tensors for missing keys
            device_map=None,
        )
        if _has_meta_params(model):
            raise RuntimeError("Model has meta params after from_pretrained")
        return model
    except Exception as e:
        print(f"Standard from_pretrained encountered an issue: {e}. Falling back to cleaned state dict loading...")

    # Try to load and clean a wrapped state dict
    cleaned_sd = _maybe_load_clean_state_dict(checkpoint_dir)
    model = LtgBertForSequenceClassification(config)

    if cleaned_sd is None:
        # Nothing to load, return initialized model (no meta tensors since constructed normally)
        return model

    # Ensure on CPU with storage allocated
    if _has_meta_params(model):
        try:
            model.to_empty(device="cpu")
        except Exception:
            # Fallback to normal .to if to_empty not available
            model = model.to("cpu")

    missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)
    if len(missing) > 0:
        print(f"Warning: {len(missing)} missing keys when loading cleaned state dict. First few: {missing[:5]}")
    if len(unexpected) > 0:
        print(f"Warning: {len(unexpected)} unexpected keys when loading cleaned state dict. First few: {unexpected[:5]}")
    return model

# NEW: helper to find newest Trainer checkpoint directory
def _find_latest_trainer_checkpoint(output_dir: str) -> Optional[str]:
    try:
        if not os.path.isdir(output_dir):
            return None
        candidates = []
        for name in os.listdir(output_dir):
            path = os.path.join(output_dir, name)
            if os.path.isdir(path) and name.startswith("checkpoint-"):
                # extract step number suffix if possible
                try:
                    step = int(name.split("checkpoint-")[-1])
                except ValueError:
                    step = -1
                candidates.append((step, path))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]
    except Exception:
        return None

def fine_tune_task(task_name, output_base_dir="./fine-tuned-models", accelerator=None):
    """Fine-tune the LTG-BERT model on a specific GLUE task"""
    if accelerator is None:
        accelerator = Accelerator()
    
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"🎯 Fine-tuning LTG-BERT on {task_name.upper()}")
        print(f"{'='*60}")
    
    # Get task configuration
    task_config = GLUE_TASKS[task_name]
    
    # Load the GLUE dataset
    if accelerator.is_main_process:
        print(f"📥 Loading {task_name} dataset...")
    dataset = load_glue_dataset(task_name)
    
    # Load the tokenizer from the checkpoint folder
    if accelerator.is_main_process:
        print("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Load the config and model from the checkpoint folder
    if accelerator.is_main_process:
        print("⚙️ Loading model configuration...")
    config = LtgBertConfig.from_pretrained(checkpoint_path)
    config.num_labels = task_config['num_labels']
    
    # Set problem type for proper loss computation
    if task_config['is_regression']:
        config.problem_type = "regression"
    elif task_config['num_labels'] == 1:
        config.problem_type = "regression"
    elif task_config['num_labels'] > 2:
        config.problem_type = "single_label_classification"
    else:
        config.problem_type = "single_label_classification"

    # Create output directory and detect existing checkpoints for this task
    output_dir = os.path.join(output_base_dir, task_name)
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    latest_ckpt = _find_latest_trainer_checkpoint(output_dir)
    if accelerator.is_main_process:
        if latest_ckpt:
            print(f"🔁 Found existing checkpoint for {task_name} at: {latest_ckpt}. Will resume training from it.")
        else:
            print("🆕 No existing checkpoint found. Starting from the base pre-trained model.")

    if accelerator.is_main_process:
        print(f"🧠 Loading pre-trained model (num_labels={task_config['num_labels']})...")
    # Use robust loader that handles wrapped checkpoints; Trainer will load checkpoint weights if resuming
    model = load_ltgbert_seqcls(checkpoint_path, config)

    if accelerator.is_main_process:
        print("✅ Model loaded successfully - backbone weights loaded; classification head initialized if needed")
    
    # Tokenize the dataset
    if accelerator.is_main_process:
        print("🔍 Tokenizing dataset...")
    tokenize_function = tokenize_function_factory(task_name, tokenizer)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Set the format for PyTorch
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # Create datasets
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    
    if accelerator.is_main_process:
        print(f"📊 Dataset info: Train={len(train_dataset)}, Val={len(eval_dataset)}")
        print(f"🖥️  Using {accelerator.num_processes} processes for training")
    
    # Define training arguments - optimized for each task
    batch_size = 16 if task_name in ['mnli', 'qqp'] else 32  # Smaller batch for large datasets
    learning_rate = 2e-5 if task_name != 'cola' else 1e-5    # Lower LR for CoLA
    epochs = 3 if task_name not in ['mnli', 'qqp'] else 2     # Fewer epochs for large datasets
    epochs = 4
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{task_config['metric']}" if task_config['metric'] != 'matthews_correlation' else "eval_matthews_correlation",
        greater_is_better=True,
        save_total_limit=2,
        seed=42,
        data_seed=42,
        remove_unused_columns=True,
        label_smoothing_factor=0.1 if task_name in ['mnli', 'qqp'] else 0.0,
        report_to=[],
        dataloader_pin_memory=False,
        max_grad_norm=1.0,  # gradient clipping for stability, helps avoid NaNs
    )
    
    # Create compute_metrics function
    compute_metrics = compute_metrics_factory(task_name)
    
    # Define the Trainer
    if accelerator.is_main_process:
        print("🏋️ Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Train the model (resume if a checkpoint exists)
    if accelerator.is_main_process:
        print("🚀 Starting training...")
    if latest_ckpt:
        trainer.train(resume_from_checkpoint=latest_ckpt)
    else:
        trainer.train()
    
    # Evaluate the model
    if accelerator.is_main_process:
        print("📈 Evaluating model...")
    eval_results = trainer.evaluate()
    if accelerator.is_main_process:
        print(f"📊 Final evaluation results for {task_name}:")
        for key, value in eval_results.items():
            print(f"  {key}: {value:.4f}")
    
    # Save the fine-tuned model (only on main process)
    if accelerator.is_main_process:
        print("💾 Saving fine-tuned model...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save evaluation results
        import json
        results_file = os.path.join(output_dir, "eval_results.json")
        with open(results_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"✅ Fine-tuning completed for {task_name}!")
        print(f"📁 Model saved to: {output_dir}")
    
    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    
    return eval_results

def main():
    # Initialize accelerator first
    accelerator = Accelerator()
    
    # Set seed for reproducibility
    set_seed(42)
    
    parser = argparse.ArgumentParser(description="Fine-tune LTG-BERT on GLUE tasks")
    parser.add_argument("--tasks", nargs="+", 
                       choices=list(GLUE_TASKS.keys()) + ["all"],
                       default=["all"],
                       help="GLUE tasks to fine-tune on")
    parser.add_argument("--output_dir", type=str, default="./fine-tuned-ltg-bert-glue",
                       help="Output directory for fine-tuned models")
    
    args = parser.parse_args()
    
    # Determine which tasks to run
    if "all" in args.tasks:
        tasks_to_run = list(GLUE_TASKS.keys())
    else:
        tasks_to_run = args.tasks
    
    if accelerator.is_main_process:
        print(f"🎯 Fine-tuning LTG-BERT on tasks: {tasks_to_run}")
        print(f"🖥️  Using {accelerator.num_processes} processes")
        print(f"🔧 Mixed precision: {accelerator.mixed_precision}")
    
    # Fine-tune on each task
    all_results = {}
    for task in tasks_to_run:
        try:
            results = fine_tune_task(task, args.output_dir, accelerator)
            all_results[task] = results
        except Exception as e:
            if accelerator.is_main_process:
                print(f"❌ Error fine-tuning {task}: {str(e)}")
                import traceback
                traceback.print_exc()
            continue
    
    # Print summary (only on main process)
    if accelerator.is_main_process:
        print(f"\n{'='*80}")
        print("📊 FINE-TUNING SUMMARY")
        print(f"{'='*80}")
        
        for task, results in all_results.items():
            task_config = GLUE_TASKS[task]
            metric_key = f"eval_{task_config['metric']}" if task_config['metric'] != 'matthews_correlation' else "eval_matthews_correlation"
            score = results.get(metric_key, "N/A")
            print(f"{task.upper():12s}: {score:.4f}" if isinstance(score, (int, float)) else f"{task.upper():12s}: {score}")
        
        print(f"\n🎉 Fine-tuning completed for {len(all_results)} tasks!")
        print(f"📁 Models saved in: {args.output_dir}")
    
    # Clean up
    accelerator.end_training()

if __name__ == "__main__":
    main()