"""
Complete training script using sentence-aware data processing for improved syntax learning.

This script orchestrates the entire pipeline from sentence extraction to model training
with proper sentence boundary preservation.
"""

import os
import sys
import logging
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_sentence_aware_pipeline(
    bnc_converted_dir: str,
    tokenizer_path: str,
    model_path: str,
    output_dir: str,
    intermediate_dir: str = "sentence_pipeline_data"
):
    """
    Run the complete sentence-aware training pipeline.
    
    Args:
        bnc_converted_dir: Directory with BNC converted markdown files
        tokenizer_path: Path to the custom tokenizer
        model_path: Path to the base model
        output_dir: Directory to save the retrained model
        intermediate_dir: Directory for intermediate data
    """
    logger.info("Starting sentence-aware training pipeline")
    
    # Create intermediate directory
    intermediate_path = Path(intermediate_dir)
    intermediate_path.mkdir(exist_ok=True)
    
    # Step 1: Extract sentences from markdown files
    logger.info("Step 1: Extracting sentences from BNC markdown files")
    sentence_dataset_path = intermediate_path / "sentence_dataset"
    
    if not sentence_dataset_path.exists():
        logger.info("Running sentence extraction...")
        cmd = f"python prepare_data_sentence_aware.py {bnc_converted_dir} {sentence_dataset_path}"
        result = os.system(cmd)
        if result != 0:
            raise RuntimeError("Sentence extraction failed")
    else:
        logger.info("Sentence dataset already exists, skipping extraction")
    
    # Step 2: Tokenize sentences with proper boundaries
    logger.info("Step 2: Tokenizing sentences with boundary preservation")
    tokenized_data_path = intermediate_path / "tokenized_sentences"
    
    if not tokenized_data_path.exists():
        logger.info("Running sentence-aware tokenization...")
        cmd = f"python tokenize_sentence_aware.py {sentence_dataset_path} {tokenizer_path} {tokenized_data_path}"
        result = os.system(cmd)
        if result != 0:
            raise RuntimeError("Sentence tokenization failed")
    else:
        logger.info("Tokenized data already exists, skipping tokenization")
    
    # Step 3: Train model with sentence-aware data
    logger.info("Step 3: Training model with sentence-aware MLM")
    train_sentence_aware_model(
        tokenized_data_path=str(tokenized_data_path),
        tokenizer_path=tokenizer_path,
        model_path=model_path,
        output_dir=output_dir
    )
    
    logger.info("Sentence-aware training pipeline complete!")

def train_sentence_aware_model(
    tokenized_data_path: str,
    tokenizer_path: str,
    model_path: str,
    output_dir: str,
    num_train_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5
):
    """
    Train the model using sentence-aware data.
    
    Args:
        tokenized_data_path: Path to tokenized sentence data
        tokenizer_path: Path to tokenizer
        model_path: Path to base model
        output_dir: Output directory for trained model
        num_train_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    """
    logger.info("Loading tokenizer and model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Load model
    model = AutoModelForMaskedLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Ensure model vocabulary matches tokenizer
    if model.config.vocab_size != len(tokenizer):
        logger.warning(f"Vocab size mismatch: model={model.config.vocab_size}, tokenizer={len(tokenizer)}")
        # Resize model embeddings
        model.resize_token_embeddings(len(tokenizer))
    
    # Create dataset and data collator
    logger.info("Setting up training data...")
    
    from sentence_aware_mlm_dataset import create_sentence_aware_dataloader
    
    train_dataloader = create_sentence_aware_dataloader(
        tokenized_data_dir=tokenized_data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        mlm_probability=0.15,
        shuffle=True
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We handle MLM in our dataset
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_steps=1000,
        save_total_limit=3,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        fp16=torch.cuda.is_available(),
        report_to=None,  # Disable wandb/tensorboard
        learning_rate=learning_rate,
    )
    
    # Create trainer with custom dataset
    class SentenceAwareTrainer(Trainer):
        def get_train_dataloader(self):
            return train_dataloader
    
    trainer = SentenceAwareTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training info
    with open(f"{output_dir}/training_info.txt", 'w') as f:
        f.write("Sentence-Aware MLM Training\n")
        f.write("============================\n")
        f.write(f"Base model: {model_path}\n")
        f.write(f"Tokenizer: {tokenizer_path}\n")
        f.write(f"Training data: {tokenized_data_path}\n")
        f.write(f"Epochs: {num_train_epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Vocabulary size: {len(tokenizer)}\n")
        f.write("Data processing: Sentence-aware with boundary preservation\n")
    
    logger.info("Training complete!")

def main():
    """Main function to run the sentence-aware training pipeline."""
    if len(sys.argv) != 5:
        print("Usage: python train_sentence_aware.py <bnc_converted_dir> <tokenizer_path> <model_path> <output_dir>")
        print("Example: python train_sentence_aware.py bnc_converted/ ./ model_babylm_bert_ltg/ model_babylm_bert_ltg_fixed/")
        sys.exit(1)
    
    bnc_converted_dir = sys.argv[1]
    tokenizer_path = sys.argv[2]
    model_path = sys.argv[3]
    output_dir = sys.argv[4]
    
    # Verify paths exist
    if not Path(bnc_converted_dir).exists():
        raise ValueError(f"BNC converted directory not found: {bnc_converted_dir}")
    
    if not Path(tokenizer_path).exists():
        raise ValueError(f"Tokenizer path not found: {tokenizer_path}")
    
    if not Path(model_path).exists():
        raise ValueError(f"Model path not found: {model_path}")
    
    # Run the pipeline
    run_sentence_aware_pipeline(
        bnc_converted_dir=bnc_converted_dir,
        tokenizer_path=tokenizer_path,
        model_path=model_path,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()
