# LTG-BERT GLUE Fine-tuning

This directory contains scripts to fine-tune the LTG-BERT model on multiple GLUE tasks.

## Supported Tasks

The script supports fine-tuning on the following 11 GLUE tasks:

| Task | Dataset | Labels | Metric | Description |
|------|---------|--------|--------|-------------|
| **boolq** | BoolQ | 2 | Accuracy | Boolean questions from Wikipedia |
| **cola** | CoLA | 2 | Matthews Correlation | Linguistic acceptability |
| **mnli** | MNLI | 3 | Accuracy | Multi-genre natural language inference |
| **mnli-mm** | MNLI-MM | 3 | Accuracy | MNLI mismatched validation |
| **mrpc** | MRPC | 2 | F1 | Microsoft Research Paraphrase Corpus |
| **multirc** | MultiRC | 2 | F1 | Multi-sentence reading comprehension |
| **qnli** | QNLI | 2 | Accuracy | Question-answering natural language inference |
| **qqp** | QQP | 2 | F1 | Quora Question Pairs |
| **rte** | RTE | 2 | Accuracy | Recognizing textual entailment |
| **sst2** | SST-2 | 2 | Accuracy | Stanford Sentiment Treebank |
| **wsc** | WSC | 2 | Accuracy | Winograd Schema Challenge |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Fine-tuning on All Tasks

```bash
./run_finetune.sh
```

This will fine-tune the LTG-BERT model on all 11 tasks and save models to `./fine-tuned-ltg-bert-glue/`.

### 3. Run Fine-tuning on Specific Tasks

```bash
python src/train_glue.py --tasks cola sst2 mrpc --output_dir ./my-models
```

## Advanced Usage

### Fine-tune Individual Tasks

```bash
# Fine-tune only on CoLA
python src/train_glue.py --tasks cola

# Fine-tune on multiple specific tasks
python src/train_glue.py --tasks boolq cola mnli --output_dir ./selected-tasks
```

### Task-Specific Optimizations

The script includes task-specific optimizations:

- **Batch sizes**: Smaller for large datasets (MNLI, QQP: 16), larger for others (32)
- **Learning rates**: Lower for CoLA (1e-5), standard for others (2e-5)
- **Epochs**: Fewer for large datasets (MNLI, QQP: 2), standard for others (3)
- **Label smoothing**: Applied to difficult tasks (MNLI, QQP)

## Output Structure

After fine-tuning, the output directory will contain:

```
fine-tuned-ltg-bert-glue/
├── boolq/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── eval_results.json
│   └── logs/
├── cola/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── eval_results.json
│   └── logs/
└── ... (other tasks)
```

## Model Loading

Load a fine-tuned model:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained(
    "./fine-tuned-ltg-bert-glue/cola",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-ltg-bert-glue/cola")

# Use the model
inputs = tokenizer("This sentence is grammatical.", return_tensors="pt")
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

## Performance Expectations

The script will output detailed metrics for each task:

- **CoLA**: Matthews correlation coefficient
- **SST-2, MNLI, QNLI, RTE, BoolQ, WSC**: Accuracy
- **MRPC, QQP, MultiRC**: F1 score

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in the script
2. **Dataset download fails**: Check internet connection and HuggingFace credentials
3. **Import errors**: Ensure all dependencies are installed

### Memory Requirements

- Minimum GPU memory: 8GB
- Recommended: 16GB+ for efficient training
- For CPU-only training, expect significantly longer training times

## Configuration

The script automatically handles task-specific configurations:

- Text column mapping for different input formats
- Appropriate metrics for each task type
- Optimized hyperparameters based on task characteristics

## Notes

- All models are saved with the HuggingFace `trust_remote_code=True` format
- Evaluation results are saved as JSON files for easy parsing
- Training logs are saved to TensorBoard format in the `logs/` subdirectory
- The script uses the pre-trained LTG-BERT checkpoint from the specified path

### Fine-tuning Script for LTG-BERT on GLUE

```python
import torch
from torch.utils.data import DataLoader
from transformers import (
    LtgBertForSequenceClassification,
    LtgBertConfig,
    AdamW,
    get_linear_schedule_with_warmup,
    GlueDataset,
    GlueDataTrainingArguments,
)
from transformers import Trainer, TrainingArguments

def main():
    # Load the configuration
    config = LtgBertConfig.from_pretrained("path/to/your/configuration_ltgbert.json")
    
    # Load the model
    model = LtgBertForSequenceClassification.from_pretrained("path/to/your/model", config=config)

    # Load the GLUE dataset
    glue_args = GlueDataTrainingArguments(
        task_name="mrpc",  # Change this to the desired GLUE task
        data_dir="path/to/glue_data/MRPC",  # Change this to the path of your GLUE dataset
        max_seq_length=128,
        overwrite_cache=True,
    )
    
    train_dataset = GlueDataset(glue_args)
    
    # Set up the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,  # You can use a separate validation dataset
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model("./fine_tuned_ltgbert")

if __name__ == "__main__":
    main()
```

### Instructions to Run the Script

1. **Install Required Libraries**: Make sure you have the `transformers` library installed. You can install it using pip:
   ```bash
   pip install transformers datasets
   ```

2. **Prepare the GLUE Dataset**: Download the GLUE dataset and place it in a directory. You can find the dataset [here](https://gluebenchmark.com/).

3. **Modify Paths**: Update the paths in the script:
   - `path/to/your/configuration_ltgbert.json`: Path to your LTG-BERT configuration file.
   - `path/to/your/model`: Path to your pre-trained LTG-BERT model.
   - `path/to/glue_data/MRPC`: Path to the specific GLUE task dataset (e.g., MRPC).

4. **Run the Script**: Execute the script using Python:
   ```bash
   python fine_tune_ltgbert.py
   ```

### Notes
- You can change the `task_name` in `GlueDataTrainingArguments` to any other GLUE task (e.g., `sst2`, `cola`, etc.).
- Adjust the `max_seq_length`, `learning_rate`, `batch_size`, and `num_train_epochs` according to your requirements and available resources.
- The script uses the `Trainer` class from the `transformers` library, which simplifies the training loop and handles evaluation and logging.