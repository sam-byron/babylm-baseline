#!/bin/bash

# Fine-tune LTG-BERT on all specified GLUE tasks
# Usage: ./run_finetune.sh [output_directory]

OUTPUT_DIR=${1:-"./fine-tuned-ltg-bert-glue"}

echo "🚀 Starting LTG-BERT fine-tuning on GLUE tasks..."
echo "📁 Output directory: $OUTPUT_DIR"

# Run the fine-tuning script
accelerate launch src/train_glue.py \
    --tasks boolq cola mnli mnli-mm mrpc multirc qnli qqp rte sst2 wsc \
    --output_dir "$OUTPUT_DIR"

echo "✅ Fine-tuning completed!"
echo "📁 Models saved in: $OUTPUT_DIR"