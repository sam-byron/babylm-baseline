#!/bin/bash

# run_glue_collection.sh
# 
# Bash script to collect and average GLUE evaluation results using collect_glue_results.py
# 
# This script defines a list of GLUE tasks and calls the Python script to process them.
# You can modify the TASKS array below to specify which tasks to include in the average.

set -e  # Exit on any error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Python script path
PYTHON_SCRIPT="$SCRIPT_DIR/src/collect_glue_results.py"

# Check if Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "❌ Error: collect_glue_results.py not found at $PYTHON_SCRIPT"
    exit 1
fi

# Define GLUE tasks to process
# Modify this array to include/exclude specific tasks
TASKS=(
    "sst2"      # Stanford Sentiment Treebank
    "mnli"      # Multi-Genre Natural Language Inference
    "mnli-mm"   # Multi-Genre NLI matched
    "qnli"      # Question Natural Language Inference
    "rte"       # Recognizing Textual Entailment
    "boolq"     # Boolean Questions
    "multirc"   # Multi-Sentence Reading Comprehension
    "wsc"       # Winograd Schema Challenge
)

# Alternative task sets - uncomment one of these if you want to use a different set:

# Core GLUE tasks only (original GLUE benchmark)
# TASKS=("cola" "sst2" "mrpc" "qqp" "mnli" "qnli" "rte")

# SuperGLUE tasks subset
# TASKS=("boolq" "multirc" "wsc" "rte")

# Small test set
# TASKS=("cola" "sst2")

echo "🚀 Running GLUE results collection..."
echo "📋 Tasks to process: ${TASKS[*]}"
echo "=" * 60

# Check if we should run all tasks or specific ones
if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "⚠️  No tasks specified, running all available tasks"
    python3 "$PYTHON_SCRIPT" --all --verbose
else
    # Run with specified tasks
    echo "🎯 Running with specified tasks..."
    python3 "$PYTHON_SCRIPT" "${TASKS[@]}" --verbose
fi

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ GLUE results collection completed successfully!"
else
    echo ""
    echo "❌ GLUE results collection failed!"
    exit 1
fi