#!/usr/bin/env bash
# 
# Stress test runner for LtgBertForMaskedLM
# This script sets up the environment and runs comprehensive tests
#

set -euo pipefail

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Parent directory contains the modules we need to import
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Set PYTHONPATH to include the parent directory
export PYTHONPATH="$PARENT_DIR:${PYTHONPATH:-}"

echo "🔧 Environment setup:"
echo "   SCRIPT_DIR: $SCRIPT_DIR"
echo "   PARENT_DIR: $PARENT_DIR" 
echo "   PYTHONPATH: $PYTHONPATH"
echo "   Conda env: ${CONDA_DEFAULT_ENV:-none}"
echo ""

# Check if we're in the right conda environment
if [[ "${CONDA_DEFAULT_ENV:-}" != "torch-black" ]]; then
    echo "⚠️  Warning: Not in torch-black environment (current: ${CONDA_DEFAULT_ENV:-none})"
    echo "   Run: conda activate torch-black"
    echo ""
fi

# Optional: raise file descriptor limit to avoid EMFILE
ulimit -n 65535 2>/dev/null || echo "   Note: Could not raise file descriptor limit"

# Set CUDA device if not already set
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "🧪 Running comprehensive stress tests..."
echo ""

# Run the stress tests
cd "$SCRIPT_DIR"
python run_stress_tests.py "$@"

echo ""
echo "✅ Stress testing completed!"