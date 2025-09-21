#!/usr/bin/env bash
# Set PYTHONPATH to parent directory so imports work
export PYTHONPATH="$(dirname "$PWD"):$PYTHONPATH"

# Run the tests
python test_ltg_bert_mlm.py
python stress_ltg_bert_mlm.py --device cpu --steps 10 --batch_sizes 4 --seq_lens 64
python fuzz_configs_ltg_bert.py