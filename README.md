
## LTG BERT Pretraining & Evaluation Workspace

This repository contains a custom BERT-style masked language model pipeline with:
* Always-on sentence packing (structural heuristics) — see `PACKING.md`.
* Fresh tokenizer over full BNC Markdown corpus with extended structural tokens — see `MIGRATION_NOTES.md`.
* Instrumented preprocessing for punctuation-final quality metrics.
* Support for large-batch LAMB training configurations.

---
## 1. Corpus Conversion
```
python convert_bnc.py  # if raw source requires normalization
```

## 2. Tokenizer Training
```
python tokenizer.py --data-dir data/pretrain/bnc --vocab_path data/pretrain/wordpiece_vocab.json --vocab_size 16384
# Optional shuffled order reproducibility test
python tokenizer.py --data-dir data/pretrain/bnc --vocab_path data/pretrain/wordpiece_vocab.json --vocab_size 16384 --shuffle-files
```
Outputs include manifest with F95 (e.g. 220) and true unknown rate (~0.081%).

## 3. Sentence Packing & Cache Generation
`prepare_data.py` now *always* uses packing; `--sentence-pack` flag retained only for backward compatibility.
```
python prepare_data.py --config_path config_sentence_aware.json
# Sanitize existing chunk cache
python prepare_data.py --config_path config_sentence_aware.json --sanitize
```
Streaming stats (optional exploratory run):
```
python stream_packing_stats.py --data-dir ./data/pretrain/bnc --tokenizer ./data/pretrain/wordpiece_vocab.json --max-length 512 --slice-lines 60000 --snapshot-every 2
```

## 4. Training (Example: Accelerate / Standard AdamW)
```
accelerate launch transformer_trainer.py --config_path model_babylm_ltg_bert.json
```

## 5. Alternative LAMB Large-Batch Configuration
See `gpt5_lamb_bs128_ga24.json` (current `block_size` 128 ablation; packing stats sourced from 512-context preprocessing). Adjust paths and launch similarly.

## 6. Model Registration (Internal Utility)
```
python ltg_bert.py
```

## 7. Evaluation (BLiMP / Other Tasks)
```
lm_eval --model hf-mlm \
    --model_args pretrained=/path/to/checkpoint.pt,backend="mlm" \
    --tasks blimp --device cuda:0 --batch_size 64
```
Add `--trust_remote_code` for custom config modules if necessary.

## 8. Data Loading / Inspection
```
python data_loader.py --config_path model_babylm_ltg_bert.json
```

## 9. LoRA Fine-Tuning Examples
```
python lora_fine_tuning.py --config_path lora_config.json

# Base + LoRA
python inter_chat_lora.py \
    --base_model_path ./model_vault/base_checkpoint.pt \
    --lora_model_path ./alpaca-lora-owt-gpt2 \
    --config_path model_open_web_full.json

# LoRA only (uses standard backbone)
python inter_chat_lora.py --lora_model_path ./alpaca-lora-owt-gpt2
```

## 10. Key Documentation
* `PACKING.md` – heuristics, metrics, rationale.
* `MIGRATION_NOTES.md` – tokenizer & special token migration details.

## 11. Reproducibility Highlights
| Aspect | Guarantee |
|--------|-----------|
| File order | Deterministic sorted traversal |
| Tokenizer manifest | Includes corpus hash + params |
| Packing stats | Streaming snapshots + final totals (planned) |
| Config pinning | JSON configs committed |

## 12. Observed Packing Quality (Representative)
| Metric | Value (stabilized) |
|--------|--------------------|
| Terminal end rate | ~0.89–0.90 |
| Dangling carry rate | ~0.16–0.18 |
| Borrow move rate | ~0.001 |
| Short trailing drop | ~0.0 |

## 13. Unknown Token Rate
True unknown substitution rate ≈ 0.081% at vocab 16,384 (literal `[UNK]` placeholders excluded via remap procedure). Increasing vocab yielded negligible reduction.

## 14. Future Enhancements
- Final global stats artifact emission
- Candidate diagnostics for short trailing drop heuristic
- Automated evaluation hook after cache build

---
*See `PACKING.md` and `MIGRATION_NOTES.md` for deeper context.*

