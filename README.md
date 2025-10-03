
## LTG BERT Pretraining & Evaluation Workspace

This repository contains a custom BERT-style masked language model pipeline with:

- Instrumented preprocessing for punctuation-final quality metrics.
- Support for large-batch LAMB training configurations.

---

## 1. Corpus Conversion

```bash
python convert_bnc.py 
```

## 2. Tokenizer Training

```bash
python tokenizer.py --data-dir data/pretrain/bnc --vocab_path data/pretrain/wordpiece_vocab.json --vocab_size 16384
```

## 3. Sentence Packing & Cache Generation

```bash
python prepare_data.py --config_path config.json
# Sanitize existing chunk cache
python prepare_data.py --config_path config.json --sanitize
```

**Streaming stats (optional exploratory run):**

```bash
python stream_packing_stats.py --data-dir ./data/pretrain/bnc --tokenizer ./data/pretrain/wordpiece_vocab.json --max-length 512 --slice-lines 60000 --snapshot-every 2
```

## 4. Training (Example: Accelerate)

```bash
accelerate launch transformer_trainer.py --config_path config.json
```

## 5. Alternative LAMB Large-Batch Configuration

See `gpt5_lamb_bs128_ga24.json` (current `block_size` 128 ablation; packing stats sourced from 512-context preprocessing). Adjust paths and launch similarly.

## 6. Model Registration (Internal Utility)

```bash
python ltg_bert.py
```

## 7. Evaluation (BLiMP / Other Tasks)

```bash
python3 blimp_sanity.py --model_path model_babylm_bert_ltg/checkpoint --normalize per_token --benchmark blimp
```
```bash
./glue-finetune/run_finetune.sh
./run_glue_collection.sh
```

## 8. Data Loading / Inspection

```bash
python data_loader.py --config_path model_babylm_ltg_bert.json
```
