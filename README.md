# LTG-BERT BNC Pretraining: Scripts, Usage, and Notes

This repository contains a lightweight, compute-conscious pipeline to prepare the British National Corpus (BNC), train a WordPiece tokenizer, pack and block data for BERT-style masked language modeling (MLM), and train/evaluate a custom LTG-BERT model.

Each script below includes purpose, how to call it, important arguments, and what’s innovative or efficient.

## Data collection and conversion

- collect_xml.sh
  - Purpose: Recursively copy all .xml files from a nested BNC source tree into a flat destination directory.
  - Usage: `./collect_xml.sh <source_dir> <dest_dir>`
  - Notes: Uses `find -print0` with a null-delimited loop to handle spaces/special characters in filenames.
  - Efficiency: Safe streaming copy; trivial but robust against odd filenames.

- convert_bnc.py
  - Purpose: Convert BNC XML into normalized markdown-like `.md` text, handling both written and spoken modalities.
  - Usage: `python convert_bnc.py`
  - Inputs: `SOURCE_FOLDER` (default `BNC_raw/Texts`), `./data/first-names.txt` for anonymizing unknown speakers.
  - Outputs: `TARGET_FOLDER` (default `raw_corpus/bnc`) with preserved directory structure.
  - Innovations: Conservative quote/spacing normalization, MosesPunctNormalizer + ftfy; streaming-style XML traversal; modality-aware formatting of titles/paragraphs/speeches.

## Tokenizer and vocabulary

- tokenizer.py
  - Purpose: Train a WordPiece tokenizer with extended special tokens `[DOC] [EOD] [SPK]` and compute F95.
  - Usage:
    - `python tokenizer.py --data-dir data/pretrain/bnc --vocab_path data/pretrain/bpe.json --vocab_size 16384 --min_frequency 10`
  - Outputs: tokenizer JSON, `<vocab_path>.manifest.json`, and `*_freqs` for frequency inspection.
  - Innovations: ByteLevel pre-tokenization for robust raw handling; reproducibility manifest (sha256 and args); corpus streaming without loading everything in memory.

- create_vocab.py
  - Purpose: End-to-end helper to shard a combined corpus, train WordPiece, and tokenize shards to gzip’d pickles.
  - Usage: `python create_vocab.py`
  - Steps: Creates shard and tokenized folders, splits train/valid, trains tokenizer, tokenizes shards.
  - Innovations: Uses fast tokenizers backend; stores tokenized docs as numpy arrays in gzip pickles for smaller footprint and faster loads.

## Packing and block creation

- sentence_packing.py
  - Purpose: Pack sentences into sequences close to a token budget using a lightweight regex-based splitter.
  - Usage:
    - Python: `from sentence_packing import pack_sentences`
  - Innovations: Borrow-pass and dangling-carry logic to improve end punctuation and fill; optional stats to inspect packing quality.

- prepare_data.py
  - Purpose: Stream raw `.md`, inject `[DOC]/[EOD]` and optional speaker tags, pack to roughly a `block_size` budget, and emit fixed-size `chunk*.pt` tensors with metadata.
  - Usage:
    - `python prepare_data.py --config_path model_babylm_ltg_bert.json [--sanitize]`
  - Key config:
    - `cache_path` (output), `block_size`, `packing_max_len` (defaults to block_size), `blocks_per_file`, `oversize_policy` (split|truncate|skip|raise).
  - Innovations: Direct packing to `block_size` avoids post-splitting; buffered emission improves I/O; optional fast sanitization to remove corrupted chunks.

## Datasets, collators, loaders

- mlm_dataset.py
  - Purpose: Masking strategies (Span/Subword/WholeWord), several dataset variants, and a SentenceAwareDataset that builds multi-sentence sequences and applies boundary-aware masking.
  - Usage: `from mlm_dataset import SentenceAwareDataset`
  - Innovations: Protected-ID mechanism so structural tokens are never masked even if special IDs are not contiguous; robust chunk loading with conservative memory usage; cached pad tensors when stitching sequences.

- dynamic_collator.py
  - Purpose: A dynamic masking collator that reuses the masking strategies in `mlm_dataset.py` at collate time.
  - Usage:
    - `from dynamic_collator import create_dynamic_collator`
    - `collate_fn = create_dynamic_collator(config, tokenizer)`
  - Key args: `masking_strategy` (span|subword|whole_word), `mask_p`, `random_p`, `keep_p`, `max_span_length` (for span).
  - Innovations: Explicit protected_ids list for structural tokens; pads and builds attention masks in one pass.

- data_loader.py
  - Purpose: Factories to produce train/val/test dataloaders. Defaults to a sentence-aware dataset + dynamic masking path.
  - Usage: `train_loader, val_loader, test_loader, collate_fn, total_tokens = data_loader(config, tokenizer, cache_path)`
  - Innovations: Preblocked dataset path with cached row-level index maps; sentence-aware path avoids storing multiple masked variants on disk.

## Training and evaluation

- transformer_trainer.py
  - Purpose: Accelerate-based training loop for LTG-BERT masked LM with optimizer/scheduler wiring, resume support, logging, and end-of-epoch validation.
  - Usage:
    - `accelerate launch transformer_trainer.py --config_path model_babylm_ltg_bert.json`
  - Innovations: Sentence-aware loader with dynamic masking; accumulation-aware cosine schedule; LR scaling capped conservatively; rank-0 JSONL logging; barrier-safe file writes.

- blimp_sanity.py
  - Purpose: Evaluate a masked LM via pseudo log-likelihood on BLiMP and BLiMP-supplement.
  - Usage:
    - `python blimp_sanity.py --model_path <checkpoint_dir> [--benchmark both] [--normalize per_token] [--max_examples 200]`
  - Innovations: Vocab clamp for tokenizer/model mismatches; worst-K example dump; robust model loading fallback to local modeling files.

## Model, configuration, and optimizer

- configuration_ltgbert.py
  - Purpose: Defines `LtgBertConfig`, a Hugging Face-compatible config class for LTG-BERT.
  - Notes: Adds `position_bucket_size` for relative attention; `model_type` is "bert" for broad tool compatibility.
  - Usage: `from configuration_ltgbert import LtgBertConfig; cfg = LtgBertConfig.from_pretrained(<ckpt_dir>)`

- modeling_ltgbert.py
  - Purpose: PyTorch implementation of LTG-BERT with relative position bucketing and GeGLU FFN; includes task heads.
  - Heads: `LtgBertForMaskedLM`, `LtgBertForSequenceClassification`, `LtgBertForTokenClassification`, `LtgBertForQuestionAnswering`, `LtgBertForMultipleChoice`.
  - Notes: Input/output embeddings tied for MLM; supports `trust_remote_code=True`; lightweight masked softmax.

- lamb.py
  - Purpose: Minimal LAMB optimizer (Layer-wise Adaptive Moments) for large-batch training.
  - Notes: No sparse gradients; decoupled weight decay; trust ratio computed per parameter tensor.
  - Usage: `from lamb import Lamb; optimizer = Lamb(model.parameters(), lr=1e-3)`

## Logging, monitoring, and training helpers

- thin_logger.py
  - Purpose: Rank-0 JSONL logger with optional TensorBoard scalars.
  - Usage: `logger = ThinLogger(log_dir, run_name="exp1", enable_tb=True, is_main=accelerator.is_main_process)`

- training_monitor.py
  - Purpose: Tracks loss/LR/grad-norm and flags spikes, oscillations, and near-explosions.
  - Usage: `monitor.update(loss, grad_norm, lr, step); monitor.check_loss_spike()`

- training_utils.py
  - Purpose: Small helpers for saving HF-compatible checkpoints, forward/grad steps, safe barriers, resume-skipping.
  - Highlights: `save_hf_checkpoint`, `save_epoch_checkpoint`, `build_batch_iterator_with_skip`, and `generate_special_tokens_map`.

## Tokenizer file utilities

- create_tokenizer_files.py
  - Purpose: Generate `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`, and copy `tokenizer.json` into a checkpoint dir.
  - Usage:
    - `python create_tokenizer_files.py --checkpoint_path <ckpt_dir> --tokenizer_path ./tokenizer.json`
  - Also patches `config.json` with an `auto_map` for AutoTokenizer if present.

- save_config.py
  - Purpose: Convert a training JSON (e.g., `./configs/base.json`) into a transformers-compatible `config.json`.
  - Usage:
    - `python save_config.py --in ./configs/base.json --out ./model_babylm_bert_ltg/checkpoint`
  - Notes: Registers auto_map entries for custom config/model classes.

## Minimal quickstart

1) Convert BNC XML to markdown

```bash
./collect_xml.sh ./temp/Texts ./data/pretrain/bnc_xml
python convert_bnc.py
```

2) Train tokenizer (WordPiece)

```bash
python tokenizer.py --data-dir data/pretrain/bnc --vocab_path data/pretrain/bpe.json --vocab_size 16384 --min_frequency 10
```

3) Prepare packed blocks

```bash
python prepare_data.py --config_path model_babylm_ltg_bert.json
```

4) Train LTG-BERT with accelerate

```bash
accelerate launch transformer_trainer.py --config_path model_babylm_ltg_bert.json
```

5) Evaluate on BLiMP (+ supplement)

```bash
python blimp_sanity.py --model_path model_vault/model_bl_bert_ltgds_regular --benchmark both --normalize per_token --max_examples 200
```

## Tips and performance notes

- Set TOKENIZERS_PARALLELISM=false when using multiprocessing-heavy code paths to avoid deadlocks.
- On Ampere+ GPUs, TF32 and fused AdamW improve throughput; the trainer enables these when available.
- Sentence-aware data path is recommended: it improves syntactic signals and saves storage by masking dynamically.
- Index maps for preblocked datasets are cached under `cache_index/` with stable hashing; the loader cleans up old cache files.

---

If you want these docs rendered as a docs site or consolidated into a single handbook, we can extract per-file sections and generate a mkdocs or Sphinx site on request.
