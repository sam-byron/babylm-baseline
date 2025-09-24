# Migration Notes: Structural Tokens & Tokenizer Refresh

## Summary
This migration introduced always-on structural tokens and a fresh tokenizer trained over the entire BNC corpus (markdown-sourced) to improve boundary integrity and contextual signals without fabricating punctuation. No legacy embedding transplant was performed; model was reinitialized (except where explicit backward compatibility was unnecessary).

## New Special Tokens
```
[UNK] [CLS] [SEP] [PAD] [MASK] [PAR] [TAB] [DOC] [EOD] [SPK]
```
- `[DOC]` / `[EOD]` mark document boundaries (inserted per file, not per arbitrary size chunk).
- `[SPK]` prefixes lines detected as dialogue speaker labels (`Name:` or similar pattern) when name in curated first-names list.
- `[PAR]`, `[TAB]` retained from earlier formatting normalization phase (paragraph/tab placeholders if present).

## Tokenizer Training
| Aspect | Value |
|--------|-------|
| Algorithm | WordPiece |
| Vocab Size | 16,384 |
| Case Handling | (Assumed) Lowercase preserve? (See manifest) |
| Digit Handling | Digits kept separate (split enabled) |
| Corpus Scope | All `.md` files under `data/pretrain/bnc` recursively |
| Ordering | Sorted deterministic file list |
| F95 | 220 (95th percentile line length pre-packing) |

A manifest file (adjacent to `wordpiece_vocab.json`) records:
- sha256 hash of streaming file concatenation (paths + contents)
- special token list & indices
- tokenizer hyperparameters
- sample preview of tail tokens

## Unknown Token Clarification
Literal `[UNK]` substrings exist in raw source (legacy placeholders). True unknown rate was measured by temporarily remapping those literal occurrences to a neutral placeholder sentinel before encoding. Observed genuine fallback unknown rate: ~0.081% (stable across vocab upsize experiments) → indicates near-complete coverage; further vocab growth provided negligible benefit.

## Packing Integration
`prepare_data.py` now always performs sentence packing with document & speaker tagging; the historical flag `--sentence-pack` is accepted but ignored (kept for backward CLI compatibility).

Key heuristics (see `PACKING.md` for full detail):
- Dangling carryover (short non-terminal tail moved forward)
- Backward borrow (repair previous non-terminal end if following short terminal fits)
- Optional short trailing drop (currently inert given distribution)
- Filler terminals list to treat interjections as boundary.

## Compatibility & Model Config
Because new special tokens expand the vocabulary and reposition indices, prior checkpoints are *not* directly compatible. Decision: reinitialize embeddings and retrain for cleanliness. If future backward compatibility becomes important, a mapping JSON can be produced referencing old vs new token ids; out of scope for this migration.

## Training Config Adjustments
- `block_size` now configurable per experiment (128 for ablation vs 512 packing context stats).
- LAMB optimizer variant config (`gpt5_lamb_bs128_ga24.json`) includes `_packing_metadata` summarizing observed terminal punctuation rates.

## Reasoning & Rejected Alternatives
| Option | Decision | Rationale |
|--------|----------|-----------|
| Partial embedding transplant | Rejected | Risk of semantic drift + index shift complexity outweighed marginal warm-start gains. |
| Artificial punctuation injection | Rejected | Inflates terminal rate artificially; better to rely on structural tokens & heuristics. |
| Aggressive short-fragment dropping | Rejected | Data loss and potential discourse coherence degradation. |

## Future Work
1. Emit final global packing stats JSON automatically at end of full preprocessing pipeline.
2. Provide script to re-align legacy checkpoints by constructing a remap table (only if needed).
3. Add candidate diagnostics for short trailing drop (why not triggered) to either prune or tune that path.
4. Integrate BLiMP / DevBench evaluation hooks into preprocessing log for automated regression tracking.

## Verification Checklist
- [x] Tokenizer manifest saved
- [x] Special tokens confirmed present in vocab
- [x] Packing instrumentation producing stable rates on full corpus slices
- [x] Unknown rate independently validated
- [x] Training config references correct `tokenizer_path`

*Last updated:* (auto-generated)
