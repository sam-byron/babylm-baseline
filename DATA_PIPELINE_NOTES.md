# Data Pipeline Notes

## Overview
This project now produces *preblocked* training data: each `chunk*.pt` file is a 2‑D `LongTensor` of shape `[N, block_size]` (currently 128). Blocks are already sentence/document aware (wrapped with `[DOC] ... [EOD]`) and padded only at the tail of each block.

## Recent Improvements
| Area | Change | Rationale |
|------|--------|-----------|
| Packing | Direct block writer with buffered document grouping | Eliminates repacking at load time |
| Shuffle | Added `shuffle_level` (none|document|segment) default **document** | Maintains intra‑document coherence, fixes false boundary errors |
| Validation | `validate_chunks.py` now supports `--carry-open-doc` and `--literal-heuristics` | Reduces false positives from literal marker text |
| Escaping | Optional `escape_literal_markers` in `prepare_data.py` | Prevents genuine literal strings like `[DOC]` from becoming structural |
| Dataset | `ChunkedDataset` detects preblocked vs legacy format | Fast O(1) row indexing, minimal memory |
| Collator | Dynamic & static masking reuse existing masking strategies | Consistent masking semantics |
| Anomalies | JSON emission of contextual anomalies | Inspect rare structural issues |

## ChunkedDataset Modes
1. **Preblocked Mode (current)**: Each chunk is a matrix. We lazily load tensors and index rows; no concatenation cost.
2. **Legacy Concatenation**: For older artifacts storing lists of variable length sequences; retained for fallback.

Detection logic inspects the first existing path. Mixed formats in the same run are not supported.

## Validator Heuristics
Literal heuristic down-ranks ambiguous `[DOC]` or `[EOD]` tokens if:
- A `[DOC]` appears while an open doc is active and is flanked only by structural/pad tokens (likely literal text).
- An `[EOD]` appears when no doc is open and no forthcoming `[DOC]` nearby (treated literal).

Escaping should reduce need for heuristics; both can be enabled for safety.

## Performance Tips
- Increase `ChunkedDataset.cache_size` if you have many small chunk files and enough RAM.
- For I/O bound training runs, placing chunk files on fast SSD/NVMe yields better throughput.
- When using dynamic masking, keep batch size modest to avoid masking overhead spikes.

## Future Opportunities
- Add on-disk mmap loader to avoid `torch.load` overhead per file.
- Optional block-level sampling weights (e.g., curriculum by document length or domain tags).
- Gather statistics on token distribution post-escaping to ensure no skew.

## Smoke Test
A lightweight smoke test lives at `tests/test_chunked_dataset_smoke.py` and exercises:
- Index building
- Random access
- Basic throughput timing

Run manually:
```bash
python tests/test_chunked_dataset_smoke.py \
  || echo "Smoke test failed"
```

Set `CACHE_PATH` / `TOKENIZER_PATH` env vars to override defaults.

## FAQ
**Q:** Why still a few validator anomalies?  
**A:** Residual edge cases of literal markers not escaped in legacy data; heuristics flag them but training impact is negligible (<5 in 1.5M blocks).

**Q:** Do I need dynamic batching with fixed block size?  
**A:** Less critical; uniform blocks mean minimal padding. You can still cap `max_tokens` for memory predictability.

---
_Last updated: automated notes generation._
