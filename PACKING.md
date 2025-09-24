# Sentence Packing & Structural Markers

## Overview
We restructure raw BNC lines into packed training sequences that minimize mid-sentence truncations while preserving document and conversational structure. Each packed sample has a single leading `[CLS]` and a trailing `[SEP]` after every sentence.

```
[CLS] Sentence one. [SEP] Sentence two? [SEP] Fragment (carried). [SEP]
```

Always-on structural tokens:
- `[DOC]` — document start
- `[EOD]` — document boundary marker
- `[SPK]` — speaker marker prefixed to detected dialogue lines

## Goals
1. Increase proportion of packs ending on terminal punctuation without fabricating punctuation.
2. Avoid discarding short trailing content unless it is a non-terminal fragment *and* the pack is already sufficiently full.
3. Preserve small fragments by carrying them forward rather than dropping them.
4. Provide instrumentation for data quality auditing and heuristic tuning.

## Heuristics
| Heuristic | Description | Key Params |
|-----------|-------------|------------|
| Dangling carryover | If last sentence lacks terminal punctuation and is short (`<= max_dangling_tokens`), pop and seed next pack. | `max_dangling_tokens=48` |
| Min fill ratio opportunistic flush | After each line, flush if `current_len / max_length >= min_fill_ratio` to promote high utilization. | `min_fill_ratio=0.85` |
| Short trailing drop (optional) | If final sentence is non-terminal, very short (`<= short_trailing_max_tokens`), and fill ratio already high, drop & carry to next pack. Currently rarely triggers. | `short_trailing_max_tokens=10` |
| Backward borrow | If previous pack ends non-terminal and next pack begins with a short terminal sentence, move that first sentence backward if it fits. | `borrow_first_sent_max_tokens=8` |
| Filler terminals | Treat interjections (`yeah, mm, erm, uh, er, ah, hmm`) as sentence-terminal to avoid over-carry of dialogue fillers. | `FILLER_TERMINALS` set |

## Instrumentation Fields
Produced when `collect_stats=True` in `pack_sentences`:
- `packs`: total packed sequences emitted.
- `terminal_end`: packs whose final sentence ended with `.?!` or filler terminal token.
- `nonterminal_end`: packs ending in a fragment (used for terminal rate denominator).
- `dangling_carried`: count of times a short non-terminal tail was moved to the next pack.
- `borrow_moves`: successful backward moves of a leading terminal sentence into previous pack.
- `short_trailing_dropped`: times a short trailing fragment was dropped (then carried) under high fill conditions.

### Derived Rates (prefix `_rate`)
`terminal_end_rate = terminal_end / packs`  
`dangling_carried_rate = dangling_carried / packs`  
`borrow_move_rate = borrow_moves / packs`  
`short_trailing_dropped_rate = short_trailing_dropped / packs`  
`nonterminal_end_rate = nonterminal_end / packs`

## Observed Metrics
| Dataset Slice | Packs | Terminal End Rate | Dangling Carried Rate | Borrow Move Rate | Short Drop Rate |
|---------------|-------|-------------------|-----------------------|------------------|-----------------|
| Debug (40k lines) | 2,067 | 0.877 | 0.166 | 0.000 | 0.000 |
| Full (stream slices 30–45 snapshot) | ~291k | 0.884–0.897 (stabilized) | 0.16–0.18 | ~0.001 | 0.000 |

*Note:* Early slices inflate terminal rate ( >0.95 ) because of more well-punctuated narrative sections; later inclusion of dialogue and lists lowers the stabilized rate.

## Rationale & Trade-offs
- **Carryover vs Drop:** Preference is to *carry* rather than drop to avoid data loss; dropping only occurs when capacity is already exploited (high fill ratio) and the fragment is very short.
- **Backward Borrow:** Low-frequency but correctness-oriented adjustment to repair non-terminal endings when a perfect terminal candidate is adjacent.
- **Short Trailing Threshold Inertness:** Current corpus distribution rarely produces very short trailing fragments in already-full packs; acceptable to leave heuristic dormant rather than relax it and risk fragment starvation.

## Future Improvements
1. Add diagnostic counters for short trailing candidates (reasons not dropped) to decide if threshold tuning is justified.
2. Provide final global stats JSON emitted at the end of full `prepare_data` run.
3. Integrate punctuation-final rate into training metadata for reproducibility.
4. Consider a dual-mode packer: *fast minimal* (no borrow) vs *quality* (full heuristics) if preprocessing throughput becomes a bottleneck.

## API Summary (`pack_sentences`)
```python
pack_sentences(lines, tokenizer, max_length,
               min_fill_ratio=0.85,
               ensure_terminal_punct=True,
               max_dangling_tokens=48,
               allow_borrow=True,
               borrow_first_sent_max_tokens=8,
               add_doc_and_speaker=True,
               collect_stats=False,
               drop_short_trailing=False,
               short_trailing_max_tokens=10)
```
Returns either `List[str]` or `(List[str], stats_dict)` when `collect_stats=True`.

## Structural Tokens & Tokenizer Manifest
Tokenizer includes: `[UNK] [CLS] [SEP] [PAD] [MASK] [PAR] [TAB] [DOC] [EOD] [SPK]`.
Manifest captures:
- vocab size, special tokens
- corpus hash (sha256 over streamed source file order)
- F95 value (95th percentile tokenized line length)
- training parameters (algorithm, lowercase, byte/digit splitting)

## Unknown Rate Clarification
Raw corpus contains literal `[UNK]` placeholders. True model unknown rate is measured after temporarily remapping literals to `[PLH]` sentinel; observed genuine UNK substitution rate ≈ 0.081% at vocab 16,384 (stable across 24.5k test variant).

## Reproducibility Checklist
1. Stream all `.md` files (recursive) in deterministic (sorted) order.
2. Persist tokenizer manifest (`wordpiece_vocab.json` + metadata JSON).
3. Record packing stats snapshot (`PACKING_STATS.json`) — TODO integrate into final pipeline.
4. Store config (`config_sentence_aware.json` or LAMB variant) referencing the tokenizer path and cache output.

## Glossary
| Term | Meaning |
|------|---------|
| F95 | 95th percentile of tokenized line length (pre-packing) used to gauge tail risk. |
| Fill Ratio | Used tokens / max_length within a pack. |
| Fragment | Non-terminal sentence segment lacking `.?!` or filler terminal. |

---
*Last updated:* (auto-generated)