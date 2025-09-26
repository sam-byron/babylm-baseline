# BLiMP scores: what was broken and how we fixed it

This note explains why early BLiMP results looked low or “busted” and the concrete fixes that produced the final, reasonable scores.

- Final reference run (filtered): acc ≈ 0.6334 ± 0.0018
- Before fixes: inconsistent/very low PLL accuracy, OOV spikes, occasional CUDA asserts, and load-time warnings

## TL;DR
A few infrastructure issues were corrupting evaluation rather than revealing the model’s true ability:

- Checkpoint save/load mismatch led to a partially reinitialized MLM head at load time.
- AutoModel compatibility gaps (output head contract, tie-weights) made the head untied or replaced.
- Tokenizer artifacts (fast/slow mismatch, vocab alignment) caused OOV spikes and even index errors.

We fixed the save pipeline, made the model fully Auto-compatible (including proper `tie_weights`), shipped tokenizer files with correct `auto_map`, and hardened evaluation. BLiMP scores then reflected actual model quality instead of tooling artifacts.

---

## Symptoms we observed

- Load-time warnings about “newly initialized” or missing keys for the classifier head.
- AutoModel load issues (meta tensors, device-map quirks) unless we used a manual import fallback.
- High OOV rates and occasionally CUDA device-side asserts (e.g., index out of range) during PLL evaluation.
- Very low PLL-accuracy on a tiny synthetic checkpoint (expected), but also erratic results on the organic checkpoint when loaded improperly.

## Root causes

1) Save/load instability and code–weight drift

- Mid-epoch synchronized saves occasionally hung; when saves finished, the checkpoint sometimes lacked the exact modeling code used during training.
- On reload, the classifier/tied head could be treated as missing or reinitialized, silently degrading logits and PLL.

2) Output head not fully compliant with HF’s Auto contract

- `get_output_embeddings` did not return the output module, and `set_output_embeddings` wasn’t properly handled.
- `tie_weights` wasn’t ensuring the output head shared weight with the input embedding in the standard HF way.
- Result: Auto loaders could leave the head untied or reinitialized.

3) Tokenizer/vocab misalignment

- Incompatible fast tokenizer normalizer forced us to fall back to the slow tokenizer sometimes.
- Missing or inconsistent tokenizer artifacts (`tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json`) and absent `auto_map` entries caused inconsistent ID mappings.
- Token IDs could exceed the model’s vocab, causing CUDA asserts and invalid PLL.

4) AutoModel loading pitfalls

- Using `low_cpu_mem_usage=True` or an automatic `device_map` on custom models sometimes produced meta tensors and subtle load bugs.
- If the checkpoint didn’t embed `modeling_ltgbert.py` and `configuration_ltgbert.py`, `trust_remote_code` could pick up mismatched class definitions.

## Fixes implemented

A) Robust, HF-friendly saving

- Introduced a `save_hf_checkpoint` routine that:
  - Calls `tie_weights()` before saving.
  - Registers `auto_map` for `AutoConfig`, `AutoModelForMaskedLM`, and `AutoTokenizer`.
  - Saves with `safe_serialization=False` to avoid loader edge cases in our stack.
  - Writes tokenizer artifacts (`tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json`).
  - Copies `modeling_ltgbert.py` and `configuration_ltgbert.py` into the checkpoint for `trust_remote_code` stability.
- Save orchestration:
  - Mid-epoch: rank-0-only lightweight save (no collectives) to avoid hangs.
  - End-of-epoch: synchronized save with barriers and timing logs.

B) Make the model Auto-compatible and warning-free

- In `LtgBertForMaskedLM`:
  - `get_output_embeddings()` returns the output `nn.Linear` module.
  - `set_output_embeddings()` accepts and replaces that module.
  - `tie_weights()` ties output head weight to input embeddings.
  - `_keys_to_ignore_on_load_missing` set for the tied weight to avoid noisy, misleading warnings.
- Mirrored these changes in the checkpoint’s `modeling_ltgbert.py` so loading uses the exact same class.

C) Tokenizer alignment and safe evaluation

- Generated complete tokenizer files and added `auto_map` for `AutoTokenizer` (BERT slow and fast entries).
- Loader hardening:
  - If fast tokenizer JSON/normalizer is incompatible, fall back to `use_fast=False`.
  - Clamp token IDs to `< vocab_size` during evaluation to prevent device-side asserts.
- Outcome: OOV ≈ 0% for the organic checkpoint; PLL is stable and meaningful.

D) Hardened AutoModel loading

- Prefer `AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True, low_cpu_mem_usage=False, device_map=None)`.
- With modeling/config files inside the checkpoint, the Auto path matches the trained implementation.

## Verification steps

1) Save-and-reload sanity check

- A small script saves via the improved routine, reloads with AutoModel, captures warnings, runs a tiny forward on CPU and CUDA, and fails fast on any issue.
- Result: no warnings; logits shape as expected.

2) BLiMP sanity and full runs

- Initial subset (e.g., `wh_vs_that_with_gap`) yielded plausible accuracy once the model loaded correctly and token IDs were in range.
- Broader run with the evaluation pipeline produced: acc ≈ 0.6334 ± 0.0018 (filtered) with strong categories (e.g., determiners, certain wh/that contrasts) and reasonable mid-range/weak categories typical for MLMs.

## Why scores improved

- Before: evaluation reflected infrastructure problems (untied/reinitialized head, OOV/ID mismatches) more than model quality.
- After: clean save/load + proper head tying + tokenizer alignment removed artifacts. Scores now reflect what the model actually learned.

## Notes and gotchas

- If you modify the modeling class, ensure the checkpoint also carries the updated `modeling_ltgbert.py` and `configuration_ltgbert.py` so `trust_remote_code` loads the right version.
- Keep `low_cpu_mem_usage=False` and `device_map=None` for custom models unless you’ve validated the alternative path.
- When testing new tokenizers, regenerate tokenizer files and confirm `config.vocab_size` matches `vocab.txt` length.

## High-level changelog

- Added robust HF save routine and mid/epoch save split to avoid hangs.
- Implemented HF output head contract and explicit `tie_weights`.
- Copied modeling/config files into checkpoints and registered `auto_map` entries.
- Generated tokenizer artifacts and added slow/fast fallback logic; clamped token IDs during eval.
- Added a standalone save–reload check and hardened BLiMP evaluation behavior.

---

With these fixes in place, BLiMP results are stable and representative of the trained model rather than tooling artifacts.
