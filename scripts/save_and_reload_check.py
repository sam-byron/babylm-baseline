#!/usr/bin/env python3
"""
Quick standalone check:
- Instantiate a tiny LtgBertForMaskedLM
- Save with training_utils.save_hf_checkpoint (safe_serialization=False, auto_map registered)
- Reload via AutoModelForMaskedLM.from_pretrained(trust_remote_code=True)
- Capture warnings/logging and exit non-zero if weight init/missing/unexpected key warnings appear

Usage:
  python scripts/save_and_reload_check.py [--out ./tmp_check_save_load]
"""

import argparse
import logging
import os
import shutil
import sys
import warnings

from pathlib import Path

from transformers import AutoModelForMaskedLM

# Ensure repo root is on sys.path for local imports when run from scripts/
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Local imports from repo
from configuration_ltgbert import LtgBertConfig
from modeling_ltgbert import LtgBertForMaskedLM
from training_utils import save_hf_checkpoint


class _LogCapture(logging.Handler):
    """Capture logging records at WARNING or above."""

    def __init__(self, level=logging.WARNING):
        super().__init__(level=level)
        self.records = []

    def emit(self, record: logging.LogRecord):
        self.records.append(record)

    def messages(self):
        return [self.format(r) if self.formatter else r.getMessage() for r in self.records]


def _install_logging_capture():
    cap = _LogCapture()
    # capture common HF loggers and root
    for name in ("transformers", "hf_hub", "huggingface_hub", "__main__", ""):
        logger = logging.getLogger(name)
        logger.addHandler(cap)
        logger.setLevel(logging.WARNING)
        logger.propagate = True
    return cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./tmp_check_save_load", help="Output checkpoint directory")
    parser.add_argument("--vocab-size", type=int, default=128, help="Tiny vocab for fast smoke test")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--intermediate", type=int, default=128)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # Build a tiny model config consistent with the architecture
    cfg = LtgBertConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.intermediate,
        max_position_embeddings=64,
        layer_norm_eps=1e-7,
        pad_token_id=0,
    )
    assert cfg.hidden_size % cfg.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"

    model = LtgBertForMaskedLM(cfg)

    # Minimal accelerator shim: mimic unwrap_model + is_main_process without importing accelerate
    class _Shim:
        is_main_process = True

        @staticmethod
        def unwrap_model(m):
            return m

    # Save checkpoint using the project's helper
    print(f"[check] Saving checkpoint to: {out}")
    save_hf_checkpoint(_Shim, model, tokenizer=None, out_dir=str(out))

    # Prepare to capture warnings and logging during load
    warn_records = []

    def _warn_hook(message, category, filename, lineno, file=None, line=None):
        warn_records.append(f"{category.__name__}: {message}")
        return warnings._showwarnmsg_impl(warnings.WarningMessage(message, category, filename, lineno, file, line))

    warnings.simplefilter("always")
    warnings.showwarning = _warn_hook
    log_cap = _install_logging_capture()

    # Reload
    print("[check] Loading with AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)")
    loaded = AutoModelForMaskedLM.from_pretrained(str(out), trust_remote_code=True)

    # Optional quick forward to ensure tensors are shaped and usable
    import torch

    loaded.eval()
    with torch.inference_mode():
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        attention_mask = torch.ones_like(input_ids)
        out_logits = loaded(input_ids=input_ids, attention_mask=attention_mask).logits
        assert out_logits.shape == (2, 8, cfg.vocab_size), f"Unexpected logits shape: {tuple(out_logits.shape)}"

    # Collect messages
    log_msgs = log_cap.messages()

    def contains_bad(msg: str) -> bool:
        msg_l = msg.lower()
        # Typical HF load-time messages when weights mismatch
        bad_keys = (
            "not initialized",
            "newly initialized",
            "missing keys",
            "unexpected keys",
            "mismatched sizes",
        )
        return any(k in msg_l for k in bad_keys)

    offending = [m for m in log_msgs if contains_bad(m)] + [m for m in warn_records if contains_bad(m)]

    if offending:
        print("\n[check] FAIL: Detected load-time warnings indicating missing/new weights:")
        for m in offending:
            print(f"  - {m}")
        print(f"\n[check] Checkpoint dir: {out}")
        sys.exit(1)

    print("\n[check] PASS: Model saved and reloaded with no initialization warnings.")
    print(f"[check] Log files in: {out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
