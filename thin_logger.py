import os
import json
import time
import datetime
from typing import Optional, Dict, Any


class ThinLogger:
    """
    Minimal, backend-neutral logger.
    Methods are no-ops on non-main ranks.
    """
    def __init__(self, log_dir: str, run_name: Optional[str] = None, enable_tb: bool = True, is_main: bool = True):
        self.is_main = bool(is_main)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = run_name or f"run-{ts}"
        self.jsonl_path = os.path.join(self.log_dir, f"{self.run_name}.jsonl")
        self._jsonl = open(self.jsonl_path, "a") if self.is_main else None

        # TensorBoard is optional; guard import
        self.tb = None
        if self.is_main and enable_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(log_dir=os.path.join(self.log_dir, "tb", self.run_name))
            except Exception:
                self.tb = None

        # Write a header line for JSONL
        if self.is_main and self._jsonl is not None:
            header = {
                "_type": "run_start",
                "run_name": self.run_name,
                "time": time.time(),
            }
            self._jsonl.write(json.dumps(header) + "\n")
            self._jsonl.flush()

    def log_config(self, cfg: Dict[str, Any]):
        if not self.is_main:
            return
        record = {"_type": "config", "time": time.time(), "config": cfg}
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()
        # Also store a copy in the log dir
        try:
            with open(os.path.join(self.log_dir, f"{self.run_name}-config.json"), "w") as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    def set_run_name(self, name: str):
        if not self.is_main:
            return
        self.run_name = name

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, prefix: Optional[str] = None):
        if not self.is_main:
            return
        rec = dict(metrics)
        if prefix:
            rec = {f"{prefix}/{k}": v for k, v in rec.items()}
        if step is not None:
            rec["step"] = int(step)
        rec["time"] = time.time()
        rec["_type"] = rec.get("_type", "metric")

        # JSONL
        self._jsonl.write(json.dumps(rec) + "\n")
        self._jsonl.flush()

        # TensorBoard scalars (only simple scalars)
        if self.tb is not None:
            for k, v in rec.items():
                if k in ("_type", "time", "step"):
                    continue
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, global_step=step if step is not None else 0)

    def log_artifact(self, path: str, name: Optional[str] = None):
        if not self.is_main:
            return
        try:
            stat = os.stat(path)
            rec = {
                "_type": "artifact",
                "time": time.time(),
                "name": name or os.path.basename(path),
                "path": os.path.abspath(path),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
            self._jsonl.write(json.dumps(rec) + "\n")
            self._jsonl.flush()
        except Exception:
            pass

    def close(self):
        if self.tb is not None:
            try:
                self.tb.flush()
                self.tb.close()
            except Exception:
                pass
        if self._jsonl is not None:
            try:
                self._jsonl.flush()
                self._jsonl.close()
            except Exception:
                pass


def compute_protected_ids(tokenizer) -> set:
    """Build a set of structural/protected token IDs from the tokenizer."""
    candidates = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "[PAR]", "[TAB]", "[DOC]", "[EOD]", "[SPK]",
        "<pad>", "<unk>", "<cls>", "<sep>", "<mask>",
        "<par>", "<tab>", "<doc>", "<eod>", "<spk>",
    ]
    vocab = tokenizer.get_vocab()
    ids = set()
    for tok in candidates:
        if tok in vocab:
            try:
                ids.add(int(vocab[tok]))
            except Exception:
                pass
    return ids


def get_logger(config: dict, checkpoint_path: str, is_main: bool) -> ThinLogger:
    """
    Factory to create a ThinLogger using config knobs.
    - config["logging_dir"] overrides default path under checkpoint.
    - config["enable_tensorboard"] toggles TB writer.
    - config["run_name"] names the run.
    """
    log_dir = config.get("logging_dir") or os.path.join(checkpoint_path, "logs")
    enable_tb = bool(config.get("enable_tensorboard", True))
    run_name = config.get("run_name")
    return ThinLogger(log_dir=log_dir, run_name=run_name, enable_tb=enable_tb, is_main=is_main)
