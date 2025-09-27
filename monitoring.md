# Monitoring, Metrics, and Visualization

This guide shows how to monitor training runs locally without external services. It covers:
- What gets logged and where
- How to visualize metrics from JSONL logs using a command-line tool (CLI)
- How to explore metrics in a Jupyter notebook

The setup is local-first: metrics are written to JSON Lines (JSONL) and can optionally be plotted live or after the run. Logging happens only on rank-0 in distributed runs to avoid duplication.


## What gets logged

The trainer uses a thin, local logger to write structured events. Typical records include:
- Train step metrics: loss, avg_loss, learning rate, throughput, step/data timing, gradient health, masking diagnostics
- Validation metrics (rank-0 only at end of epoch): average loss, masked LM accuracy
- Epoch summaries: data/length distributions, truncation rates, uniqueness stats
- Meta: host, user, world size

Notes:
- Training events include a prefix (e.g., `train`) and validation events use `val`.
- Epoch markers are recorded with `_type: "epoch_end"` along with summary metrics.
- Records always include a `step` and often an `epoch` for easier plotting.

The JSONL file is typically written under your checkpoint directory (for example: `<checkpoint>/logs/metrics.jsonl`). If your run writes to a different file name or location, adjust the paths below accordingly.


## CLI interface (scripts/plot_jsonl.py)

Use the CLI when you want fast, repeatable plots from one or many runs.

- Location: `scripts/plot_jsonl.py`
- Dependencies: `pandas`, `matplotlib` (already standard in most Python envs). Gzipped JSONL (`.jsonl.gz`) is supported.

### Basic usage

```bash
# Single run: plot training (avg) and validation loss with a robust x-axis
python scripts/plot_jsonl.py \
  --input /path/to/metrics.jsonl \
  --metrics train.loss val.loss \
  --x-key global_step \
  --prefer-avg \
  --out outputs/loss_curve.png

# Multiple runs: compare two experiments on the same chart
python scripts/plot_jsonl.py \
  --input /expA/metrics.jsonl /expB/metrics.jsonl \
  --metrics train.loss \
  --x-key global_step \
  --smooth 20 \
  --out outputs/compare_train_loss.png

# Make validation a connected line and further reduce noise
python scripts/plot_jsonl.py \
  --input /path/to/metrics.jsonl \
  --metrics train.loss val.loss \
  --x-key global_step \
  --prefer-avg \
  --force-line \
  --downsample 50 \
  --smooth 200 --smooth-mode centered \
  --clip-low-pct 1 --clip-pct 99 \
  --out outputs/loss_curve_smooth.png

# Show an interactive window instead of (or in addition to) saving to a file
python scripts/plot_jsonl.py \
  --input /path/to/metrics.jsonl \
  --metrics train.loss val.mlm_acc \
  --x-key global_step \
  --show
```

### Common flags

- `--input`: One or more JSONL files. You can pass multiple files to compare runs.
- `--metrics`: Space-separated list of metric names to plot. Examples:
  - `train.loss`, `train.avg_loss`
  - `val.loss`, `val.mlm_acc`
  - If your logs use slash notation (e.g., `train/loss`), pass those literal names.
- `--x-key`: X-axis column. Defaults to `step`. Use `global_step` to avoid per-epoch overlap on the x-axis.
- `--smooth`: Moving average window (integer). Use `0` or omit to disable smoothing. See also `--smooth-mode`.
- `--out`: Output PNG path. If omitted and `--show` is used, the plot displays without saving.
- `--show`: Display an interactive window.
- `--prefer-avg`: Prefer `train.avg_loss` over `train.loss` when present (less noisy).
- `--force-line`: Force line plots (connect sparse points like validation into a continuous line).
- `--smooth-mode`: Smoother type when `--smooth>0`. Options: `mean` (default), `centered`, `ema`, `median`.
- `--clip-pct`, `--clip-low-pct`: Clip metric values to upper/lower percentiles (e.g., `--clip-low-pct 1 --clip-pct 99`). Useful to suppress rare spikes before smoothing.
- `--downsample`: Downsample by x-key bin size (e.g., `--downsample 50` groups 50 adjacent points) prior to smoothing.

### Data handling details

- The parser expands nested `metrics` dicts in each JSON line so you can address metrics as flat names (e.g., `train.loss`). The plotter also aliases dotted and slashed names both ways (e.g., `train.loss` <-> `train/loss`, `val.loss` <-> `val/loss`).
- If your file is empty or contains only non-metric entries, the tool will warn you.
- When comparing multiple inputs, the legend labels are taken from file names.
- If both `epoch` and `step` are present, choosing `--x-key global_step` computes a monotonic index across epochs to prevent overlap. The data is sorted by the selected x-key before plotting.
- With `--prefer-avg`, the tool will map `train.loss` to `train.avg_loss` for plotting and also create a shadow column so downstream tools that expect `train.loss` still work.

### Troubleshooting

- “Stripes” or steps overlapped across epochs: use `--x-key global_step`.
- Validation points not connected: add `--force-line`.
- Loss is too noisy: add `--prefer-avg`, increase `--smooth`, try `--smooth-mode centered` or `--smooth-mode ema`, and optionally `--downsample 50`.
- Sharp spikes remain: clip tails before smoothing: `--clip-low-pct 1 --clip-pct 99`.
- “Empty DataFrame provided to plot_metrics”: your file had no recognized metric columns. Ensure records contain a `metrics` object or top-level metric keys.
- Metric names differ: inspect a few lines in the JSONL to confirm the exact keys (e.g., `train.loss` vs `train/loss`). The plotter aliases common dotted/slashed variants automatically.


## Jupyter workflow (notebooks/metrics_viewer.ipynb)

Use the notebook for exploratory analysis, custom plots, and quick experiments.

- Location: `notebooks/metrics_viewer.ipynb`
- Utilities: the notebook relies on the same parsing and plotting utilities used by the CLI.

### Typical flow

1) Open the notebook and set the paths to one or more JSONL logs in the first code cell:

```python
LOGS = [
    "/path/to/metrics.jsonl",  # or metrics.jsonl.gz
    # "/path/to/another_run/metrics.jsonl",
]
```

2) Select metrics and smoothing:

```python
METRICS = ["train.loss", "val.loss", "val.mlm_acc"]
SMOOTH = 20  # 0 to disable
XKEY = "step"  # or "epoch" if available in your logs
```

3) Run the cell that loads and normalizes the logs. It will:
- Read JSONL (streaming for large files)
- Expand nested `metrics`
- Flatten columns for easy filtering (e.g., `train.loss`)

4) Plot the selected metrics. The notebook provides simple helpers to:
- Filter by metric names or prefixes
- Apply moving-average smoothing
- Compare multiple runs on the same axes
- Save figures to disk (e.g., `outputs/metrics.png`)

### Example code (in the notebook)

```python
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from metrics_utils import read_jsonl, filter_metrics, moving_average, plot_metrics, save_plot

# 1) Load
frames = [read_jsonl(path) for path in LOGS]
# 2) Filter by the metrics you care about
frames = [filter_metrics(df, METRICS) for df in frames]
# 3) Optional smoothing
if SMOOTH and SMOOTH > 0:
    frames = [moving_average(df, window=SMOOTH) for df in frames]
# 4) Plot
fig = plot_metrics(frames, metrics=METRICS, x_key=XKEY, labels=[Path(p).stem for p in LOGS])
# 5) Save (optional)
os.makedirs("outputs", exist_ok=True)
save_plot(fig, "outputs/metrics.png")
```

### Tips

- Large files: The loader uses a streaming path when needed; still, you can downsample to speed up plotting.
- Metric discovery: After loading, `df.columns` shows available metric columns; use that to refine `METRICS`.
- Validation cadence: In distributed runs, validation is logged from rank-0 at epoch end. If you’re not seeing `val.*` lines every N steps, that’s by design.


## FAQ

- Where is the log file?
  - Typically under your run’s checkpoint directory, inside a `logs/` subfolder. Adjust paths if you customized logging.

- What if I only want command-line plots?
  - Use `scripts/plot_jsonl.py` exclusively. It works without Jupyter and can run in CI to produce artifacts.

- Can I add new metrics?
  - Yes. Add them to the trainer’s logging calls (they’ll be included under the `metrics` dict) and they will appear in both CLI and notebook flows automatically.

- Do I need W&B or TensorBoard?
  - No. This workflow is local-first. You can still mirror metrics to other systems later if you like.


## Next steps (optional)

- Directory mode for the CLI (auto-pick latest JSONL in a folder)
- Auto-detect common metric names for first-time plots
- Live plotting during training (tailing the JSONL)