#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

# add src to path when running standalone
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from metrics_utils import read_jsonl, plot_metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description='Plot metrics from one or more JSONL files')
    ap.add_argument('--input', '-i', nargs='+', required=True, help='One or more JSONL files')
    ap.add_argument('--metrics', '-m', nargs='+', required=True, help='Metric names to plot')
    ap.add_argument('--x-key', default='step', help='x-axis key (default: step). Use global_step to avoid per-epoch overlap.')
    ap.add_argument('--smooth', type=int, default=0, help='EMA/moving average window')
    ap.add_argument('--out', '-o', type=str, default=None, help='Output PNG path')
    ap.add_argument('--show', action='store_true', help='Show the figure interactively')
    ap.add_argument('--prefer-avg', action='store_true', help='Prefer train.avg_loss over train.loss when available')
    ap.add_argument('--force-line', action='store_true', help='Force line plots (connect points) for all metrics')
    ap.add_argument('--smooth-mode', choices=['mean', 'centered', 'ema', 'median'], default='mean',
                    help='Smoothing mode when --smooth>0 (default: mean)')
    ap.add_argument('--clip-pct', type=float, default=None,
                    help='Clip metric values to percentile (e.g., 99.0). Use with --clip-low-pct for symmetric clipping.')
    ap.add_argument('--clip-low-pct', type=float, default=None,
                    help='Lower percentile for clipping (e.g., 1.0).')
    ap.add_argument('--downsample', type=int, default=None,
                    help='Downsample by x-key bin size (e.g., 50 groups adjacent points). Applied before smoothing.')
    args = ap.parse_args(argv)

    frames = []
    for p in args.input:
        df = read_jsonl(p)
        df['__source__'] = str(p)
        frames.append(df)
    if not frames:
        print('No inputs', file=sys.stderr)
        return 2
    df_all = pd.concat(frames, ignore_index=True)

    # If requested, map train.loss to train.avg_loss when present to reduce noise.
    # Do this both logically (args.metrics) and physically (create alias column) so plotter finds it.
    if args.prefer_avg and 'train.avg_loss' in df_all.columns:
        # Create a shadow column so downstream plotters can use 'train.loss' without knowing about avg
        df_all['train.loss'] = df_all.get('train.avg_loss', df_all.get('train/loss', None))
        # Also update requested metrics to point to the smoother variant when applicable
        args.metrics = [
            ('train.avg_loss' if m in ('train.loss', 'train/loss') else m) for m in args.metrics
        ]

    # Compute a monotonic global_step to prevent overlapping steps across epochs
    if 'epoch' in df_all.columns and 'step' in df_all.columns:
        try:
            df_tmp = df_all[['epoch', 'step']].dropna().copy()
            df_tmp['epoch'] = df_tmp['epoch'].astype(int)
            df_tmp['step'] = df_tmp['step'].astype(int)
            # Determine per-epoch max step and build cumulative offsets
            max_per_epoch = df_tmp.groupby('epoch')['step'].max().sort_index()
            offsets = {}
            cumulative = 0
            for ep, mx in max_per_epoch.items():
                offsets[ep] = cumulative
                cumulative += int(mx) + 1
            # Apply mapping to full df
            def _global_step_row(row):
                e = row.get('epoch')
                s = row.get('step')
                if pd.isna(e) or pd.isna(s):
                    return None
                try:
                    return int(offsets.get(int(e), 0) + int(s))
                except Exception:
                    return None
            df_all['global_step'] = df_all.apply(_global_step_row, axis=1)
            if args.x_key == 'global_step':
                # If user selected global_step, ensure it exists
                if df_all['global_step'].isna().all():
                    print('Warning: global_step could not be computed; falling back to step', file=sys.stderr)
                    args.x_key = 'step'
        except Exception as e:
            print(f'Warning: failed to compute global_step: {e}', file=sys.stderr)

    # Ensure commonly-used aliases exist so callers can request either dotted or slashed names
    alias_pairs = [
        ('val.loss', 'val/loss'),
        ('train.loss', 'train/loss'),
        ('train.avg_loss', 'train/avg_loss'),
    ]
    for left, right in alias_pairs:
        if left not in df_all.columns and right in df_all.columns:
            df_all[left] = df_all[right]
        if right not in df_all.columns and left in df_all.columns:
            df_all[right] = df_all[left]

    # Sort by the chosen x-axis to avoid zig-zags when concatenating frames
    if args.x_key in df_all.columns:
        df_all = df_all.sort_values(args.x_key, kind='mergesort')

    if args.force_line:
        # Direct line plotting to ensure points are connected
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        title = '; '.join(args.metrics)
        for m in args.metrics:
            if m not in df_all.columns:
                print(f"Warning: metric '{m}' not found in data columns", file=sys.stderr)
                continue
            sub = df_all[[args.x_key, m]].dropna().copy()
            if sub.empty:
                continue
            # Optional downsampling by x-bin
            if args.downsample and args.downsample > 1:
                try:
                    # Bin by floor(x/bin)*bin
                    bin_size = int(args.downsample)
                    sub['__bin__'] = (sub[args.x_key] // bin_size) * bin_size
                    sub = sub.groupby('__bin__', as_index=False).agg({args.x_key: 'mean', m: 'mean'})
                except Exception as e:
                    print(f"Warning: downsample failed for {m}: {e}", file=sys.stderr)
            # Optional clipping
            if args.clip_pct is not None or args.clip_low_pct is not None:
                try:
                    lo = args.clip_low_pct
                    hi = args.clip_pct
                    if lo is None and hi is not None:
                        lo = 100.0 - float(hi)
                    if hi is None and lo is not None:
                        hi = 100.0 - float(lo)
                    lo_v = sub[m].quantile(lo/100.0) if lo is not None else None
                    hi_v = sub[m].quantile(hi/100.0) if hi is not None else None
                    if lo_v is not None:
                        sub[m] = sub[m].clip(lower=lo_v)
                    if hi_v is not None:
                        sub[m] = sub[m].clip(upper=hi_v)
                except Exception as e:
                    print(f"Warning: clipping failed for {m}: {e}", file=sys.stderr)
            # Optional smoothing
            if args.smooth and args.smooth > 0:
                try:
                    mode = args.smooth_mode
                    if mode == 'mean':
                        sub[m] = sub[m].rolling(window=args.smooth, min_periods=1).mean()
                    elif mode == 'centered':
                        sub[m] = sub[m].rolling(window=args.smooth, min_periods=1, center=True).mean()
                    elif mode == 'median':
                        sub[m] = sub[m].rolling(window=args.smooth, min_periods=1).median()
                    elif mode == 'ema':
                        # Convert window to alpha similar to EMA span behavior
                        # alpha ~ 2/(N+1)
                        alpha = 2.0 / (args.smooth + 1.0)
                        sub[m] = sub[m].ewm(alpha=alpha, adjust=False).mean()
                except Exception as e:
                    print(f"Warning: smoothing failed for {m}: {e}", file=sys.stderr)
            ax.plot(sub[args.x_key], sub[m], label=m, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel(args.x_key)
        ax.set_ylabel('value')
        ax.grid(True, alpha=0.3)
        ax.legend()
        if args.out:
            fig.savefig(args.out, bbox_inches='tight')
    else:
        fig, ax = plot_metrics(
            df_all,
            metrics=args.metrics,
            x=args.x_key,
            smooth=args.smooth,
            title='; '.join(args.metrics),
            save_to=args.out,
        )

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
