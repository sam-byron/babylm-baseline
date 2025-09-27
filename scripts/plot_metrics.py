#!/usr/bin/env python3
import argparse
import json
import os
import sys
import glob
from datetime import datetime


def read_jsonl(path):
    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception:
                continue
    return records


def ema(series, alpha=0.9):
    if not series:
        return []
    out = []
    m = series[0]
    for x in series:
        m = alpha * m + (1 - alpha) * x
        out.append(m)
    return out


def find_latest_jsonl(directory):
    candidates = glob.glob(os.path.join(directory, '*.jsonl'))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def parse_args():
    ap = argparse.ArgumentParser(description='Plot metrics from JSONL log file produced by ThinLogger.')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--jsonl', type=str, help='Path to a JSONL metrics file')
    src.add_argument('--dir', type=str, help='Directory containing JSONL files (uses latest)')
    ap.add_argument('--outdir', type=str, default=None, help='Output directory for plots (default: alongside jsonl)')
    ap.add_argument('--ema', type=float, default=0.9, help='EMA smoothing factor in [0,1] (default 0.9)')
    ap.add_argument('--show', action='store_true', help='Show interactive plots if matplotlib backend allows')
    return ap.parse_args()


def main():
    args = parse_args()
    if args.dir:
        jsonl_path = find_latest_jsonl(args.dir)
        if not jsonl_path:
            print(f"No JSONL files found in {args.dir}", file=sys.stderr)
            sys.exit(1)
    else:
        jsonl_path = args.jsonl

    records = read_jsonl(jsonl_path)
    if not records:
        print(f"No records found in {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    # Extract metric points
    points = []
    for r in records:
        if r.get('_type') not in (None, 'metric'):
            # Skip non-metric housekeeping entries unless they carry scalars
            pass
        step = r.get('step')
        time_ts = r.get('time')
        # Allow prefixed keys like train/loss etc.
        for k, v in r.items():
            if k in ('_type', 'time', 'step'):
                continue
            if isinstance(v, (int, float)):
                points.append({'key': k, 'step': step, 'value': float(v), 'time': time_ts})

    if not points:
        print("No scalar metrics found.", file=sys.stderr)
        sys.exit(1)

    # Optional pandas/matplotlib
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        have_plot = True
    except Exception:
        have_plot = False

    base_outdir = args.outdir or os.path.dirname(jsonl_path)
    os.makedirs(base_outdir, exist_ok=True)

    # Always write a CSV dump for quick inspection
    try:
        import csv
        csv_path = os.path.join(base_outdir, 'metrics_dump.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['key', 'step', 'value', 'time'])
            w.writeheader()
            for p in points:
                w.writerow(p)
        print(f"Wrote CSV dump: {csv_path}")
    except Exception as e:
        print(f"Warning: failed to write CSV dump: {e}")

    if not have_plot:
        print("matplotlib/pandas not found. Install with: pip install matplotlib pandas", file=sys.stderr)
        # Print a small summary
        keys = sorted(set(p['key'] for p in points))
        print("Available metric keys:")
        for k in keys:
            vals = [p['value'] for p in points if p['key'] == k and p['value'] is not None]
            if vals:
                print(f"- {k}: n={len(vals)} min={min(vals):.4f} max={max(vals):.4f}")
        sys.exit(0)

    # Plot a few common charts
    df = pd.DataFrame(points)
    # Coerce steps
    df = df.dropna(subset=['step'])
    df['step'] = df['step'].astype(int)
    groups = {
        'loss': ['train/loss', 'train/avg_loss', 'val/loss'],
        'accuracy': ['train/mlm_acc', 'val/mlm_acc'],
        'throughput': ['train/tokens_per_sec', 'train/samples_per_sec'],
        'optimization': ['train/lr', 'train/grad_norm'],
        'masking': ['train/masked_count', 'train/mask_token_count', 'train/keep_count', 'train/random_count', 'train/protected_label_leak'],
    }

    for group_name, keys in groups.items():
        plt.figure(figsize=(10, 6))
        plotted_any = False
        for key in keys:
            sub = df[df['key'] == key].sort_values('step')
            if sub.empty:
                continue
            y = sub['value'].tolist()
            if args.ema and 0.0 < args.ema < 1.0:
                y_smooth = ema(y, alpha=args.ema)
            else:
                y_smooth = y
            plt.plot(sub['step'], y_smooth, label=key)
            plotted_any = True
        if not plotted_any:
            plt.close()
            continue
        plt.title(f"{group_name} — {os.path.basename(jsonl_path)}")
        plt.xlabel('step')
        plt.legend()
        plt.grid(True, alpha=0.3)
        out_path = os.path.join(base_outdir, f"{group_name}.png")
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"Saved: {out_path}")
        if args.show:
            try:
                plt.show()
            except Exception:
                pass
        plt.close()

    print("Done.")


if __name__ == '__main__':
    main()
