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
    ap.add_argument('--x-key', default='step', help='x-axis key (default: step)')
    ap.add_argument('--smooth', type=int, default=0, help='EMA/moving average window')
    ap.add_argument('--out', '-o', type=str, default=None, help='Output PNG path')
    ap.add_argument('--show', action='store_true', help='Show the figure interactively')
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

    fig, ax = plot_metrics(df_all, metrics=args.metrics, x=args.x_key, smooth=args.smooth, title='; '.join(args.metrics), save_to=args.out)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
