
from __future__ import annotations
import json, time, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

# ---------------------- JSONL utilities ----------------------

def read_jsonl(path: Path | str, fast: bool = True) -> pd.DataFrame:
    """Read a JSONL file into a normalized DataFrame with columns:
    - step (int)
    - metric (str)
    - value (float)
    - timestamp (float)
    Also preserves any extra flattened columns when present.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if fast:
        try:
            df = pd.read_json(path, lines=True)
            # If the fast path produced a 'metrics' column (nested dicts), expand it
            if isinstance(df, pd.DataFrame) and 'metrics' in df.columns:
                rows: List[Dict[str, Any]] = []
                # Expand each row via flatten_metrics for consistency
                for _, row in df.iterrows():
                    try:
                        rec = row.to_dict()
                        rows.extend(flatten_metrics(rec))
                    except Exception as e:
                        logger.warning("Skipping bad row during fast expand: %s", e)
                df = pd.DataFrame(rows)
            df = normalize_records(df)
            # If after normalization we still have no values (e.g., schema unexpected),
            # fall back to robust streaming parse
            if df.empty or df['value'].notna().sum() == 0:
                raise ValueError("fast path yielded empty/invalid frame; fallback")
            return df
        except Exception as e:
            logger.warning("fast read_json failed: %s; falling back to streaming", e)

    rows: List[Dict[str, Any]] = []
    with path.open('r') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                flat = flatten_metrics(rec)
                rows.extend(flat)
            except Exception as e:
                logger.warning("Skipping bad line %d: %s", i, e)
    if not rows:
        return pd.DataFrame(columns=['timestamp', 'step', 'metric', 'value'])
    df = pd.DataFrame(rows)
    df = normalize_records(df)
    return df


def flatten_metrics(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a single raw record into one or many metric rows.
    If the record already has metric/value, keep as-is.
    If the record has a 'metrics' dict, expand to multiple rows.
    """
    ts = record.get('timestamp') or record.get('time') or time.time()
    step = record.get('step') or record.get('global_step') or record.get('iteration')

    base = {k: v for k, v in record.items() if k not in ('metrics',)}
    base['timestamp'] = float(ts)
    if step is not None:
        try:
            base['step'] = int(step)
        except Exception:
            pass

    if 'metric' in record and 'value' in record:
        try:
            v = float(record['value'])
        except Exception:
            return []
        out = dict(base)
        out['metric'] = str(record['metric'])
        out['value'] = v
        return [out]

    # Expand explicit nested metrics dict when present and non-empty
    metrics = record.get('metrics', None)
    if isinstance(metrics, dict) and len(metrics) > 0:
        out: List[Dict[str, Any]] = []
        for k, v in metrics.items():
            try:
                vv = float(v)
            except Exception:
                continue
            row = dict(base)
            row['metric'] = str(k)
            row['value'] = vv
            out.append(row)
        return out

    # As a last resort, try to interpret all float-like leaf keys
    out = []
    for k, v in record.items():
        if k in ('timestamp', 'time', 'metrics', 'metric', 'value'):
            continue
        try:
            vv = float(v)
        except Exception:
            continue
        row = dict(base)
        row['metric'] = str(k)
        row['value'] = vv
        out.append(row)
    return out


def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns and dtypes, drop NaNs in value, sort by timestamp/step."""
    if df.empty:
        return pd.DataFrame(columns=['timestamp', 'step', 'metric', 'value'])

    # If the input already resembles flattened rows, keep only needed + passthrough
    cols = list(df.columns)
    required = ['timestamp', 'step', 'metric', 'value']
    for c in required:
        if c not in df.columns:
            df[c] = pd.NA

    # Coerce dtypes
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['step'] = pd.to_numeric(df['step'], errors='coerce').astype('Int64')
    df['metric'] = df['metric'].astype('string')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    df = df.dropna(subset=['metric', 'value'])
    df = df.sort_values(['timestamp', 'step'], na_position='last', kind='mergesort')
    return df.reset_index(drop=True)

# ---------------------- Aggregation utilities ----------------------

def filter_metrics(df: pd.DataFrame, include: Optional[Sequence[str]] = None, exclude: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Filter by metric names with a small convenience: treat '.' and '/' as interchangeable.
    This allows requesting 'train.loss' for a metric logged as 'train/loss'.
    """
    out = df
    if include:
        include = list(include)
        canon_include = {m.replace('.', '/') for m in include}
        mask_inc = out['metric'].isin(include) | out['metric'].str.replace('.', '/', regex=False).isin(canon_include)
        out = out[mask_inc]
    if exclude:
        exclude = list(exclude)
        canon_exclude = {m.replace('.', '/') for m in exclude}
        mask_exc = out['metric'].isin(exclude) | out['metric'].str.replace('.', '/', regex=False).isin(canon_exclude)
        out = out[~mask_exc]
    return out


def pivot_metrics(df: pd.DataFrame, index: str = 'step', columns: str = 'metric', values: str = 'value') -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    pv = df.pivot_table(index=index, columns=columns, values=values, aggfunc='mean')
    pv = pv.sort_index()
    return pv


def moving_average(series, window: int = 10):
    import numpy as np
    if window is None or window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=False).mean()


def resample_by_step(df: pd.DataFrame, every: int = 10, agg: str = 'mean') -> pd.DataFrame:
    if df.empty:
        return df
    if 'step' not in df:
        return df
    grp = (df['step'] // max(1, int(every))) * max(1, int(every))
    df2 = df.copy()
    df2['step_bucket'] = grp
    out = df2.groupby(['step_bucket', 'metric'], as_index=False)['value'].agg(agg)
    out = out.rename(columns={'step_bucket': 'step'})
    return out

# ---------------------- Plotting utilities ----------------------

def save_plot(fig, path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight', dpi=150)


def plot_metrics(df: pd.DataFrame, metrics: Sequence[str], x: str = 'step', smooth: Optional[int] = None, title: Optional[str] = None, save_to: Optional[str | Path] = None):
    if df.empty:
        raise ValueError('Empty DataFrame provided to plot_metrics')
    sub = filter_metrics(df, include=metrics)
    if sub.empty:
        raise ValueError('No matching metrics to plot')
    pv = pivot_metrics(sub, index=x, columns='metric', values='value')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = 0
    # Canonicalize requested names to match columns (support dot or slash)
    cols = set(pv.columns.astype(str))
    for m in metrics:
        candidates = [m, m.replace('.', '/'), m.replace('/', '.')]
        col = next((c for c in candidates if c in cols), None)
        if col is None:
            continue
        y = pv[col]
        if smooth and smooth > 1:
            y = moving_average(y, window=int(smooth))
        # Use the original requested name as the label for readability
        ax.plot(pv.index, y, label=m)
        plotted += 1

    ax.set_xlabel(x)
    ax.set_ylabel('value')
    if title:
        ax.set_title(title)
    if plotted == 0:
        raise ValueError('No series plotted; requested metrics not found in pivoted data')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_to:
        save_plot(fig, save_to)
    return fig, ax
