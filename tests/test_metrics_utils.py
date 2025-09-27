import json
from pathlib import Path
import pandas as pd
from src.metrics_utils import read_jsonl, flatten_metrics, normalize_records, moving_average, filter_metrics, pivot_metrics

def make_tmp_jsonl(tmp_path: Path):
    p = tmp_path / 't.jsonl'
    rows = [
        {"timestamp": 1.0, "step": 1, "metrics": {"loss": 1.2, "accuracy": 0.1}},
        {"timestamp": 2.0, "step": 2, "metric": "loss", "value": 1.1},
        {"timestamp": 3.0, "step": 3, "lr": 0.001, "grad_norm": 3.0},
    ]
    with p.open('w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return p


def test_read_jsonl_valid(tmp_path: Path):
    p = make_tmp_jsonl(tmp_path)
    df = read_jsonl(p)
    assert not df.empty
    assert set(['metric', 'value']).issubset(df.columns)


def test_flatten_and_normalize(tmp_path: Path):
    rec = {"timestamp": 1.0, "step": 1, "metrics": {"a": 1, "b": 2}}
    flat = flatten_metrics(rec)
    assert len(flat) == 2
    df = pd.DataFrame(flat)
    df2 = normalize_records(df)
    assert set(['timestamp', 'step', 'metric', 'value']).issubset(df2.columns)


def test_moving_average_edges():
    import pandas as pd
    s = pd.Series([1,2,3,4])
    out = moving_average(s, window=2)
    assert len(out) == 4
    assert abs(out.iloc[-1] - 3.5) < 1e-6


def test_filter_and_pivot(tmp_path: Path):
    p = make_tmp_jsonl(tmp_path)
    df = read_jsonl(p)
    sub = filter_metrics(df, include=['loss'])
    pv = pivot_metrics(sub)
    assert 'loss' in pv.columns
