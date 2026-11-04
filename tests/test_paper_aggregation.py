"""Aggregation rules behind the paper's leaderboards (scripts/paper/aggregate_results.py)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "aggregate_results", ROOT / "scripts" / "paper" / "aggregate_results.py"
)
agg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agg)


def _row(sample, method, recall, ssim=0.5, subset="mono", **extra):
    metrics = {"vol_ssim": ssim, "recall": recall, "precision": recall, "rmsd": 0.3, **extra}
    return {"subset": subset, "task": "T1", "method": method, "sample_id": sample, "metrics": metrics}


def _aggregate(tmp_path, rows):
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return agg.aggregate(*agg.load_latest(path))


def test_reruns_count_once_per_sample(tmp_path):
    rows = [_row("Cu-0165-00", "x2", 0.99) for _ in range(10)] + [_row("Cu-0279-01", "x2", 0.01)]
    out = _aggregate(tmp_path, rows)
    assert out["mono"]["x2"]["n"] == 2
    assert abs(out["mono"]["x2"]["recall"][0] - 0.50) < 1e-9


def test_latest_run_wins(tmp_path):
    rows = [_row("Cu-0165-00", "fista", 0.0, ssim=float("nan")), _row("Cu-0165-00", "fista", 1.0, ssim=0.55)]
    out = _aggregate(tmp_path, rows)
    assert out["mono"]["fista"]["recall"][0] == 1.0
    assert out["mono"]["fista"]["vol_ssim"][0] == 0.55


def test_augmented_samples_excluded_from_mono_and_bi(tmp_path):
    rows = [_row("Cu-0165-00", "sirt", 0.9), _row("Cu-0165-s03", "sirt", 0.0)]
    out = _aggregate(tmp_path, rows)
    assert out["mono"]["sirt"]["n"] == 1
    assert out["mono"]["sirt"]["recall"][0] == 0.9


def test_atom_metrics_require_v2_eval(tmp_path):
    rows = [
        _row("Cu-0165-00", "insta", 0.002, n_pred=0),  # broken pre-v2 atom eval
        _row("Cu-0279-01", "sirt", 0.8, n_pred=10, atom_eval="v2"),
    ]
    out = _aggregate(tmp_path, rows)
    assert out["mono"]["insta"]["recall"] is None
    assert out["mono"]["insta"]["vol_ssim"] is not None
    assert out["mono"]["sirt"]["recall"][0] == 0.8
