"""Smoke tests for scripts/run_benchmark.py and the CI utility.

These tests stay fast (<30 s total) by picking the smallest subset sample
and the cheapest method (WBP), and by pointing the results writer at a
temp file instead of the canonical jsonl.
"""
from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
BENCH = ROOT / "data/aet_bench"
PY = sys.executable


def _smallest_mono_sample() -> str:
    samples = sorted((BENCH / "mono").iterdir())
    if not samples:
        pytest.skip("mono tier not present")
    return samples[0].name


def test_run_benchmark_wbp_mono_t1(tmp_path):
    """WBP/T1 on one mono sample appends one well-formed jsonl row."""
    if importlib.util.find_spec("skimage") is None:
        pytest.skip("scikit-image is required for WBP smoke reconstruction")
    out = tmp_path / "smoke.jsonl"
    sid = _smallest_mono_sample()
    r = subprocess.run(
        [PY, str(SCRIPTS / "run_benchmark.py"),
         "--subset", "mono", "--sample", sid,
         "--method", "wbp", "--task", "T1",
         "--results", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stderr
    assert out.exists()
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert lines, f"empty results file; stdout={r.stdout[-500:]}"
    row = json.loads(lines[-1])
    for k in ("subset", "sample_id", "method", "task", "metrics"):
        assert k in row
    assert row["subset"] == "mono"
    assert row["method"] == "wbp"
    assert 0.0 <= row["metrics"]["vol_ssim"] <= 1.0


def test_bootstrap_ci_basic():
    """bootstrap_ci on a constant returns (value, value, value) with no variance."""
    sys.path.insert(0, str(SCRIPTS))
    from bench_bootstrap import bootstrap_ci
    m, lo, hi = bootstrap_ci([7.0, 7.0, 7.0, 7.0])
    assert m == pytest.approx(7.0)
    assert lo == pytest.approx(7.0)
    assert hi == pytest.approx(7.0)


def test_bootstrap_ci_filters_nan():
    sys.path.insert(0, str(SCRIPTS))
    from bench_bootstrap import bootstrap_ci
    vals = [1.0, 2.0, float("nan"), 3.0, None, float("inf")]
    m, lo, hi = bootstrap_ci(vals, n_resample=200, seed=0)
    assert m == pytest.approx(2.0)
    assert lo < hi
    assert lo <= 2.0 <= hi


def test_bootstrap_ci_empty():
    sys.path.insert(0, str(SCRIPTS))
    from bench_bootstrap import bootstrap_ci
    assert bootstrap_ci([]) == (None, None, None)
    assert bootstrap_ci([float("nan"), None]) == (None, None, None)


def test_results_jsonl_schema():
    """Every row in the canonical results.jsonl parses and has core fields."""
    rf = BENCH / "results.jsonl"
    if not rf.exists():
        pytest.skip("no results.jsonl yet")
    n = 0
    with rf.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            for k in ("subset", "sample_id", "method", "task"):
                assert k in r, f"missing {k} in row {n}: {r}"
            assert isinstance(r.get("metrics", {}), dict)
            n += 1
    assert n > 0


def test_generate_paper_assets_importable():
    """generate_paper_assets imports cleanly (it's wired to bench_bootstrap)."""
    sys.path.insert(0, str(SCRIPTS))
    import importlib
    mod = importlib.import_module("generate_paper_assets")
    assert callable(mod.generate_latex_table)
    assert callable(mod.write_all_tables)
