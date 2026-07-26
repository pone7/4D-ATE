#!/usr/bin/env python3
"""Bootstrap 95% confidence intervals for per-(subset, method, metric) means.

Used by generate_paper_assets.py so every table cell carries a CI alongside
the point estimate. Standalone CLI reports CIs for one (subset, method, metric)
triple for sanity checking.

Usage (CLI):
    python scripts/bench_bootstrap.py --subset mono --method tomopy_mlem \
           --metric vol_psnr [--n 1000] [--ci 0.95]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

RESULTS = Path("/data/aet-bench/data/aet_bench/results.jsonl")


def _clean(vals: Iterable) -> list[float]:
    out = []
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        out.append(float(v))
    return out


def bootstrap_ci(
    vals: Sequence, n_resample: int = 1000, ci: float = 0.95, seed: int = 0
) -> tuple[float | None, float | None, float | None]:
    """Percentile bootstrap CI for the mean.

    Returns (mean, lo, hi). If vals has zero valid entries, returns (None, None, None).
    For a single valid value, returns (mean, mean, mean) — no variance to resample.
    """
    clean = _clean(vals)
    if not clean:
        return None, None, None
    mean = float(np.mean(clean))
    if len(clean) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    arr = np.asarray(clean, dtype=float)
    idx = rng.integers(0, len(arr), size=(n_resample, len(arr)))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot_means, alpha))
    hi = float(np.quantile(boot_means, 1.0 - alpha))
    return mean, lo, hi


def _load_rows(path: Path = RESULTS) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--task", default="T1")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ci", type=float, default=0.95)
    args = ap.parse_args()

    rows = _load_rows()
    vals = [
        r["metrics"].get(args.metric)
        for r in rows
        if r["subset"] == args.subset
        and r["method"] == args.method
        and r.get("task", "T1") == args.task
    ]
    mean, lo, hi = bootstrap_ci(vals, n_resample=args.n, ci=args.ci)
    n = len(_clean(vals))
    if mean is None:
        print("no valid values")
        return
    print(
        f"subset={args.subset} method={args.method} metric={args.metric} "
        f"task={args.task} n={n}"
    )
    print(f"  mean = {mean:.6g}")
    print(f"  {int(args.ci*100)}% CI = [{lo:.6g}, {hi:.6g}]")
    print(f"  half-width = {(hi - lo) / 2.0:.6g}")


if __name__ == "__main__":
    main()
