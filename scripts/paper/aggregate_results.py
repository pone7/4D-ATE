#!/usr/bin/env python3
"""Aggregate results.jsonl into the per-tier numbers reported in the paper.

Aggregation rules (Appendix E of the paper):

* Canonical tier sets: Mono and Bi use only the base samples (sample ids
  without an ``-sNN`` augmentation suffix: 50 and 30 samples); 4D, Multislice,
  Real, and Brownian-Pt use every released sample.
* One result per (subset, task, method, sample): the most recently logged row.
  Voxel metrics come from that row; atom metrics come from the most recent row
  whose atom evaluation is valid (rows that carry ``n_pred`` must also carry
  ``atom_eval == "v2"``, the same guard used by generate_paper_assets.py).
* Uncertainty: percentile bootstrap of the mean over samples, 1,000 resamples,
  95% interval, reported as a half-width.

Usage:
    python scripts/paper/aggregate_results.py \
        --results results.jsonl --out paper_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

TIERS = [
    ("mono", "T1"),
    ("bi", "T1"),
    ("4d", "T7"),
    ("multislice", "T1"),
    ("real", "T1"),
    ("brownian_pt", "T1"),
]
VOXEL_METRICS = {"vol_ssim", "vol_psnr"}
METRICS = [
    "vol_ssim",
    "recall",
    "precision",
    "rmsd",
    "atom_identity_persistence",
    "trajectory_rmsd",
]


def is_canonical(row: dict) -> bool:
    """Mono / Bi leaderboards use base samples only."""
    return row["subset"] not in ("mono", "bi") or "-s" not in row["sample_id"]


def atom_eval_valid(metrics: dict) -> bool:
    return "n_pred" not in metrics or metrics.get("atom_eval") == "v2"


def is_finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def bootstrap_half_width(vals: list[float], n_resample: int = 1000, seed: int = 0) -> float:
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(rng.choice(vals) for _ in range(n)) / n for _ in range(n_resample))
    lo = means[int(0.025 * n_resample) - 1]
    hi = means[int(0.975 * n_resample) - 1]
    return (hi - lo) / 2


def load_latest(path: Path) -> tuple[dict, dict]:
    """Return (latest row metrics, latest atom-valid row metrics) per key."""
    latest: dict = {}
    latest_atom: dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not is_canonical(r):
                continue
            key = (r["subset"], r.get("task", "T1"), r["method"], r["sample_id"])
            latest[key] = r["metrics"]
            if atom_eval_valid(r["metrics"]):
                latest_atom[key] = r["metrics"]
    return latest, latest_atom


def aggregate(latest: dict, latest_atom: dict) -> dict:
    out: dict = defaultdict(dict)
    for subset, task in TIERS:
        methods = sorted({k[2] for k in latest if k[0] == subset and k[1] == task})
        for m in methods:
            entry: dict = {"n": len({k[3] for k in latest if k[:3] == (subset, task, m)})}
            for metric in METRICS:
                src = latest if metric in VOXEL_METRICS else latest_atom
                raw = [v.get(metric) for k, v in src.items() if k[:3] == (subset, task, m)]
                vals = [x for x in raw if is_finite(x)]
                # [mean, CI half-width, n finite, n rows considered]
                entry[metric] = (
                    [sum(vals) / len(vals), bootstrap_half_width(vals), len(vals), len(raw)]
                    if vals
                    else None
                )
            out[subset][m] = entry

    per_sample: dict = {}
    for k, m in latest_atom.items():
        if k[0] == "multislice":
            per_sample.setdefault(k[3], {})[k[2]] = [
                latest[k].get("vol_ssim"),
                m.get("recall"),
                m.get("rmsd"),
            ]
    out["multislice_per_sample"] = per_sample
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, required=True, help="path to results.jsonl")
    p.add_argument("--out", type=Path, default=Path("paper_results.json"))
    args = p.parse_args()

    latest, latest_atom = load_latest(args.results)
    out = aggregate(latest, latest_atom)
    args.out.write_text(json.dumps(out, indent=1))

    for subset, _ in TIERS:
        print(f"## {subset}")
        rows = sorted(out[subset].items(), key=lambda kv: -(kv[1]["vol_ssim"] or [0])[0])
        for m, d in rows:
            fmt = lambda x: "  -  " if x is None else f"{x[0]:.3f}±{x[1]:.3f}"
            print(f"  {m:15s} n={d['n']:3d}  SSIM {fmt(d['vol_ssim'])}  recall {fmt(d['recall'])}  RMSD {fmt(d['rmsd'])}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
