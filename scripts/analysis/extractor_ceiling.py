#!/usr/bin/env python
"""Extractor ceiling curve: atom recall vs NMS min-distance.

Central diagnostic for the NeurIPS 3869 revision. Running the shared peak
extractor on the *ground-truth* volume bounds the recall any method can
achieve through the shared harness. Sweeping the NMS min-distance shows the
harness is the bottleneck when the exclusion window is too coarse (it merges
neighbouring atoms) and is not the bottleneck at the pinned operating point
(min-distance ~1.0 A), where GT recovery approaches 1.0.

We overlay the same curve for representative reconstruction methods so the
figure directly answers: (i) is the low atom recall a harness artefact?
(ii) at a fair operating point, do method gaps persist?

Output: data/aet_bench/analysis/extractor_ceiling.json + figure via
summarize_sensitivity.py.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analysis.extraction_sensitivity import (  # noqa: E402
    _min_distance_vox,
    _normalise,
    _tune_threshold_for_count,
    compute_atom_metrics,
    voxel_to_physical,
)
from analysis.method_sensitivity import canonical_mono_samples, load_reference  # noqa: E402

DATA_ROOT = ROOT / "data" / "aet_bench"
OUT_PATH = DATA_ROOT / "analysis" / "extractor_ceiling.json"

MIN_DIST_GRID = (0.8, 1.0, 1.3, 1.6, 2.0, 2.5)
TAU = 0.5

METHOD_VOL = {
    "4dgs": "output/aet_bench_4dgs/{sid}/model_4d/point_cloud/iteration_8000/vol_pred_T0.npy",
    "r2gaussian": "output/aet_bench_r2g/{sid}/model/point_cloud/iteration_8000/vol_pred.npy",
    "fista_tv": "output/aet_bench_sweepvols/fista_tv/{sid}.npy",
    "tomopy_mlem": "output/aet_bench_sweepvols/tomopy_mlem/{sid}.npy",
    "astra_cgls": "output/aet_bench_sweepvols/astra_cgls/{sid}.npy",
}


def _recall_at(vol, atoms, cell, min_dist_ang):
    vn = _normalise(np.asarray(vol, dtype=np.float32))
    md = _min_distance_vox(vn.shape, cell, min_dist_ang)
    peaks, _ = _tune_threshold_for_count(vn, md, len(atoms), 0.05)
    pred = voxel_to_physical(peaks, vn.shape, cell)
    return compute_atom_metrics(pred, atoms[:, :3], TAU)["recall"]


def _gt_volume(sid):
    for backend in ("analytical",):
        p = DATA_ROOT / "mono" / sid / backend / "data.pkl"
        if p.exists():
            with p.open("rb") as f:
                d = pickle.load(f)
            img = d["image"]
            return img[0] if img.ndim == 4 else img
    return None


def run(limit: int, output: Path) -> int:
    samples = canonical_mono_samples(limit)
    series = {name: {md: [] for md in MIN_DIST_GRID} for name in ["GT", *METHOD_VOL]}
    n_done = 0
    for sid in samples:
        ref = load_reference(sid)
        if ref is None:
            continue
        atoms, cell = ref
        gt = _gt_volume(sid)
        if gt is None:
            continue
        for md in MIN_DIST_GRID:
            series["GT"][md].append(_recall_at(gt, atoms, cell, md))
        for name, tmpl in METHOD_VOL.items():
            p = ROOT / tmpl.format(sid=sid)
            if not p.exists():
                continue
            v = np.load(p)
            if v.ndim == 4:
                v = v[0]
            for md in MIN_DIST_GRID:
                series[name][md].append(_recall_at(v, atoms, cell, md))
        n_done += 1
        print(f"  ceiling {n_done}: {sid}", flush=True)

    summary = {
        "min_dist_grid": list(MIN_DIST_GRID),
        "tau": TAU,
        "n_samples": n_done,
        "mean_recall": {
            name: {str(md): (float(np.mean(v)) if v else None) for md, v in md_map.items()}
            for name, md_map in series.items()
        },
        "n_per_series": {name: len(md_map[MIN_DIST_GRID[0]]) for name, md_map in series.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(f"Wrote ceiling curve to {output}")
    print("GT recall by min-dist:", summary["mean_recall"]["GT"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--output", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)
    return run(args.limit, args.output)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
