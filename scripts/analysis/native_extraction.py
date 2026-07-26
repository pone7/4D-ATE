#!/usr/bin/env python
"""Representation-aware atom extraction for AET-Bench.

Reviewer requirement (NeurIPS 3869): the shared volume-to-atom extractor
(peak detection + NMS on a rasterized volume) may unfairly penalize methods
whose native output is a set of Gaussian primitives or a continuous field.
This module adds two things the reviewers asked for:

  1. ``extract_atoms_from_gaussians`` -- a *native* extractor that clusters
     the method's own Gaussian primitive centres (density-weighted) instead
     of rasterizing then peak-picking. Gaussian-splatting methods
     (R2-Gaussian, 4D-GS, X2-Gaussian) store ~10^5 primitives that
     collectively form the density; a single atom is represented by many
     primitives, so a density-weighted greedy merge recovers atom centres.

  2. A ``shared_subvoxel`` comparison point: the same shared peak extractor
     but with centre-of-mass sub-voxel refinement (WQbd's question).

For each mono sample we compare three extractors on the same reconstruction:
  * ``shared``          -- integer-voxel peak detection (paper default)
  * ``shared_subvoxel`` -- shared + CoM sub-voxel refinement
  * ``native``          -- Gaussian primitive clustering (Gaussian methods)

Output: ``data/aet_bench/analysis/native_extraction.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analysis.extraction_sensitivity import (  # noqa: E402
    _min_distance_vox,
    _normalise,
    _peak_local_max_nd,
    _tune_threshold_for_count,
    compute_atom_metrics,
    refine_subvoxel_com,
    voxel_to_physical,
)
from analysis.method_sensitivity import (  # noqa: E402
    canonical_mono_samples,
    load_reference,
)

DATA_ROOT = ROOT / "data" / "aet_bench"
OUT_PATH = DATA_ROOT / "analysis" / "native_extraction.jsonl"
SUMMARY_PATH = DATA_ROOT / "analysis" / "native_extraction_summary.json"

TAU_GRID = (0.3, 0.5, 0.7, 1.0)
# Pinned to the corrected operating point (1.0 A); see run_benchmark.task_t1_t2
# and tests/test_extractor_ceiling.py. The previous 2.0 A window merged
# neighbouring atoms and depressed *every* method's shared-extractor recall.
DEFAULT_NMS_RADIUS = 1.0
DEFAULT_THRESHOLD = 0.05

# Gaussian methods: (rendered volume, primitive checkpoint) path templates.
GAUSSIAN_PATHS = {
    "r2gaussian": {
        "vol": ["output/aet_bench_r2g/{sid}/model/point_cloud/iteration_8000/vol_pred.npy"],
        "pc": ["output/aet_bench_r2g/{sid}/model/point_cloud/iteration_8000/point_cloud.pickle"],
    },
    "4dgs": {
        "vol": [
            "output/aet_bench_4dgs/{sid}/model_4d/point_cloud/iteration_8000/vol_pred_T0.npy",
            "output/aet_bench_4dgs/{sid}/model_4d/point_cloud/iteration_5000/vol_pred_T0.npy",
        ],
        "pc": [
            "output/aet_bench_4dgs/{sid}/model_4d/point_cloud/iteration_8000/point_cloud.pickle",
            "output/aet_bench_4dgs/{sid}/model_4d/point_cloud/iteration_5000/point_cloud.pickle",
        ],
    },
    "x2gaussian": {
        "vol": [
            "output/aet_bench/{sid}/model_static/point_cloud/iteration_8000/vol_pred.npy",
            "output/aet_bench/{sid}/model_static/point_cloud/iteration_5000/vol_pred.npy",
        ],
        "pc": [
            "output/aet_bench/{sid}/model_static/point_cloud/iteration_8000/point_cloud.pickle",
            "output/aet_bench/{sid}/model_static/point_cloud/iteration_5000/point_cloud.pickle",
        ],
    },
}


def _first_existing(templates: list[str], sid: str) -> Path | None:
    for template in templates:
        path = ROOT / template.format(sid=sid)
        if path.exists():
            return path
    return None


def extract_atoms_from_gaussians(
    xyz_norm: np.ndarray,
    density: np.ndarray,
    *,
    cell_size_angstrom: float,
    mode: str,
    num_expected: int | None,
    merge_radius_angstrom: float = 1.5,
    density_percentile: float = 60.0,
) -> np.ndarray:
    """Cluster Gaussian primitive centres into atom positions (physical A).

    Gaussian-splatting checkpoints store primitive centres in a normalized
    ``[-1, 1]`` cube; physical position is ``xyz_norm * cell_size / 2``.
    Primitives are density-weighted; we greedily seed clusters at the
    highest-density remaining primitive and merge all primitives within
    ``merge_radius_angstrom``, taking the density-weighted centroid.

    mode == 'oracle_count': keep the ``num_expected`` highest-weight clusters.
    mode == 'blind_fixed':  keep clusters whose summed density exceeds a
                            percentile of the primitive density distribution.
    """
    xyz = np.asarray(xyz_norm, dtype=float)
    dens = np.asarray(density, dtype=float).ravel()
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        return np.empty((0, 3), dtype=float)

    phys = xyz * (float(cell_size_angstrom) / 2.0)

    # Drop clearly-background primitives before clustering.
    if mode == "blind_fixed":
        cutoff = np.percentile(dens, density_percentile)
        keep = dens >= cutoff
    else:
        # oracle_count: keep a generous pool (background is pruned by weight ranking).
        cutoff = np.percentile(dens, max(0.0, density_percentile - 30.0))
        keep = dens >= cutoff
    phys = phys[keep]
    dens = dens[keep]
    if len(phys) == 0:
        return np.empty((0, 3), dtype=float)
    # Shift densities positive so they act as merge weights.
    weights = dens - dens.min() + 1e-6

    from scipy.spatial import cKDTree

    order = np.argsort(weights)[::-1]
    tree = cKDTree(phys)
    assigned = np.zeros(len(phys), dtype=bool)
    r = float(merge_radius_angstrom)
    clusters: list[tuple[np.ndarray, float]] = []
    for idx in order:
        if assigned[idx]:
            continue
        neigh = tree.query_ball_point(phys[idx], r)
        neigh = [n for n in neigh if not assigned[n]]
        if not neigh:
            neigh = [idx]
        w = weights[neigh]
        centroid = np.average(phys[neigh], axis=0, weights=w)
        clusters.append((centroid, float(w.sum())))
        assigned[neigh] = True

    if not clusters:
        return np.empty((0, 3), dtype=float)
    centroids = np.asarray([c for c, _ in clusters])
    cweights = np.asarray([w for _, w in clusters])

    if mode == "oracle_count" and num_expected is not None and num_expected > 0:
        keep_idx = np.argsort(cweights)[::-1][:num_expected]
        centroids = centroids[keep_idx]
    return centroids


def _shared_peaks(volume: np.ndarray, cell: float, num_expected: int, mode: str) -> np.ndarray:
    vol_norm = _normalise(volume)
    min_distance = _min_distance_vox(vol_norm.shape, cell, DEFAULT_NMS_RADIUS)
    if mode == "oracle_count":
        peaks_vox, _ = _tune_threshold_for_count(vol_norm, min_distance, num_expected, DEFAULT_THRESHOLD)
    else:
        peaks_vox = _peak_local_max_nd(vol_norm, min_distance, DEFAULT_THRESHOLD)
    return vol_norm, peaks_vox


def evaluate_sample_method(
    method: str,
    sample_id: str,
    atoms_gt: np.ndarray,
    cell: float,
) -> Iterable[dict[str, object]]:
    gt = np.asarray(atoms_gt, dtype=float)[:, :3]
    n_expected = len(gt)
    vol_path = _first_existing(GAUSSIAN_PATHS[method]["vol"], sample_id)
    pc_path = _first_existing(GAUSSIAN_PATHS[method]["pc"], sample_id)
    if vol_path is None:
        return

    volume = np.load(vol_path)
    if volume.ndim == 4:
        volume = volume[0]

    for mode in ("oracle_count", "blind_fixed"):
        vol_norm, peaks_vox = _shared_peaks(volume, cell, n_expected, mode)

        shared_pred = voxel_to_physical(peaks_vox, vol_norm.shape, cell)
        sub_vox = refine_subvoxel_com(vol_norm, peaks_vox, radius=1)
        sub_pred = voxel_to_physical(sub_vox, vol_norm.shape, cell)

        native_pred = None
        if pc_path is not None:
            with pc_path.open("rb") as f:
                pc = pickle.load(f)
            if "xyz" in pc and "density" in pc:
                native_pred = extract_atoms_from_gaussians(
                    np.asarray(pc["xyz"]),
                    np.asarray(pc["density"]),
                    cell_size_angstrom=cell,
                    mode=mode,
                    num_expected=n_expected,
                )

        extractors = {"shared": shared_pred, "shared_subvoxel": sub_pred}
        if native_pred is not None:
            extractors["native"] = native_pred

        for extractor, pred in extractors.items():
            for tau in TAU_GRID:
                metrics = compute_atom_metrics(pred, gt, tau)
                yield {
                    "sample_id": sample_id,
                    "method": method,
                    "extractor": extractor,
                    "mode": mode,
                    "tau_angstrom": tau,
                    "metrics": metrics,
                }


def summarize(rows: list[dict]) -> dict:
    from collections import defaultdict

    # mean recall per (method, extractor, mode, tau)
    agg: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        recall = row["metrics"].get("recall")
        if recall is None or (isinstance(recall, float) and math.isnan(recall)):
            continue
        key = (row["method"], row["extractor"], row["mode"], row["tau_angstrom"])
        agg[key].append(float(recall))
    summary = []
    for (method, extractor, mode, tau), vals in sorted(agg.items()):
        summary.append(
            {
                "method": method,
                "extractor": extractor,
                "mode": mode,
                "tau_angstrom": tau,
                "mean_recall": float(np.mean(vals)),
                "n": len(vals),
            }
        )
    return {"cells": summary}


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        if math.isinf(v):
            return "inf"
        if math.isnan(v):
            return "nan"
        return v
    return value


def run(methods: list[str], limit: int, output: Path, summary_path: Path) -> int:
    samples = canonical_mono_samples(limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    n_rows = 0
    with output.open("w") as f:
        for sample_id in samples:
            ref = load_reference(sample_id)
            if ref is None:
                continue
            atoms_gt, cell = ref
            for method in methods:
                for row in evaluate_sample_method(method, sample_id, atoms_gt, cell):
                    rows.append(row)
                    f.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")
                    n_rows += 1
            print(f"  {sample_id}: {n_rows} rows", flush=True)
    summary = summarize(rows)
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2))
    print(f"Wrote {n_rows} rows to {output}")
    print(f"Summary -> {summary_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--methods", default=",".join(GAUSSIAN_PATHS))
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip() and m.strip() in GAUSSIAN_PATHS]
    return run(methods, args.limit, args.output, args.summary)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
