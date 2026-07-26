#!/usr/bin/env python
"""Method-level extraction-sensitivity sweep for AET-Bench.

Reviewer requirement (NeurIPS 3869): show that the SSIM-vs-recall ranking
inversion on the analytical mono tier is *not* an artefact of a single set
of extraction / matching hyperparameters. Unlike
``extraction_sensitivity.py`` (which sweeps hyperparameters on the reference
ground-truth volume), this script runs the identical peak-detection and
Hungarian-matching code on the *reconstructed* volume produced by each
method, then reports the full method ranking at every point of the
hyperparameter grid.

Volume providers:
  * classical / iterative methods  -> recomputed via ``run_benchmark``
    ``METHOD_REGISTRY`` and cached under ``output/aet_bench_sweepvols/``.
  * Gaussian-splatting methods      -> the already-rendered ``vol_pred.npy``
    produced during the main benchmark run (loaded, never retrained).

The grids (TAU_GRID, NMS_RADIUS_GRID, THRESHOLD_GRID, MODES) are imported
from ``extraction_sensitivity`` so the two analyses stay in lock-step.
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
    MODES,
    NMS_RADIUS_GRID,
    TAU_GRID,
    THRESHOLD_GRID,
    _min_distance_vox,
    _normalise,
    _peak_local_max_nd,
    _tune_threshold_for_count,
    compute_atom_metrics,
    voxel_to_physical,
)

DATA_ROOT = ROOT / "data" / "aet_bench"
OUT_PATH = DATA_ROOT / "analysis" / "method_sensitivity.jsonl"
SUMMARY_PATH = DATA_ROOT / "analysis" / "method_sensitivity_summary.json"
CACHE_ROOT = ROOT / "output" / "aet_bench_sweepvols"

# Gaussian-splatting methods: load the rendered volume the main run stored.
GAUSSIAN_VOL_PATHS = {
    "r2gaussian": [
        "output/aet_bench_r2g/{sid}/model/point_cloud/iteration_8000/vol_pred.npy",
    ],
    "4dgs": [
        "output/aet_bench_4dgs/{sid}/model_4d/point_cloud/iteration_8000/vol_pred_T0.npy",
        "output/aet_bench_4dgs/{sid}/model_4d/point_cloud/iteration_5000/vol_pred_T0.npy",
    ],
    "x2gaussian": [
        "output/aet_bench/{sid}/model_static/point_cloud/iteration_8000/vol_pred.npy",
        "output/aet_bench/{sid}/model_static/point_cloud/iteration_5000/vol_pred.npy",
    ],
}

# Classical / regularized methods that are cheap enough to recompute on CPU.
CLASSICAL_METHODS = (
    "fista_tv",
    "wbp",
    "astra_cgls",
    "astra_sirt",
    "astra_sart",
    "tomopy_mlem",
    "dart",
)


def canonical_mono_samples(limit: int | None = None) -> list[str]:
    root = DATA_ROOT / "mono"
    dirs = [
        p.name
        for p in sorted(root.iterdir())
        if p.is_dir() and "-s" not in p.name and "-d" not in p.name
    ]
    return dirs[:limit] if limit else dirs


def load_reference(sample_id: str) -> tuple[np.ndarray, float] | None:
    """Return (atoms_gt[:, :3], cell_size_angstrom) for a mono sample."""
    sample_dir = DATA_ROOT / "mono" / sample_id
    atoms_path = sample_dir / "atoms_gt.npy"
    meta_path = sample_dir / "metadata.json"
    if not atoms_path.exists() or not meta_path.exists():
        return None
    atoms = np.load(atoms_path, allow_pickle=True)
    meta = json.loads(meta_path.read_text())
    cell = float(meta.get("cell_size_angstrom") or 0.0)
    if cell <= 0:
        return None
    return np.asarray(atoms, dtype=float)[:, :3], cell


def _gaussian_volume(method: str, sample_id: str) -> np.ndarray | None:
    for template in GAUSSIAN_VOL_PATHS[method]:
        path = ROOT / template.format(sid=sample_id)
        if path.exists():
            vol = np.load(path)
            if vol.ndim == 4:
                vol = vol[0]
            return np.asarray(vol, dtype=np.float32)
    return None


def _classical_volume(method: str, sample_id: str, backend: str = "analytical") -> np.ndarray | None:
    cache = CACHE_ROOT / method / f"{sample_id}.npy"
    if cache.exists():
        return np.load(cache)
    from run_benchmark import METHOD_REGISTRY, METHOD_REGISTRY_LEGACY, load_sample

    registry = {**METHOD_REGISTRY, **METHOD_REGISTRY_LEGACY}
    fn = registry[method]
    data, _, _ = load_sample("mono", sample_id, backend)
    try:
        vol = fn(data, sample_id=sample_id)
    except TypeError:
        vol = fn(data)
    vol = np.asarray(vol, dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, vol)
    return vol


def get_volume(method: str, sample_id: str) -> np.ndarray | None:
    if method in GAUSSIAN_VOL_PATHS:
        return _gaussian_volume(method, sample_id)
    return _classical_volume(method, sample_id)


def sweep_volume(
    volume: np.ndarray,
    atoms_gt: np.ndarray,
    *,
    cell_size_angstrom: float,
    sample_id: str,
    method: str,
) -> Iterable[dict[str, object]]:
    """Yield one metrics row per (mode, tau, nms_radius, threshold).

    Peak detection depends only on (mode, nms_radius, threshold); tau only
    affects matching. We therefore extract peaks once per detection setting
    and reuse them across the tau grid.
    """
    gt = np.asarray(atoms_gt, dtype=float)[:, :3]
    vol_norm = _normalise(volume)
    for mode in MODES:
        for nms_radius in NMS_RADIUS_GRID:
            min_distance = _min_distance_vox(vol_norm.shape, cell_size_angstrom, nms_radius)
            for threshold_rel in THRESHOLD_GRID:
                if mode == "oracle_count":
                    peaks_vox, _ = _tune_threshold_for_count(
                        vol_norm, min_distance, len(gt), float(threshold_rel)
                    )
                else:
                    peaks_vox = _peak_local_max_nd(vol_norm, min_distance, float(threshold_rel))
                pred = voxel_to_physical(peaks_vox, vol_norm.shape, cell_size_angstrom)
                for tau in TAU_GRID:
                    metrics = compute_atom_metrics(pred, gt, tau)
                    yield {
                        "sample_id": sample_id,
                        "method": method,
                        "mode": mode,
                        "tau_angstrom": tau,
                        "nms_radius_angstrom": nms_radius,
                        "threshold_rel": threshold_rel,
                        "metrics": metrics,
                    }


def _setting_key(row: dict) -> tuple:
    return (
        row["mode"],
        row["tau_angstrom"],
        row["nms_radius_angstrom"],
        row["threshold_rel"],
    )


def summarize_rankings(rows: list[dict]) -> dict:
    """Per-setting mean recall per method + ordering-persistence counts.

    Returns a dict with:
      settings: [{mode, tau, nms, threshold, mean_recall:{method:val}}]
      n_settings_ordering_holds: {"<fast>><slow>": count} for every method
          pair, where the pair label is derived from the global-mean order.
    """
    from collections import defaultdict

    per_setting: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    global_recall: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = _setting_key(row)
        recall = row["metrics"].get("recall")
        if recall is None or (isinstance(recall, float) and math.isnan(recall)):
            continue
        per_setting[key][row["method"]].append(float(recall))
        global_recall[row["method"]].append(float(recall))

    global_mean = {m: float(np.mean(v)) for m, v in global_recall.items() if v}
    order = sorted(global_mean, key=lambda m: -global_mean[m])

    settings = []
    for key, method_vals in sorted(per_setting.items()):
        mean_recall = {m: float(np.mean(v)) for m, v in method_vals.items()}
        mode, tau, nms, threshold = key
        settings.append(
            {
                "mode": mode,
                "tau_angstrom": tau,
                "nms_radius_angstrom": nms,
                "threshold_rel": threshold,
                "mean_recall": mean_recall,
            }
        )

    ordering_holds: dict[str, int] = {}
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            hi, lo = order[i], order[j]
            label = f"{hi}>{lo}"
            count = 0
            for setting in settings:
                mr = setting["mean_recall"]
                if hi in mr and lo in mr and mr[hi] > mr[lo]:
                    count += 1
            ordering_holds[label] = count

    return {
        "global_mean_recall": global_mean,
        "global_order_by_recall": order,
        "n_total_settings": len(settings),
        "settings": settings,
        "n_settings_ordering_holds": ordering_holds,
    }


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


def run(
    methods: list[str],
    limit: int,
    output: Path,
    summary_path: Path,
) -> int:
    samples = canonical_mono_samples(limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    skipped: dict[str, int] = {}
    n_rows = 0
    with output.open("w") as f:
        for sample_id in samples:
            ref = load_reference(sample_id)
            if ref is None:
                continue
            atoms_gt, cell = ref
            for method in methods:
                vol = get_volume(method, sample_id)
                if vol is None:
                    skipped[method] = skipped.get(method, 0) + 1
                    continue
                for row in sweep_volume(
                    vol, atoms_gt, cell_size_angstrom=cell,
                    sample_id=sample_id, method=method,
                ):
                    rows.append(row)
                    f.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")
                    n_rows += 1
            print(f"  swept {sample_id}: {n_rows} rows so far", flush=True)

    summary = summarize_rankings(rows)
    summary["skipped_missing_volume"] = skipped
    summary["methods_requested"] = methods
    summary["n_samples"] = len(samples)
    summary_path.write_text(json.dumps(summary, default=_json_default, indent=2))
    print(f"Wrote {n_rows} rows to {output}")
    print(f"Global recall order: {summary['global_order_by_recall']}")
    print(f"Summary -> {summary_path}")
    if skipped:
        print(f"Skipped (missing volume): {skipped}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--methods",
        default=",".join(list(CLASSICAL_METHODS) + list(GAUSSIAN_VOL_PATHS)),
        help="Comma-separated method keys.",
    )
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    return run(methods, args.limit, args.output, args.summary)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
