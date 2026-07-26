#!/usr/bin/env python
"""Sweep atom-extraction hyperparameters for AET-Bench volumes.

The analysis is intentionally separated from ``evaluate_baselines.py`` so it
can run in CPU-only review environments without scikit-image.  It uses the
same metric definitions as the main evaluator: voxel peaks are converted into
physical coordinates, then Hungarian/KD-tree matching reports recall,
precision, F1, RMSD, and counts.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "aet_bench"
OUT_PATH = DATA_ROOT / "analysis" / "extraction_sensitivity.jsonl"

TAU_GRID = (0.3, 0.5, 0.7, 1.0)
NMS_RADIUS_GRID = (1.5, 2.0, 2.5)
THRESHOLD_GRID = (0.03, 0.05, 0.10)
MODES = ("oracle_count", "blind_fixed")

DEFAULT_LIMITS = {
    "mono": 50,
    "bi": 30,
    "4d": 10,
    "multislice": 3,
}
SMOKE_LIMITS = {
    "mono": 3,
    "bi": 2,
    "4d": 1,
    "multislice": 1,
}


@dataclass(frozen=True)
class VolumeCase:
    subset: str
    sample_id: str
    phase: int | None
    volume: np.ndarray
    atoms_gt: np.ndarray
    cell_size_angstrom: float
    volume_source: str


def _as_float_volume(volume: np.ndarray) -> np.ndarray:
    vol = np.asarray(volume, dtype=np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    if vol.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {vol.shape}")
    return vol


def _normalise(volume: np.ndarray) -> np.ndarray:
    vol = _as_float_volume(volume)
    vol = vol - float(np.min(vol))
    max_val = float(np.max(vol))
    if max_val <= 1e-12:
        return np.zeros_like(vol, dtype=np.float32)
    return (vol / max_val).astype(np.float32, copy=False)


def _voxel_size_angstrom(volume_shape: tuple[int, int, int], cell_size_angstrom: float) -> float:
    return float(cell_size_angstrom) / float(max(volume_shape))


def _min_distance_vox(
    volume_shape: tuple[int, int, int],
    cell_size_angstrom: float,
    nms_radius_angstrom: float,
) -> int:
    voxel_size = _voxel_size_angstrom(volume_shape, cell_size_angstrom)
    return max(1, int(round(float(nms_radius_angstrom) / max(voxel_size, 1e-12))))


def _peak_local_max_nd(volume_norm: np.ndarray, min_distance: int, threshold_rel: float) -> np.ndarray:
    if float(np.max(volume_norm)) <= 1e-12:
        return np.empty((0, 3), dtype=int)

    size = max(1, 2 * int(min_distance) + 1)
    local_max = maximum_filter(volume_norm, size=size, mode="constant") == volume_norm
    candidates = np.argwhere(local_max & (volume_norm > float(threshold_rel)))
    if len(candidates) == 0:
        return np.empty((0, 3), dtype=int)

    values = volume_norm[tuple(candidates.T)]
    order = np.argsort(values)[::-1]
    max_candidates = 10000
    if len(order) > max_candidates:
        order = order[:max_candidates]

    selected: list[np.ndarray] = []
    min_dist2 = float(min_distance * min_distance)
    for idx in order:
        cand = candidates[idx]
        if all(float(np.sum((cand - prev) ** 2)) >= min_dist2 for prev in selected):
            selected.append(cand)
    if not selected:
        return np.empty((0, 3), dtype=int)
    return np.asarray(selected, dtype=int)


def _tune_threshold_for_count(
    volume_norm: np.ndarray,
    min_distance: int,
    num_expected: int,
    initial_threshold: float,
) -> tuple[np.ndarray, float]:
    if num_expected <= 0:
        return _peak_local_max_nd(volume_norm, min_distance, initial_threshold), initial_threshold

    lo, hi = 0.005, 0.8
    best_peaks = _peak_local_max_nd(volume_norm, min_distance, initial_threshold)
    best_threshold = float(initial_threshold)
    best_gap = abs(len(best_peaks) - num_expected)
    for _ in range(14):
        mid = (lo + hi) / 2.0
        peaks = _peak_local_max_nd(volume_norm, min_distance, mid)
        gap = abs(len(peaks) - num_expected)
        if gap < best_gap:
            best_peaks = peaks
            best_threshold = mid
            best_gap = gap
        if len(peaks) > num_expected * 1.5:
            lo = mid
        elif len(peaks) < max(1, num_expected * 0.5):
            hi = mid
        else:
            best_peaks = peaks
            best_threshold = mid
            break
    return best_peaks, best_threshold


def voxel_to_physical(
    peaks_voxel: np.ndarray,
    volume_shape: tuple[int, int, int],
    cell_size_angstrom: float,
) -> np.ndarray:
    """Convert voxel coordinates to centered Angstrom coordinates."""
    if len(peaks_voxel) == 0:
        return np.empty((0, 3), dtype=float)
    half = float(cell_size_angstrom) / 2.0
    return peaks_voxel.astype(float) / np.asarray(volume_shape, dtype=float) * float(cell_size_angstrom) - half


def refine_subvoxel_com(
    volume_norm: np.ndarray,
    peaks_voxel: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    """Refine integer peak positions to sub-voxel accuracy.

    For each integer peak, compute the intensity-weighted centre of mass over
    a ``(2*radius+1)^3`` neighbourhood. Returns float voxel coordinates. This
    addresses reviewer WQbd's question about whether the shared extractor's
    lack of sub-voxel refinement is what caps the atom scores.
    """
    if len(peaks_voxel) == 0:
        return np.empty((0, 3), dtype=float)
    shape = np.asarray(volume_norm.shape)
    refined = np.empty((len(peaks_voxel), 3), dtype=float)
    for i, peak in enumerate(np.asarray(peaks_voxel, dtype=int)):
        lo = np.maximum(peak - radius, 0)
        hi = np.minimum(peak + radius + 1, shape)
        block = volume_norm[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        total = float(block.sum())
        if total <= 1e-12:
            refined[i] = peak.astype(float)
            continue
        coords = [np.arange(lo[d], hi[d], dtype=float) for d in range(3)]
        gx, gy, gz = np.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
        refined[i, 0] = float((gx * block).sum() / total)
        refined[i, 1] = float((gy * block).sum() / total)
        refined[i, 2] = float((gz * block).sum() / total)
    return refined


def extract_atoms_from_volume(
    volume: np.ndarray,
    *,
    cell_size_angstrom: float,
    nms_radius_angstrom: float,
    threshold_rel: float,
    mode: str,
    num_expected: int | None,
) -> np.ndarray:
    """Extract atom candidates in physical coordinates.

    ``blind_fixed`` ignores ``num_expected`` completely. ``oracle_count`` uses
    it only to tune the peak threshold toward the expected atom count.
    """
    if mode not in MODES:
        raise ValueError(f"unknown extraction mode {mode!r}; expected one of {MODES}")

    vol_norm = _normalise(volume)
    min_distance = _min_distance_vox(vol_norm.shape, cell_size_angstrom, nms_radius_angstrom)
    if mode == "oracle_count":
        if num_expected is None:
            raise ValueError("oracle_count mode requires num_expected")
        peaks, _ = _tune_threshold_for_count(
            vol_norm, min_distance, int(num_expected), float(threshold_rel)
        )
    else:
        peaks = _peak_local_max_nd(vol_norm, min_distance, float(threshold_rel))
    return voxel_to_physical(peaks, vol_norm.shape, cell_size_angstrom)


def _greedy_match(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    max_dist: float | None,
) -> tuple[list[int], list[int], np.ndarray]:
    tree_pred = cKDTree(pred_positions)
    k = min(10, len(pred_positions))
    dists_all, idxs_all = tree_pred.query(gt_positions, k=k)
    if dists_all.ndim == 1:
        dists_all = dists_all[:, None]
        idxs_all = idxs_all[:, None]

    matched_pred: list[int] = []
    matched_gt: list[int] = []
    distances: list[float] = []
    used_pred: set[int] = set()
    for gi in np.argsort(dists_all[:, 0]):
        for ki in range(dists_all.shape[1]):
            pi = int(idxs_all[gi, ki])
            d = float(dists_all[gi, ki])
            if pi not in used_pred and (max_dist is None or d <= max_dist):
                matched_pred.append(pi)
                matched_gt.append(int(gi))
                distances.append(d)
                used_pred.add(pi)
                break
    return matched_pred, matched_gt, np.asarray(distances, dtype=float)


def _match_atoms(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    max_dist: float | None,
) -> tuple[list[int], list[int], np.ndarray]:
    if len(pred_positions) == 0 or len(gt_positions) == 0:
        return [], [], np.asarray([], dtype=float)
    if len(pred_positions) > 2000 or len(gt_positions) > 2000:
        return _greedy_match(pred_positions, gt_positions, max_dist)

    cost_matrix = cdist(pred_positions, gt_positions)
    pred_idx, gt_idx = linear_sum_assignment(cost_matrix)
    distances = cost_matrix[pred_idx, gt_idx]
    if max_dist is not None:
        valid = distances <= max_dist
        pred_idx = pred_idx[valid]
        gt_idx = gt_idx[valid]
        distances = distances[valid]
    return list(pred_idx), list(gt_idx), np.asarray(distances, dtype=float)


def compute_atom_metrics(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    tau_angstrom: float,
) -> dict[str, float | int]:
    if len(pred_positions) == 0:
        return {
            "rmsd": float("inf"),
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "n_pred": 0,
            "n_gt": int(len(gt_positions)),
            "n_matched": 0,
        }

    _, _, distances = _match_atoms(pred_positions, gt_positions, max_dist=float(tau_angstrom) * 3.0)
    within = int(np.sum(distances <= float(tau_angstrom))) if len(distances) else 0
    recall = within / len(gt_positions) if len(gt_positions) else 0.0
    precision = within / len(pred_positions) if len(pred_positions) else 0.0
    f1 = 2.0 * recall * precision / (recall + precision + 1e-10)
    return {
        "rmsd": float(np.sqrt(np.mean(distances ** 2))) if len(distances) else float("inf"),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "n_pred": int(len(pred_positions)),
        "n_gt": int(len(gt_positions)),
        "n_matched": within,
    }


def evaluate_extraction_setting(
    volume: np.ndarray,
    atoms_gt: np.ndarray,
    *,
    cell_size_angstrom: float,
    tau_angstrom: float,
    nms_radius_angstrom: float,
    threshold_rel: float,
    mode: str,
) -> dict[str, float | int]:
    gt_positions = np.asarray(atoms_gt, dtype=float)[:, :3]
    pred_positions = extract_atoms_from_volume(
        volume,
        cell_size_angstrom=cell_size_angstrom,
        nms_radius_angstrom=nms_radius_angstrom,
        threshold_rel=threshold_rel,
        mode=mode,
        num_expected=len(gt_positions),
    )
    return compute_atom_metrics(pred_positions, gt_positions, tau_angstrom)


def _sample_dirs(subset: str, limit: int) -> list[Path]:
    root = DATA_ROOT / ("sim4d" if subset == "4d" else subset)
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if subset == "mono":
        canonical = [p for p in dirs if "-s" not in p.name and "-d" not in p.name]
        dirs = canonical or dirs
    return dirs[:limit]


def _load_case_from_reference(sample_dir: Path, subset: str) -> Iterable[VolumeCase]:
    backend = "multislice" if subset == "multislice" else "analytical"
    pkl_path = sample_dir / backend / "data.pkl"
    if not pkl_path.exists():
        return []
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    meta_path = sample_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    cell_size = float(meta.get("cell_size_angstrom") or max(data.get("nVoxel", [128])) * 1.0)
    image = np.asarray(data["image"])
    atoms_path = sample_dir / "atoms_gt.npy"
    if not atoms_path.exists():
        return []
    atoms = np.load(atoms_path, allow_pickle=True)

    if image.ndim == 4:
        # Static tiers in this repository sometimes store repeated time slots;
        # sim4d stores true phases. Sweep every phase only for the 4D tier.
        phase_indices = range(image.shape[0]) if subset == "4d" else (0,)
        cases = []
        for phase in phase_indices:
            phase_atoms = atoms
            phase_path = sample_dir / f"atoms_gt_phase{phase}.npy"
            if phase_path.exists():
                phase_atoms = np.load(phase_path, allow_pickle=True)
            cases.append(
                VolumeCase(
                    subset=subset,
                    sample_id=sample_dir.name,
                    phase=phase if subset == "4d" else None,
                    volume=_as_float_volume(image[phase]),
                    atoms_gt=np.asarray(phase_atoms),
                    cell_size_angstrom=cell_size,
                    volume_source="reference_volume",
                )
            )
        return cases

    return [
        VolumeCase(
            subset=subset,
            sample_id=sample_dir.name,
            phase=None,
            volume=_as_float_volume(image),
            atoms_gt=np.asarray(atoms),
            cell_size_angstrom=cell_size,
            volume_source="reference_volume",
        )
    ]


def iter_reference_cases(limits: dict[str, int]) -> Iterable[VolumeCase]:
    for subset, limit in limits.items():
        for sample_dir in _sample_dirs(subset, limit):
            yield from _load_case_from_reference(sample_dir, subset)


def sweep_case(case: VolumeCase) -> Iterable[dict[str, object]]:
    for mode in MODES:
        for tau in TAU_GRID:
            for nms_radius in NMS_RADIUS_GRID:
                for threshold_rel in THRESHOLD_GRID:
                    metrics = evaluate_extraction_setting(
                        case.volume,
                        case.atoms_gt,
                        cell_size_angstrom=case.cell_size_angstrom,
                        tau_angstrom=tau,
                        nms_radius_angstrom=nms_radius,
                        threshold_rel=threshold_rel,
                        mode=mode,
                    )
                    yield {
                        "subset": case.subset,
                        "sample_id": case.sample_id,
                        "phase": case.phase,
                        "volume_source": case.volume_source,
                        "mode": mode,
                        "tau_angstrom": tau,
                        "nms_radius_angstrom": nms_radius,
                        "threshold_rel": threshold_rel,
                        "metrics": metrics,
                    }


def parse_limits(text: str | None, *, smoke: bool) -> dict[str, int]:
    limits = dict(SMOKE_LIMITS if smoke else DEFAULT_LIMITS)
    if not text:
        return limits
    for item in text.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        if key not in limits:
            raise ValueError(f"unknown subset in limits: {key}")
        limits[key] = int(value)
    return limits


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if math.isinf(v):
            return "inf"
        if math.isnan(v):
            return "nan"
        return v
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a tiny representative subset for fast local paper regeneration.",
    )
    parser.add_argument(
        "--limits",
        default=None,
        help="Comma-separated subset limits, e.g. mono=50,bi=30,4d=10,multislice=3.",
    )
    args = parser.parse_args(argv)

    limits = parse_limits(args.limits, smoke=args.smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_cases = 0
    with args.output.open("w") as f:
        for case in iter_reference_cases(limits):
            n_cases += 1
            for row in sweep_case(case):
                f.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")
                n_rows += 1
    print(f"Wrote {n_rows} sensitivity rows from {n_cases} volume cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
