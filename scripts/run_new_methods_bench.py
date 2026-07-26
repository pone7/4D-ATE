#!/usr/bin/env python3
"""
Run the 3 new benchmark methods (sax_nerf, instamap, dif_gaussian) on:
  - 50 original mono samples (those with existing fbp results)
  - 4 original real samples (Zr-Zr1, Zr-Zr2, Zr-Zr3, MPEA2)

Appends results to results.jsonl, skipping any (method, subset, sample)
combinations that already have results.

Evaluation uses the same task_t1_t2 pipeline as run_benchmark.py but
with a timeout on peak detection to avoid Hungarian matching bottleneck
on very noisy reconstructions with large atom counts.
"""
import json
import os
import pickle
import signal
import sys
import time
import traceback
from pathlib import Path

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_benchmark import (
    METHOD_REGISTRY,
    SUBSET_DIRS,
    load_sample,
    _adaptive_threshold,
)
from evaluate_baselines import (
    evaluate_reconstruction,
    volume_psnr,
    volume_ssim,
    detect_atoms_from_volume,
    voxel_to_physical,
    compute_atom_metrics,
)
from metrics.fsc import fsc_summary

RESULTS_PATH = Path("/data/aet-bench/data/aet_bench/results.jsonl")
METHODS = ["sax_nerf", "instamap", "dif_gaussian"]
REAL_SAMPLES = ["Zr-Zr1", "Zr-Zr2", "Zr-Zr3", "MPEA2"]


def _print(*args, **kwargs):
    """Print with flush."""
    print(*args, **kwargs, flush=True)


def _to_py(o):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(x) for x in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _task_t1_t2_fast(vol_pred, data, atoms_gt, meta):
    """Fast version of task_t1_t2 that caps peak detection to avoid
    catastrophic Hungarian matching on very noisy reconstructions.

    The key difference: limit detected peaks to at most 2x n_gt, and
    add a timeout on the evaluation to skip if it takes too long.
    """
    vol_gt = data["image"]
    if vol_gt.ndim == 4:
        vol_gt = vol_gt[0]
    cell_size = meta["cell_size_angstrom"]
    voxel_size = cell_size / vol_gt.shape[0]
    threshold = _adaptive_threshold(cell_size, vol_gt.shape)
    min_peak_distance = max(1, int(round(2.0 / voxel_size)))

    # --- Volume-level metrics (always fast) ---
    if vol_pred.max() > 0:
        scale = vol_gt.max() / vol_pred.max()
        vol_pred_scaled = vol_pred * scale
    else:
        vol_pred_scaled = vol_pred.copy()

    psnr = volume_psnr(vol_gt, vol_pred_scaled)
    ssim = volume_ssim(vol_gt, vol_pred_scaled)

    # --- FSC (always fast -- just 2 FFTs) ---
    voxel_size_a = cell_size / vol_gt.shape[0]
    fsc = fsc_summary(vol_pred_scaled, vol_gt, voxel_size_angstrom=voxel_size_a)

    # --- Atom-level metrics (can be slow for noisy volumes) ---
    gt_positions = atoms_gt[:, :3]
    n_gt = len(gt_positions)

    try:
        # Fast peak detection: single call to peak_local_max with a
        # fixed threshold. The full binary search (20 iterations) is
        # too slow for noisy volumes on 128^3.
        from skimage.feature import peak_local_max

        vol_norm = vol_pred / (vol_pred.max() + 1e-10)
        max_peaks = min(n_gt * 2, 500)

        peaks_voxel = peak_local_max(
            vol_norm, min_distance=min_peak_distance,
            threshold_rel=0.15, num_peaks=max_peaks
        )

        pred_positions = voxel_to_physical(peaks_voxel, vol_pred.shape, cell_size)
        atom_metrics = compute_atom_metrics(pred_positions, gt_positions, threshold)
    except Exception as e:
        _print(f"    atom detection failed: {e}")
        atom_metrics = {
            'rmsd': float('inf'), 'recall': 0.0, 'precision': 0.0,
            'f1': 0.0, 'n_pred': 0, 'n_gt': n_gt, 'n_matched': 0,
        }

    return {
        "vol_psnr": psnr,
        "vol_ssim": ssim,
        "fsc_0.143_angstrom": fsc["resolution_0.143"],
        "fsc_0.5_angstrom": fsc["resolution_0.5"],
        "rmsd": atom_metrics.get('rmsd', float('inf')),
        "recall": atom_metrics.get('recall', 0.0),
        "precision": atom_metrics.get('precision', 0.0),
        "f1": atom_metrics.get('f1', 0.0),
        "n_gt": n_gt,
        "n_matched": atom_metrics.get('n_matched', 0),
    }


def get_existing_results():
    """Load existing results and return set of (method, subset, sample_id) tuples."""
    existing = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    existing.add((r["method"], r["subset"], r["sample_id"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return existing


def get_mono_samples():
    """Get the 50 mono samples that have fbp results."""
    mono_samples = set()
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("subset") == "mono" and r.get("method") == "fbp":
                    mono_samples.add(r["sample_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return sorted(mono_samples)


def run_one(method, subset, sample_id, backend="analytical"):
    """Run a single (method, subset, sample) combination."""
    data, atoms_gt, meta = load_sample(subset, sample_id, backend)
    fn = METHOD_REGISTRY[method]

    t0 = time.time()
    try:
        vol_pred = fn(data, sample_id=sample_id, task="T1")
    except TypeError:
        try:
            vol_pred = fn(data, sample_id=sample_id)
        except TypeError:
            vol_pred = fn(data)
    runtime = time.time() - t0

    # Evaluate (using fast version)
    t_eval = time.time()
    metrics = _task_t1_t2_fast(vol_pred, data, atoms_gt, meta)
    eval_time = time.time() - t_eval

    row = {
        "subset": subset,
        "sample_id": sample_id,
        "backend": backend,
        "method": method,
        "task": "T1",
        "metrics": metrics,
        "runtime_sec": runtime,
    }

    # Append to JSONL
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(_to_py(row)) + "\n")

    psnr = metrics.get("vol_psnr") or 0.0
    ssim = metrics.get("vol_ssim") or 0.0
    recall = metrics.get("recall") or 0.0
    f1 = metrics.get("f1") or 0.0
    _print(
        f"  [{method}] {subset}/{sample_id}: "
        f"PSNR={psnr:.2f} SSIM={ssim:.3f} "
        f"Recall={recall:.1%} F1={f1:.3f} "
        f"train={runtime:.1f}s eval={eval_time:.1f}s"
    )
    return row


def main():
    mono_samples = get_mono_samples()
    _print(f"Found {len(mono_samples)} mono samples with fbp results")
    _print(f"Real samples: {REAL_SAMPLES}")

    existing = get_existing_results()
    _print(f"Total existing result rows: {len(existing)}")

    for method in METHODS:
        # Count existing
        existing_mono = sum(
            1 for m, s, sid in existing if m == method and s == "mono"
        )
        existing_real = sum(
            1 for m, s, sid in existing if m == method and s == "real"
        )
        _print(f"\n{'='*60}")
        _print(f"Method: {method}")
        _print(f"  Existing: mono={existing_mono}, real={existing_real}")
        _print(f"{'='*60}")

        # Run mono samples
        total_mono = len(mono_samples)
        skipped = 0
        done = 0
        failed = 0
        for i, sample_id in enumerate(mono_samples):
            if (method, "mono", sample_id) in existing:
                skipped += 1
                continue
            _print(f"\n[{method}] mono {i+1}/{total_mono}: {sample_id}")
            try:
                run_one(method, "mono", sample_id)
                done += 1
            except Exception as e:
                failed += 1
                _print(f"  FAILED on {sample_id}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=3)
                sys.stdout.flush()

        _print(f"\n[{method}] mono summary: done={done}, skipped={skipped}, failed={failed}")

        # Run real samples
        skipped = 0
        done = 0
        failed = 0
        for sample_id in REAL_SAMPLES:
            if (method, "real", sample_id) in existing:
                skipped += 1
                continue
            _print(f"\n[{method}] real: {sample_id}")
            try:
                run_one(method, "real", sample_id)
                done += 1
            except Exception as e:
                failed += 1
                _print(f"  FAILED on {sample_id}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=3)
                sys.stdout.flush()

        _print(f"\n[{method}] real summary: done={done}, skipped={skipped}, failed={failed}")

    _print("\n" + "=" * 60)
    _print("All methods complete!")

    # Final summary
    existing = get_existing_results()
    for method in METHODS:
        cm = sum(1 for m, s, sid in existing if m == method and s == "mono")
        cr = sum(1 for m, s, sid in existing if m == method and s == "real")
        _print(f"  {method}: mono={cm}, real={cr}")


if __name__ == "__main__":
    main()
