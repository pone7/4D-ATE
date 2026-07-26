#!/usr/bin/env python
"""
Batch runner: 7 neural methods x 19 real AET samples.

Methods (Python-based, run in aet4dgs env):
  neat, implicit_et, isonet, sax_nerf, instamap, dif_gaussian

Method requiring separate env (aet_bench_r2g):
  x_gaussian  -- attempted last, failures noted

Usage:
    source /opt/conda/bin/activate aet4dgs
    cd /data/aet-bench/scripts
    python run_real_neural_batch.py
"""
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from evaluate_baselines import evaluate_reconstruction
from metrics.fsc import fsc_summary

RESULTS_PATH = Path("/data/aet-bench/data/aet_bench/results.jsonl")
REAL_DIR = Path("/data/aet-bench/data/aet_bench/real")

SAMPLES = [
    "AuPd-DH", "AuPd-ICNP1", "AuPd-ICNP2", "AuPd-ICNP3", "AuPd-ICNP4",
    "AuPd-ICNP5", "AuPd-ICNP6", "AuPd-ICNP7", "AuPd-ICNP8",
    "AuPd-PC1", "AuPd-PC2", "AuPd-PC3", "AuPd-SF",
    "NiPt-MultiGrain",
    "PdPt-Yang-P1", "PdPt-Yang-P2",
    "PdPt-Zhou-EPB", "PdPt-Zhou-PB", "PdPt-Zhou-TO",
]

# Python-based methods first (run in aet4dgs env)
METHODS_PYTHON = ["neat", "implicit_et", "isonet", "sax_nerf", "instamap", "dif_gaussian"]
# x_gaussian needs aet_bench_r2g env -- attempt last
METHODS_R2G = ["x_gaussian"]


def load_sample(sample_id):
    sample_dir = REAL_DIR / sample_id
    pkl_path = sample_dir / "analytical" / "data.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    atoms_gt = np.load(sample_dir / "atoms_gt.npy")
    with open(sample_dir / "metadata.json") as f:
        meta = json.load(f)
    return data, atoms_gt, meta


def get_method_fn(method_name):
    """Import and return the method function."""
    from run_benchmark import METHOD_REGISTRY
    return METHOD_REGISTRY[method_name]


def evaluate_result(vol_pred, data, atoms_gt, meta):
    """Evaluate a reconstruction volume. Returns metrics dict."""
    vol_gt = data["image"]
    if vol_gt.ndim == 4:
        vol_gt = vol_gt[0]
    cell_size = meta["cell_size_angstrom"]
    voxel_size = cell_size / vol_gt.shape[0]
    threshold = max(0.5, 1.5 * voxel_size)
    min_peak_distance = max(1, int(round(2.0 / voxel_size)))

    res = evaluate_reconstruction(
        vol_pred, vol_gt, atoms_gt, cell_size,
        method_name="benchmark",
        threshold=threshold,
        min_peak_distance=min_peak_distance,
    )

    # FSC
    voxel_size_a = cell_size / vol_gt.shape[0]
    if vol_pred.max() > 0:
        scale = vol_gt.max() / vol_pred.max()
        vol_pred_s = vol_pred * scale
    else:
        vol_pred_s = vol_pred
    fsc = fsc_summary(vol_pred_s, vol_gt, voxel_size_angstrom=voxel_size_a)

    return {
        "vol_psnr": res["vol_psnr"],
        "vol_ssim": res["vol_ssim"],
        "fsc_0.143_angstrom": fsc["resolution_0.143"],
        "fsc_0.5_angstrom": fsc["resolution_0.5"],
        "rmsd": res["rmsd"],
        "recall": res["recall"],
        "precision": res["precision"],
        "f1": res["f1"],
        "n_gt": res["n_gt"],
        "n_matched": res["n_matched"],
    }


def to_python(o):
    """Convert numpy types for JSON serialization."""
    if isinstance(o, dict):
        return {k: to_python(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_python(x) for x in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def append_result(row):
    """Append a result row to the JSONL file."""
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(to_python(row)) + "\n")


def run_method_sample(method_name, sample_id, data, atoms_gt, meta):
    """Run a single method on a single sample. Returns (row, error)."""
    fn = get_method_fn(method_name)
    t0 = time.time()
    try:
        vol_pred = fn(data, sample_id=sample_id, task="T1")
    except TypeError:
        try:
            vol_pred = fn(data, sample_id=sample_id)
        except TypeError:
            vol_pred = fn(data)
    runtime = time.time() - t0

    metrics = evaluate_result(vol_pred, data, atoms_gt, meta)

    row = {
        "subset": "real",
        "sample_id": sample_id,
        "backend": "analytical",
        "method": method_name,
        "task": "T1",
        "metrics": metrics,
        "runtime_sec": runtime,
    }
    return row


def check_already_done(method_name, sample_id):
    """Check if this method+sample already exists in results."""
    if not RESULTS_PATH.exists():
        return False
    with open(RESULTS_PATH) as f:
        for line in f:
            try:
                row = json.loads(line)
                if (row.get("subset") == "real" and
                    row.get("sample_id") == sample_id and
                    row.get("method") == method_name):
                    return True
            except json.JSONDecodeError:
                continue
    return False


def main():
    # Preload the already-done set to skip duplicates
    done_set = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("subset") == "real":
                        done_set.add((row["sample_id"], row["method"]))
                except json.JSONDecodeError:
                    continue

    all_methods = METHODS_PYTHON + METHODS_R2G
    total = len(all_methods) * len(SAMPLES)
    done_count = 0
    skip_count = 0
    fail_count = 0
    failures = []

    print(f"=== Batch run: {len(all_methods)} methods x {len(SAMPLES)} samples = {total} jobs ===")
    print(f"Results file: {RESULTS_PATH}")
    print()

    for method_name in all_methods:
        print(f"\n{'='*60}")
        print(f"METHOD: {method_name}")
        print(f"{'='*60}")

        for sample_id in SAMPLES:
            if (sample_id, method_name) in done_set:
                print(f"  [SKIP] {sample_id} -- already in results")
                skip_count += 1
                continue

            print(f"  [{method_name}] {sample_id} ... ", end="", flush=True)
            try:
                data, atoms_gt, meta = load_sample(sample_id)
                row = run_method_sample(method_name, sample_id, data, atoms_gt, meta)
                append_result(row)

                m = row["metrics"]
                print(f"PSNR={m['vol_psnr']:.2f}  Recall={m['recall']:.1%}  "
                      f"F1={m['f1']:.3f}  runtime={row['runtime_sec']:.1f}s")
                done_count += 1

            except Exception as e:
                print(f"FAILED: {type(e).__name__}: {e}")
                traceback.print_exc(limit=5)
                fail_count += 1
                failures.append((method_name, sample_id, str(e)))

            # Free GPU memory between runs
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {done_count} succeeded, {skip_count} skipped, {fail_count} failed")
    if failures:
        print(f"\nFailed runs:")
        for method, sample, err in failures:
            print(f"  {method}/{sample}: {err}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
