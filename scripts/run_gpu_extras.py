#!/usr/bin/env python3
"""Fill in missing GPU-dependent baselines (phase 2 補齐).

Runs after the main queue finishes. Covers:
  - GPU classical: astra_sart, astra_ossart, fista_tv, dart
  - GS-based:      r2gaussian, x2gaussian, fourdgs

Results go to results_gpu_extras.jsonl; merge later with merge_gpu_extras.py
or merge manually into results.jsonl once all phase-2 jobs finish.

Launch four copies with GPU_SLICE=0..3 and N_GPUS=4 to shard across workers:
    for i in 0 1 2 3; do
        GPU_SLICE=$i N_GPUS=4 nohup \\
            python scripts/run_gpu_extras.py >> logs/gpu_extras.$i.log 2>&1 &
    done
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path("/data/aet-bench")
B = ROOT / "data/aet_bench"
RF_MAIN = B / "results.jsonl"
RF_EXTRAS = B / "results_gpu_extras.jsonl"

sys.path.insert(0, str(ROOT / "scripts"))
from metrics.fsc import fsc_summary  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402
from evaluate_baselines import (  # noqa: E402
    detect_atoms_from_volume, voxel_to_physical, compute_atom_metrics,
)

GPU_SLICE = int(os.environ.get("GPU_SLICE", "0"))
N_GPUS = int(os.environ.get("N_GPUS", "1"))

METHODS_CLASSICAL_GPU = ["astra_sart", "astra_ossart", "fista_tv", "dart"]
# x_gaussian is in paper Table 2 but has zero rows in results.jsonl, so
# we explicitly include it in the GS补齐 queue.
METHODS_GS = ["r2gaussian", "x2gaussian", "x_gaussian", "fourdgs"]
SUBSETS = ["mono", "bi", "multislice", "real", "brownian_pt"]


def _csv_env(name: str) -> set[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def _selected_methods(methods: list[str]) -> list[str]:
    only = _csv_env("GPU_METHODS")
    skip = _csv_env("SKIP_METHODS")
    out = []
    for method in methods:
        if only and method not in only:
            continue
        if method in skip:
            continue
        out.append(method)
    return out


def _to_py(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def fast_eval(vol_pred, vol_gt, atoms_gt, voxel_size, cell_size):
    if vol_pred.max() > 0:
        vps = vol_pred * (vol_gt.max() / vol_pred.max())
    else:
        vps = vol_pred
    mse = np.mean((vol_gt - vps) ** 2)
    psnr = 10 * np.log10(vol_gt.max() ** 2 / (mse + 1e-10)) if mse > 0 else 100.0
    ssim_vals = []
    step = max(1, vol_gt.shape[0] // 32)
    dr = float(vol_gt.max() - vol_gt.min())
    for i in range(0, vol_gt.shape[0], step):
        ssim_vals.append(structural_similarity(vol_gt[i], vps[i], data_range=dr))
    ssim = float(np.mean(ssim_vals))
    try:
        fsc = fsc_summary(vol_pred, vol_gt, voxel_size)
    except Exception:
        fsc = {}
    n_gt = int(len(atoms_gt))
    try:
        gt_pos = atoms_gt[:, :3] if atoms_gt.ndim == 2 else atoms_gt
        peaks_voxel = detect_atoms_from_volume(vps, min_distance=3, num_expected=n_gt)
        pred_pos = voxel_to_physical(peaks_voxel, vps.shape, cell_size)
        am = compute_atom_metrics(pred_pos, gt_pos, threshold=0.5)
    except Exception:
        am = {"rmsd": float("nan"), "recall": 0.0, "precision": 0.0, "f1": 0.0,
              "n_pred": 0, "n_gt": n_gt, "n_matched": 0}
    return {
        "vol_psnr": float(psnr),
        "vol_ssim": float(ssim),
        "rmsd": float(am.get("rmsd", float("nan"))),
        "recall": float(am.get("recall", 0.0)),
        "precision": float(am.get("precision", 0.0)),
        "f1": float(am.get("f1", 0.0)),
        "n_pred": int(am.get("n_pred", 0)),
        "n_gt": int(am.get("n_gt", n_gt)),
        "n_matched": int(am.get("n_matched", 0)),
        "atom_eval": "v2",
        **fsc,
    }


def load_done() -> set:
    done = set()
    for path in [RF_MAIN, RF_EXTRAS]:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in r:
                    continue
                if path == RF_EXTRAS and r.get("metrics", {}).get("atom_eval") != "v2":
                    continue
                done.add((r["subset"], r["sample_id"], r["method"]))
    return done


def main():
    done = load_done()
    subsets = {}
    for sub in SUBSETS:
        d = B / sub
        if d.exists():
            subsets[sub] = sorted([p.name for p in d.iterdir() if p.is_dir()])

    # GS first (user-prioritized, slowest — don't want to get stuck on classical tail),
    # then classical cleanup.
    JOBS = []
    methods_gs = _selected_methods(METHODS_GS)
    methods_classical = _selected_methods(METHODS_CLASSICAL_GPU)
    print(
        f"[GPU{GPU_SLICE}] methods_gs={methods_gs} "
        f"methods_classical={methods_classical}",
        flush=True,
    )

    for sub, samples in subsets.items():
        for m in methods_gs:
            for s in samples:
                if (sub, s, m) not in done:
                    JOBS.append((sub, s, m))
    for sub, samples in subsets.items():
        for m in methods_classical:
            for s in samples:
                if (sub, s, m) not in done:
                    JOBS.append((sub, s, m))

    MY_JOBS = JOBS[GPU_SLICE::N_GPUS]
    print(f"[GPU{GPU_SLICE}] {len(MY_JOBS)}/{len(JOBS)} jobs to run", flush=True)

    from run_benchmark import METHOD_REGISTRY, METHOD_REGISTRY_LEGACY
    ALL = {**METHOD_REGISTRY, **METHOD_REGISTRY_LEGACY}

    for i, (sub, sid, method) in enumerate(MY_JOBS, 1):
        try:
            with open(B / sub / sid / "analytical" / "data.pkl", "rb") as f:
                data = pickle.load(f)
            atoms_gt = np.load(B / sub / sid / "atoms_gt.npy")
            meta = json.load(open(B / sub / sid / "metadata.json"))
            t0 = time.time()
            vol = ALL[method](data, sid)
            vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
            if (vol > 0).any():
                vol = np.clip(vol, 0, float(np.percentile(vol[vol > 0], 99.9)) * 2)
            else:
                vol = np.clip(vol, 0, 1)
            vg = data["image"][0] if data["image"].ndim == 4 else data["image"]
            cs = meta.get("cell_size_angstrom", 25.0)
            vs = cs / vg.shape[0]
            me = fast_eval(vol, vg, atoms_gt, vs, cs)
            me = {k: _to_py(x) for k, x in me.items()}
            rt = time.time() - t0
            row = {
                "subset": sub, "sample_id": sid, "backend": "analytical",
                "method": method, "task": "T1", "metrics": me, "runtime_sec": rt,
            }
            with open(RF_EXTRAS, "a") as f:
                f.write(json.dumps(row, default=_to_py) + "\n")
            print(
                f"[GPU{GPU_SLICE}][{i}/{len(MY_JOBS)}] {method:15s} "
                f"{sid:25s} PSNR={me.get('vol_psnr', 0):.1f} ({rt:.0f}s)",
                flush=True,
            )
        except Exception as e:
            err_row = {
                "error": str(e),
                "tb": traceback.format_exc(),
                "subset": sub, "sample_id": sid, "method": method,
            }
            with open(RF_EXTRAS, "a") as f:
                f.write(json.dumps(err_row) + "\n")
            print(
                f"[GPU{GPU_SLICE}][{i}/{len(MY_JOBS)}] {method:15s} "
                f"{sid:25s} FAILED: {e}",
                flush=True,
            )
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"[GPU{GPU_SLICE}] DONE!", flush=True)


if __name__ == "__main__":
    main()
