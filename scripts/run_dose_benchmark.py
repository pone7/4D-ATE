#!/usr/bin/env python3
"""Run the dose/noise pilot benchmark.

Executes 5 representative methods on the 60 dose samples (20 samples × 3 doses).
Total ≈ 300 jobs. Shardable across GPUs using GPU_SLICE/N_GPUS env vars.

Launch in parallel:
    for i in 0 1 2 3; do
        GPU_SLICE=$i N_GPUS=4 nohup \\
            python scripts/run_dose_benchmark.py \\
            > logs/dose.$i.log 2>&1 &
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
BENCH = ROOT / "data/aet_bench"
DOSE = BENCH / "dose"
RF = BENCH / "results_dose.jsonl"

sys.path.insert(0, str(ROOT / "scripts"))
from metrics.fsc import fsc_summary  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402
from evaluate_baselines import (  # noqa: E402
    detect_atoms_from_volume, voxel_to_physical, compute_atom_metrics,
)

GPU_SLICE = int(os.environ.get("GPU_SLICE", "0"))
N_GPUS = int(os.environ.get("N_GPUS", "1"))
METHODS = ["wbp", "tomopy_mlem", "astra_cgls", "isonet", "x2gaussian"]


def _to_py(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
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
        "vol_psnr": float(psnr), "vol_ssim": float(ssim),
        "rmsd": float(am.get("rmsd", float("nan"))),
        "recall": float(am.get("recall", 0.0)),
        "precision": float(am.get("precision", 0.0)),
        "f1": float(am.get("f1", 0.0)),
        "n_pred": int(am.get("n_pred", 0)),
        "n_gt": int(am.get("n_gt", n_gt)),
        "n_matched": int(am.get("n_matched", 0)),
        "atom_eval": "v2", **fsc,
    }


def load_done() -> set:
    done = set()
    if not RF.exists():
        return done
    for line in RF.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in r:
            continue
        if r.get("metrics", {}).get("atom_eval") != "v2":
            continue
        done.add((r["sample_id"], r["method"]))
    return done


def main():
    done = load_done()
    samples = sorted([p.name for p in DOSE.iterdir() if p.is_dir()])
    jobs = [(s, m) for m in METHODS for s in samples if (s, m) not in done]
    mine = jobs[GPU_SLICE::N_GPUS]
    print(f"[GPU{GPU_SLICE}] {len(mine)}/{len(jobs)} jobs", flush=True)

    from run_benchmark import METHOD_REGISTRY, METHOD_REGISTRY_LEGACY
    ALL = {**METHOD_REGISTRY, **METHOD_REGISTRY_LEGACY}

    for i, (sid, method) in enumerate(mine, 1):
        try:
            with open(DOSE / sid / "analytical/data.pkl", "rb") as f:
                data = pickle.load(f)
            atoms_gt = np.load(DOSE / sid / "atoms_gt.npy")
            meta = json.load(open(DOSE / sid / "metadata.json"))
            dose = meta.get("dose_electrons_per_ang2")
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
                "subset": "dose", "sample_id": sid, "backend": "analytical",
                "method": method, "task": "T1", "dose": dose,
                "metrics": me, "runtime_sec": rt,
            }
            with RF.open("a") as f:
                f.write(json.dumps(row, default=_to_py) + "\n")
            print(
                f"[GPU{GPU_SLICE}][{i}/{len(mine)}] {method:12s} {sid:22s} "
                f"dose={dose:g} PSNR={me.get('vol_psnr', 0):.1f} ({rt:.0f}s)",
                flush=True,
            )
        except Exception as e:
            err = {"error": str(e), "tb": traceback.format_exc(),
                   "subset": "dose", "sample_id": sid, "method": method}
            with RF.open("a") as f:
                f.write(json.dumps(err) + "\n")
            print(f"[GPU{GPU_SLICE}][{i}/{len(mine)}] {method:12s} {sid:22s} FAILED: {e}",
                  flush=True)
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"[GPU{GPU_SLICE}] DONE!", flush=True)


if __name__ == "__main__":
    main()
