#!/usr/bin/env python3
"""Run the submission-critical GPU subset.

This is intentionally smaller than run_gpu_extras.py. It fills the rows that
directly affect the current paper draft:
  - 4D-GS on the 50 mono canonical samples used by Table 1 / size scaling.
  - 4D-GS on all multislice pilot samples.
  - X2-Gaussian and 4D-GS on representative real samples.

Rows are appended to results_gpu_critical.jsonl so they can be audited before
merging into the main result log.
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path("/data/aet-bench")
B = ROOT / "data/aet_bench"
RF_MAIN = B / "results.jsonl"
RF_EXTRAS = B / "results_gpu_extras.jsonl"
RF_CRITICAL = B / "results_gpu_critical.jsonl"

sys.path.insert(0, str(ROOT / "scripts"))
from run_gpu_extras import fast_eval, _to_py  # noqa: E402

GPU_SLICE = int(os.environ.get("GPU_SLICE", "0"))
N_GPUS = int(os.environ.get("N_GPUS", "1"))


def mono_canonical() -> list[str]:
    samples = []
    for p in (B / "mono").iterdir():
        if p.is_dir() and re.match(r"^[A-Za-z]+-[0-9]+-[0-9]+$", p.name):
            samples.append(p.name)
    return sorted(samples)


def load_done() -> set[tuple[str, str, str]]:
    done = set()
    for path in [RF_MAIN, RF_EXTRAS, RF_CRITICAL]:
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
                if r.get("task", "T1") != "T1":
                    continue
                done.add((r["subset"], r["sample_id"], r["method"]))
    return done


def build_jobs() -> list[tuple[str, str, str]]:
    jobs: list[tuple[str, str, str]] = []
    for sid in mono_canonical():
        jobs.append(("mono", sid, "fourdgs"))
    for sid in sorted(p.name for p in (B / "multislice").iterdir() if p.is_dir()):
        jobs.append(("multislice", sid, "fourdgs"))
    for sid in ["Zr-Zr1", "PdPt-Zhou-PB", "AuPd-ICNP1", "NiPt-MultiGrain", "MPEA2"]:
        jobs.append(("real", sid, "x2gaussian"))
        jobs.append(("real", sid, "fourdgs"))
    return jobs


def main() -> None:
    from run_benchmark import METHOD_REGISTRY, METHOD_REGISTRY_LEGACY

    all_methods = {**METHOD_REGISTRY, **METHOD_REGISTRY_LEGACY}
    done = load_done()
    jobs = [j for j in build_jobs() if j not in done]
    my_jobs = jobs[GPU_SLICE::N_GPUS]
    print(f"[critical GPU{GPU_SLICE}] {len(my_jobs)}/{len(jobs)} jobs to run", flush=True)

    for i, (sub, sid, method) in enumerate(my_jobs, 1):
        try:
            with open(B / sub / sid / "analytical" / "data.pkl", "rb") as f:
                data = pickle.load(f)
            atoms_gt = np.load(B / sub / sid / "atoms_gt.npy")
            meta = json.load(open(B / sub / sid / "metadata.json"))

            t0 = time.time()
            fn = all_methods[method]
            try:
                vol = fn(data, sample_id=sid, task="T1")
            except TypeError:
                vol = fn(data, sample_id=sid)
            vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
            if (vol > 0).any():
                vol = np.clip(vol, 0, float(np.percentile(vol[vol > 0], 99.9)) * 2)
            else:
                vol = np.clip(vol, 0, 1)

            vol_gt = data["image"][0] if data["image"].ndim == 4 else data["image"]
            cell_size = meta.get("cell_size_angstrom", 25.0)
            voxel_size = cell_size / vol_gt.shape[0]
            metrics = fast_eval(vol, vol_gt, atoms_gt, voxel_size, cell_size)
            metrics = {k: _to_py(v) for k, v in metrics.items()}
            runtime = time.time() - t0
            row = {
                "subset": sub,
                "sample_id": sid,
                "backend": "analytical",
                "method": method,
                "task": "T1",
                "metrics": metrics,
                "runtime_sec": runtime,
                "critical_subset": True,
            }
            with open(RF_CRITICAL, "a") as f:
                f.write(json.dumps(row, default=_to_py) + "\n")
            print(
                f"[critical GPU{GPU_SLICE}][{i}/{len(my_jobs)}] {method:10s} "
                f"{sub:10s} {sid:20s} PSNR={metrics.get('vol_psnr', 0):.1f} "
                f"recall={metrics.get('recall', 0):.3f} ({runtime:.0f}s)",
                flush=True,
            )
        except Exception as exc:
            with open(RF_CRITICAL, "a") as f:
                f.write(json.dumps({
                    "error": str(exc),
                    "tb": traceback.format_exc(),
                    "subset": sub,
                    "sample_id": sid,
                    "method": method,
                    "critical_subset": True,
                }) + "\n")
            print(
                f"[critical GPU{GPU_SLICE}][{i}/{len(my_jobs)}] {method:10s} "
                f"{sub:10s} {sid:20s} FAILED: {exc}",
                flush=True,
            )
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"[critical GPU{GPU_SLICE}] DONE", flush=True)


if __name__ == "__main__":
    main()
