#!/usr/bin/env python3
"""Fill in missing CPU-only classical baselines (fbp/sirt/tomopy_gridrec/genfire).

These methods run entirely on CPU so they won't steal GPU cycles from the
neural workers still running on the H200. Results are appended to a
*separate* file (results_cpu_extras.jsonl) to avoid racing with the
GPU-side run_all_missing.py that writes results.jsonl.

Run:
    /opt/conda/envs/aet4dgs/bin/python \\
        scripts/run_cpu_extras.py [--workers N] [--methods fbp,sirt,...] \\
        [--limit N] [--dry-run]

When finished, merge with:
    python scripts/merge_cpu_extras.py
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

# Keep BLAS single-threaded so our process-level parallelism isn't shadowed
# by each worker trying to saturate the machine.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np  # noqa: E402

ROOT = Path("/data/aet-bench")
B = ROOT / "data/aet_bench"
RF_MAIN = B / "results.jsonl"
RF_EXTRAS = B / "results_cpu_extras.jsonl"

CPU_METHODS = ["fbp", "sirt", "tomopy_gridrec", "genfire"]
SUBSETS = ["mono", "bi", "multislice", "real", "brownian_pt"]


# ---------------------------------------------------------------------------
# method dispatch (imports lazy, per worker process)
# ---------------------------------------------------------------------------

def _method_fbp(data, sample_id):
    from skimage.transform import iradon, resize
    train = data["train"]
    angles_deg = np.degrees(train["angles"])
    projs = train["projections"]
    vol_size = int(data["nVoxel"][0])
    if projs.shape[1] != vol_size or projs.shape[2] != vol_size:
        projs = np.stack([resize(p, (vol_size, vol_size), preserve_range=True)
                          for p in projs]).astype(np.float32)
    vol = np.zeros((vol_size, vol_size, vol_size), dtype=np.float32)
    for y in range(vol_size):
        sino = projs[:, y, :].T
        vol[:, y, :] = iradon(sino, theta=angles_deg, output_size=vol_size,
                              filter_name="ramp", circle=True).astype(np.float32)
    return vol


def _method_sirt(data, sample_id, n_iter=10):
    from skimage.transform import iradon_sart, resize
    train = data["train"]
    angles_deg = np.degrees(train["angles"])
    projs = train["projections"]
    vol_size = int(data["nVoxel"][0])
    if projs.shape[1] != vol_size or projs.shape[2] != vol_size:
        projs = np.stack([resize(p, (vol_size, vol_size), preserve_range=True)
                          for p in projs]).astype(np.float32)
    vol = np.zeros((vol_size, vol_size, vol_size), dtype=np.float32)
    for y in range(vol_size):
        sino = projs[:, y, :].T
        recon = None
        for _ in range(n_iter):
            recon = iradon_sart(sino, theta=angles_deg, image=recon)
        vol[:, y, :] = recon.astype(np.float32)
    return vol


def _method_tomopy_gridrec(data, sample_id):
    import tomopy
    from skimage.transform import resize
    train = data["train"]
    projs = np.asarray(train["projections"], dtype=np.float32)
    angles_rad = np.asarray(train["angles"], dtype=np.float64)
    vol_size = int(data["nVoxel"][0])
    if projs.shape[1] != vol_size or projs.shape[2] != vol_size:
        projs = np.stack([resize(p, (vol_size, vol_size), preserve_range=True)
                          for p in projs]).astype(np.float32)
    center = vol_size / 2.0
    recon = tomopy.recon(projs, angles_rad, center=center,
                         algorithm="gridrec", ncore=1)
    return recon.astype(np.float32)


def _method_genfire(data, sample_id, n_iter=50, oversampling=2):
    # Reuse the battle-tested baseline wrapper so this stays in lock-step
    # with the results generated earlier.
    sys.path.insert(0, str(ROOT / "scripts"))
    from baselines.run_genfire import run_genfire
    return run_genfire(data, n_iter=n_iter, oversampling=oversampling)


METHODS = {
    "fbp": _method_fbp,
    "sirt": _method_sirt,
    "tomopy_gridrec": _method_tomopy_gridrec,
    "genfire": _method_genfire,
}


# ---------------------------------------------------------------------------
# evaluation (matches run_all_missing.py fast_eval)
# ---------------------------------------------------------------------------

def _fast_eval(vol_pred, vol_gt, atoms_gt, voxel_size):
    # Lazy imports per-worker
    sys.path.insert(0, str(ROOT / "scripts"))
    from metrics.fsc import fsc_summary
    from skimage.metrics import structural_similarity

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
        ssim_vals.append(
            structural_similarity(vol_gt[i], vps[i], data_range=dr)
        )
    ssim = float(np.mean(ssim_vals))
    try:
        fsc = fsc_summary(vol_pred, vol_gt, voxel_size)
    except Exception:
        fsc = {}
    return {
        "vol_psnr": float(psnr),
        "vol_ssim": float(ssim),
        "rmsd": float("nan"),
        "recall": 0.0,
        "precision": 0.0,
        "f1": 0.0,
        "n_pred": 0,
        "n_gt": int(len(atoms_gt)),
        "n_matched": 0,
        **fsc,
    }


def _to_py(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, np.float64, np.float32)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# ---------------------------------------------------------------------------
# worker entry
# ---------------------------------------------------------------------------

def run_job(args):
    sub, sid, method = args
    try:
        pkl_path = B / sub / sid / "analytical" / "data.pkl"
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        atoms_gt = np.load(B / sub / sid / "atoms_gt.npy")
        meta = json.load(open(B / sub / sid / "metadata.json"))
        t0 = time.time()
        vol = METHODS[method](data, sid)
        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
        if (vol > 0).any():
            vol = np.clip(vol, 0, float(np.percentile(vol[vol > 0], 99.9)) * 2)
        else:
            vol = np.clip(vol, 0, 1)
        vg = data["image"][0] if data["image"].ndim == 4 else data["image"]
        cs = meta.get("cell_size_angstrom", 25.0)
        vs = cs / vg.shape[0]
        me = _fast_eval(vol, vg, atoms_gt, vs)
        me = {k: _to_py(x) for k, x in me.items()}
        rt = time.time() - t0
        row = {
            "subset": sub, "sample_id": sid, "backend": "analytical",
            "method": method, "task": "T1", "metrics": me, "runtime_sec": rt,
        }
        return row
    except Exception as e:
        return {
            "error": str(e),
            "tb": traceback.format_exc(),
            "subset": sub, "sample_id": sid, "method": method,
        }
    finally:
        gc.collect()


# ---------------------------------------------------------------------------

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
                done.add((r["subset"], r["sample_id"], r["method"]))
    return done


def build_jobs(methods, subsets, limit=None):
    done = load_done()
    jobs = []
    for sub in subsets:
        d = B / sub
        if not d.exists():
            continue
        samples = sorted([p.name for p in d.iterdir() if p.is_dir()])
        for m in methods:
            for s in samples:
                if (sub, s, m) not in done:
                    jobs.append((sub, s, m))
    if limit:
        jobs = jobs[:limit]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--methods", default=",".join(CPU_METHODS))
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        print(f"Unknown methods: {bad}. Valid: {list(METHODS)}", file=sys.stderr)
        sys.exit(2)

    jobs = build_jobs(methods, subsets, args.limit)
    print(f"CPU extras runner: {len(jobs)} jobs; {args.workers} workers; methods={methods}")

    if args.dry_run:
        from collections import Counter
        c = Counter((sub, m) for sub, _, m in jobs)
        for (sub, m), n in sorted(c.items()):
            print(f"  {sub:12s} {m:18s} {n}")
        return

    RF_EXTRAS.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    ok = err = 0

    with mp.Pool(args.workers, maxtasksperchild=20) as pool:
        with open(RF_EXTRAS, "a") as fout:
            for i, row in enumerate(pool.imap_unordered(run_job, jobs), 1):
                if "error" in row:
                    err += 1
                    line = json.dumps(row, default=_to_py)
                else:
                    ok += 1
                    line = json.dumps(row, default=_to_py)
                fout.write(line + "\n")
                fout.flush()
                if i % 10 == 0 or i == len(jobs):
                    elapsed = time.time() - start
                    rate = i / elapsed * 60 if elapsed > 0 else 0
                    eta = (len(jobs) - i) / (rate / 60) / 60 if rate else float("inf")
                    print(
                        f"  [{i}/{len(jobs)}] ok={ok} err={err} "
                        f"rate={rate:.1f}/min eta={eta:.1f}min",
                        flush=True,
                    )

    print(f"DONE. ok={ok} err={err} total={len(jobs)} "
          f"elapsed={(time.time()-start)/60:.1f}min")


if __name__ == "__main__":
    main()
