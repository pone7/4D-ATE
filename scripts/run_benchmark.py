#!/usr/bin/env python
"""
AET-Bench unified benchmark runner.

Dispatches a single (subset, sample, method, task) combination, produces a
recon volume, evaluates it, and appends a result row to a JSONL log.

Usage:
    python run_benchmark.py --subset mono --method fbp --task T1
    python run_benchmark.py --subset mono --sample Cu-0309-00 --method fbp --task T2

Result schema (one JSON object per line in results.jsonl):
    {
        "subset": "mono", "sample_id": "...", "backend": "analytical",
        "method": "fbp", "task": "T1",
        "metrics": {"vol_psnr": ..., "vol_ssim": ..., "fsc_0.143": ...,
                    "rmsd": ..., "recall": ..., "precision": ..., "f1": ...},
        "runtime_sec": ..., "iter": ...,
    }
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from evaluate_baselines import evaluate_reconstruction  # noqa: E402
from metrics.fsc import fsc_summary  # noqa: E402


SUBSET_DIRS = {
    "mono": Path("/data/aet-bench/data/aet_bench/mono"),
    "bi":   Path("/data/aet-bench/data/aet_bench/bi"),
    "4d":   Path("/data/aet-bench/data/aet_bench/sim4d"),
    "real": Path("/data/aet-bench/data/aet_bench/real"),
    "multislice": Path("/data/aet-bench/data/aet_bench/multislice"),
    "brownian_pt": Path("/data/aet-bench/data/aet_bench/brownian_pt"),
    "dose": Path("/data/aet-bench/data/aet_bench/dose"),
}


def list_samples(subset):
    root = SUBSET_DIRS[subset]
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def load_sample(subset, sample_id, backend):
    sample_dir = SUBSET_DIRS[subset] / sample_id
    pkl = sample_dir / backend / "data.pkl"
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    atoms_gt = np.load(sample_dir / "atoms_gt.npy")
    with open(sample_dir / "metadata.json") as f:
        meta = json.load(f)
    return data, atoms_gt, meta


# ---------- Method registry ----------

def method_fbp(data):
    """Slice-wise filtered back projection (scikit-image iradon)."""
    from skimage.transform import iradon
    train = data["train"]
    angles_deg = np.degrees(train["angles"])
    projs = train["projections"]  # (N, H, W)
    n_a, h, w = projs.shape
    vol_size = data["nVoxel"][0]
    if h != vol_size:
        from skimage.transform import resize
        projs = np.stack([resize(p, (vol_size, vol_size), preserve_range=True)
                          for p in projs])
    # For each y-slice, sinogram[angle, x] -> reconstruct (x, z) plane
    vol = np.zeros((vol_size, vol_size, vol_size), dtype=np.float32)
    for y in range(vol_size):
        sino = projs[:, y, :].T  # (W, n_angles)
        recon = iradon(sino, theta=angles_deg, output_size=vol_size,
                       filter_name="ramp", circle=True)
        vol[:, y, :] = recon.astype(np.float32)
    return vol


def method_sirt(data, n_iter=10):
    """Slice-wise SIRT via skimage iradon_sart (one SART iter per sweep)."""
    from skimage.transform import iradon_sart, resize
    train = data["train"]
    angles_deg = np.degrees(train["angles"])
    projs = train["projections"]
    n_a, h, w = projs.shape
    vol_size = data["nVoxel"][0]
    if h != vol_size:
        projs = np.stack([resize(p, (vol_size, vol_size), preserve_range=True)
                          for p in projs])
    vol = np.zeros((vol_size, vol_size, vol_size), dtype=np.float32)
    for y in range(vol_size):
        sino = projs[:, y, :].T  # (W, n_angles)
        recon = None
        for _ in range(n_iter):
            recon = iradon_sart(sino, theta=angles_deg, image=recon)
        vol[:, y, :] = recon.astype(np.float32)
    return vol


def method_x2gaussian(data, sample_id="sample", task="T1"):
    """Train X²-Gaussian (HAADF-adapted) and return its volume(s)."""
    from baselines.run_x2gaussian import run_x2gaussian
    work_root = Path("/data/aet-bench/output/aet_bench") / sample_id
    mode = "4d" if task == "T7" else "static"
    return run_x2gaussian(data, sample_id, work_root, mode=mode)


def method_genfire(data):
    from baselines.run_genfire import run_genfire
    return run_genfire(data, n_iter=50, oversampling=2)


def method_r2gaussian(data, sample_id="sample"):
    from baselines.run_r2gaussian import run_r2gaussian
    work_root = Path("/data/aet-bench/output/aet_bench_r2g") / sample_id
    return run_r2gaussian(data, sample_id, work_root)


def method_implicit_et(data, sample_id="sample"):
    from baselines.run_implicit_et import run_implicit_et
    work_root = Path("/data/aet-bench/output/aet_bench_inr") / sample_id
    return run_implicit_et(data, sample_id, work_root)


def method_isonet(data, sample_id="sample"):
    from baselines.run_isonet import run_isonet
    work_root = Path("/data/aet-bench/output/aet_bench_isonet") / sample_id
    return run_isonet(data, sample_id, work_root)


def method_4dgs(data, sample_id="sample", task="T1"):
    from baselines.run_4dgs import run_4dgs
    work_root = Path("/data/aet-bench/output/aet_bench_4dgs") / sample_id
    out = run_4dgs(data, sample_id, work_root)
    if task == "T7":
        return out
    if isinstance(out, dict) and "vols" in out:
        return out["vols"][0]
    return out


def method_astra_sirt(data, sample_id="sample"):
    from baselines.run_astra import run_astra
    return run_astra(data, sample_id,
                     Path("/data/aet-bench/output/aet_bench_astra") / sample_id,
                     algorithm="SIRT", n_iter=150)


def method_astra_sart(data, sample_id="sample"):
    from baselines.run_astra import run_astra
    return run_astra(data, sample_id,
                     Path("/data/aet-bench/output/aet_bench_astra") / sample_id,
                     algorithm="SART", n_iter=150)


def method_astra_cgls(data, sample_id="sample"):
    from baselines.run_astra import run_astra
    return run_astra(data, sample_id,
                     Path("/data/aet-bench/output/aet_bench_astra") / sample_id,
                     algorithm="CGLS", n_iter=50)


def method_astra_ossart(data, sample_id="sample"):
    from baselines.run_astra import run_astra
    return run_astra(data, sample_id,
                     Path("/data/aet-bench/output/aet_bench_astra") / sample_id,
                     algorithm="OS-SART", n_iter=100)


def method_tomopy_gridrec(data, sample_id="sample"):
    from baselines.run_tomopy import run_tomopy
    return run_tomopy(data, sample_id, Path("/tmp/tomopy") / sample_id,
                       algorithm="gridrec")


def method_tomopy_mlem(data, sample_id="sample"):
    from baselines.run_tomopy import run_tomopy
    return run_tomopy(data, sample_id, Path("/tmp/tomopy") / sample_id,
                       algorithm="mlem")


def method_wbp(data, sample_id="sample"):
    from baselines.run_classical_extras import run_wbp
    return run_wbp(data, sample_id, Path("/tmp/wbp") / sample_id)


def method_fista_tv(data, sample_id="sample"):
    from baselines.run_classical_extras import run_fista_tv
    return run_fista_tv(data, sample_id,
                        Path("/data/aet-bench/output/aet_bench_fista") / sample_id)


def method_dart(data, sample_id="sample"):
    from baselines.run_classical_extras import run_dart
    return run_dart(data, sample_id,
                    Path("/data/aet-bench/output/aet_bench_dart") / sample_id)


def method_neat(data, sample_id="sample"):
    from baselines.run_neat import run_neat
    return run_neat(data, sample_id,
                    Path("/data/aet-bench/output/aet_bench_neat") / sample_id)


def method_sax_nerf(data, sample_id="sample"):
    from baselines.run_sax_nerf import run_sax_nerf
    return run_sax_nerf(data, sample_id,
                        Path("/data/aet-bench/output/aet_bench_sax_nerf") / sample_id)


def method_x_gaussian(data, sample_id="sample"):
    from baselines.run_x_gaussian import run_x_gaussian
    return run_x_gaussian(data, sample_id,
                          Path("/data/aet-bench/output/aet_bench_x_gaussian") / sample_id)


def method_dif_gaussian(data, sample_id="sample"):
    from baselines.run_dif_gaussian import run_dif_gaussian
    return run_dif_gaussian(data, sample_id,
                            Path("/data/aet-bench/output/aet_bench_dif_gaussian") / sample_id)


def method_instamap(data, sample_id="sample"):
    from baselines.run_instamap import run_instamap
    return run_instamap(data, sample_id,
                        Path("/data/aet-bench/output/aet_bench_instamap") / sample_id)


METHOD_REGISTRY = {
    # --- Classical (5) ---
    "wbp": method_wbp,
    "astra_sirt": method_astra_sirt,
    "astra_cgls": method_astra_cgls,
    "tomopy_mlem": method_tomopy_mlem,
    "genfire": method_genfire,
    # --- Neural existing (5) ---
    "neat": method_neat,
    "implicit_et": method_implicit_et,
    "r2gaussian": method_r2gaussian,
    "x2gaussian": method_x2gaussian,
    "isonet": method_isonet,
    # --- Neural new (4) ---
    "sax_nerf": method_sax_nerf,
    "x_gaussian": method_x_gaussian,
    "dif_gaussian": method_dif_gaussian,
    "instamap": method_instamap,
}

# Legacy aliases kept for backward compatibility with existing results.jsonl
METHOD_REGISTRY_LEGACY = {
    "fbp": method_fbp, "sirt": method_sirt, "astra_sart": method_astra_sart,
    "astra_ossart": method_astra_ossart, "tomopy_gridrec": method_tomopy_gridrec,
    "fista_tv": method_fista_tv, "dart": method_dart, "fourdgs": method_4dgs,
}


# ---------- Task evaluators ----------

def _adaptive_threshold(cell_size: float, vol_shape) -> float:
    """Per-sample atom-matching threshold in Å.

    For simulated nanoparticles (cell ~25 Å, 128³ → 0.2 Å/voxel) the
    default 0.5 Å threshold is a tight but physically-meaningful target.
    For real experimental data downsampled to 128³ with much larger
    physical cells (~120 Å → ~1 Å/voxel), the per-voxel resolution is
    the hard ceiling, so we scale the threshold proportionally:
    `threshold = max(0.5, 1.5 * voxel_size)`.
    """
    voxel_size = cell_size / vol_shape[0]
    return max(0.5, 1.5 * voxel_size)


def task_t1_t2(vol_pred, data, atoms_gt, meta):
    """T1 (volume metrics) + T2 (atom localization). Returns merged dict."""
    vol_gt = data["image"]
    if vol_gt.ndim == 4:
        vol_gt = vol_gt[0]
    cell_size = meta["cell_size_angstrom"]
    voxel_size = cell_size / vol_gt.shape[0]
    threshold = _adaptive_threshold(cell_size, vol_gt.shape)
    # Physical min-peak-distance for non-maximum suppression. To *resolve*
    # two atoms separated by the nearest-neighbour distance d_NN, the NMS
    # exclusion window (whose full width is 2*min_peak_distance+1 voxels)
    # must be narrower than d_NN. Close-packed metals have d_NN ~2.5-2.9 Å
    # (Cu 2.55, Ag/Au 2.89), so the exclusion radius must stay below
    # d_NN/2 ~1.3 Å; larger values merge adjacent atoms and cap recall.
    # We pin 1.0 Å (well inside the resolvable plateau; see
    # scripts/analysis/extraction_sensitivity.py for the full sweep, and
    # tests/test_extractor_ceiling.py for the GT-recovery guard).
    min_peak_distance = max(1, int(round(1.0 / voxel_size)))
    res = evaluate_reconstruction(vol_pred, vol_gt, atoms_gt, cell_size,
                                   method_name="benchmark",
                                   threshold=threshold,
                                   min_peak_distance=min_peak_distance)
    voxel_size_a = cell_size / vol_gt.shape[0]
    # FSC needs same scale; rescale pred to GT max like evaluate_reconstruction
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


def task_t7_4d(vol_pred, data, atoms_gt, meta):
    """T7 evaluator.

    vol_pred shapes accepted:
        (vol, vol, vol)           — static method, replicated across 10 phases
        (10, vol, vol, vol)       — 4D method, volumes only
        dict {'vols': (10,...), 'gauss_xyz': (10, M, 3) or None}
    """
    from metrics.trajectory import trajectory_rmsd, atom_identity_persistence

    sample_dir = SUBSET_DIRS["4d"] / meta["sample_id"]
    atoms_gt_4d = np.load(sample_dir / "atoms_gt_4d.npy")  # (n_phases, N, 4)
    n_phases = atoms_gt_4d.shape[0]
    cell_size = meta["cell_size_angstrom"]

    vol_gt_4d = data["image"]  # (n_phases, vol, vol, vol)
    if vol_gt_4d.ndim == 3:
        vol_gt_4d = np.broadcast_to(vol_gt_4d[None],
                                     (n_phases,) + vol_gt_4d.shape).copy()

    gauss_xyz_4d = None
    if isinstance(vol_pred, dict):
        gauss_xyz_4d = vol_pred.get("gauss_xyz")
        vol_pred = vol_pred["vols"]
    if vol_pred.ndim == 3:
        vol_pred_4d = np.broadcast_to(vol_pred[None], vol_gt_4d.shape).copy()
    else:
        vol_pred_4d = vol_pred

    threshold = _adaptive_threshold(cell_size, vol_gt_4d.shape[1:])
    per_phase = []
    for t in range(n_phases):
        atoms_gt_t = atoms_gt_4d[t]
        res = evaluate_reconstruction(
            vol_pred_4d[t], vol_gt_4d[t], atoms_gt_t, cell_size,
            method_name=f"benchmark_T{t}", threshold=threshold,
        )
        voxel_size_a = cell_size / vol_gt_4d.shape[1]
        scale = vol_gt_4d[t].max() / max(vol_pred_4d[t].max(), 1e-12)
        fsc = fsc_summary(vol_pred_4d[t] * scale, vol_gt_4d[t],
                           voxel_size_angstrom=voxel_size_a)
        per_phase.append({
            "vol_psnr": res["vol_psnr"],
            "vol_ssim": res["vol_ssim"],
            "rmsd": res["rmsd"],
            "recall": res["recall"],
            "precision": res["precision"],
            "f1": res["f1"],
            "fsc_0.143_angstrom": fsc["resolution_0.143"],
        })

    def _mean(key):
        vals = [p[key] for p in per_phase
                if p[key] is not None and p[key] != float("inf")]
        return float(np.mean(vals)) if vals else None

    out = {
        "vol_psnr": _mean("vol_psnr"),
        "vol_ssim": _mean("vol_ssim"),
        "rmsd": _mean("rmsd"),
        "recall": _mean("recall"),
        "precision": _mean("precision"),
        "f1": _mean("f1"),
        "fsc_0.143_angstrom": _mean("fsc_0.143_angstrom"),
        "per_phase": per_phase,
    }

    if gauss_xyz_4d is not None:
        gt_xyz_4d = atoms_gt_4d[..., :3].astype(np.float32)
        out["trajectory_rmsd"] = trajectory_rmsd(gauss_xyz_4d, gt_xyz_4d)
        out["atom_identity_persistence"] = atom_identity_persistence(
            gauss_xyz_4d, gt_xyz_4d
        )
    return out


TASK_REGISTRY = {
    "T1": task_t1_t2,
    "T2": task_t1_t2,  # share evaluator; differ only in which metrics matter
    "T7": task_t7_4d,
}


# ---------- Driver ----------

def run_one(subset, sample_id, backend, method, task, results_path):
    data, atoms_gt, meta = load_sample(subset, sample_id, backend)
    fn = METHOD_REGISTRY[method]
    t0 = time.time()
    try:
        vol_pred = fn(data, sample_id=sample_id, task=task)
    except TypeError:
        try:
            vol_pred = fn(data, sample_id=sample_id)
        except TypeError:
            vol_pred = fn(data)
    runtime = time.time() - t0
    metrics = TASK_REGISTRY[task](vol_pred, data, atoms_gt, meta)

    row = {
        "subset": subset, "sample_id": sample_id, "backend": backend,
        "method": method, "task": task, "metrics": metrics,
        "runtime_sec": runtime,
    }

    def _to_py(o):
        if isinstance(o, dict):
            return {k: _to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_py(x) for x in o]
        if isinstance(o, np.generic):
            return o.item()
        return o

    with open(results_path, "a") as f:
        f.write(json.dumps(_to_py(row)) + "\n")
    psnr = metrics.get('vol_psnr') or 0.0
    ssim = metrics.get('vol_ssim') or 0.0
    recall = metrics.get('recall') or 0.0
    print(f"[{method}/{task}] {sample_id} ({backend}): "
          f"PSNR={psnr:.2f} SSIM={ssim:.3f} "
          f"Recall={recall:.1%} runtime={runtime:.1f}s")
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", choices=list(SUBSET_DIRS), required=True)
    p.add_argument("--sample", default=None,
                   help="sample_id; if omitted, runs all samples in subset")
    p.add_argument("--backend", default="analytical",
                   choices=["analytical", "abtem"])
    p.add_argument("--method", required=True, choices=list(METHOD_REGISTRY))
    p.add_argument("--task", required=True, choices=list(TASK_REGISTRY))
    p.add_argument("--results", default="/data/aet-bench/data/aet_bench/results.jsonl")
    args = p.parse_args()

    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    samples = [args.sample] if args.sample else list_samples(args.subset)
    if not samples:
        print(f"No samples found in subset {args.subset}")
        return
    print(f"Running {args.method}/{args.task} on {len(samples)} sample(s)")
    for s in samples:
        try:
            run_one(args.subset, s, args.backend, args.method, args.task,
                    args.results)
        except Exception as e:
            import traceback
            print(f"  FAILED on {s}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)


if __name__ == "__main__":
    main()
