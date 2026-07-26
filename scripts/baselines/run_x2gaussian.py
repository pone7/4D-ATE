"""X²-Gaussian (HAADF) wrapper for AET-Bench.

Modes:
    static — 3D-only training (no_grid + all deformation channels off)
    4d     — HexPlane deformation field on for dx (position) only
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

X2_DIR = Path("/data/aet-bench/x2-gaussian")
PYTHON = "/opt/conda/envs/aet4dgs/bin/python"

# open3d requires libGL from the env; make sure the subprocess finds it.
_ENV_LD = str(Path("/opt/conda/envs/aet4dgs/lib"))


STATIC_CFG = {
    "haadf_gamma": 1.7,
    "iterations": 8000,
    "densify_from_iter": 300,
    "densify_until_iter": 5000,
    "densification_interval": 100,
    "densify_grad_threshold": 5.0e-5,
    "max_num_gaussians": 100000,
    "lambda_dssim": 0.25,
    "lambda_tv": 0.05,
    "tv_vol_size": 32,
    "no_dx": True, "no_ds": True, "no_dr": True, "no_do": True,
    "no_grid": True,
    "time_smoothness_weight": 0.0,
    "plane_tv_weight": 0.0,
    "l1_time_planes": 0.0,
}

DEFORM_4D_CFG = {
    "haadf_gamma": 1.7,
    "iterations": 8000,
    "densify_from_iter": 300,
    "densify_until_iter": 5000,
    "densification_interval": 100,
    "densify_grad_threshold": 5.0e-5,
    "max_num_gaussians": 100000,
    "lambda_dssim": 0.25,
    "lambda_tv": 0.05,
    "tv_vol_size": 32,
    # Position deformation only — see baseline expansion design.
    "no_dx": False, "no_ds": True, "no_dr": True, "no_do": True,
    "no_grid": False,
    "time_smoothness_weight": 0.001,
    "plane_tv_weight": 0.0001,
    "l1_time_planes": 0.0001,
    # Physics losses stay off for benchmark.
    "lambda_physics": 0.0,
}


def _make_config(data_pkl: Path, model_dir: Path, cfg_path: Path,
                  base_cfg: dict, overrides=None):
    cfg = dict(base_cfg)
    if overrides:
        cfg.update(overrides)
    cfg["source_path"] = str(data_pkl)
    cfg["model_path"] = str(model_dir)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_x2gaussian(data: dict, sample_id: str, work_root: Path,
                    iterations: int = 8000, mode: str = "static"):
    """Train X²-Gaussian and return reconstruction(s).

    mode="static": returns (vol, vol, vol) float32
    mode="4d":     returns dict with keys
        'vols':      (10, vol, vol, vol) float32
        'gauss_xyz': (10, M, 3) float32, identity-preserving across phases
    """
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    tmp_pkl = work_root / "data.pkl"
    with open(tmp_pkl, "wb") as f:
        pickle.dump(data, f, protocol=4)

    init_path = work_root / "init_data.npy"
    if not init_path.exists():
        rng = np.random.default_rng(0)
        n_points = 50_000
        init = np.concatenate([
            (rng.random((n_points, 3)) - 0.5) * 2.0,
            rng.random((n_points, 1)) * 0.5,
        ], axis=1).astype(np.float32)
        np.save(init_path, init)

    base_cfg = STATIC_CFG if mode == "static" else DEFORM_4D_CFG
    cfg_path = work_root / f"config_{mode}.yaml"
    model_dir = work_root / f"model_{mode}"
    _make_config(
        tmp_pkl, model_dir, cfg_path, base_cfg,
        overrides={
            "iterations": iterations,
            "densify_until_iter": min(5000, iterations - 1000),
        },
    )

    cmd = [
        PYTHON, str(X2_DIR / "train.py"),
        "--config", str(cfg_path),
        "--quiet",
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Do not inherit the cluster/global LD_LIBRARY_PATH. It can point at a
    # different Python's torch libraries and crash on import.
    env["LD_LIBRARY_PATH"] = _ENV_LD
    print(f"  [x2g/{mode}] training {sample_id} → {model_dir}")
    res = subprocess.run(cmd, cwd=str(X2_DIR), env=env, capture_output=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr.decode("utf-8", errors="ignore")[-2000:])
        raise RuntimeError(f"x2gaussian/{mode} train failed for {sample_id}")

    iter_dir = model_dir / "point_cloud" / f"iteration_{iterations}"
    if not iter_dir.exists():
        cands = sorted((model_dir / "point_cloud").glob("iteration_*"))
        if not cands:
            raise FileNotFoundError(f"no iteration dirs in {model_dir}")
        iter_dir = cands[-1]

    if mode == "static":
        vol = np.load(iter_dir / "vol_pred_T0.npy").astype(np.float32)
        return vol

    # mode == "4d"
    vol_paths = sorted(iter_dir.glob("vol_pred_T*.npy"),
                       key=lambda p: int(p.stem.split("T")[-1]))
    if len(vol_paths) < 10:
        raise FileNotFoundError(
            f"expected 10 vol_pred_T*.npy under {iter_dir}, "
            f"found {len(vol_paths)}"
        )
    vols = np.stack([np.load(p) for p in vol_paths[:10]]).astype(np.float32)

    xyz_paths = sorted(iter_dir.glob("gauss_xyz_T*.npy"),
                       key=lambda p: int(p.stem.split("T")[-1]))
    gauss_xyz = None
    if len(xyz_paths) >= 10:
        # Identity-preserving: same index across all phases (guaranteed by
        # scene.save() writing all phases from the same model snapshot).
        # Downsample to top-K indices at t=0 to bound downstream memory.
        all_xyz = [np.load(p).astype(np.float32) for p in xyz_paths[:10]]
        n_total = all_xyz[0].shape[0]
        K = min(10_000, n_total)
        idx = np.arange(K)
        gauss_xyz = np.stack([xyz[idx] for xyz in all_xyz], axis=0)

    return {"vols": vols, "gauss_xyz": gauss_xyz}
