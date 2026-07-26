"""R²-Gaussian (HAADF) wrapper for AET-Bench.

R²-Gaussian (Zha et al., NeurIPS 2024) is the static 3DGS tomography
framework that X²-Gaussian extends to 4D. We run it as a separate
baseline so the benchmark can distinguish "3D Gaussian splatting (static)"
from "4D Gaussian splatting" as distinct entries.

Environment: aet_bench_r2g, a clone of aet4dgs. The clone inherits our
HAADF-patched CUDA rasterizer (alpha = w*exp(power), without the X-ray
μ factor), so R²-Gaussian here uses the HAADF forward model. It does
NOT apply the Z^γ non-linearity that we added to X²-Gaussian's
render_query — R²-Gaussian therefore effectively runs at γ=1.0, a
distinct parameter regime from X²-Gaussian static (γ=1.7).

The wrapper materializes a temporary data.pkl whose `image` field is
the single (vol, vol, vol) GT volume (R²-G does not handle stacked
phases), then invokes external/r2_gaussian/train.py.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

R2G_DIR = Path("/data/aet-bench/external/r2_gaussian")
R2G_PY = Path("/opt/conda/envs/aet_bench_r2g/bin/python")
R2G_LD = (
    "/opt/conda/envs/aet_bench_r2g/lib/python3.9/site-packages/torch/lib:"
    "/opt/conda/envs/aet_bench_r2g/lib"
)


DEFAULT_CFG = {
    "iterations": 8000,
    "densify_from_iter": 300,
    "densify_until_iter": 5000,
    "densification_interval": 100,
    "densify_grad_threshold": 5.0e-5,
    "max_num_gaussians": 100000,
    "lambda_dssim": 0.25,
    "lambda_tv": 0.05,
    "tv_vol_size": 32,
}


def _make_config(data_pkl: Path, model_dir: Path, cfg_path: Path,
                  overrides=None):
    cfg = dict(DEFAULT_CFG)
    if overrides:
        cfg.update(overrides)
    cfg["source_path"] = str(data_pkl)
    cfg["model_path"] = str(model_dir)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _ensure_single_phase_image(data: dict) -> dict:
    """R²-Gaussian expects `data["image"]` as a 3D volume. Our mono/bi/4d
    pickles carry a stacked (10, V, V, V) image for X²-G's evaluator. Strip
    it to a single phase before handing off.
    """
    d = dict(data)
    img = d["image"]
    if hasattr(img, "ndim") and img.ndim == 4:
        d["image"] = img[0]
    return d


def run_r2gaussian(data: dict, sample_id: str, work_root: Path,
                    iterations: int = 8000) -> np.ndarray:
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    tmp_pkl = work_root / "data.pkl"
    with open(tmp_pkl, "wb") as f:
        pickle.dump(_ensure_single_phase_image(data), f, protocol=4)

    # R²-Gaussian expects init_<name>.npy next to the pickle (same
    # convention as X²-Gaussian's NAF loader).
    init_path = work_root / "init_data.npy"
    if not init_path.exists():
        rng = np.random.default_rng(0)
        n_points = 50_000
        init = np.concatenate([
            (rng.random((n_points, 3)) - 0.5) * 2.0,
            rng.random((n_points, 1)) * 0.5,
        ], axis=1).astype(np.float32)
        np.save(init_path, init)

    cfg_path = work_root / "config.yaml"
    model_dir = work_root / "model"
    _make_config(
        tmp_pkl, model_dir, cfg_path,
        overrides={
            "iterations": iterations,
            "densify_until_iter": min(5000, iterations - 1000),
        },
    )

    cmd = [
        str(R2G_PY), str(R2G_DIR / "train.py"),
        "--config", str(cfg_path),
        "--quiet",
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Keep this subprocess isolated from global torch/CUDA libraries.
    env["LD_LIBRARY_PATH"] = R2G_LD
    env["PYTHONPATH"] = str(R2G_DIR) + ":" + env.get("PYTHONPATH", "")
    print(f"  [r2g] training {sample_id} → {model_dir}")
    res = subprocess.run(cmd, cwd=str(R2G_DIR), env=env, capture_output=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr.decode("utf-8", errors="ignore")[-2000:])
        raise RuntimeError(f"r2gaussian train failed for {sample_id}")

    iter_dir = model_dir / "point_cloud" / f"iteration_{iterations}"
    if not iter_dir.exists():
        cands = sorted((model_dir / "point_cloud").glob("iteration_*"))
        if not cands:
            raise FileNotFoundError(f"no iteration dirs in {model_dir}")
        iter_dir = cands[-1]

    # R²-Gaussian saves a single vol_pred.npy (not per-phase)
    vol_path = iter_dir / "vol_pred.npy"
    if not vol_path.exists():
        cands = list(iter_dir.glob("vol_pred*.npy"))
        if not cands:
            raise FileNotFoundError(f"no vol_pred in {iter_dir}")
        vol_path = cands[0]
    return np.load(vol_path).astype(np.float32)
