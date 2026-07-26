"""X-Gaussian (parallel-beam adaptation) wrapper for AET-Bench.

X-Gaussian (Wang et al., ECCV 2024) is a radiative Gaussian splatting
method for X-ray CT reconstruction.  It differs from R2-Gaussian
(which the AET-Bench already includes) in two key respects:

  1. **Isotropic Gaussians** -- each Gaussian has a single scalar scale
     (σ) rather than three axis-aligned scales (s_x, s_y, s_z) with
     independent rotations.  This produces smoother reconstructions and
     is a better inductive bias for blob-like atoms where anisotropic
     covariance is unnecessary.

  2. **Angle-pose initialization** -- initial Gaussian positions are
     seeded from the measured projection angles via a coarse back-
     projection estimate, rather than purely random or grid sampling.

The original X-Gaussian codebase (ECCV 2024) uses a standard 3DGS
alpha-compositing rasterizer with cone-beam geometry.  Since AET-Bench
requires **parallel-beam X-ray integration** (line integrals, not
alpha compositing), we cannot directly invoke the upstream CUDA
rasterizer.

Instead, this wrapper re-implements X-Gaussian's two key ideas on
top of R2-Gaussian's X-ray parallel-beam infrastructure:

  - Training: uses the R2-Gaussian CUDA rasterizer + voxelizer
    (already compiled in the aet_bench_r2g env) with isotropic
    covariance constraints enforced during optimization.
  - Voxelization: R2-Gaussian's GaussianVoxelizer produces the
    final 3D volume.

The isotropic constraint is enforced by reparameterizing the
per-Gaussian scaling as `scales = exp(log_s_scalar).expand(3)`,
tying all three axes.  Rotations become irrelevant for isotropic
Gaussians and are frozen.

Environment: aet_bench_r2g (same conda env as R2-Gaussian baseline).

Reference repo: /data/aet-bench/x-gaussian/
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

# ---------- paths to the R2-Gaussian infrastructure ----------
R2G_DIR = Path("/data/aet-bench/external/r2_gaussian")
R2G_PY = Path("/opt/conda/envs/aet_bench_r2g/bin/python")
R2G_LD = (
    "/opt/conda/envs/aet_bench_r2g/lib/python3.9/site-packages/torch/lib:"
    "/opt/conda/envs/aet_bench_r2g/lib"
)

# We ship a thin training script (see _TRAIN_SCRIPT below) that imports
# R2-Gaussian's Scene / render / query machinery but enforces isotropic
# Gaussians and uses angle-pose initialization.
_TRAIN_SCRIPT_NAME = "train_x_gaussian.py"

DEFAULT_CFG = {
    "iterations": 3000,
    "densify_from_iter": 300,
    "densify_until_iter": 2000,
    "densification_interval": 100,
    "densify_grad_threshold": 5.0e-5,
    "max_num_gaussians": 100000,
    "lambda_dssim": 0.2,
    "lambda_tv": 0.05,
    "tv_vol_size": 32,
    # X-Gaussian-specific: isotropic constraint
    "isotropic": True,
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
    """Strip multi-phase image arrays down to a single 3D volume.

    X-Gaussian (like R2-Gaussian) handles only static single-phase
    reconstruction. If the data dict carries a (10, V, V, V) stacked
    volume (for X2-Gaussian's multi-phase evaluator), extract phase 0.
    """
    d = dict(data)
    img = d["image"]
    if hasattr(img, "ndim") and img.ndim == 4:
        d["image"] = img[0]
    return d


def _angle_pose_init(data: dict, n_points: int = 50_000,
                     rng_seed: int = 0) -> np.ndarray:
    """X-Gaussian-style angle-pose initialization.

    Instead of purely uniform random initialization, seed Gaussian
    positions using a coarse back-projection estimate from the measured
    projections. This concentrates initial points near high-density
    regions.

    Strategy:
      1. Compute a crude SIRT reconstruction (or simple back-projection
         if ASTRA is unavailable) at low resolution (32^3).
      2. Sample points from the resulting intensity distribution
         (intensity-weighted sampling) to place more Gaussians where
         signal is stronger.
      3. Pad with uniform random points to ensure coverage.

    Falls back to uniform random if back-projection fails.
    """
    rng = np.random.default_rng(rng_seed)

    # Try intensity-weighted initialization from projections
    try:
        train = data["train"]
        angles = np.asarray(train["angles"], dtype=np.float64)
        projs = np.asarray(train["projections"], dtype=np.float32)
        vol_size = int(data["nVoxel"][0])

        # Coarse back-projection: resize projections to 32^3 and
        # accumulate simple back-projection (no filter)
        from skimage.transform import resize as sk_resize

        coarse = 32
        projs_small = np.stack([
            sk_resize(p, (coarse, coarse), preserve_range=True)
            for p in projs
        ]).astype(np.float32)

        # Simple back-projection into a coarse volume
        vol_bp = np.zeros((coarse, coarse, coarse), dtype=np.float32)
        for i, angle in enumerate(angles):
            sino_slice = projs_small[i]  # (H, W) = (y, x) projection
            c, s = np.cos(angle), np.sin(angle)
            # For each voxel, find its projected 2D coordinate
            coords_1d = np.linspace(-1.0, 1.0, coarse)
            xg, yg, zg = np.meshgrid(coords_1d, coords_1d, coords_1d,
                                      indexing="ij")
            # Rotation about Y axis: u = x*cos + z*sin, v = y
            u = xg * c + zg * s
            v = yg
            # Map u, v from [-1,1] to pixel indices [0, coarse-1]
            ui = ((u + 1) / 2 * (coarse - 1)).clip(0, coarse - 1).astype(int)
            vi = ((v + 1) / 2 * (coarse - 1)).clip(0, coarse - 1).astype(int)
            vol_bp += sino_slice[vi, ui]

        vol_bp = np.clip(vol_bp, 0, None)
        vol_bp /= max(vol_bp.sum(), 1e-12)

        # Intensity-weighted sampling: place 70% of points via
        # importance sampling, 30% uniformly for coverage
        n_importance = int(n_points * 0.7)
        n_uniform = n_points - n_importance

        # Importance sampling from the back-projection
        flat = vol_bp.ravel()
        prob = flat / flat.sum()
        idx = rng.choice(flat.size, size=n_importance, replace=True, p=prob)
        ix, iy, iz = np.unravel_index(idx, vol_bp.shape)
        pos_imp = np.stack([
            ix / (coarse - 1) * 2 - 1,
            iy / (coarse - 1) * 2 - 1,
            iz / (coarse - 1) * 2 - 1,
        ], axis=1).astype(np.float32)
        # Add small jitter within a voxel
        jitter = rng.uniform(-1.0 / coarse, 1.0 / coarse,
                             size=pos_imp.shape).astype(np.float32)
        pos_imp += jitter

        # Uniform random
        pos_uni = (rng.random((n_uniform, 3)) - 0.5).astype(np.float32) * 2.0

        positions = np.concatenate([pos_imp, pos_uni], axis=0)

        # Initial opacity (smaller for densely sampled regions)
        opacities = rng.random((n_points, 1)).astype(np.float32) * 0.3

        return np.concatenate([positions, opacities], axis=1)

    except Exception:
        # Fallback: uniform random initialization
        return np.concatenate([
            (rng.random((n_points, 3)) - 0.5).astype(np.float32) * 2.0,
            rng.random((n_points, 1)).astype(np.float32) * 0.5,
        ], axis=1)


def _write_train_script(script_path: Path):
    """Write the thin training script that enforces X-Gaussian's
    isotropic constraint on top of R2-Gaussian's infrastructure.

    The script is written to disk and invoked as a subprocess so it
    runs inside the aet_bench_r2g conda environment.
    """
    script = r'''#!/usr/bin/env python
"""X-Gaussian training script (isotropic Gaussians on R2-G infra).

Enforces isotropic scaling: after every optimizer.step(), the three
scale components are averaged so that s_x = s_y = s_z. Rotations
are frozen to identity (irrelevant for isotropic Gaussians).
"""
import os
import os.path as osp
import sys
import torch
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml

sys.path.insert(0, "./")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.utils.log_utils import prepare_output_and_logger
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss


def _enforce_isotropic(gaussians):
    """Enforce isotropic constraint: set all three log-scales to their
    mean, making the Gaussian spherically symmetric."""
    with torch.no_grad():
        s = gaussians._scaling  # (N, 3) in log-space
        mean_s = s.mean(dim=1, keepdim=True)  # (N, 1)
        s.copy_(mean_s.expand_as(s))
        # Also freeze rotation to identity (irrelevant for isotropic)
        gaussians._rotation[:, 0] = 1.0
        gaussians._rotation[:, 1:] = 0.0


def training(
    dataset, opt, pipe,
    testing_iterations,
    saving_iterations,
):
    first_iter = 0

    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    # Set up parameters
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound)
    initialize_gaussian(gaussians, dataset, None)
    scene.gaussians = gaussians
    gaussians.training_setup(opt)

    # Enforce isotropic from the start
    _enforce_isotropic(gaussians)

    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv:
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    # Train
    viewpoint_stack = None
    progress_bar = tqdm(range(0, opt.iterations), desc="Train X-Gaussian", leave=False)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render X-ray projection
        render_pkg = render(viewpoint_cam, gaussians, pipe)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Compute loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss = {"total": 0.0}
        render_loss = l1_loss(image, gt_image)
        loss["render"] = render_loss
        loss["total"] += loss["render"]
        if opt.lambda_dssim > 0:
            loss_dssim = 1.0 - ssim(image, gt_image)
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim
        # 3D TV loss
        if use_tv:
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            vol_pred = query(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe,
            )["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv

        loss["total"].backward()

        with torch.no_grad():
            # Adaptive control
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if iteration < opt.densify_until_iter:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                    )
            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            # Optimization step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # ** X-Gaussian key constraint: enforce isotropy after every step **
            _enforce_isotropic(gaussians)

            # Save gaussians
            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)

            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="X-Gaussian training (isotropic)")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load configuration
    args_dict = vars(args)
    if args.config is not None:
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    print("Optimizing " + args.model_path)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
    )
    print("\nTraining complete.")
'''
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script)


def run_x_gaussian(data: dict, sample_id: str, out_dir: Path,
                   iterations: int = 3000) -> np.ndarray:
    """Train X-Gaussian (isotropic, angle-pose init) and return a 3D volume.

    Parameters
    ----------
    data : dict
        NAF-format data dict. Must contain at minimum:
          data["train"]["angles"]      -- (N_proj,) tilt angles in radians
          data["train"]["projections"] -- (N_proj, H, W) projection images
          data["nVoxel"]               -- (3,) voxel grid dimensions
          data["image"]                -- (V, V, V) or (10, V, V, V) GT volume
          plus scanner geometry keys (DSD, DSO, dVoxel, etc.)
    sample_id : str
        Identifier for logging / path naming.
    out_dir : Path
        Working directory for config, pickle, and model outputs.
    iterations : int
        Number of training iterations (default 3000).

    Returns
    -------
    vol : np.ndarray, shape (V, V, V), dtype float32
        Reconstructed 3D volume.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write the NAF data pickle (single-phase)
    tmp_pkl = out_dir / "data.pkl"
    with open(tmp_pkl, "wb") as f:
        pickle.dump(_ensure_single_phase_image(data), f, protocol=4)

    # 2. Angle-pose initialization (X-Gaussian's distinguishing feature)
    init_path = out_dir / "init_data.npy"
    if not init_path.exists():
        init_pts = _angle_pose_init(data, n_points=50_000)
        np.save(init_path, init_pts)

    # 3. Write the training script
    train_script = out_dir / _TRAIN_SCRIPT_NAME
    _write_train_script(train_script)

    # 4. Write config
    cfg_path = out_dir / "config.yaml"
    model_dir = out_dir / "model"
    _make_config(
        tmp_pkl, model_dir, cfg_path,
        overrides={
            "iterations": iterations,
            "densify_until_iter": min(2000, iterations - 500),
        },
    )

    # 5. Launch training subprocess in the R2-Gaussian environment
    cmd = [
        str(R2G_PY), str(train_script),
        "--config", str(cfg_path),
        "--quiet",
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    # Keep this subprocess isolated from global torch/CUDA libraries.
    env["LD_LIBRARY_PATH"] = R2G_LD
    env["PYTHONPATH"] = str(R2G_DIR) + ":" + env.get("PYTHONPATH", "")
    # Unset proxy env vars that may cause conda/pip slowdowns
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(proxy_var, None)

    print(f"  [x_gaussian] training {sample_id} -> {model_dir}")
    res = subprocess.run(cmd, cwd=str(R2G_DIR), env=env, capture_output=True)
    if res.returncode != 0:
        stderr = res.stderr.decode("utf-8", errors="ignore")[-3000:]
        stdout = res.stdout.decode("utf-8", errors="ignore")[-1000:]
        sys.stderr.write(f"--- X-Gaussian stderr ---\n{stderr}\n")
        sys.stderr.write(f"--- X-Gaussian stdout ---\n{stdout}\n")
        raise RuntimeError(f"x_gaussian train failed for {sample_id}")

    # 6. Load the voxelized volume from the latest saved iteration
    iter_dir = model_dir / "point_cloud" / f"iteration_{iterations}"
    if not iter_dir.exists():
        cands = sorted((model_dir / "point_cloud").glob("iteration_*"))
        if not cands:
            raise FileNotFoundError(
                f"no iteration dirs in {model_dir / 'point_cloud'}"
            )
        iter_dir = cands[-1]

    vol_path = iter_dir / "vol_pred.npy"
    if not vol_path.exists():
        cands = list(iter_dir.glob("vol_pred*.npy"))
        if not cands:
            raise FileNotFoundError(f"no vol_pred in {iter_dir}")
        vol_path = cands[0]

    return np.load(vol_path).astype(np.float32)
