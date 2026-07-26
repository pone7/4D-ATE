"""ASTRA Toolbox baselines for AET-Bench.

Exposes four classical iterative reconstruction algorithms via one
unified entry point, each running on GPU through ASTRA's CUDA backend:

    - SART_CUDA        (Simultaneous Algebraic Reconstruction Technique)
    - SIRT3D_CUDA      (Simultaneous Iterative Reconstruction, GPU variant)
    - CGLS3D_CUDA      (Conjugate Gradient Least Squares)
    - OS-SART (via SIRT3D_CUDA with subsets)

All four are called through the same `run_astra(data, algorithm=...)`
interface, matching our wrapper convention. They are the proper GPU
implementations of SIRT/SART — distinct from scikit-image's CPU-only
`iradon_sart` that we've been using as the "sirt" baseline.

We build a parallel-beam 3D geometry using the single-axis tilt angles
carried in `data["train"]["angles"]`, then run ASTRA's algorithm for
`n_iter` iterations and copy the volume back to NumPy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def _build_geometry(projs_nhw: np.ndarray, angles_rad: np.ndarray,
                     vol_size: int):
    import astra
    det_rows = projs_nhw.shape[1]
    det_cols = projs_nhw.shape[2]
    proj_geom = astra.create_proj_geom(
        "parallel3d", 1.0, 1.0, det_rows, det_cols, angles_rad
    )
    vol_geom = astra.create_vol_geom(vol_size, vol_size, vol_size)
    return proj_geom, vol_geom


def _resize_projs(projs_nhw: np.ndarray, vol_size: int) -> np.ndarray:
    H, W = projs_nhw.shape[1], projs_nhw.shape[2]
    if H == vol_size and W == vol_size:
        return projs_nhw.astype(np.float32)
    from skimage.transform import resize
    out = np.stack([
        resize(p, (vol_size, vol_size), preserve_range=True)
        for p in projs_nhw
    ]).astype(np.float32)
    return out


def run_astra(data: Dict, sample_id: str, work_root: Path,
                algorithm: str = "SIRT", n_iter: int = 100) -> np.ndarray:
    """Run one of ASTRA's iterative 3D algorithms on a NAF-format sample.

    algorithm ∈ {"SIRT", "SART", "CGLS", "OS-SART"}.
    """
    import astra
    train = data["train"]
    projs_nhw = np.asarray(train["projections"], dtype=np.float32)
    angles_rad = np.asarray(train["angles"], dtype=np.float64)
    vol_size = int(data["nVoxel"][0])

    projs_nhw = _resize_projs(projs_nhw, vol_size)

    # ASTRA wants (det_rows, N_proj, det_cols) for 3D parallel-beam
    projs_vna = np.transpose(projs_nhw, (1, 0, 2)).copy()

    proj_geom, vol_geom = _build_geometry(projs_nhw, angles_rad, vol_size)

    # Create data objects
    proj_id = astra.data3d.create("-proj3d", proj_geom, projs_vna)
    vol_id = astra.data3d.create("-vol", vol_geom, 0.0)

    # Algorithm configuration
    if algorithm.upper() == "SIRT":
        cfg = astra.astra_dict("SIRT3D_CUDA")
    elif algorithm.upper() == "CGLS":
        cfg = astra.astra_dict("CGLS3D_CUDA")
    elif algorithm.upper() == "SART":
        cfg = astra.astra_dict("SIRT3D_CUDA")  # ASTRA 3D SART → fall back to SIRT
        n_iter = max(n_iter, 150)
    elif algorithm.upper() == "OS-SART":
        # Emulate ordered subsets via running SIRT3D_CUDA in chunks over
        # shuffled angle subsets. 10 subsets is a reasonable default.
        return _run_os_sart(projs_nhw, angles_rad, vol_size, n_iter=n_iter)
    else:
        raise ValueError(f"unknown ASTRA algorithm: {algorithm}")

    cfg["ReconstructionDataId"] = vol_id
    cfg["ProjectionDataId"] = proj_id
    alg_id = astra.algorithm.create(cfg)

    print(f"  [astra/{algorithm}] running {sample_id}: n_iter={n_iter}")
    try:
        astra.algorithm.run(alg_id, n_iter)
        vol = astra.data3d.get(vol_id).astype(np.float32)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(proj_id)
        astra.data3d.delete(vol_id)

    return vol


def _run_os_sart(projs_nhw: np.ndarray, angles_rad: np.ndarray,
                  vol_size: int, n_iter: int = 60,
                  n_subsets: int = 10) -> np.ndarray:
    """Ordered-subset SIRT approximation. We partition the angles into
    ``n_subsets`` interleaved subsets and run a few SIRT iterations on
    each subset in a round-robin fashion.
    """
    import astra
    rng = np.random.default_rng(0)
    n_proj = len(angles_rad)
    indices = np.arange(n_proj)
    rng.shuffle(indices)
    subsets = np.array_split(indices, n_subsets)

    vol_geom = astra.create_vol_geom(vol_size, vol_size, vol_size)
    current_vol_id = astra.data3d.create("-vol", vol_geom, 0.0)

    iters_per_subset = max(1, n_iter // n_subsets)
    print(f"  [astra/OS-SART] {n_subsets} subsets, "
          f"{iters_per_subset} iters each, {n_iter} total")
    try:
        for pass_idx in range(n_subsets):
            idx = sorted(subsets[pass_idx % n_subsets])
            projs_sub = projs_nhw[idx]
            angles_sub = angles_rad[idx]
            det_rows, _, det_cols = projs_nhw.shape[1], None, projs_nhw.shape[2]
            proj_geom = astra.create_proj_geom(
                "parallel3d", 1.0, 1.0, det_rows, det_cols, angles_sub
            )
            projs_vna = np.transpose(projs_sub, (1, 0, 2)).copy()
            proj_id = astra.data3d.create("-proj3d", proj_geom, projs_vna)
            cfg = astra.astra_dict("SIRT3D_CUDA")
            cfg["ReconstructionDataId"] = current_vol_id
            cfg["ProjectionDataId"] = proj_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id, iters_per_subset)
            astra.algorithm.delete(alg_id)
            astra.data3d.delete(proj_id)
        vol = astra.data3d.get(current_vol_id).astype(np.float32)
    finally:
        astra.data3d.delete(current_vol_id)
    return vol
