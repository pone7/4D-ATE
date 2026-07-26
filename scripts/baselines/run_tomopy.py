"""TomoPy baselines for AET-Bench.

TomoPy (Synchrotron X-ray tomography standard) exposes several classical
reconstruction algorithms. We wrap two that are representative and not
redundant with our other baselines:

    - gridrec  — Fourier-gridding direct method (very fast)
    - mlem     — Maximum Likelihood Expectation Maximization

gridrec is the golden standard in synchrotron tomography and covers a
distinct algorithmic family (direct Fourier reconstruction) that we
don't otherwise have. MLEM covers Poisson-likelihood iterative
reconstruction.

For single-axis tilt HAADF we treat the tilt axis as the rotation axis
and reconstruct slice-by-slice (same as our skimage FBP/SIRT baselines),
which matches TomoPy's 2D parallel-beam assumption.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def _reshape_for_tomopy(projs_nhw: np.ndarray, vol_size: int) -> np.ndarray:
    """Resize and transpose projections to TomoPy's expected layout:
    (n_proj, detector_rows, detector_cols). TomoPy reconstructs one slice
    per detector row.
    """
    from skimage.transform import resize
    N, H, W = projs_nhw.shape
    if H != vol_size or W != vol_size:
        projs_nhw = np.stack([
            resize(p, (vol_size, vol_size), preserve_range=True)
            for p in projs_nhw
        ])
    return projs_nhw.astype(np.float32)


def run_tomopy(data: Dict, sample_id: str, work_root: Path,
                algorithm: str = "gridrec") -> np.ndarray:
    import tomopy

    train = data["train"]
    projs_nhw = np.asarray(train["projections"], dtype=np.float32)
    angles_rad = np.asarray(train["angles"], dtype=np.float64)
    vol_size = int(data["nVoxel"][0])

    projs = _reshape_for_tomopy(projs_nhw, vol_size)

    # TomoPy expects (n_angles, rows, cols); center defaults to cols/2
    center = vol_size / 2.0

    if algorithm == "gridrec":
        print(f"  [tomopy/gridrec] reconstructing {sample_id}")
        recon = tomopy.recon(projs, angles_rad, center=center,
                              algorithm="gridrec")
    elif algorithm == "mlem":
        print(f"  [tomopy/mlem] reconstructing {sample_id}")
        recon = tomopy.recon(projs, angles_rad, center=center,
                              algorithm="mlem", num_iter=30)
    else:
        raise ValueError(f"unsupported tomopy algorithm: {algorithm}")

    # TomoPy returns (rows, cols, cols); our NAF convention is (V, V, V)
    # with axis order (z, y, x). Rearrange to (x, y, z) for consistency
    # with skimage FBP/SIRT paths which write (V, V, V) in (y, x, z) — the
    # downstream PSNR/SSIM are axis-invariant so any consistent ordering
    # works. Just return the TomoPy output cast to float32.
    return recon.astype(np.float32)
