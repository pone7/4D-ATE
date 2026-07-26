"""GENFIRE wrapper for AET-Bench.

GENFIRE (Pryor et al. 2017) is a Fourier-iterative reconstruction algorithm
widely used in atomic electron tomography. We use it via its Python API by
materializing per-sample projection/angle .npy files in a tmp dir and
calling `genfire.main.main`.

Single-axis tilt around y → Euler (phi, theta, psi) = (0, tilt, 0).
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np


def run_genfire(data: dict, n_iter: int = 50,
                oversampling: int = 2) -> np.ndarray:
    import genfire.fileio  # lazy import (warns about pyFFTW)
    from genfire.main import main as gf_main
    from genfire.reconstruct import ReconstructionParameters

    train = data["train"]
    angles_rad = np.asarray(train["angles"], dtype=np.float64)
    projs = np.asarray(train["projections"], dtype=np.float32)  # (N, H, W)
    vol_size = data["nVoxel"][0]

    # GENFIRE expects projections as (H, W, N) and a square detector matching vol_size
    if projs.shape[1] != vol_size or projs.shape[2] != vol_size:
        from skimage.transform import resize
        projs = np.stack([resize(p, (vol_size, vol_size), preserve_range=True)
                          for p in projs]).astype(np.float32)
    projs_hwn = np.moveaxis(projs, 0, -1).astype(np.float32)

    # Single-axis y-tilt → ZXZ Euler (phi=0, theta=tilt_deg, psi=0).
    angles_deg = np.degrees(angles_rad)
    # GENFIRE Euler convention: phi (col 0) is the dominant single-tilt angle
    # for most published AET datasets (Pryor 2017 supplementary).
    eulers = np.zeros((len(angles_deg), 3), dtype=np.float64)
    eulers[:, 0] = angles_deg

    work = Path(tempfile.mkdtemp(prefix="genfire_"))
    try:
        proj_path = work / "projections.npy"
        ang_path = work / "angles.npy"
        out_path = work / "result.mrc"
        np.save(proj_path, projs_hwn)
        np.save(ang_path, eulers)

        params = ReconstructionParameters()
        params.projections = str(proj_path)
        params.eulerAngles = str(ang_path)
        params.numIterations = n_iter
        params.oversamplingRatio = oversampling
        params.interpolationCutoffDistance = 0.5
        params.resultsFilename = str(out_path)
        params.useDefaultSupport = True
        params.calculateRfree = False
        params.griddingMethod = "FFT"
        params.constraintPositivity = True
        params.constraintSupport = True
        params.enforceResolutionCircle = True
        params.permitMultipleGridding = True
        # Disable interactive figure
        params.displayFigure.DisplayFigureON = False
        params.displayFigure.DisplayErrorFigureON = False

        gf_main(params)

        vol = genfire.fileio.readVolume(str(out_path))
        # readVolume may return padded volume; crop to nVoxel
        if vol.shape[0] != vol_size:
            pad = (vol.shape[0] - vol_size) // 2
            vol = vol[pad:pad + vol_size, pad:pad + vol_size, pad:pad + vol_size]
        return vol.astype(np.float32)
    finally:
        shutil.rmtree(work, ignore_errors=True)
