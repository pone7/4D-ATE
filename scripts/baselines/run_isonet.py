"""IsoNet baseline for AET-Bench (Gerchberg–Papoulis fallback).

Rationale: the upstream IsoNet (Liu 2022, Nat Commun) is a
TensorFlow/GPU pipeline designed for cryo-electron tomography, with a
multi-step workflow (prepare_star → deconv → extract → refine → predict)
that requires TF 2.3 + CUDA 10 — incompatible with the current GPU
driver in this environment. Proper integration is out of scope for the
time budget.

IsoNet's scientific role in the comparison is: "deep-learning based
post-processing of an existing reconstruction that corrects the
missing-wedge / anisotropic-resolution artifact." We preserve that role
with a training-free Gerchberg–Papoulis iterative Fourier inpainting
applied to our SIRT output. This is the classic pre-DL missing-wedge
filler; its numbers are a floor that DL methods like IsoNet are
expected to beat. We label it "IsoFill (GP)" in the paper table with a
footnote explaining the substitution.

Limitation: this is NOT the IsoNet DL model. It is a missing-wedge
correction of the same *category* (post-processing an existing
reconstruction). Include it as a baseline; expect it to be weaker than
true DL approaches. If/when IsoNet integration becomes feasible, this
file can be replaced without touching run_benchmark.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np

FALLBACK_USED = True


def _missing_wedge_mask(shape, tilt_range_deg: float) -> np.ndarray:
    """Binary Fourier mask marking the populated wedge in k-space.

    Single-axis tilt around y: rotation in the x–z plane. Frequencies
    inside the double cone around the missing axis (ky-axis) are
    unpopulated. This returns 1.0 where data exists, 0.0 where missing.
    """
    nz, ny, nx = shape
    kz = np.fft.fftshift(np.fft.fftfreq(nz))
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    # Angle relative to the kz-kx plane (the tilt plane):
    theta = np.arctan2(np.abs(KZ), np.abs(KX) + 1e-12)  # 0..π/2
    cutoff = np.deg2rad(tilt_range_deg)
    return (theta <= cutoff).astype(np.float32)


def _sirt_recon(data: Dict) -> np.ndarray:
    """Inline copy of the SIRT method from run_benchmark to avoid a circular
    import."""
    from skimage.transform import iradon_sart, resize
    train = data["train"]
    angles_deg = np.degrees(np.asarray(train["angles"]))
    projs = np.asarray(train["projections"], dtype=np.float32)
    vol_size = int(data["nVoxel"][0])
    if projs.shape[1] != vol_size or projs.shape[2] != vol_size:
        projs = np.stack([resize(p, (vol_size, vol_size), preserve_range=True)
                          for p in projs]).astype(np.float32)
    vol = np.zeros((vol_size, vol_size, vol_size), dtype=np.float32)
    for y in range(vol_size):
        sino = projs[:, y, :].T
        recon = None
        for _ in range(10):
            recon = iradon_sart(sino, theta=angles_deg, image=recon)
        vol[:, y, :] = recon.astype(np.float32)
    return vol


def run_isonet(data: Dict, sample_id: str, work_root: Path,
                n_iter: int = 20) -> np.ndarray:
    """SIRT + Gerchberg–Papoulis missing-wedge fill."""
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    print(f"  [isonet/gp] computing SIRT seed for {sample_id}")
    vol = _sirt_recon(data)

    # Derive tilt-range from training angles
    angles_rad = np.asarray(data["train"]["angles"], dtype=np.float32)
    tilt_range_deg = float(np.degrees(np.abs(angles_rad).max()))
    print(f"  [isonet/gp] tilt range ±{tilt_range_deg:.1f}°, {n_iter} GP iter")

    mask = _missing_wedge_mask(vol.shape, tilt_range_deg)
    # Iterative Fourier inpainting: keep measured frequencies, enforce
    # positivity in real space. Converges to a MW-filled reconstruction.
    current = vol.astype(np.float32).copy()
    measured_kspace = np.fft.fftshift(np.fft.fftn(current)) * mask
    for _ in range(n_iter):
        k = np.fft.fftshift(np.fft.fftn(current))
        k = measured_kspace + k * (1.0 - mask)
        current = np.real(np.fft.ifftn(np.fft.ifftshift(k))).astype(np.float32)
        current = np.clip(current, 0.0, None)

    return current
