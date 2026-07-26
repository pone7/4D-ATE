"""Fourier Shell Correlation between two 3D volumes.

FSC(k) = <F1(k) · conj(F2(k))> / sqrt(<|F1(k)|^2> · <|F2(k)|^2>)

Returns the FSC curve plus the resolution at conventional thresholds 0.143
(gold-standard cryo-EM) and 0.5. Resolution is reported in voxels (or in Å
if `voxel_size_angstrom` is supplied).
"""
from __future__ import annotations

import numpy as np


def _radial_bins(shape):
    nz, ny, nx = shape
    kz = np.fft.fftfreq(nz) * nz
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.fftfreq(nx) * nx
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    r = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    return r


def fourier_shell_correlation(vol1: np.ndarray, vol2: np.ndarray):
    """Compute the radially-averaged FSC curve.

    Returns
    -------
    k : (n_shells,) shell index (integer voxel frequency)
    fsc : (n_shells,) correlation per shell
    """
    assert vol1.shape == vol2.shape, "volumes must match shape"
    F1 = np.fft.fftn(vol1)
    F2 = np.fft.fftn(vol2)
    r = _radial_bins(vol1.shape)
    r_int = r.astype(np.int32)
    n_shells = int(min(vol1.shape) // 2)

    num = np.bincount(r_int.ravel(),
                      weights=np.real(F1 * np.conj(F2)).ravel(),
                      minlength=n_shells + 1)
    p1 = np.bincount(r_int.ravel(), weights=(np.abs(F1) ** 2).ravel(),
                     minlength=n_shells + 1)
    p2 = np.bincount(r_int.ravel(), weights=(np.abs(F2) ** 2).ravel(),
                     minlength=n_shells + 1)
    denom = np.sqrt(p1 * p2) + 1e-12
    fsc = (num / denom)[:n_shells]
    k = np.arange(n_shells)
    return k, fsc


def resolution_at_threshold(k, fsc, threshold, voxel_size_angstrom=None,
                             vol_size=None):
    """Find the first shell where FSC drops below `threshold`.

    Returns the resolution in voxels (period = vol_size / k_cross), or in Å
    if voxel_size_angstrom and vol_size are provided.
    """
    below = np.where(fsc < threshold)[0]
    if len(below) == 0:
        k_cross = float(k[-1])
    else:
        k_cross = float(below[0])
    if k_cross < 1:
        return float("inf")
    if voxel_size_angstrom is not None and vol_size is not None:
        return (vol_size * voxel_size_angstrom) / k_cross
    return vol_size / k_cross if vol_size is not None else 1.0 / k_cross


def fsc_summary(vol_recon, vol_gt, voxel_size_angstrom=None):
    """Compute FSC and report resolutions at 0.5 and 0.143."""
    vol_size = vol_gt.shape[0]
    k, fsc = fourier_shell_correlation(vol_recon, vol_gt)
    return {
        "fsc_curve": fsc.tolist(),
        "k": k.tolist(),
        "resolution_0.5": resolution_at_threshold(
            k, fsc, 0.5, voxel_size_angstrom, vol_size),
        "resolution_0.143": resolution_at_threshold(
            k, fsc, 0.143, voxel_size_angstrom, vol_size),
    }


if __name__ == "__main__":
    # Self-test: identical volumes -> FSC = 1; shifted -> FSC drops
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 32, 32)).astype(np.float32)
    b = a.copy()
    k, fsc = fourier_shell_correlation(a, b)
    assert np.all(fsc[1:] > 0.999), "identical volumes should give FSC≈1"

    b_shift = np.roll(a, shift=2, axis=0)
    _, fsc_shift = fourier_shell_correlation(a, b_shift)
    assert fsc_shift[-1] < fsc[-1] - 0.1, "shifted volume should reduce FSC"
    print("FSC self-test passed.")
    print(f"  identical FSC mean: {fsc.mean():.4f}")
    print(f"  shifted   FSC mean: {fsc_shift.mean():.4f}")
