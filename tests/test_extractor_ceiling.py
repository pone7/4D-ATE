"""Guard against extractor mis-tuning that merges neighbouring atoms.

Regression test for the NeurIPS 3869 revision. The shared peak extractor
must recover essentially all atoms when run on the *ground-truth* volume;
otherwise the harness — not the reconstruction method — caps atom recall,
which is precisely the failure mode reviewers flagged. This test builds a
close-packed FCC lattice (nearest-neighbour ~2.6 A), rasterizes it as the
GT density would be, and asserts the pinned extraction setting recovers
>= 95% of atoms. It also asserts the mis-tuned 2.0 A window (the previous
regression) does NOT, so the guard is meaningful.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analysis.extraction_sensitivity import (  # noqa: E402
    _normalise,
    _tune_threshold_for_count,
    compute_atom_metrics,
    voxel_to_physical,
)


def _fcc_gt_volume(vol_size: int = 128, cell_ang: float = 24.0, a: float = 3.6):
    """Rasterize an FCC nanocrystal into a Gaussian-blob density volume.

    a=3.6 A gives d_NN = a/sqrt(2) ~2.55 A (Cu-like close packing).
    """
    fcc_basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    atoms = []
    reach = 2
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            for k in range(-reach, reach + 1):
                for b in fcc_basis:
                    pos = (np.array([i, j, k]) + b) * a
                    if np.linalg.norm(pos) <= 7.0:  # ~7 A radius nanoparticle
                        atoms.append(pos)
    atoms = np.unique(np.round(np.asarray(atoms), 4), axis=0)

    vol = np.zeros((vol_size,) * 3, dtype=np.float32)
    voxel = cell_ang / vol_size
    sigma_vox = 0.5 / voxel  # 0.5 A blobs
    centers_vox = (atoms + cell_ang / 2.0) / voxel
    rad = int(np.ceil(3 * sigma_vox))
    for c in centers_vox:
        ci = np.round(c).astype(int)
        lo = np.maximum(ci - rad, 0)
        hi = np.minimum(ci + rad + 1, vol_size)
        gx, gy, gz = np.meshgrid(
            np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), np.arange(lo[2], hi[2]),
            indexing="ij",
        )
        vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += np.exp(
            -(((gx - c[0]) ** 2 + (gy - c[1]) ** 2 + (gz - c[2]) ** 2) / (2 * sigma_vox ** 2))
        ).astype(np.float32)
    return vol, atoms, cell_ang


def _gt_self_recall(vol, atoms, cell, min_dist_ang, tau=0.5):
    vn = _normalise(vol)
    voxel = cell / max(vn.shape)
    md = max(1, int(round(min_dist_ang / voxel)))
    peaks, _ = _tune_threshold_for_count(vn, md, len(atoms), 0.05)
    pred = voxel_to_physical(peaks, vn.shape, cell)
    return compute_atom_metrics(pred, atoms[:, :3], tau)["recall"]


def test_pinned_extractor_recovers_gt_atoms():
    vol, atoms, cell = _fcc_gt_volume()
    recall = _gt_self_recall(vol, atoms, cell, min_dist_ang=1.0)
    assert recall >= 0.95, f"pinned 1.0 A extractor GT recall too low: {recall:.3f}"


def test_mistuned_window_merges_atoms():
    # The previous regression (2.0 A window) must visibly fail on the same GT,
    # otherwise the ceiling guard above would be vacuous.
    vol, atoms, cell = _fcc_gt_volume()
    recall = _gt_self_recall(vol, atoms, cell, min_dist_ang=2.0)
    assert recall < 0.6, f"2.0 A window unexpectedly fine ({recall:.3f}); guard is vacuous"
