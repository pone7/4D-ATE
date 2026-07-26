"""Tests for representation-aware (native) atom extraction + sub-voxel refinement.

Reviewer requirement (NeurIPS 3869, j8nP headline weakness + WQbd question):
Gaussian / implicit methods should be given an extractor matched to their own
output type, and sub-voxel peak refinement should be tested, to show whether
the SSIM-vs-recall inversion is a property of the methods or the harness.
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
    _peak_local_max_nd,
    refine_subvoxel_com,
)
from analysis.native_extraction import extract_atoms_from_gaussians  # noqa: E402


def _offset_blob_volume(center_vox: tuple[float, float, float]) -> np.ndarray:
    grid = np.arange(32, dtype=float)
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")
    cx, cy, cz = center_vox
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) / (2.0 * 1.2 ** 2)).astype(np.float32)


def test_subvoxel_refinement_beats_integer_peak():
    true_center = (15.6, 16.4, 16.0)
    vol = _normalise(_offset_blob_volume(true_center))
    peaks = _peak_local_max_nd(vol, min_distance=3, threshold_rel=0.2)
    assert len(peaks) >= 1
    truth = np.asarray(true_center)
    int_err = np.min(np.linalg.norm(peaks.astype(float) - truth, axis=1))
    refined = refine_subvoxel_com(vol, peaks, radius=2)
    ref_err = np.min(np.linalg.norm(refined - truth, axis=1))
    assert ref_err < int_err
    assert ref_err < 0.4  # sub-voxel accuracy on a clean blob


def test_native_extraction_merges_primitives_per_atom():
    # Three Gaussian primitives clustered around each of two atoms, in the
    # normalized [-1, 1] cube frame Gaussian-splatting checkpoints use.
    cell = 20.0
    atoms_phys = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    atoms_norm = atoms_phys / (cell / 2.0)
    rng = np.random.default_rng(0)
    xyz = []
    density = []
    for a in atoms_norm:
        for _ in range(3):
            xyz.append(a + rng.normal(scale=0.01, size=3))
            density.append(5.0)
    # background low-density primitives
    for _ in range(20):
        xyz.append(rng.uniform(-1, 1, size=3))
        density.append(0.2)
    xyz = np.asarray(xyz)
    density = np.asarray(density)

    pred = extract_atoms_from_gaussians(
        xyz,
        density,
        cell_size_angstrom=cell,
        mode="oracle_count",
        num_expected=2,
        merge_radius_angstrom=1.5,
    )
    assert len(pred) == 2
    # both atoms recovered within tau = 0.5 A
    from scipy.spatial.distance import cdist

    dists = cdist(pred, atoms_phys)
    matched = (dists.min(axis=1) <= 0.5).sum()
    assert matched == 2
