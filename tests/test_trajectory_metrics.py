"""Tests for scripts/metrics/trajectory.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from metrics.trajectory import trajectory_rmsd, atom_identity_persistence


def _make_identity_traj(n_phases=10, n_atoms=20, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(-5, 5, size=(n_atoms, 3))
    return np.broadcast_to(base, (n_phases, n_atoms, 3)).copy()


def test_identity_trajectory_zero_rmsd():
    gt = _make_identity_traj()
    pred = gt.copy()
    rmsd = trajectory_rmsd(pred, gt)
    assert rmsd == pytest.approx(0.0, abs=1e-6)


def test_identity_persistence_full():
    gt = _make_identity_traj()
    pred = gt.copy()
    p = atom_identity_persistence(pred, gt)
    assert p == pytest.approx(1.0)


def test_linear_drift_rmsd():
    gt = _make_identity_traj(n_phases=10, n_atoms=5, seed=1)
    # Predicted Gaussians are shifted by exactly 0.3 Å in +x at every phase.
    pred = gt.copy()
    pred[..., 0] += 0.3
    rmsd = trajectory_rmsd(pred, gt)
    assert rmsd == pytest.approx(0.3, abs=1e-6)


def test_rmsd_uses_root_mean_square_not_mean():
    """Two atoms shifted by 0 and 1 Å in +x: mean = 0.5, RMSD = sqrt(0.5) ≈ 0.7071."""
    n_phases = 4
    n_atoms = 2
    base = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    gt = np.broadcast_to(base, (n_phases, n_atoms, 3)).copy()
    pred = gt.copy()
    pred[..., 1, 0] += 1.0  # shift only the second atom by 1 Å
    rmsd = trajectory_rmsd(pred, gt)
    assert rmsd == pytest.approx(np.sqrt(0.5), abs=1e-6)


def test_swap_breaks_persistence():
    gt = _make_identity_traj(n_phases=2, n_atoms=4, seed=2)
    pred = gt.copy()
    # Swap atom 0 and 1 in phase 1 → identity persistence drops.
    pred[1, [0, 1]] = pred[1, [1, 0]]
    p = atom_identity_persistence(pred, gt)
    assert p == pytest.approx(0.5, abs=1e-6)  # 2 of 4 atoms still consistent


def test_extra_predicted_gaussians_dont_break_matching():
    gt = _make_identity_traj(n_phases=3, n_atoms=5, seed=3)
    extra = np.random.default_rng(99).uniform(-10, 10, size=(3, 100, 3))
    pred = np.concatenate([gt, extra], axis=1)  # 5 real + 100 noise
    rmsd = trajectory_rmsd(pred, gt)
    assert rmsd == pytest.approx(0.0, abs=1e-6)
