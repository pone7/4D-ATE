"""Trajectory metrics for 4D atomic reconstruction.

These exploit the identity-preserving property of Gaussian splatting:
the Gaussian at index i represents the same physical entity across phases.

Inputs are arrays of shape (n_phases, n_pred, 3) and (n_phases, n_gt, 3).
n_pred can be >= n_gt; matching uses the GT-as-query direction so noise
Gaussians do not corrupt the metric.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _match_at_phase0(pred_xyz_4d: np.ndarray,
                      gt_xyz_4d: np.ndarray) -> np.ndarray:
    """Hungarian matching at phase 0. Returns indices into pred for each GT atom.

    Both inputs are (n_phases, N, 3); only phase 0 is used here. The result is
    an array of length n_gt giving the chosen pred-Gaussian index per GT atom.
    """
    pred0 = pred_xyz_4d[0]
    gt0 = gt_xyz_4d[0]
    cost = cdist(gt0, pred0)  # (n_gt, n_pred)
    # GT atoms are rows; we pick one pred per GT (allow pred reuse only if
    # n_pred < n_gt, which we don't expect).
    if pred0.shape[0] >= gt0.shape[0]:
        row_ind, col_ind = linear_sum_assignment(cost)
        # Ensure outputs are ordered by GT index.
        order = np.argsort(row_ind)
        return col_ind[order]
    # Degenerate: more GT than pred. Each GT just picks its closest pred.
    return cost.argmin(axis=1)


def trajectory_rmsd(pred_xyz_4d: np.ndarray,
                     gt_xyz_4d: np.ndarray) -> float:
    """Root mean square deviation between matched-Gaussian trajectories and GT
    atom trajectories. RMSD = sqrt(mean(||pred_t,i - gt_t,i||^2)) over all
    phases t and matched atoms i.
    """
    if pred_xyz_4d.ndim != 3 or gt_xyz_4d.ndim != 3:
        raise ValueError("Both inputs must be (n_phases, N, 3).")
    if pred_xyz_4d.shape[0] != gt_xyz_4d.shape[0]:
        raise ValueError("Phase counts must match.")

    matched_idx = _match_at_phase0(pred_xyz_4d, gt_xyz_4d)
    matched_traj = pred_xyz_4d[:, matched_idx, :]   # (n_phases, n_gt, 3)
    diffs = matched_traj - gt_xyz_4d                # (n_phases, n_gt, 3)
    sq_dist = np.sum(diffs ** 2, axis=-1)           # (n_phases, n_gt)
    return float(np.sqrt(sq_dist.mean()))


def atom_identity_persistence(pred_xyz_4d: np.ndarray,
                                gt_xyz_4d: np.ndarray) -> float:
    """Fraction of GT atoms whose phase-0 Gaussian remains the closest pred to
    the GT atom in *every* later phase.
    """
    matched_idx = _match_at_phase0(pred_xyz_4d, gt_xyz_4d)
    n_phases, n_gt = gt_xyz_4d.shape[0], gt_xyz_4d.shape[1]
    n_persistent = 0
    for g in range(n_gt):
        chosen = matched_idx[g]
        ok = True
        for t in range(n_phases):
            dists_t = np.linalg.norm(pred_xyz_4d[t] - gt_xyz_4d[t, g], axis=-1)
            if dists_t.argmin() != chosen:
                ok = False
                break
        if ok:
            n_persistent += 1
    return n_persistent / max(1, n_gt)
