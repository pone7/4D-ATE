#!/usr/bin/env python
"""
Unified evaluation framework for AET reconstruction methods.

Metrics:
1. Volume PSNR / SSIM (3D volume vs GT)
2. Atom RMSD (Hungarian matching of detected peaks vs GT atoms)
3. Recall / Precision at threshold (e.g., 0.5 Angstrom)
4. Element accuracy (for multi-element systems)

Usage:
    python evaluate_baselines.py --vol_pred pred.npy --vol_gt gt.npy \
        --atoms_gt atoms_gt.npy --method_name "FBP"
"""

import numpy as np
import argparse
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
from skimage.metrics import structural_similarity as ssim_metric


# ---------- Volume metrics ----------

def volume_psnr(vol_gt, vol_pred):
    """Compute PSNR between two volumes."""
    mse = np.mean((vol_gt - vol_pred) ** 2)
    if mse < 1e-10:
        return float('inf')
    max_val = vol_gt.max()
    return 10 * np.log10(max_val ** 2 / mse)


def volume_ssim(vol_gt, vol_pred):
    """Compute mean SSIM across slices."""
    ssim_vals = []
    for axis in range(3):
        n_slices = vol_gt.shape[axis]
        for i in range(n_slices):
            if axis == 0:
                gt_slice = vol_gt[i, :, :]
                pred_slice = vol_pred[i, :, :]
            elif axis == 1:
                gt_slice = vol_gt[:, i, :]
                pred_slice = vol_pred[:, i, :]
            else:
                gt_slice = vol_gt[:, :, i]
                pred_slice = vol_pred[:, :, i]

            if gt_slice.max() < 1e-8:
                continue
            data_range = gt_slice.max() - gt_slice.min()
            if data_range < 1e-8:
                continue
            s = ssim_metric(gt_slice, pred_slice, data_range=data_range)
            ssim_vals.append(s)
    return np.mean(ssim_vals) if ssim_vals else 0.0


# ---------- Atom detection ----------

def detect_atoms_from_volume(volume, min_distance=3, threshold_rel=0.05,
                              num_expected=None):
    """
    Detect atomic positions from a reconstructed 3D volume.

    Uses peak_local_max for local maxima detection.

    Args:
        volume: (Nx, Ny, Nz) 3D array
        min_distance: minimum distance between peaks (voxels)
        threshold_rel: relative threshold for peak detection
        num_expected: if set, adjust threshold to get close to this count

    Returns:
        peaks_voxel: (M, 3) peak positions in voxel coordinates
    """
    # Normalize volume
    vol_norm = volume / (volume.max() + 1e-10)

    # Scale min_distance for large volumes to keep peak detection tractable
    vol_size = volume.shape[0]
    if vol_size > 192:
        min_distance = max(min_distance, vol_size // 48)

    # Raise threshold for near-uniform volumes to avoid finding millions of peaks
    mean_norm = vol_norm.mean()
    if mean_norm > 0.3:
        threshold_rel = max(threshold_rel, mean_norm * 0.8)

    if num_expected is not None and num_expected <= 500:
        # Binary search only for small atom counts (fast)
        lo, hi = 0.01, 0.5
        for _ in range(15):
            mid = (lo + hi) / 2
            peaks = peak_local_max(vol_norm, min_distance=min_distance,
                                    threshold_rel=mid)
            if len(peaks) > num_expected * 1.5:
                lo = mid
            elif len(peaks) < num_expected * 0.5:
                hi = mid
            else:
                break
        threshold_rel = mid

    # Single call with fixed threshold for large counts or no num_expected
    peaks = peak_local_max(vol_norm, min_distance=min_distance,
                            threshold_rel=threshold_rel)
    return peaks


def voxel_to_physical(peaks_voxel, vol_shape, cell_size):
    """Convert voxel coordinates to physical coordinates (centered)."""
    half = cell_size / 2
    physical = peaks_voxel.astype(float) / np.array(vol_shape) * cell_size - half
    return physical


# ---------- Atom matching ----------

def _greedy_match(pred_positions, gt_positions, max_dist=None):
    """Greedy nearest-neighbor matching for large atom counts (>5000).

    For each GT atom, find nearest unmatched predicted atom. O(N*M) with
    KD-tree, tractable up to ~100K atoms.
    """
    from scipy.spatial import cKDTree
    tree_pred = cKDTree(pred_positions)
    tree_gt = cKDTree(gt_positions)

    matched_pred, matched_gt, distances = [], [], []
    used_pred = set()

    # Query: for each GT atom, find closest pred atoms
    dists_all, idxs_all = tree_pred.query(gt_positions, k=min(10, len(pred_positions)))
    if dists_all.ndim == 1:
        dists_all = dists_all[:, None]
        idxs_all = idxs_all[:, None]

    # Sort GT atoms by nearest distance (match closest first)
    gt_order = np.argsort(dists_all[:, 0])
    for gi in gt_order:
        for ki in range(dists_all.shape[1]):
            pi = idxs_all[gi, ki]
            d = dists_all[gi, ki]
            if pi not in used_pred and (max_dist is None or d <= max_dist):
                matched_pred.append(int(pi))
                matched_gt.append(int(gi))
                distances.append(float(d))
                used_pred.add(pi)
                break

    unmatched_pred = [i for i in range(len(pred_positions)) if i not in used_pred]
    unmatched_gt = [i for i in range(len(gt_positions)) if i not in set(matched_gt)]
    return matched_pred, matched_gt, np.array(distances), unmatched_pred, unmatched_gt


def match_atoms_hungarian(pred_positions, gt_positions, max_dist=None):
    """
    Match predicted atoms to GT atoms using Hungarian algorithm.
    Falls back to greedy matching when either side has >5000 atoms
    (Hungarian is O(n³) and infeasible for large counts).

    Args:
        pred_positions: (M, 3) predicted atom positions
        gt_positions: (N, 3) ground truth atom positions
        max_dist: maximum allowed matching distance

    Returns:
        matched_pred: indices of matched predicted atoms
        matched_gt: indices of matched GT atoms
        distances: distances of matched pairs
        unmatched_pred: indices of unmatched predicted atoms
        unmatched_gt: indices of unmatched GT atoms
    """
    if len(pred_positions) == 0 or len(gt_positions) == 0:
        return [], [], [], list(range(len(pred_positions))), list(range(len(gt_positions)))

    # Greedy fallback for large atom counts (Hungarian is O(n³))
    if len(pred_positions) > 2000 or len(gt_positions) > 2000:
        return _greedy_match(pred_positions, gt_positions, max_dist)

    cost_matrix = cdist(pred_positions, gt_positions)

    # Hungarian matching
    pred_idx, gt_idx = linear_sum_assignment(cost_matrix)

    # Filter by max distance
    distances = cost_matrix[pred_idx, gt_idx]
    if max_dist is not None:
        valid = distances <= max_dist
        pred_idx = pred_idx[valid]
        gt_idx = gt_idx[valid]
        distances = distances[valid]

    matched_pred = list(pred_idx)
    matched_gt = list(gt_idx)
    unmatched_pred = [i for i in range(len(pred_positions)) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(gt_positions)) if i not in matched_gt]

    return matched_pred, matched_gt, distances, unmatched_pred, unmatched_gt


def compute_atom_metrics(pred_positions, gt_positions, threshold=0.5):
    """
    Compute atom-level metrics.

    Args:
        pred_positions: (M, 3) predicted positions (Angstroms)
        gt_positions: (N, 3) ground truth positions (Angstroms)
        threshold: distance threshold for recall/precision (Angstroms)

    Returns:
        metrics dict
    """
    if len(pred_positions) == 0:
        return {
            'rmsd': float('inf'),
            'recall': 0.0,
            'precision': 0.0,
            'f1': 0.0,
            'n_pred': 0,
            'n_gt': len(gt_positions),
            'n_matched': 0,
        }

    # Match with threshold
    matched_pred, matched_gt, distances, unmatched_pred, unmatched_gt = \
        match_atoms_hungarian(pred_positions, gt_positions, max_dist=threshold * 3)

    # RMSD of matched pairs
    if len(distances) > 0:
        rmsd = np.sqrt(np.mean(distances ** 2))
    else:
        rmsd = float('inf')

    # Recall & Precision at threshold
    within_threshold = (distances <= threshold).sum() if len(distances) > 0 else 0
    recall = within_threshold / len(gt_positions) if len(gt_positions) > 0 else 0
    precision = within_threshold / len(pred_positions) if len(pred_positions) > 0 else 0
    f1 = 2 * recall * precision / (recall + precision + 1e-10)

    return {
        'rmsd': float(rmsd),
        'recall': float(recall),
        'precision': float(precision),
        'f1': float(f1),
        'n_pred': len(pred_positions),
        'n_gt': len(gt_positions),
        'n_matched': int(within_threshold),
        'mean_dist': float(distances.mean()) if len(distances) > 0 else float('inf'),
        'median_dist': float(np.median(distances)) if len(distances) > 0 else float('inf'),
    }


# ---------- Full evaluation ----------

def evaluate_reconstruction(vol_pred, vol_gt, atoms_gt, cell_size,
                             method_name="Unknown", threshold=0.5,
                             min_peak_distance=3):
    """
    Full evaluation of a reconstruction method.

    Args:
        vol_pred: (Nx, Ny, Nz) reconstructed volume
        vol_gt: (Nx, Ny, Nz) ground truth volume
        atoms_gt: (N, 4) GT atoms [x, y, z, Z] in Angstroms
        cell_size: physical size of volume in Angstroms
        method_name: name for display
        threshold: atom matching threshold in Angstroms
        min_peak_distance: min distance between peaks in voxels

    Returns:
        results dict
    """
    # Normalize pred to GT scale
    if vol_pred.max() > 0:
        scale = vol_gt.max() / vol_pred.max()
        vol_pred_scaled = vol_pred * scale
    else:
        vol_pred_scaled = vol_pred

    # Volume metrics
    psnr = volume_psnr(vol_gt, vol_pred_scaled)
    ssim = volume_ssim(vol_gt, vol_pred_scaled)

    # Atom detection — skip for very large volumes to avoid excessive runtime
    gt_positions = atoms_gt[:, :3]
    n_gt = len(gt_positions)
    vol_size = vol_pred.shape[0]

    # Skip atom detection when volume is too large or too uniform
    vol_dynamic_range = (vol_pred.max() - vol_pred.min()) / (vol_pred.mean() + 1e-10)
    skip_atoms = vol_size > 192 or vol_dynamic_range < 0.5

    if skip_atoms:
        atom_metrics = {
            'rmsd': float('nan'), 'recall': 0.0, 'precision': 0.0,
            'f1': 0.0, 'n_pred': 0, 'n_gt': n_gt, 'n_matched': 0,
        }
    else:
        peaks_voxel = detect_atoms_from_volume(
            vol_pred, min_distance=min_peak_distance,
            num_expected=n_gt
        )
        pred_positions = voxel_to_physical(peaks_voxel, vol_pred.shape, cell_size)
        atom_metrics = compute_atom_metrics(pred_positions, gt_positions, threshold)

    results = {
        'method': method_name,
        'vol_psnr': psnr,
        'vol_ssim': ssim,
        **atom_metrics,
    }

    return results


def print_results(results):
    """Print results in a readable format."""
    print(f"\n{'='*50}")
    print(f"Method: {results['method']}")
    print(f"{'='*50}")
    print(f"Volume PSNR:  {results['vol_psnr']:.2f} dB")
    print(f"Volume SSIM:  {results['vol_ssim']:.4f}")
    print(f"Atom RMSD:    {results['rmsd']:.3f} Å")
    print(f"Recall@{0.5}Å:  {results['recall']:.1%} ({results['n_matched']}/{results['n_gt']})")
    print(f"Precision:    {results['precision']:.1%} ({results['n_matched']}/{results['n_pred']})")
    print(f"F1 Score:     {results['f1']:.3f}")
    print(f"Mean Dist:    {results.get('mean_dist', 'N/A')}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vol_pred", type=str, required=True)
    parser.add_argument("--vol_gt", type=str, required=True)
    parser.add_argument("--atoms_gt", type=str, required=True)
    parser.add_argument("--cell_size", type=float, default=40.0)
    parser.add_argument("--method_name", type=str, default="Unknown")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    vol_pred = np.load(args.vol_pred)
    vol_gt = np.load(args.vol_gt)
    atoms_gt = np.load(args.atoms_gt)

    results = evaluate_reconstruction(
        vol_pred, vol_gt, atoms_gt, args.cell_size,
        args.method_name, args.threshold
    )
    print_results(results)
