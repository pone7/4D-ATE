#!/usr/bin/env python
"""
Run all baseline reconstruction methods and compare with AET-4DGS.

Methods:
1. FBP (scikit-image iradon) — classical filtered back-projection
2. SIRT (iterative) — if TIGRE available
3. GENFIRE — if installed
4. AET-4DGS — our method (loads from existing output)

Usage:
    python run_all_baselines.py --data_dir ../data/cu_nano \
        --our_output ../output/cu_nano_haadf_gamma \
        --our_iter 10000
"""

import numpy as np
import pickle
import argparse
import json
from pathlib import Path
from evaluate_baselines import (
    evaluate_reconstruction, print_results,
    detect_atoms_from_volume, voxel_to_physical
)


# ---------- FBP Baseline (scikit-image) ----------

def run_fbp(projections, angles_rad, vol_size=128):
    """
    3D FBP reconstruction using scikit-image iradon.
    Reconstructs slice-by-slice along each axis.

    Args:
        projections: (N_angles, H, W) projection images
        angles_rad: (N_angles,) tilt angles in radians
        vol_size: output volume size per dimension

    Returns:
        volume: (vol_size, vol_size, vol_size) reconstructed volume
    """
    from skimage.transform import iradon

    angles_deg = np.degrees(angles_rad)
    n_angles, det_h, det_w = projections.shape

    # FBP reconstructs 2D slices from a sinogram
    # For parallel beam tilt around Y axis:
    # - Each row of projections forms a sinogram for one Z-slice
    # - sinogram[angle_idx, pixel_x] -> reconstruct XZ plane at fixed Y

    # We reconstruct the XZ plane for each Y position
    volume = np.zeros((vol_size, vol_size, vol_size), dtype=np.float32)

    # Resize projections to match vol_size if needed
    if det_w != vol_size or det_h != vol_size:
        from skimage.transform import resize
        projs_resized = np.zeros((n_angles, vol_size, vol_size), dtype=np.float32)
        for i in range(n_angles):
            projs_resized[i] = resize(projections[i], (vol_size, vol_size),
                                       preserve_range=True)
        projections = projs_resized

    # For each Y slice, create sinogram and reconstruct XZ plane
    for y in range(vol_size):
        # sinogram: (n_angles, det_w) — the y-th row from each projection
        sinogram = projections[:, y, :]  # (n_angles, vol_size)

        # iradon expects sinogram as (n_pixels, n_angles)
        sino_t = sinogram.T  # (vol_size, n_angles)

        recon_slice = iradon(sino_t, theta=angles_deg,
                              output_size=vol_size, filter_name='ramp')
        volume[:, y, :] = recon_slice

    # Clip negative values
    volume = np.maximum(volume, 0)
    return volume


# ---------- SIRT Baseline (simple iterative) ----------

def run_sirt_simple(projections, angles_rad, vol_size=128, n_iter=50):
    """
    Simple SIRT-like iterative reconstruction.
    Uses forward/back projection via Radon transform.

    Args:
        projections: (N_angles, H, W) projection images
        angles_rad: (N_angles,) tilt angles in radians
        vol_size: output volume size
        n_iter: number of iterations

    Returns:
        volume: (vol_size, vol_size, vol_size) reconstructed volume
    """
    from skimage.transform import iradon, radon

    angles_deg = np.degrees(angles_rad)
    n_angles = len(angles_rad)

    # Resize projections
    if projections.shape[1] != vol_size or projections.shape[2] != vol_size:
        from skimage.transform import resize
        projs = np.zeros((n_angles, vol_size, vol_size), dtype=np.float32)
        for i in range(n_angles):
            projs[i] = resize(projections[i], (vol_size, vol_size),
                               preserve_range=True)
    else:
        projs = projections.copy()

    # Initialize with FBP
    volume = run_fbp(projs, angles_rad, vol_size)

    # SIRT iterations (slice-by-slice)
    print(f"  Running SIRT ({n_iter} iterations)...")
    for it in range(n_iter):
        for y in range(vol_size):
            # Forward project current estimate
            xz_slice = volume[:, y, :]
            sino_fwd = radon(xz_slice, theta=angles_deg)

            # Compute residual
            sino_gt = projs[:, y, :].T  # (vol_size, n_angles)
            # Resize sino_fwd to match
            if sino_fwd.shape[0] != sino_gt.shape[0]:
                from skimage.transform import resize as resize2
                sino_fwd = resize2(sino_fwd, sino_gt.shape, preserve_range=True)

            residual = sino_gt - sino_fwd

            # Back project residual
            correction = iradon(residual, theta=angles_deg,
                                 output_size=vol_size, filter_name=None)
            volume[:, y, :] += correction / (n_angles + 1e-8) * 0.5

        volume = np.maximum(volume, 0)

        if (it + 1) % 10 == 0:
            print(f"    Iter {it+1}/{n_iter}")

    return volume


# ---------- Load data ----------

def load_naf_data(pkl_path):
    """Load projection data from NAF pickle."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Combine train + val projections
    train_projs = data['train']['projections']
    train_angles = data['train']['angles']
    val_projs = data['val']['projections']
    val_angles = data['val']['angles']

    all_projs = np.concatenate([train_projs, val_projs], axis=0)
    all_angles = np.concatenate([train_angles, val_angles], axis=0)

    # Sort by angle
    sort_idx = np.argsort(all_angles)
    all_projs = all_projs[sort_idx]
    all_angles = all_angles[sort_idx]

    gt_vol = data['image']
    if gt_vol.ndim == 4:
        gt_vol = gt_vol[0]  # Take first phase for static comparison

    return all_projs, all_angles, gt_vol


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/data/aet-bench/data/cu_nano")
    parser.add_argument("--our_output", type=str,
                        default="/data/aet-bench/output/cu_nano_haadf_gamma")
    parser.add_argument("--our_iter", type=int, default=10000)
    parser.add_argument("--cell_size", type=float, default=40.0,
                        help="Physical size of volume in Angstroms")
    parser.add_argument("--vol_size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Atom matching threshold in Angstroms")
    parser.add_argument("--sirt_iter", type=int, default=30)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    our_output = Path(args.our_output)

    # Load data
    print("Loading data...")
    pkl_path = data_dir / "data.pkl"
    atoms_gt = np.load(data_dir / "atoms_gt.npy")
    all_projs, all_angles, gt_vol = load_naf_data(pkl_path)

    print(f"  Projections: {all_projs.shape}")
    print(f"  Angles: {all_angles.shape}, range [{np.degrees(all_angles.min()):.0f}, {np.degrees(all_angles.max()):.0f}]")
    print(f"  GT volume: {gt_vol.shape}")
    print(f"  GT atoms: {atoms_gt.shape}")

    all_results = []

    # ----- Method 1: FBP -----
    print("\n" + "="*60)
    print("Running FBP (scikit-image iradon)...")
    print("="*60)
    vol_fbp = run_fbp(all_projs, all_angles, args.vol_size)
    np.save(data_dir / "recon_fbp.npy", vol_fbp)
    results_fbp = evaluate_reconstruction(
        vol_fbp, gt_vol, atoms_gt, args.cell_size, "FBP", args.threshold
    )
    print_results(results_fbp)
    all_results.append(results_fbp)

    # ----- Method 2: SIRT -----
    print("\n" + "="*60)
    print(f"Running SIRT ({args.sirt_iter} iterations)...")
    print("="*60)
    vol_sirt = run_sirt_simple(all_projs, all_angles, args.vol_size, args.sirt_iter)
    np.save(data_dir / "recon_sirt.npy", vol_sirt)
    results_sirt = evaluate_reconstruction(
        vol_sirt, gt_vol, atoms_gt, args.cell_size, f"SIRT-{args.sirt_iter}", args.threshold
    )
    print_results(results_sirt)
    all_results.append(results_sirt)

    # ----- Method 3: AET-4DGS (ours) -----
    print("\n" + "="*60)
    print("Loading AET-4DGS results...")
    print("="*60)
    pc_dir = our_output / f"point_cloud/iteration_{args.our_iter}"
    vol_ours_path = pc_dir / "vol_pred_T0.npy"
    if vol_ours_path.exists():
        vol_ours = np.load(vol_ours_path)
        results_ours = evaluate_reconstruction(
            vol_ours, gt_vol, atoms_gt, args.cell_size,
            "AET-4DGS (ours)", args.threshold
        )
        print_results(results_ours)
        all_results.append(results_ours)
    else:
        print(f"  WARNING: {vol_ours_path} not found, skipping")

    # ----- Method 4: GENFIRE (if available) -----
    try:
        import genfire
        print("\n" + "="*60)
        print("Running GENFIRE...")
        print("="*60)
        # GENFIRE expects projections as (N, N, num_proj)
        projs_gf = np.transpose(all_projs, (1, 2, 0))
        angles_gf = np.degrees(all_angles)
        vol_gf = genfire.reconstruct(projs_gf, angles_gf)
        np.save(data_dir / "recon_genfire.npy", vol_gf)
        results_gf = evaluate_reconstruction(
            vol_gf, gt_vol, atoms_gt, args.cell_size, "GENFIRE", args.threshold
        )
        print_results(results_gf)
        all_results.append(results_gf)
    except ImportError:
        print("\n  GENFIRE not installed, skipping")
    except Exception as e:
        print(f"\n  GENFIRE error: {e}")

    # ----- Summary table -----
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    header = f"{'Method':<20} {'PSNR':>8} {'SSIM':>8} {'RMSD':>8} {'Recall':>8} {'Prec':>8} {'F1':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        rmsd_str = f"{r['rmsd']:.3f}" if r['rmsd'] < 100 else "N/A"
        print(f"{r['method']:<20} {r['vol_psnr']:>8.2f} {r['vol_ssim']:>8.4f} "
              f"{rmsd_str:>8} {r['recall']:>7.1%} {r['precision']:>7.1%} {r['f1']:>8.3f}")

    # Save results
    results_path = data_dir / "baseline_comparison.json"
    # Convert to serializable
    for r in all_results:
        for k, v in r.items():
            if isinstance(v, np.floating):
                r[k] = float(v)
            elif isinstance(v, np.integer):
                r[k] = int(v)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
