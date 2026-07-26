#!/usr/bin/env python
"""
Prepare real AET experimental data for AET-4DGS pipeline.

Converts Yang et al. 2017 FePt dataset (MATLAB format) to NAF pickle format.

Usage:
    python prepare_real_data.py --data_dir ../data/real/FePt_Yang2017 \
        --out_dir ../data/real/FePt_Yang2017_prepared
"""

import numpy as np
import pickle
import argparse
import json
from pathlib import Path
from scipy.io import loadmat
from skimage.transform import resize


def load_fept_data(data_dir):
    """Load Yang 2017 FePt dataset from MATLAB .mat files."""
    data_dir = Path(data_dir)

    # Load denoised projections
    proj_file = data_dir / "FePt_Denoised_Projections.mat"
    print(f"Loading projections from {proj_file}...")
    proj_data = loadmat(str(proj_file))
    print(f"  Keys: {list(proj_data.keys())}")

    # The projections are typically stored as a 3D array
    # Try common key names
    projs = None
    for key in proj_data:
        if not key.startswith('_'):
            val = proj_data[key]
            if isinstance(val, np.ndarray) and val.ndim >= 2:
                print(f"  Found array '{key}': shape={val.shape}, dtype={val.dtype}")
                if val.ndim == 3 and projs is None:
                    projs = val

    if projs is None:
        # Try to find the largest array
        max_size = 0
        for key in proj_data:
            if not key.startswith('_'):
                val = proj_data[key]
                if isinstance(val, np.ndarray) and val.size > max_size:
                    max_size = val.size
                    projs = val

    # Load tilt angles
    angle_file = data_dir / "FePt_final_Angles.mat"
    if angle_file.exists():
        print(f"Loading angles from {angle_file}...")
        angle_data = loadmat(str(angle_file))
        angles = None
        for key in angle_data:
            if not key.startswith('_'):
                val = angle_data[key]
                if isinstance(val, np.ndarray):
                    print(f"  Found '{key}': shape={val.shape}")
                    if val.size > 1 and angles is None:
                        angles = val.flatten()
    else:
        angles = None

    # Load reconstruction volume (for comparison)
    vol_file = data_dir / "FePt_reconstruction_volume.mat"
    vol = None
    if vol_file.exists():
        print(f"Loading reconstruction volume from {vol_file}...")
        vol_data = loadmat(str(vol_file))
        for key in vol_data:
            if not key.startswith('_'):
                val = vol_data[key]
                if isinstance(val, np.ndarray) and val.ndim == 3:
                    print(f"  Found volume '{key}': shape={val.shape}")
                    vol = val

    # Load final atomic model
    atom_coords = None
    model_dir = data_dir / "Final_Model"
    if not model_dir.exists():
        # Try unzipping
        zip_file = data_dir / "Final_Model.zip"
        if zip_file.exists():
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(data_dir)
            print(f"Extracted Final_Model.zip")

    if model_dir.exists():
        print(f"Looking for atomic coordinates in {model_dir}...")
        for f in model_dir.glob("*.mat"):
            mat = loadmat(str(f))
            for key in mat:
                if not key.startswith('_'):
                    val = mat[key]
                    if isinstance(val, np.ndarray):
                        print(f"  {f.name}/{key}: shape={val.shape}")

    return projs, angles, vol, atom_coords


def convert_to_naf(projs, angles_deg, vol_gt, out_dir, detector_size=256, vol_size=128):
    """Convert to NAF pickle format."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_projs = projs.shape[-1] if projs.ndim == 3 else projs.shape[0]

    # Determine projection layout
    if projs.ndim == 3:
        if projs.shape[2] == n_projs:
            # (H, W, N) format — transpose to (N, H, W)
            projs = np.transpose(projs, (2, 0, 1))
        elif projs.shape[0] == n_projs:
            pass  # already (N, H, W)

    print(f"Projections: {projs.shape}")
    print(f"Angles: {len(angles_deg)} from {angles_deg.min():.1f} to {angles_deg.max():.1f} deg")

    # Resize projections to target size
    if projs.shape[1] != detector_size or projs.shape[2] != detector_size:
        print(f"Resizing projections from {projs.shape[1]}x{projs.shape[2]} to {detector_size}x{detector_size}")
        projs_resized = np.zeros((len(projs), detector_size, detector_size), dtype=np.float32)
        for i in range(len(projs)):
            projs_resized[i] = resize(projs[i].astype(float), (detector_size, detector_size),
                                       preserve_range=True)
        projs = projs_resized

    # Resize GT volume
    if vol_gt is not None:
        if vol_gt.shape != (vol_size, vol_size, vol_size):
            print(f"Resizing volume from {vol_gt.shape} to ({vol_size},{vol_size},{vol_size})")
            vol_gt = resize(vol_gt.astype(float), (vol_size, vol_size, vol_size),
                             preserve_range=True).astype(np.float32)
        vol_10 = np.stack([vol_gt] * 10, axis=0)
    else:
        vol_10 = np.zeros((10, vol_size, vol_size, vol_size), dtype=np.float32)

    # Convert angles to radians
    angles_rad = np.radians(angles_deg).astype(np.float64)

    # Sort by angle
    sort_idx = np.argsort(angles_rad)
    projs = projs[sort_idx]
    angles_rad = angles_rad[sort_idx]

    # Normalize projections
    projs = projs.astype(np.float32)
    projs = projs / (projs.max() + 1e-8) * 1000  # Scale to reasonable range

    # Split train/test
    n_total = len(angles_rad)
    n_test = max(1, n_total // 7)
    test_step = max(1, n_total // n_test)
    test_idx = sorted(set(range(0, n_total, test_step)))
    train_idx = sorted(set(range(n_total)) - set(test_idx))

    # Physical dimensions (approximate for FePt NP)
    # Yang 2017: ~8nm particle, pixel size ~0.35 Å
    pixel_size_angstrom = 0.35
    fov_angstrom = detector_size * pixel_size_angstrom  # ~90 Å
    dVoxel_mm = fov_angstrom / vol_size * 1e-7  # Å → mm
    dDet_mm = pixel_size_angstrom * 1e-7

    data = {
        "DSD": 10000.0,
        "DSO": 5000.0,
        "nVoxel": [vol_size] * 3,
        "dVoxel": [2.0] * 3,  # Use same units as simulated data
        "nDetector": [detector_size, detector_size],
        "dDetector": [1.0, 1.0],
        "offOrigin": [0.0, 0.0, 0.0],
        "offDetector": [0.0, 0.0],
        "totalAngle": float(angles_rad.max() - angles_rad.min()),
        "startAngle": float(angles_rad.min()),
        "accuracy": 0.5,
        "mode": "parallel",
        "numTrain": len(train_idx),
        "numVal": len(test_idx),
        "image": vol_10,
        "train": {
            "angles": angles_rad[train_idx],
            "projections": projs[train_idx],
            "time": np.zeros(len(train_idx), dtype=np.float32),
            "phase": np.zeros(len(train_idx), dtype=np.int32),
        },
        "val": {
            "angles": angles_rad[test_idx],
            "projections": projs[test_idx],
            "time": np.zeros(len(test_idx), dtype=np.float32),
            "phase": np.zeros(len(test_idx), dtype=np.int32),
        },
    }

    pkl_path = out_dir / "data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f, protocol=4)

    # Random init
    n_points = 50000
    init = np.concatenate([
        (np.random.rand(n_points, 3) - 0.5) * 2,
        np.random.rand(n_points, 1) * 0.5
    ], axis=1).astype(np.float32)
    np.save(out_dir / "init_data.npy", init)

    print(f"Saved NAF data: {pkl_path}")
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"  Projection range: [{projs.min():.1f}, {projs.max():.1f}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/data/aet-bench/data/real/FePt_Yang2017")
    parser.add_argument("--out_dir", type=str,
                        default="/data/aet-bench/data/real/FePt_Yang2017_prepared")
    parser.add_argument("--detector_size", type=int, default=256)
    parser.add_argument("--vol_size", type=int, default=128)
    args = parser.parse_args()

    projs, angles, vol, atoms = load_fept_data(args.data_dir)

    if projs is None:
        print("ERROR: Could not load projections!")
        return

    if angles is None:
        # Generate equal-slope angles if not found
        n = projs.shape[-1] if projs.ndim == 3 else projs.shape[0]
        print(f"No angle file found. Generating {n} equal-slope angles from -70 to +70 deg")
        angles = np.linspace(-70, 70, n)

    convert_to_naf(projs, angles, vol, args.out_dir,
                    args.detector_size, args.vol_size)


if __name__ == "__main__":
    main()
