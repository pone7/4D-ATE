#!/usr/bin/env python
"""
Generate NAF-format pickle data directly compatible with numpy 1.x.
Run this with the aet4dgs env (numpy 1.24).
"""
import numpy as np
import pickle
import json
import sys
from pathlib import Path


def main(data_dir):
    data_dir = Path(data_dir)

    # Load the projections and GT from the Blender format
    with open(data_dir / "meta_data.json") as f:
        meta = json.load(f)

    gt_vol = np.load(data_dir / "gt_volume.npy")
    print(f"GT volume: {gt_vol.shape}, range=[{gt_vol.min():.2f}, {gt_vol.max():.2f}]")

    # Load all projections
    all_train = meta["proj_train"]
    all_test = meta["proj_test"]

    train_projs = np.stack([np.load(data_dir / p["file_path"]) for p in all_train])
    test_projs = np.stack([np.load(data_dir / p["file_path"]) for p in all_test])
    train_angles = np.array([p["angle"] for p in all_train], dtype=np.float64)
    test_angles = np.array([p["angle"] for p in all_test], dtype=np.float64)

    n_train = len(train_angles)
    n_test = len(test_angles)
    det_h, det_w = train_projs.shape[1], train_projs.shape[2]
    n_vox = gt_vol.shape[0]

    print(f"Train: {n_train} projections, Test: {n_test} projections")
    print(f"Detector: {det_h}x{det_w}, Volume: {n_vox}^3")

    # Build NAF format pickle
    # Physical units in mm (NAF reader divides by 1000 to get meters)
    dVoxel_mm = 2.0  # 2mm per voxel
    dDet_mm = 1.0    # 1mm per detector pixel

    # For 10-phase volume (static = repeat 10x)
    vol_10 = np.stack([gt_vol] * 10, axis=0)

    data = {
        "DSD": 10000.0,    # mm, large for parallel beam
        "DSO": 5000.0,     # mm
        "nVoxel": [n_vox, n_vox, n_vox],
        "dVoxel": [dVoxel_mm, dVoxel_mm, dVoxel_mm],
        "nDetector": [det_h, det_w],
        "dDetector": [dDet_mm, dDet_mm],
        "offOrigin": [0.0, 0.0, 0.0],
        "offDetector": [0.0, 0.0],
        "totalAngle": float(train_angles.max() - train_angles.min()),
        "startAngle": float(train_angles.min()),
        "accuracy": 0.5,
        "mode": "parallel",
        "numTrain": n_train,
        "numVal": n_test,
        "image": vol_10.astype(np.float32),
        "train": {
            "angles": train_angles.astype(np.float64),
            "projections": train_projs.astype(np.float32),
            "time": np.zeros(n_train, dtype=np.float32),
            "phase": np.zeros(n_train, dtype=np.int32),
        },
        "val": {
            "angles": test_angles.astype(np.float64),
            "projections": test_projs.astype(np.float32),
            "time": np.zeros(n_test, dtype=np.float32),
            "phase": np.zeros(n_test, dtype=np.int32),
        },
    }

    pkl_path = data_dir / "data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f, protocol=4)
    print(f"Saved NAF data to {pkl_path}")

    # Generate random initialization
    init_path = data_dir / "init_data_debug.npy"
    n_points = 50000

    # After NAF reader: sVoxel = nVoxel * dVoxel / 1000 (meters)
    # scene_scale = 2 / max(sVoxel)
    sVoxel_m = n_vox * dVoxel_mm / 1000  # meters
    scene_scale = 2.0 / sVoxel_m

    # Random points in [-1, 1]^3 (post-scene-scale coordinates)
    positions = (np.random.rand(n_points, 3) - 0.5) * 2
    densities = np.random.rand(n_points, 1) * 0.5

    init_data = np.concatenate([positions, densities], axis=1).astype(np.float32)
    np.save(init_path, init_data)
    print(f"Saved init ({n_points} points) to {init_path}")
    print(f"  scene_scale={scene_scale:.2f}, sVoxel={sVoxel_m:.4f}m")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/aet-bench/data/cu_nano"
    main(data_dir)
