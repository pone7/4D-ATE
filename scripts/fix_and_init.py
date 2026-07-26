#!/usr/bin/env python
"""Fix NAF data format and generate random initialization for X2-Gaussian."""

import numpy as np
import pickle
import sys
from pathlib import Path


def main(data_dir):
    data_dir = Path(data_dir)
    pkl_path = data_dir / "data.pkl"

    # Load pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Fix 1: vol_gt must be (10, Nx, Ny, Nz) for 10 phases
    image = data["image"]
    if image.ndim == 3:
        print(f"Fixing vol_gt: {image.shape} -> (10, *shape)")
        data["image"] = np.stack([image] * 10, axis=0)

    # Fix 2: Use proper mm units for NAF format
    # NAF reader divides all lengths by 1000 (mm -> m), then applies scene_scale = 2/max(sVoxel)
    # For parallel beam: DSD/DSO are arbitrary (large values = more parallel)
    # Key: sVoxel and dDetector control the scale
    nVoxel = data["nVoxel"]  # [128, 128, 128]
    nDet = data["nDetector"]  # [256, 256]

    # Set physical dimensions (mm)
    dVoxel_mm = 2.0  # 2mm per voxel -> sVoxel = 256mm
    dDet_mm = 1.0    # 1mm per detector pixel -> sDetector = 256mm

    data["DSD"] = 10000.0  # 10m source-detector (large for parallel beam)
    data["DSO"] = 5000.0   # 5m source-origin
    data["dVoxel"] = [dVoxel_mm] * 3
    data["dDetector"] = [dDet_mm] * 2
    data["offOrigin"] = [0.0, 0.0, 0.0]
    data["offDetector"] = [0.0, 0.0]

    sVoxel = [n * dVoxel_mm for n in nVoxel]
    print(f"Scanner config (mm): sVoxel={sVoxel}, dVoxel={dVoxel_mm}, dDet={dDet_mm}")
    print(f"  DSD={data['DSD']}, DSO={data['DSO']}")

    # Convert all numpy arrays to use protocol compatible with numpy 1.x
    # by converting to Python lists then back
    def downgrade_arrays(obj):
        if isinstance(obj, np.ndarray):
            return np.array(obj.tolist(), dtype=obj.dtype)
        elif isinstance(obj, dict):
            return {k: downgrade_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [downgrade_arrays(v) for v in obj]
        return obj

    data = downgrade_arrays(data)

    # Save fixed pickle with protocol 4 (compatible with Python 3.8+)
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f, protocol=4)
    print(f"Saved fixed data to {pkl_path}")

    # Generate random initialization
    init_path = data_dir / "init_data_debug.npy"

    n_points = 50000
    sVoxel_m = np.array(sVoxel) / 1000  # mm -> m
    scene_scale = 2.0 / max(sVoxel_m)

    # Random points in scaled volume (after scene_scale, volume is in [-1,1]^3)
    positions_scaled = (np.random.rand(n_points, 3) - 0.5) * 2  # [-1, 1]
    densities = np.random.rand(n_points) * 0.5

    init_data = np.column_stack([positions_scaled, densities]).astype(np.float32)
    np.save(init_path, init_data)
    print(f"Saved random init ({n_points} points) to {init_path}")
    print(f"  Position range: [{positions_scaled.min():.3f}, {positions_scaled.max():.3f}]")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/aet-bench/data/cu_nano"
    main(data_dir)
