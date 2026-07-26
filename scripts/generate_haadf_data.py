#!/usr/bin/env python
"""
Generate simulated HAADF-STEM tilt series data for AET-4DGS.

Creates a nanoparticle atomic structure using ASE, then simulates
HAADF-STEM projections using a Z-contrast Gaussian model:
    I(x,y) = sum_i  Z_i^gamma * exp(-|r - r_i|^2 / (2*sigma_i^2))

Output format: X2-Gaussian compatible (meta_data.json + .npy projections).

Usage:
    python generate_haadf_data.py --element Cu --radius 15 --out_dir ../data/cu_nano
    python generate_haadf_data.py --element Pt --radius 12 --out_dir ../data/pt_nano
    python generate_haadf_data.py --element Cu,Pt --radius 15 --out_dir ../data/cupt_nano --alloy_ratio 0.5
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path


# ---------- Nanoparticle construction ----------

def make_nanoparticle(element="Cu", radius=15.0, lattice_type="fcc"):
    """
    Create a spherical nanoparticle using ASE.

    Args:
        element: Chemical symbol (Cu, Pt, Au, etc.) or "Cu,Pt" for alloy
        radius: Nanoparticle radius in Angstroms
        lattice_type: Crystal structure type

    Returns:
        positions: (N, 3) atomic positions in Angstroms
        atomic_numbers: (N,) atomic numbers
        cell_size: float, bounding box side length
    """
    from ase.cluster import FaceCenteredCubic
    from ase.data import atomic_numbers as Z_TABLE, covalent_radii

    # Lattice constants (Angstroms)
    LATTICE_CONST = {
        "Cu": 3.615, "Pt": 3.924, "Au": 4.078, "Ag": 4.085,
        "Fe": 2.866, "Ni": 3.524, "Pd": 3.890, "Al": 4.050,
    }

    elements = element.split(",")
    primary = elements[0]
    a = LATTICE_CONST.get(primary, 3.6)

    # Determine number of layers from radius
    # For FCC: nearest-neighbor distance = a/sqrt(2)
    nn_dist = a / np.sqrt(2)
    n_layers = max(2, int(radius / nn_dist))

    # Build FCC cluster with octahedral surfaces
    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    layers = [n_layers, n_layers, n_layers]

    atoms = FaceCenteredCubic(primary, surfaces, layers, latticeconstant=a)

    # Trim to sphere
    center = atoms.positions.mean(axis=0)
    atoms.positions -= center
    distances = np.linalg.norm(atoms.positions, axis=2 if atoms.positions.ndim > 2 else 1)
    mask = distances <= radius
    atoms = atoms[mask]

    positions = atoms.positions.copy()
    atomic_nums = atoms.get_atomic_numbers().copy()

    # For alloy: randomly replace some atoms
    if len(elements) > 1:
        secondary = elements[1]
        Z_sec = Z_TABLE[secondary]
        n_replace = len(atoms) // 2  # 50% by default
        replace_idx = np.random.choice(len(atoms), n_replace, replace=False)
        atomic_nums[replace_idx] = Z_sec

    # Bounding box with padding
    cell_size = 2 * radius + 10.0  # 5 Angstrom padding each side

    print(f"Nanoparticle: {len(positions)} atoms, radius={radius} A")
    print(f"  Elements: {np.unique(atomic_nums)} (counts: {np.bincount(atomic_nums)[np.unique(atomic_nums)]})")
    print(f"  Bounding box: {cell_size:.1f} A")

    return positions, atomic_nums, cell_size


# ---------- HAADF-STEM projection (analytical Z-contrast model) ----------

def project_haadf(positions, atomic_numbers, angle_deg,
                   detector_size=256, pixel_size=0.2,
                   sigma=0.8, gamma=1.7, tilt_axis="y"):
    """
    Compute HAADF-STEM projection at a given tilt angle.

    The HAADF signal is modeled as:
        I(x,y) = sum_i Z_i^gamma * G(x - x_i, y - y_i; sigma_i)

    where G is a 2D Gaussian representing the projected atomic potential,
    and Z^gamma accounts for Z-contrast (gamma ~ 1.7 for HAADF).

    Args:
        positions: (N, 3) atomic positions in Angstroms
        atomic_numbers: (N,) atomic numbers
        angle_deg: tilt angle in degrees
        detector_size: number of pixels per side
        pixel_size: Angstrom per pixel
        sigma: Gaussian width in Angstroms (thermal vibration + probe)
        gamma: Z-contrast exponent (~1.7 for HAADF)
        tilt_axis: rotation axis ("x" or "y")

    Returns:
        projection: (detector_size, detector_size) 2D image
    """
    angle_rad = np.radians(angle_deg)

    # Rotation matrix around tilt axis
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if tilt_axis == "y":
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif tilt_axis == "x":
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        raise ValueError(f"Unknown tilt axis: {tilt_axis}")

    # Rotate positions
    rotated = positions @ R.T  # (N, 3)

    # Project along z-axis: keep (x, y), discard z
    proj_x = rotated[:, 0]
    proj_y = rotated[:, 1]

    # Z-contrast weights
    weights = atomic_numbers.astype(np.float64) ** gamma

    # Create detector grid
    half_size = detector_size * pixel_size / 2
    x_grid = np.linspace(-half_size, half_size, detector_size)
    y_grid = np.linspace(-half_size, half_size, detector_size)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")

    # Accumulate Gaussian contributions
    # Process in batches for memory efficiency
    projection = np.zeros((detector_size, detector_size), dtype=np.float64)
    batch_size = 200
    sigma2 = 2 * sigma ** 2

    for start in range(0, len(proj_x), batch_size):
        end = min(start + batch_size, len(proj_x))
        dx = X[np.newaxis, :, :] - proj_x[start:end, np.newaxis, np.newaxis]
        dy = Y[np.newaxis, :, :] - proj_y[start:end, np.newaxis, np.newaxis]
        gauss = np.exp(-(dx**2 + dy**2) / sigma2)
        w = weights[start:end, np.newaxis, np.newaxis]
        projection += (w * gauss).sum(axis=0)

    return projection.astype(np.float32)


# ---------- Ground truth volume ----------

def create_gt_volume(positions, atomic_numbers, cell_size,
                     n_voxels=128, sigma=0.8, gamma=1.7):
    """
    Create a 3D ground truth volume from atomic positions.

    Each atom contributes a 3D Gaussian weighted by Z^gamma.

    Returns:
        volume: (n_voxels, n_voxels, n_voxels) float32 array
    """
    half = cell_size / 2
    grid = np.linspace(-half, half, n_voxels)

    volume = np.zeros((n_voxels, n_voxels, n_voxels), dtype=np.float64)
    weights = atomic_numbers.astype(np.float64) ** gamma
    sigma2 = 2 * sigma ** 2

    print(f"Creating GT volume ({n_voxels}^3)...")

    # Process atom by atom (volume is large)
    for i in range(len(positions)):
        x, y, z = positions[i]
        w = weights[i]

        # Only compute within 4*sigma of atom center
        r_cut = 4 * sigma
        ix_min = max(0, int((x - r_cut + half) / cell_size * n_voxels))
        ix_max = min(n_voxels, int((x + r_cut + half) / cell_size * n_voxels) + 1)
        iy_min = max(0, int((y - r_cut + half) / cell_size * n_voxels))
        iy_max = min(n_voxels, int((y + r_cut + half) / cell_size * n_voxels) + 1)
        iz_min = max(0, int((z - r_cut + half) / cell_size * n_voxels))
        iz_max = min(n_voxels, int((z + r_cut + half) / cell_size * n_voxels) + 1)

        if ix_min >= ix_max or iy_min >= iy_max or iz_min >= iz_max:
            continue

        gx = grid[ix_min:ix_max]
        gy = grid[iy_min:iy_max]
        gz = grid[iz_min:iz_max]

        dx = gx - x
        dy = gy - y
        dz = gz - z

        gauss = np.exp(-dx[:, None, None]**2 / sigma2) * \
                np.exp(-dy[None, :, None]**2 / sigma2) * \
                np.exp(-dz[None, None, :]**2 / sigma2)

        volume[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max] += w * gauss

    return volume.astype(np.float32)


# ---------- Package for X2-Gaussian ----------

def save_x2gaussian_format(out_dir, projections, angles_deg,
                            gt_volume, scanner_cfg,
                            train_ratio=0.85):
    """
    Save data in X2-Gaussian Blender format (meta_data.json + .npy).

    Args:
        out_dir: output directory
        projections: list of (H, W) arrays
        angles_deg: list of tilt angles in degrees
        gt_volume: (Nx, Ny, Nz) ground truth volume
        scanner_cfg: scanner configuration dict
        train_ratio: fraction of projections for training
    """
    out_dir = Path(out_dir)
    proj_dir = out_dir / "projections"
    proj_dir.mkdir(parents=True, exist_ok=True)

    # Save ground truth volume
    vol_path = "gt_volume.npy"
    np.save(out_dir / vol_path, gt_volume)
    print(f"Saved GT volume: {gt_volume.shape}")

    # Save projections and build metadata
    all_projs = []
    for i, (proj, angle) in enumerate(zip(projections, angles_deg)):
        fname = f"proj_{i:04d}.npy"
        np.save(proj_dir / fname, proj)
        all_projs.append({
            "file_path": f"projections/{fname}",
            "angle": float(np.radians(angle)),
        })

    # Split train/test
    n_total = len(all_projs)
    n_train = int(n_total * train_ratio)

    # Interleave: pick every k-th for test
    indices = np.arange(n_total)
    test_step = max(1, n_total // (n_total - n_train))
    test_indices = set(indices[::test_step][:n_total - n_train])
    train_indices = [i for i in range(n_total) if i not in test_indices]
    test_indices = sorted(test_indices)

    proj_train = [all_projs[i] for i in train_indices]
    proj_test = [all_projs[i] for i in test_indices]

    print(f"Split: {len(proj_train)} train, {len(proj_test)} test")

    # Build meta_data.json
    meta = {
        "vol": vol_path,
        "scanner": scanner_cfg,
        "proj_train": proj_train,
        "proj_test": proj_test,
    }

    with open(out_dir / "meta_data.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved meta_data.json to {out_dir}")

    # Also save atomic positions for ground truth comparison
    return out_dir / "meta_data.json"


def save_naf_format(out_dir, projections, angles_deg,
                     gt_volume, scanner_cfg, train_ratio=0.85):
    """
    Save data in NAF pickle format (alternative to Blender format).
    This format includes time/phase fields needed for 4D-GS.
    """
    import pickle

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    projs = np.stack(projections, axis=0)  # (N, H, W)
    angles = np.array([np.radians(a) for a in angles_deg])

    n_total = len(angles)
    n_train = int(n_total * train_ratio)

    # Interleave split
    test_step = max(1, n_total // (n_total - n_train))
    test_indices = set(range(0, n_total, test_step))
    while len(test_indices) > n_total - n_train:
        test_indices.discard(max(test_indices))
    train_indices = sorted(set(range(n_total)) - test_indices)
    test_indices = sorted(test_indices)

    # For static reconstruction: all time=0, phase=0
    data = {
        "DSD": scanner_cfg["DSD"],
        "DSO": scanner_cfg["DSO"],
        "nVoxel": scanner_cfg["nVoxel"],
        "dVoxel": scanner_cfg["dVoxel"],
        "nDetector": scanner_cfg["nDetector"],
        "dDetector": scanner_cfg["dDetector"],
        "offOrigin": scanner_cfg["offOrigin"],
        "offDetector": scanner_cfg["offDetector"],
        "totalAngle": float(angles.max() - angles.min()),
        "startAngle": float(angles.min()),
        "accuracy": 0.5,
        "mode": scanner_cfg["mode"],
        "numTrain": len(train_indices),
        "numVal": len(test_indices),
        "image": gt_volume,
        "train": {
            "angles": angles[train_indices],
            "projections": projs[train_indices],
            "time": np.zeros(len(train_indices), dtype=np.float32),
            "phase": np.zeros(len(train_indices), dtype=np.int32),
        },
        "val": {
            "angles": angles[test_indices],
            "projections": projs[test_indices],
            "time": np.zeros(len(test_indices), dtype=np.float32),
            "phase": np.zeros(len(test_indices), dtype=np.int32),
        },
    }

    pkl_path = out_dir / "data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved NAF format to {pkl_path}")
    return pkl_path


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Generate HAADF-STEM tilt series data")
    parser.add_argument("--element", type=str, default="Cu",
                        help="Element symbol (e.g. Cu, Pt, or Cu,Pt for alloy)")
    parser.add_argument("--radius", type=float, default=15.0,
                        help="Nanoparticle radius in Angstroms")
    parser.add_argument("--out_dir", type=str, default="../data/cu_nano",
                        help="Output directory")
    parser.add_argument("--tilt_range", type=float, default=70.0,
                        help="Max tilt angle in degrees (symmetric +-)")
    parser.add_argument("--tilt_step", type=float, default=2.0,
                        help="Tilt step in degrees")
    parser.add_argument("--detector_size", type=int, default=256,
                        help="Detector size in pixels")
    parser.add_argument("--pixel_size", type=float, default=0.2,
                        help="Pixel size in Angstroms")
    parser.add_argument("--sigma", type=float, default=0.8,
                        help="Gaussian width in Angstroms")
    parser.add_argument("--gamma", type=float, default=1.7,
                        help="Z-contrast exponent")
    parser.add_argument("--vol_size", type=int, default=128,
                        help="Ground truth volume voxels per side")
    parser.add_argument("--format", type=str, default="both",
                        choices=["blender", "naf", "both"],
                        help="Output format")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 1. Create nanoparticle
    print("=" * 60)
    print("Step 1: Creating nanoparticle")
    print("=" * 60)
    positions, atomic_nums, cell_size = make_nanoparticle(
        args.element, args.radius
    )

    # 2. Generate tilt series
    print("\n" + "=" * 60)
    print("Step 2: Generating HAADF-STEM projections")
    print("=" * 60)
    angles = np.arange(-args.tilt_range, args.tilt_range + args.tilt_step/2, args.tilt_step)
    print(f"Tilt angles: {angles[0]:.0f} to {angles[-1]:.0f} deg, "
          f"{len(angles)} projections")

    projections = []
    for i, angle in enumerate(angles):
        proj = project_haadf(
            positions, atomic_nums, angle,
            detector_size=args.detector_size,
            pixel_size=args.pixel_size,
            sigma=args.sigma,
            gamma=args.gamma,
        )
        projections.append(proj)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(angles)}] angle={angle:.0f} deg, "
                  f"proj range=[{proj.min():.2f}, {proj.max():.2f}]")

    # 3. Generate ground truth volume
    print("\n" + "=" * 60)
    print("Step 3: Creating ground truth volume")
    print("=" * 60)
    gt_volume = create_gt_volume(
        positions, atomic_nums, cell_size,
        n_voxels=args.vol_size, sigma=args.sigma, gamma=args.gamma
    )
    print(f"Volume range: [{gt_volume.min():.2f}, {gt_volume.max():.2f}]")

    # 4. Scanner config
    # Convert to meters for X2-Gaussian compatibility
    cell_m = cell_size * 1e-10  # Angstrom -> meter
    pixel_m = args.pixel_size * 1e-10
    voxel_m = cell_m / args.vol_size
    det_size_m = args.detector_size * pixel_m

    scanner_cfg = {
        "DSD": 1.0,   # Parallel beam: DSD >> DSO (effectively infinity)
        "DSO": 0.5,   # Arbitrary for parallel beam
        "nVoxel": [args.vol_size, args.vol_size, args.vol_size],
        "sVoxel": [cell_m, cell_m, cell_m],
        "dVoxel": [voxel_m, voxel_m, voxel_m],
        "nDetector": [args.detector_size, args.detector_size],
        "sDetector": [det_size_m, det_size_m],
        "dDetector": [pixel_m, pixel_m],
        "offOrigin": [0.0, 0.0, 0.0],
        "offDetector": [0.0, 0.0],
        "mode": "parallel",
    }

    # 5. Save
    print("\n" + "=" * 60)
    print("Step 4: Saving data")
    print("=" * 60)

    out_dir = Path(args.out_dir)

    if args.format in ("blender", "both"):
        save_x2gaussian_format(
            out_dir, projections, angles, gt_volume, scanner_cfg
        )

    if args.format in ("naf", "both"):
        save_naf_format(
            out_dir, projections, angles, gt_volume, scanner_cfg
        )

    # Save atomic positions as ground truth
    atom_data = np.column_stack([positions, atomic_nums])
    np.save(out_dir / "atoms_gt.npy", atom_data)
    print(f"Saved ground truth atoms: {atom_data.shape}")

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Output: {out_dir}")
    print(f"  Atoms: {len(positions)}")
    print(f"  Projections: {len(projections)} ({args.detector_size}x{args.detector_size})")
    print(f"  Volume: {args.vol_size}^3")
    print("=" * 60)


if __name__ == "__main__":
    main()
