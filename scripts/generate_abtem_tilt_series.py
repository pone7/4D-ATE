#!/usr/bin/env python
"""
Generate HAADF-STEM tilt series using abTEM multislice simulation.

Plan Phase 1b.2: 200 keV, convergence 25 mrad, HAADF 50-200 mrad.
71 angles (-70° to +70°, 2° step).

Usage:
    python generate_abtem_tilt_series.py --structure Cu-923 --angles 71
    python generate_abtem_tilt_series.py --structure Cu-309 --angles 71 --quick
"""

import numpy as np
import argparse
import json
import pickle
import time as timem
from pathlib import Path
from ase.io import read
from ase.build import make_supercell


def rotate_atoms(atoms, angle_deg, axis="y"):
    """Rotate atoms around an axis through the center."""
    from ase.build import rotate
    atoms = atoms.copy()
    center = atoms.positions.mean(axis=0)
    atoms.positions -= center
    # Rotation matrix
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == "y":
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == "x":
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    atoms.positions = atoms.positions @ R.T
    atoms.positions += center
    return atoms


def simulate_haadf_abtem(atoms, energy=200e3, semiangle=25.0,
                          inner_angle=50.0, outer_angle=200.0,
                          sampling=0.1, scan_step=0.5,
                          slice_thickness=2.0):
    """
    Simulate a single HAADF-STEM image using abTEM multislice.

    Args:
        atoms: ASE Atoms object
        energy: electron energy in eV
        semiangle: probe convergence semi-angle in mrad
        inner_angle: HAADF detector inner angle in mrad
        outer_angle: HAADF detector outer angle in mrad
        sampling: wave function sampling in Angstroms
        scan_step: STEM scan step size in Angstroms
        slice_thickness: multislice slice thickness in Angstroms

    Returns:
        image: 2D HAADF image as numpy array
        extent: (x_range, y_range) in Angstroms
    """
    import abtem

    # Prepare atoms: need a cell with vacuum
    atoms = atoms.copy()
    positions = atoms.positions
    margin = 5.0  # Angstrom vacuum around particle

    # Set cell to contain all atoms + margin
    x_range = positions[:, 0].max() - positions[:, 0].min() + 2 * margin
    y_range = positions[:, 1].max() - positions[:, 1].min() + 2 * margin
    z_range = positions[:, 2].max() - positions[:, 2].min() + 2 * margin

    # Center atoms in cell
    center = positions.mean(axis=0)
    atoms.positions -= center
    atoms.positions += np.array([x_range / 2, y_range / 2, z_range / 2])

    atoms.cell = [x_range, y_range, z_range]
    atoms.pbc = [True, True, False]  # periodic in xy, not z

    # Create potential
    potential = abtem.Potential(
        atoms,
        sampling=sampling,
        slice_thickness=slice_thickness,
    )

    # Create STEM probe
    probe = abtem.Probe(
        energy=energy,
        semiangle_cutoff=semiangle,
        sampling=sampling,
    )

    # Create HAADF detector
    detector = abtem.AnnularDetector(inner=inner_angle, outer=outer_angle)

    # Create scan grid
    n_scan_x = max(4, int(x_range / scan_step))
    n_scan_y = max(4, int(y_range / scan_step))

    scan = abtem.GridScan(
        start=(margin * 0.5, margin * 0.5),
        end=(x_range - margin * 0.5, y_range - margin * 0.5),
        gpts=(n_scan_x, n_scan_y),
    )

    # Run multislice STEM simulation
    measurement = probe.scan(potential, scan=scan, detectors=detector)
    image = measurement.array.squeeze()

    return image, (x_range, y_range)


def simulate_haadf_analytical(atoms, detector_size=256, pixel_size=0.15,
                               sigma=0.5, gamma=1.7):
    """
    Fast analytical HAADF projection (fallback if abTEM is too slow).
    """
    positions = atoms.positions
    atomic_nums = atoms.get_atomic_numbers()
    weights = atomic_nums.astype(np.float64) ** gamma

    half = detector_size * pixel_size / 2
    x_grid = np.linspace(-half, half, detector_size)
    y_grid = np.linspace(-half, half, detector_size)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")

    # Center positions
    center = positions.mean(axis=0)
    pos = positions - center

    projection = np.zeros((detector_size, detector_size), dtype=np.float64)
    sigma2 = 2 * sigma ** 2

    for start in range(0, len(pos), 200):
        end = min(start + 200, len(pos))
        dx = X[np.newaxis] - pos[start:end, 0, np.newaxis, np.newaxis]
        dy = Y[np.newaxis] - pos[start:end, 1, np.newaxis, np.newaxis]
        gauss = np.exp(-(dx**2 + dy**2) / sigma2)
        w = weights[start:end, np.newaxis, np.newaxis]
        projection += (w * gauss).sum(axis=0)

    return projection.astype(np.float32)


def generate_tilt_series(structure_dir, n_angles=71, tilt_range=70.0,
                          use_abtem=True, detector_size=256,
                          quick=False):
    """Generate full tilt series for a structure."""
    structure_dir = Path(structure_dir)
    atoms_original = read(str(structure_dir / "structure.xyz"))

    with open(structure_dir / "metadata.json") as f:
        meta = json.load(f)

    name = meta["name"]
    n_atoms = meta["n_atoms"]
    cell_size = meta["cell_size_angstrom"]

    print(f"\nGenerating tilt series for {name} ({n_atoms} atoms)")
    print(f"  Cell size: {cell_size:.1f} Å")
    print(f"  Angles: {n_angles} from -{tilt_range}° to +{tilt_range}°")
    print(f"  Method: {'abTEM multislice' if use_abtem else 'analytical'}")

    angles_deg = np.linspace(-tilt_range, tilt_range, n_angles)
    projections = []
    pixel_size = cell_size / detector_size

    t0 = timem.time()
    for i, angle in enumerate(angles_deg):
        atoms_rot = rotate_atoms(atoms_original, angle, axis="y")

        if use_abtem:
            try:
                scan_step = 0.5 if not quick else 1.0
                img, extent = simulate_haadf_abtem(
                    atoms_rot,
                    scan_step=scan_step,
                    sampling=0.1 if not quick else 0.2,
                )
                # Resize to detector_size
                from skimage.transform import resize
                img = resize(img, (detector_size, detector_size),
                              preserve_range=True).astype(np.float32)
            except Exception as e:
                print(f"  abTEM failed at angle {angle:.0f}°: {e}")
                print(f"  Falling back to analytical projection")
                img = simulate_haadf_analytical(
                    atoms_rot, detector_size, pixel_size
                )
        else:
            img = simulate_haadf_analytical(
                atoms_rot, detector_size, pixel_size
            )

        projections.append(img)

        elapsed = timem.time() - t0
        if (i + 1) % 5 == 0 or i == 0:
            eta = elapsed / (i + 1) * (n_angles - i - 1)
            print(f"  [{i+1}/{n_angles}] angle={angle:.0f}° "
                  f"range=[{img.min():.1f}, {img.max():.1f}] "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    projections = np.stack(projections)
    print(f"  Total time: {timem.time() - t0:.1f}s")
    print(f"  Projections shape: {projections.shape}")

    return projections, angles_deg


def save_as_naf(projections, angles_deg, gt_vol, atoms_gt, meta,
                 out_dir, vol_size=128):
    """Save in NAF pickle format for X²-Gaussian."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    angles_rad = np.radians(angles_deg)
    n_total = len(angles_rad)
    n_test = max(1, n_total // 7)
    test_step = max(1, n_total // n_test)
    test_indices = set(range(0, n_total, test_step))
    train_indices = sorted(set(range(n_total)) - test_indices)
    test_indices = sorted(test_indices)

    det_h, det_w = projections.shape[1], projections.shape[2]
    dVoxel_mm = 2.0
    dDet_mm = 1.0

    # 10-phase GT volume (static, repeated)
    vol_10 = np.stack([gt_vol] * 10, axis=0)

    data = {
        "DSD": 10000.0,
        "DSO": 5000.0,
        "nVoxel": [vol_size] * 3,
        "dVoxel": [dVoxel_mm] * 3,
        "nDetector": [det_h, det_w],
        "dDetector": [dDet_mm, dDet_mm],
        "offOrigin": [0.0, 0.0, 0.0],
        "offDetector": [0.0, 0.0],
        "totalAngle": float(angles_rad.max() - angles_rad.min()),
        "startAngle": float(angles_rad.min()),
        "accuracy": 0.5,
        "mode": "parallel",
        "numTrain": len(train_indices),
        "numVal": len(test_indices),
        "image": vol_10.astype(np.float32),
        "train": {
            "angles": angles_rad[train_indices].astype(np.float64),
            "projections": projections[train_indices].astype(np.float32),
            "time": np.zeros(len(train_indices), dtype=np.float32),
            "phase": np.zeros(len(train_indices), dtype=np.int32),
        },
        "val": {
            "angles": angles_rad[test_indices].astype(np.float64),
            "projections": projections[test_indices].astype(np.float32),
            "time": np.zeros(len(test_indices), dtype=np.float32),
            "phase": np.zeros(len(test_indices), dtype=np.int32),
        },
    }

    with open(out_dir / "data.pkl", "wb") as f:
        pickle.dump(data, f, protocol=4)

    # Save GT atoms and volume
    np.save(out_dir / "atoms_gt.npy", atoms_gt.astype(np.float32))
    np.save(out_dir / "gt_volume.npy", gt_vol.astype(np.float32))

    # Random init point cloud
    n_points = min(50000, meta["n_atoms"] * 100)
    init = np.concatenate([
        (np.random.rand(n_points, 3) - 0.5) * 2,
        np.random.rand(n_points, 1) * 0.5
    ], axis=1).astype(np.float32)
    np.save(out_dir / "init_data.npy", init)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved NAF data to {out_dir}")
    print(f"  Train: {len(train_indices)}, Test: {len(test_indices)}")


def create_gt_volume(atoms, cell_size, vol_size=128, sigma=0.5, gamma=1.7):
    """Create GT volume from atomic positions."""
    positions = atoms.positions - atoms.positions.mean(axis=0)
    atomic_nums = atoms.get_atomic_numbers()
    weights = atomic_nums.astype(np.float64) ** gamma
    half = cell_size / 2
    grid = np.linspace(-half, half, vol_size)
    sigma2 = 2 * sigma ** 2
    r_cut = 4 * sigma

    volume = np.zeros((vol_size, vol_size, vol_size), dtype=np.float64)
    for i in range(len(positions)):
        x, y, z = positions[i]
        w = weights[i]
        ix_min = max(0, int((x - r_cut + half) / cell_size * vol_size))
        ix_max = min(vol_size, int((x + r_cut + half) / cell_size * vol_size) + 1)
        iy_min = max(0, int((y - r_cut + half) / cell_size * vol_size))
        iy_max = min(vol_size, int((y + r_cut + half) / cell_size * vol_size) + 1)
        iz_min = max(0, int((z - r_cut + half) / cell_size * vol_size))
        iz_max = min(vol_size, int((z + r_cut + half) / cell_size * vol_size) + 1)
        if ix_min >= ix_max or iy_min >= iy_max or iz_min >= iz_max:
            continue
        gx = grid[ix_min:ix_max] - x
        gy = grid[iy_min:iy_max] - y
        gz = grid[iz_min:iz_max] - z
        gauss = (np.exp(-gx[:, None, None]**2 / sigma2) *
                 np.exp(-gy[None, :, None]**2 / sigma2) *
                 np.exp(-gz[None, None, :]**2 / sigma2))
        volume[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max] += w * gauss

    return volume.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", type=str, default="Cu-309",
                        help="Structure name (Cu-309, Cu-923, Pt-923, FePt-923, Cu-2057)")
    parser.add_argument("--angles", type=int, default=71)
    parser.add_argument("--tilt_range", type=float, default=70.0)
    parser.add_argument("--detector_size", type=int, default=256)
    parser.add_argument("--vol_size", type=int, default=128)
    parser.add_argument("--use_abtem", action="store_true",
                        help="Use abTEM multislice (slower but more realistic)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: lower resolution for fast testing")
    parser.add_argument("--all", action="store_true",
                        help="Generate for all structures")
    args = parser.parse_args()

    base_dir = Path("/data/aet-bench/data/simulated")

    if args.all:
        structures = ["Cu-309", "Cu-923", "Pt-923", "FePt-923"]
    else:
        structures = [args.structure]

    for struct_name in structures:
        struct_dir = base_dir / struct_name
        if not struct_dir.exists():
            print(f"Structure {struct_name} not found at {struct_dir}")
            continue

        with open(struct_dir / "metadata.json") as f:
            meta = json.load(f)

        atoms = read(str(struct_dir / "structure.xyz"))
        cell_size = meta["cell_size_angstrom"]

        # Generate tilt series
        projections, angles = generate_tilt_series(
            struct_dir, n_angles=args.angles, tilt_range=args.tilt_range,
            use_abtem=args.use_abtem, detector_size=args.detector_size,
            quick=args.quick,
        )

        # Create GT volume
        print(f"  Creating GT volume ({args.vol_size}³)...")
        gt_vol = create_gt_volume(atoms, cell_size, args.vol_size)

        # Prepare atoms_gt
        positions = atoms.positions - atoms.positions.mean(axis=0)
        atoms_gt = np.column_stack([positions, atoms.get_atomic_numbers()])

        # Save
        out_dir = base_dir / struct_name
        save_as_naf(projections, angles, gt_vol, atoms_gt, meta,
                     out_dir, args.vol_size)

    print("\nDone!")


if __name__ == "__main__":
    main()
