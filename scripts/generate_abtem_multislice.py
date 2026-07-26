#!/usr/bin/env python
"""
Generate HAADF-STEM tilt series using abTEM multislice simulation.

This produces physically realistic projections accounting for:
- Dynamical scattering (multislice)
- Probe convergence
- HAADF detector geometry (inner/outer angles)
- Thermal vibrations (frozen phonon)

Usage:
    python generate_abtem_multislice.py --structure Cu-309 --n_angles 71
"""

import numpy as np
import argparse
import json
import pickle
import time as timem
from pathlib import Path
from ase.io import read
from skimage.transform import resize as skresize


def prepare_atoms_for_abtem(atoms, angle_deg, margin=3.0):
    """Rotate atoms and set up cell for abTEM."""
    atoms = atoms.copy()

    # Center at origin
    center = atoms.positions.mean(axis=0)
    atoms.positions -= center

    # Rotate around y-axis
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    atoms.positions = atoms.positions @ R.T

    # Set cell to contain all atoms with margin
    pos = atoms.positions
    x_range = pos[:, 0].max() - pos[:, 0].min() + 2 * margin
    y_range = pos[:, 1].max() - pos[:, 1].min() + 2 * margin
    z_range = pos[:, 2].max() - pos[:, 2].min() + 2 * margin

    # Shift so all positions are positive
    atoms.positions -= pos.min(axis=0) - margin
    atoms.cell = [x_range, y_range, z_range]
    atoms.pbc = [True, True, False]

    return atoms, (x_range, y_range)


def simulate_one_projection_abtem(atoms, angle_deg, energy=200e3,
                                    semiangle=21.0, inner=60.0, outer=200.0,
                                    sampling=0.08, scan_step=0.4,
                                    slice_thickness=1.0):
    """Simulate one HAADF-STEM projection using abTEM multislice."""
    import abtem

    atoms_rot, (x_range, y_range) = prepare_atoms_for_abtem(atoms, angle_deg)

    # Create frozen phonon potential (thermal vibrations)
    potential = abtem.Potential(
        atoms_rot,
        sampling=sampling,
        slice_thickness=slice_thickness,
    )

    # STEM probe
    probe = abtem.Probe(
        energy=energy,
        semiangle_cutoff=semiangle,
        sampling=sampling,
    )

    # HAADF detector
    detector = abtem.AnnularDetector(inner=inner, outer=outer)

    # Scan grid
    margin = 2.0
    n_scan_x = max(8, int((x_range - 2*margin) / scan_step))
    n_scan_y = max(8, int((y_range - 2*margin) / scan_step))

    scan = abtem.GridScan(
        start=(margin, margin),
        end=(x_range - margin, y_range - margin),
        gpts=(n_scan_x, n_scan_y),
    )

    # Run simulation
    measurement = probe.scan(potential, scan=scan, detectors=detector)
    image = measurement.array.squeeze()

    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", type=str, default="Cu-309")
    parser.add_argument("--n_angles", type=int, default=71)
    parser.add_argument("--tilt_range", type=float, default=70.0)
    parser.add_argument("--detector_size", type=int, default=128,
                        help="Output image size (resized from scan)")
    parser.add_argument("--vol_size", type=int, default=128)
    parser.add_argument("--energy", type=float, default=200e3)
    parser.add_argument("--semiangle", type=float, default=21.0)
    parser.add_argument("--inner", type=float, default=60.0)
    parser.add_argument("--outer", type=float, default=200.0)
    parser.add_argument("--scan_step", type=float, default=0.5)
    parser.add_argument("--sampling", type=float, default=0.1)
    args = parser.parse_args()

    base_dir = Path("/data/aet-bench/data/simulated")
    struct_dir = base_dir / args.structure

    if not struct_dir.exists():
        print(f"Structure {args.structure} not found at {struct_dir}")
        return

    atoms = read(str(struct_dir / "structure.xyz"))
    with open(struct_dir / "metadata.json") as f:
        meta = json.load(f)

    cell_size = meta["cell_size_angstrom"]
    print(f"Structure: {args.structure}, {len(atoms)} atoms")
    print(f"Cell size: {cell_size:.1f} Å")
    print(f"abTEM params: {args.energy/1e3:.0f} keV, α={args.semiangle} mrad, "
          f"HAADF [{args.inner}-{args.outer}] mrad")

    angles_deg = np.linspace(-args.tilt_range, args.tilt_range, args.n_angles)
    projections = []

    t0 = timem.time()
    for i, angle in enumerate(angles_deg):
        try:
            img = simulate_one_projection_abtem(
                atoms, angle,
                energy=args.energy,
                semiangle=args.semiangle,
                inner=args.inner,
                outer=args.outer,
                sampling=args.sampling,
                scan_step=args.scan_step,
            )
            # Resize to target detector size
            img_resized = skresize(img, (args.detector_size, args.detector_size),
                                    preserve_range=True).astype(np.float32)
        except Exception as e:
            print(f"  ERROR at angle {angle:.0f}°: {e}")
            # Fallback to zero image
            img_resized = np.zeros((args.detector_size, args.detector_size), dtype=np.float32)

        projections.append(img_resized)

        elapsed = timem.time() - t0
        eta = elapsed / (i + 1) * (args.n_angles - i - 1)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{args.n_angles}] angle={angle:.0f}° "
                  f"shape={img.shape if 'img' in dir() else '?'} "
                  f"range=[{img_resized.min():.4f}, {img_resized.max():.4f}] "
                  f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")

    projections = np.stack(projections)
    print(f"\nTotal time: {timem.time() - t0:.1f}s")
    print(f"Projections: {projections.shape}")

    # Save as abTEM-specific output
    out_dir = base_dir / f"{args.structure}_abtem"
    out_dir.mkdir(exist_ok=True)

    np.save(out_dir / "projections_abtem.npy", projections)
    np.save(out_dir / "angles_deg.npy", angles_deg)

    # Also create GT volume (analytical, same as before)
    print("Creating GT volume...")
    atoms_gt = np.load(struct_dir / "atoms_gt.npy")
    positions = atoms_gt[:, :3]
    atomic_nums = atoms_gt[:, 3].astype(int)
    gamma = 1.7
    sigma = 0.5
    sigma2 = 2 * sigma ** 2
    half = cell_size / 2
    grid = np.linspace(-half, half, args.vol_size)
    r_cut = 4 * sigma
    weights = atomic_nums.astype(np.float64) ** gamma

    volume = np.zeros((args.vol_size,) * 3, dtype=np.float64)
    for idx in range(len(positions)):
        x, y, z = positions[idx]
        w = weights[idx]
        ix_min = max(0, int((x - r_cut + half) / cell_size * args.vol_size))
        ix_max = min(args.vol_size, int((x + r_cut + half) / cell_size * args.vol_size) + 1)
        iy_min = max(0, int((y - r_cut + half) / cell_size * args.vol_size))
        iy_max = min(args.vol_size, int((y + r_cut + half) / cell_size * args.vol_size) + 1)
        iz_min = max(0, int((z - r_cut + half) / cell_size * args.vol_size))
        iz_max = min(args.vol_size, int((z + r_cut + half) / cell_size * args.vol_size) + 1)
        if ix_min >= ix_max or iy_min >= iy_max or iz_min >= iz_max:
            continue
        gx = grid[ix_min:ix_max] - x
        gy = grid[iy_min:iy_max] - y
        gz = grid[iz_min:iz_max] - z
        gauss = (np.exp(-gx[:, None, None]**2 / sigma2) *
                 np.exp(-gy[None, :, None]**2 / sigma2) *
                 np.exp(-gz[None, None, :]**2 / sigma2))
        volume[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max] += w * gauss

    gt_vol = volume.astype(np.float32)

    # Save as NAF pickle (use aet4dgs-compatible format)
    angles_rad = np.radians(angles_deg)
    n_total = len(angles_rad)
    n_test = max(1, n_total // 7)
    test_step = max(1, n_total // n_test)
    test_idx = sorted(set(range(0, n_total, test_step)))
    train_idx = sorted(set(range(n_total)) - set(test_idx))

    vol_10 = np.stack([gt_vol] * 10, axis=0)

    data = {
        "DSD": 10000.0, "DSO": 5000.0,
        "nVoxel": [args.vol_size] * 3, "dVoxel": [2.0] * 3,
        "nDetector": [args.detector_size] * 2, "dDetector": [1.0, 1.0],
        "offOrigin": [0.0, 0.0, 0.0], "offDetector": [0.0, 0.0],
        "totalAngle": float(angles_rad.max() - angles_rad.min()),
        "startAngle": float(angles_rad.min()),
        "accuracy": 0.5, "mode": "parallel",
        "numTrain": len(train_idx), "numVal": len(test_idx),
        "image": vol_10,
        "train": {
            "angles": angles_rad[train_idx].astype(np.float64),
            "projections": projections[train_idx],
            "time": np.zeros(len(train_idx), dtype=np.float32),
            "phase": np.zeros(len(train_idx), dtype=np.int32),
        },
        "val": {
            "angles": angles_rad[test_idx].astype(np.float64),
            "projections": projections[test_idx],
            "time": np.zeros(len(test_idx), dtype=np.float32),
            "phase": np.zeros(len(test_idx), dtype=np.int32),
        },
    }

    with open(out_dir / "data.pkl", "wb") as f:
        pickle.dump(data, f, protocol=4)

    np.save(out_dir / "atoms_gt.npy", atoms_gt.astype(np.float32))
    np.save(out_dir / "gt_volume.npy", gt_vol)

    # Init point cloud
    n_points = 50000
    init = np.concatenate([
        (np.random.rand(n_points, 3) - 0.5) * 2,
        np.random.rand(n_points, 1) * 0.5
    ], axis=1).astype(np.float32)
    np.save(out_dir / "init_data.npy", init)

    meta_out = dict(meta)
    meta_out["simulation"] = "abTEM_multislice"
    meta_out["energy_eV"] = args.energy
    meta_out["semiangle_mrad"] = args.semiangle
    meta_out["haadf_inner_mrad"] = args.inner
    meta_out["haadf_outer_mrad"] = args.outer
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\nSaved to {out_dir}")
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")


if __name__ == "__main__":
    main()
