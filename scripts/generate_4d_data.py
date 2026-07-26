#!/usr/bin/env python
"""
Generate 4D dynamic HAADF-STEM data simulating Cu nanoparticle annealing.

Dynamics: surface atoms gradually migrate/diffuse, core atoms have thermal vibrations.
Output: NAF pickle with time-varying projections and 10-phase ground truth volumes.
"""

import numpy as np
import pickle
import json
import argparse
from pathlib import Path


def make_cu_nanoparticle(radius=15.0):
    """Create FCC Cu nanoparticle."""
    from ase.cluster import FaceCenteredCubic
    a = 3.615  # Cu lattice constant
    nn_dist = a / np.sqrt(2)
    n_layers = max(2, int(radius / nn_dist))
    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    layers = [n_layers, n_layers, n_layers]
    atoms = FaceCenteredCubic("Cu", surfaces, layers, latticeconstant=a)
    center = atoms.positions.mean(axis=0)
    atoms.positions -= center
    distances = np.linalg.norm(atoms.positions, axis=1)
    mask = distances <= radius
    atoms = atoms[mask]
    return atoms.positions.copy(), atoms.get_atomic_numbers().copy()


def simulate_annealing(positions, n_phases=10, max_displacement=1.5, seed=42):
    """
    Simulate annealing dynamics: surface atoms migrate, core vibrates.

    Returns:
        positions_4d: (n_phases, N, 3) — positions at each phase
    """
    rng = np.random.RandomState(seed)
    N = len(positions)
    distances = np.linalg.norm(positions, axis=1)
    r_max = distances.max()

    # Classify atoms: surface (outer 30%) vs core
    surface_mask = distances > 0.7 * r_max
    n_surface = surface_mask.sum()
    n_core = N - n_surface
    print(f"  Atoms: {N} total, {n_surface} surface, {n_core} core")

    positions_4d = np.zeros((n_phases, N, 3))

    for t in range(n_phases):
        frac = t / (n_phases - 1)  # 0 → 1

        # Core atoms: small thermal vibrations (Debye-Waller)
        thermal_sigma = 0.15  # Angstroms, typical at room temp
        core_vibration = rng.randn(N, 3) * thermal_sigma

        # Surface atoms: directional migration (inward relaxation + tangential diffusion)
        surface_displacement = np.zeros((N, 3))
        if n_surface > 0:
            # Inward relaxation: surface atoms move toward center over time
            directions = -positions[surface_mask] / (distances[surface_mask, None] + 1e-8)
            inward = directions * max_displacement * frac * 0.5

            # Tangential diffusion: random walk on the surface
            tangent1 = np.cross(directions, rng.randn(n_surface, 3))
            tangent1 /= np.linalg.norm(tangent1, axis=1, keepdims=True) + 1e-8
            tangent_disp = tangent1 * max_displacement * frac * 0.3 * rng.randn(n_surface, 1)

            surface_displacement[surface_mask] = inward + tangent_disp

        positions_4d[t] = positions + core_vibration + surface_displacement

    return positions_4d


def project_haadf_4d(positions, atomic_numbers, angle_deg,
                      detector_size=256, pixel_size=0.2,
                      sigma=0.8, gamma=1.7):
    """Project 3D positions to 2D HAADF image at given angle."""
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # rotate around y

    rotated = positions @ R.T
    proj_x = rotated[:, 0]
    proj_y = rotated[:, 1]
    weights = atomic_numbers.astype(np.float64) ** gamma

    half_size = detector_size * pixel_size / 2
    x_grid = np.linspace(-half_size, half_size, detector_size)
    y_grid = np.linspace(-half_size, half_size, detector_size)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")

    projection = np.zeros((detector_size, detector_size), dtype=np.float64)
    sigma2 = 2 * sigma ** 2
    batch_size = 200

    for start in range(0, len(proj_x), batch_size):
        end = min(start + batch_size, len(proj_x))
        dx = X[np.newaxis] - proj_x[start:end, np.newaxis, np.newaxis]
        dy = Y[np.newaxis] - proj_y[start:end, np.newaxis, np.newaxis]
        gauss = np.exp(-(dx**2 + dy**2) / sigma2)
        w = weights[start:end, np.newaxis, np.newaxis]
        projection += (w * gauss).sum(axis=0)

    return projection.astype(np.float32)


def create_gt_volume_4d(positions_4d, atomic_numbers, cell_size,
                         n_voxels=128, sigma=0.8, gamma=1.7):
    """Create 10-phase ground truth volumes."""
    n_phases = positions_4d.shape[0]
    volumes = np.zeros((n_phases, n_voxels, n_voxels, n_voxels), dtype=np.float32)

    half = cell_size / 2
    grid = np.linspace(-half, half, n_voxels)
    weights = atomic_numbers.astype(np.float64) ** gamma
    sigma2 = 2 * sigma ** 2
    r_cut = 4 * sigma

    for t in range(n_phases):
        print(f"  GT volume phase {t}/{n_phases}...")
        positions = positions_4d[t]
        for i in range(len(positions)):
            x, y, z = positions[i]
            w = weights[i]
            ix_min = max(0, int((x - r_cut + half) / cell_size * n_voxels))
            ix_max = min(n_voxels, int((x + r_cut + half) / cell_size * n_voxels) + 1)
            iy_min = max(0, int((y - r_cut + half) / cell_size * n_voxels))
            iy_max = min(n_voxels, int((y + r_cut + half) / cell_size * n_voxels) + 1)
            iz_min = max(0, int((z - r_cut + half) / cell_size * n_voxels))
            iz_max = min(n_voxels, int((z + r_cut + half) / cell_size * n_voxels) + 1)
            if ix_min >= ix_max or iy_min >= iy_max or iz_min >= iz_max:
                continue
            gx = grid[ix_min:ix_max] - x
            gy = grid[iy_min:iy_max] - y
            gz = grid[iz_min:iz_max] - z
            gauss = np.exp(-gx[:, None, None]**2 / sigma2) * \
                    np.exp(-gy[None, :, None]**2 / sigma2) * \
                    np.exp(-gz[None, None, :]**2 / sigma2)
            volumes[t, ix_min:ix_max, iy_min:iy_max, iz_min:iz_max] += w * gauss

    return volumes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="/data/aet-bench/data/cu_anneal_4d")
    parser.add_argument("--radius", type=float, default=15.0)
    parser.add_argument("--n_phases", type=int, default=10)
    parser.add_argument("--n_angles", type=int, default=71)
    parser.add_argument("--tilt_range", type=float, default=70.0)
    parser.add_argument("--detector_size", type=int, default=256)
    parser.add_argument("--vol_size", type=int, default=128)
    parser.add_argument("--max_displacement", type=float, default=1.5,
                        help="Max surface atom displacement in Angstroms")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create nanoparticle
    print("=" * 60)
    print("Step 1: Creating Cu nanoparticle")
    positions, atomic_nums = make_cu_nanoparticle(args.radius)
    cell_size = 2 * args.radius + 10.0
    print(f"  {len(positions)} atoms, cell_size={cell_size:.1f} A")

    # 2. Simulate annealing dynamics
    print("\nStep 2: Simulating annealing dynamics")
    positions_4d = simulate_annealing(
        positions, args.n_phases, args.max_displacement, args.seed
    )
    for t in range(args.n_phases):
        disp = np.linalg.norm(positions_4d[t] - positions, axis=1)
        print(f"  Phase {t}: mean disp={disp.mean():.3f}, max={disp.max():.3f} A")

    # 3. Generate projections at all angles and times
    print("\nStep 3: Generating 4D HAADF-STEM projections")
    angles = np.linspace(-args.tilt_range, args.tilt_range, args.n_angles)

    # Simulate a scan: projections acquired sequentially
    # Total scan = 60s, breathing cycle = 3s
    # Each projection has a time stamp
    scan_time = 60.0
    breath_cycle = 3.0
    n_total_projs = args.n_angles  # one full tilt series
    dt = scan_time / n_total_projs  # time per projection

    # Generate projections with time-varying positions
    all_projs = []
    all_angles = []
    all_times = []
    all_phases = []

    for i, angle in enumerate(angles):
        t_sec = i * dt  # time in seconds
        # Which phase? Use periodic breathing cycle
        phase_float = (t_sec % breath_cycle) / breath_cycle * args.n_phases
        phase_idx = int(phase_float) % args.n_phases
        phase_frac = phase_float - int(phase_float)

        # Interpolate positions between phases
        next_phase = (phase_idx + 1) % args.n_phases
        pos_interp = (1 - phase_frac) * positions_4d[phase_idx] + \
                     phase_frac * positions_4d[next_phase]

        proj = project_haadf_4d(pos_interp, atomic_nums, angle,
                                 args.detector_size, sigma=0.8, gamma=1.7)
        all_projs.append(proj)
        all_angles.append(np.radians(angle))
        all_times.append(t_sec / scan_time)  # normalize to [0, 1)
        all_phases.append(phase_idx)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args.n_angles}] angle={angle:.0f}° t={t_sec:.1f}s phase={phase_idx}")

    # 4. Generate GT volumes for 10 phases
    print("\nStep 4: Creating 10-phase GT volumes")
    gt_volumes = create_gt_volume_4d(
        positions_4d, atomic_nums, cell_size,
        args.vol_size, sigma=0.8, gamma=1.7
    )
    print(f"  GT volumes shape: {gt_volumes.shape}")

    # 5. Split train/test
    n_total = len(all_projs)
    n_test = max(1, n_total // 7)
    test_step = n_total // n_test
    test_indices = set(range(0, n_total, test_step))
    train_indices = sorted(set(range(n_total)) - test_indices)
    test_indices = sorted(test_indices)

    projs = np.stack(all_projs)
    angles_arr = np.array(all_angles)
    times_arr = np.array(all_times, dtype=np.float32)
    phases_arr = np.array(all_phases, dtype=np.int32)

    print(f"\n  Train: {len(train_indices)}, Test: {len(test_indices)}")

    # 6. Build NAF pickle
    print("\nStep 5: Saving NAF format")
    dVoxel_mm = 2.0
    dDet_mm = 1.0

    data = {
        "DSD": 10000.0,
        "DSO": 5000.0,
        "nVoxel": [args.vol_size] * 3,
        "dVoxel": [dVoxel_mm] * 3,
        "nDetector": [args.detector_size, args.detector_size],
        "dDetector": [dDet_mm, dDet_mm],
        "offOrigin": [0.0, 0.0, 0.0],
        "offDetector": [0.0, 0.0],
        "totalAngle": float(angles_arr.max() - angles_arr.min()),
        "startAngle": float(angles_arr.min()),
        "accuracy": 0.5,
        "mode": "parallel",
        "numTrain": len(train_indices),
        "numVal": len(test_indices),
        "image": gt_volumes.astype(np.float32),
        "train": {
            "angles": angles_arr[train_indices],
            "projections": projs[train_indices],
            "time": times_arr[train_indices],
            "phase": phases_arr[train_indices],
        },
        "val": {
            "angles": angles_arr[test_indices],
            "projections": projs[test_indices],
            "time": times_arr[test_indices],
            "phase": phases_arr[test_indices],
        },
    }

    pkl_path = out_dir / "data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f, protocol=4)
    print(f"  Saved to {pkl_path}")

    # 7. Create init point cloud
    n_points = 50000
    positions_scaled = (np.random.rand(n_points, 3) - 0.5) * 2
    densities = np.random.rand(n_points, 1) * 0.5
    init_data = np.concatenate([positions_scaled, densities], axis=1).astype(np.float32)
    init_path = out_dir / "init_data.npy"
    np.save(init_path, init_data)

    # Save atoms GT
    atom_data = np.column_stack([positions, atomic_nums])
    np.save(out_dir / "atoms_gt.npy", atom_data)
    np.save(out_dir / "positions_4d.npy", positions_4d)

    print(f"\n{'='*60}")
    print(f"Done! 4D HAADF-STEM data generated:")
    print(f"  Atoms: {len(positions)}")
    print(f"  Phases: {args.n_phases}")
    print(f"  Projections: {n_total} ({args.detector_size}x{args.detector_size})")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
