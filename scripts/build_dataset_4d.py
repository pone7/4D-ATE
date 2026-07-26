#!/usr/bin/env python
"""
AET-Bench: build the AET-Sim-4D subset.

Generates 10 dynamic trajectories (Cu nanoparticle annealing) with 10
phases each by reusing scripts/generate_4d_data.py. Varies particle radius
and random seed across trajectories so the benchmark covers different sizes
and dynamic realizations.

Output layout:
    data/aet_bench/sim4d/<traj_id>/
        analytical/data.pkl    # NAF format with time/phase fields
        atoms_gt.npy           # phase-0 atoms (xyz + Z)
        atoms_gt_4d.npy        # (n_phases, N, 4) full trajectory
        metadata.json
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from generate_4d_data import (  # noqa: E402
    make_cu_nanoparticle, simulate_annealing,
    project_haadf_4d, create_gt_volume_4d,
)


# 10 trajectories: vary radius (size) and seed (dynamics)
TRAJECTORIES = [
    # (traj_idx, radius_A, seed, max_displacement)
    (0, 8.0,  1, 1.0),
    (1, 10.0, 2, 1.0),
    (2, 10.0, 3, 1.5),
    (3, 12.0, 4, 1.0),
    (4, 12.0, 5, 1.5),
    (5, 12.0, 6, 2.0),
    (6, 14.0, 7, 1.5),
    (7, 14.0, 8, 2.0),
    (8, 16.0, 9, 1.5),
    (9, 16.0, 10, 2.0),
]


def build_one(traj_idx, radius, seed, max_disp,
              n_phases, n_angles, tilt_range,
              detector_size, vol_size, out_root):
    print(f"\n=== Trajectory {traj_idx}: r={radius} Å, seed={seed}, "
          f"max_disp={max_disp} Å ===")
    np.random.seed(seed)

    positions, atomic_nums = make_cu_nanoparticle(radius)
    cell_size = 2 * radius + 10.0

    positions_4d = simulate_annealing(positions, n_phases, max_disp, seed)

    angles = np.linspace(-tilt_range, tilt_range, n_angles)
    pixel_size = cell_size / detector_size  # ensure FOV matches volume
    all_projs, all_angles, all_times, all_phases = [], [], [], []
    for i, ang in enumerate(angles):
        # spread time uniformly across the trajectory: each angle picks a phase
        t_norm = i / max(1, n_angles - 1)
        phase_float = t_norm * (n_phases - 1)
        phase_idx = int(phase_float)
        frac = phase_float - phase_idx
        next_phase = min(phase_idx + 1, n_phases - 1)
        pos_interp = (1 - frac) * positions_4d[phase_idx] + frac * positions_4d[next_phase]

        proj = project_haadf_4d(pos_interp, atomic_nums, ang,
                                 detector_size, pixel_size,
                                 sigma=0.8, gamma=1.7)
        all_projs.append(proj)
        all_angles.append(np.radians(ang))
        all_times.append(t_norm)
        all_phases.append(phase_idx)

    gt_volumes = create_gt_volume_4d(positions_4d, atomic_nums, cell_size,
                                      vol_size, sigma=0.8, gamma=1.7)

    projs = np.stack(all_projs).astype(np.float32)
    angles_arr = np.array(all_angles, dtype=np.float64)
    times_arr = np.array(all_times, dtype=np.float32)
    phases_arr = np.array(all_phases, dtype=np.int32)

    # Train/test split: every 7th angle reserved
    n_total = len(angles_arr)
    test_idx = sorted(set(range(0, n_total, 7)))
    train_idx = sorted(set(range(n_total)) - set(test_idx))

    data = {
        "DSD": 10000.0, "DSO": 5000.0,
        "nVoxel": [vol_size] * 3, "dVoxel": [2.0] * 3,
        "nDetector": [detector_size, detector_size], "dDetector": [1.0, 1.0],
        "offOrigin": [0.0, 0.0, 0.0], "offDetector": [0.0, 0.0],
        "totalAngle": float(angles_arr.max() - angles_arr.min()),
        "startAngle": float(angles_arr.min()),
        "accuracy": 0.5, "mode": "parallel",
        "numTrain": len(train_idx), "numVal": len(test_idx),
        "image": gt_volumes,  # (n_phases, vol, vol, vol)
        "train": {
            "angles": angles_arr[train_idx],
            "projections": projs[train_idx],
            "time": times_arr[train_idx],
            "phase": phases_arr[train_idx],
        },
        "val": {
            "angles": angles_arr[test_idx],
            "projections": projs[test_idx],
            "time": times_arr[test_idx],
            "phase": phases_arr[test_idx],
        },
    }

    traj_id = f"Cu4D-{int(radius):02d}A-{traj_idx:02d}"
    sample_dir = out_root / traj_id
    (sample_dir / "analytical").mkdir(parents=True, exist_ok=True)
    with open(sample_dir / "analytical" / "data.pkl", "wb") as f:
        pickle.dump(data, f, protocol=4)

    # GT atom trajectories
    atoms0 = np.column_stack([positions_4d[0],
                               atomic_nums.astype(np.float32)])
    np.save(sample_dir / "atoms_gt.npy", atoms0.astype(np.float32))
    atoms_4d = np.concatenate(
        [positions_4d,
         np.broadcast_to(atomic_nums[None, :, None],
                         (n_phases, len(atomic_nums), 1)).astype(np.float32)],
        axis=2,
    )
    np.save(sample_dir / "atoms_gt_4d.npy", atoms_4d.astype(np.float32))

    meta = {
        "sample_id": traj_id,
        "element": "Cu",
        "n_atoms": int(len(positions)),
        "radius_angstrom": radius,
        "cell_size_angstrom": cell_size,
        "n_phases": n_phases,
        "n_angles": n_angles,
        "tilt_range_deg": tilt_range,
        "detector_size": detector_size,
        "vol_size": vol_size,
        "max_displacement_angstrom": max_disp,
        "seed": seed,
        "backends": ["analytical"],
    }
    with open(sample_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", type=str,
                   default="/data/aet-bench/data/aet_bench/sim4d")
    p.add_argument("--n_phases", type=int, default=10)
    p.add_argument("--n_angles", type=int, default=71)
    p.add_argument("--tilt_range", type=float, default=70.0)
    p.add_argument("--detector_size", type=int, default=128)
    p.add_argument("--vol_size", type=int, default=128)
    p.add_argument("--n_trajectories", type=int, default=10)
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for traj in TRAJECTORIES[:args.n_trajectories]:
        idx, radius, seed, max_disp = traj
        meta = build_one(idx, radius, seed, max_disp,
                          args.n_phases, args.n_angles, args.tilt_range,
                          args.detector_size, args.vol_size, out_root)
        manifest.append(meta)

    with open(out_root / "manifest.json", "w") as f:
        json.dump({"n_samples": len(manifest), "samples": manifest}, f, indent=2)
    print(f"\nDone. {len(manifest)} 4D trajectories → {out_root}")


if __name__ == "__main__":
    main()
