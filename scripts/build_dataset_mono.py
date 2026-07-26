#!/usr/bin/env python
"""
AET-Bench: build the AET-Sim-Mono subset.

Generates ~50 single-element FCC nanoparticle samples covering Cu/Pt/Au/Pd/Ag
across a range of sizes (200-2000 atoms). For each sample, produces an
analytical Z-contrast HAADF tilt series (fast) and optionally an abTEM
multislice tilt series (slow, opt-in via --backend both).

Output layout:
    data/aet_bench/mono/<sample_id>/
        atoms_gt.npy            # (N, 4): xyz + Z
        metadata.json           # element, n_atoms, generator params
        analytical/data.pkl     # NAF-format tilt series (always)
        abtem/data.pkl          # NAF-format tilt series (if --backend both)

Usage:
    python build_dataset_mono.py --backend analytical
    python build_dataset_mono.py --backend both --n_samples 10
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.cluster import FaceCenteredCubic
from ase.data import atomic_numbers as Z_TABLE

# Reuse projection / volume / save helpers from the existing analytical generator.
sys.path.insert(0, str(Path(__file__).parent))
from generate_haadf_data import (  # noqa: E402
    project_haadf,
    create_gt_volume,
    save_naf_format,
)


LATTICE_CONST = {
    "Cu": 3.615, "Pt": 3.924, "Au": 4.078, "Pd": 3.890, "Ag": 4.085,
}

# (element, target_n_atoms) — 50 samples = 5 elements x 10 sizes
SIZE_LADDER = [201, 309, 459, 561, 711, 923, 1157, 1415, 1697, 2057]
ELEMENTS = ["Cu", "Pt", "Au", "Pd", "Ag"]


def build_octahedral(element, target_n, seed=0):
    """Build an FCC octahedral cluster sized to ~target_n atoms."""
    a = LATTICE_CONST[element]
    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    best = None
    for n in range(2, 18):
        atoms = FaceCenteredCubic(element, surfaces, [n, n, n], latticeconstant=a)
        if best is None or abs(len(atoms) - target_n) < abs(len(best) - target_n):
            best = atoms
        if len(atoms) >= target_n:
            break
    best.positions -= best.positions.mean(axis=0)
    return best


def make_sample_id(element, n_atoms, idx):
    return f"{element}-{n_atoms:04d}-{idx:02d}"


def generate_analytical(positions, atomic_nums, cell_size, out_dir,
                        tilt_range=70.0, n_angles=71,
                        detector_size=128, vol_size=128,
                        sigma=0.8, gamma=1.7):
    """Render analytical tilt series + GT volume and save NAF pickle."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pixel_size = cell_size / detector_size

    angles = np.linspace(-tilt_range, tilt_range, n_angles)
    projections = [
        project_haadf(positions, atomic_nums, a,
                       detector_size=detector_size, pixel_size=pixel_size,
                       sigma=sigma, gamma=gamma)
        for a in angles
    ]
    gt_vol = create_gt_volume(positions, atomic_nums, cell_size,
                               n_voxels=vol_size, sigma=sigma, gamma=gamma)

    # Use the same arbitrary-unit scanner config that has been validated with
    # X²-Gaussian (matches generate_abtem_multislice.py). Meter-scale configs
    # cause numerical issues for small cells.
    scanner_cfg = {
        "DSD": 10000.0, "DSO": 5000.0,
        "nVoxel": [vol_size] * 3, "dVoxel": [2.0] * 3,
        "nDetector": [detector_size, detector_size], "dDetector": [1.0, 1.0],
        "offOrigin": [0.0, 0.0, 0.0], "offDetector": [0.0, 0.0],
        "mode": "parallel",
    }
    save_naf_format(out_dir, projections, angles, gt_vol, scanner_cfg)
    # X²-Gaussian's evaluator iterates 10 phases and indexes vol_gt[t]; for the
    # static benchmark we replicate the single GT volume across 10 phases and
    # add zero time/phase fields so the same data.pkl works for both static and
    # 4D-trained methods.
    import pickle as _pkl
    pkl_path = out_dir / "data.pkl"
    with open(pkl_path, "rb") as f:
        d = _pkl.load(f)
    if d["image"].ndim == 3:
        d["image"] = np.stack([d["image"]] * 10, axis=0)
    for split in ("train", "val"):
        n = len(d[split]["angles"])
        d[split]["time"] = np.zeros(n, dtype=np.float32)
        d[split]["phase"] = np.zeros(n, dtype=np.int32)
    with open(pkl_path, "wb") as f:
        _pkl.dump(d, f, protocol=4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", type=str,
                   default="/data/aet-bench/data/aet_bench/mono")
    p.add_argument("--backend", choices=["analytical", "both"],
                   default="analytical",
                   help="'analytical' (fast) or 'both' (also abTEM multislice)")
    p.add_argument("--n_samples", type=int, default=50,
                   help="Cap number of samples (default 50 = 5 elem x 10 sizes)")
    p.add_argument("--n_angles", type=int, default=71)
    p.add_argument("--tilt_range", type=float, default=70.0)
    p.add_argument("--detector_size", type=int, default=128)
    p.add_argument("--vol_size", type=int, default=128)
    p.add_argument("--abtem_script", type=str,
                   default="/data/aet-bench/scripts/generate_abtem_multislice.py")
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    samples = []
    for elem in ELEMENTS:
        for n in SIZE_LADDER:
            samples.append((elem, n))
            if len(samples) >= args.n_samples:
                break
        if len(samples) >= args.n_samples:
            break

    print(f"Building {len(samples)} Mono samples → {out_root}")
    print(f"  backend={args.backend}, n_angles={args.n_angles}, "
          f"detector={args.detector_size}, vol={args.vol_size}")

    manifest = []
    for idx, (elem, target_n) in enumerate(samples):
        atoms = build_octahedral(elem, target_n, seed=idx)
        positions = atoms.positions.copy()
        atomic_nums = atoms.get_atomic_numbers().copy()
        n_atoms = len(atoms)

        sample_id = make_sample_id(elem, n_atoms, idx)
        sample_dir = out_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # GT atom coordinates
        atom_data = np.column_stack([positions, atomic_nums]).astype(np.float32)
        np.save(sample_dir / "atoms_gt.npy", atom_data)

        # Cell size: bounding sphere + 5 Å padding each side
        r_max = float(np.linalg.norm(positions, axis=1).max())
        cell_size = 2 * r_max + 10.0

        meta = {
            "sample_id": sample_id,
            "element": elem,
            "n_atoms": int(n_atoms),
            "atomic_number": int(Z_TABLE[elem]),
            "lattice_constant": LATTICE_CONST[elem],
            "morphology": "octahedral_fcc",
            "radius_angstrom": r_max,
            "cell_size_angstrom": cell_size,
            "n_angles": args.n_angles,
            "tilt_range_deg": args.tilt_range,
            "detector_size": args.detector_size,
            "vol_size": args.vol_size,
            "backends": [],
        }

        print(f"\n[{idx+1}/{len(samples)}] {sample_id}: {n_atoms} atoms, "
              f"r={r_max:.1f} Å")

        # Analytical (always)
        analytical_dir = sample_dir / "analytical"
        generate_analytical(
            positions, atomic_nums, cell_size, analytical_dir,
            tilt_range=args.tilt_range, n_angles=args.n_angles,
            detector_size=args.detector_size, vol_size=args.vol_size,
        )
        meta["backends"].append("analytical")

        # abTEM (opt-in, very slow — delegates to existing script)
        if args.backend == "both":
            abtem_dir = sample_dir / "abtem"
            cmd = [
                sys.executable, args.abtem_script,
                "--structure", sample_id,  # script reads structure.xyz
                "--n_angles", str(args.n_angles),
                "--tilt_range", str(args.tilt_range),
                "--detector_size", str(args.detector_size),
                "--vol_size", str(args.vol_size),
            ]
            # generate_abtem_multislice.py expects structure.xyz + metadata.json
            # in data/simulated/<structure>/. Easiest: drop them there.
            staging = Path("/data/aet-bench/data/simulated") / sample_id
            staging.mkdir(parents=True, exist_ok=True)
            from ase.io import write as ase_write
            ase_write(str(staging / "structure.xyz"), atoms)
            np.save(staging / "atoms_gt.npy", atom_data)
            with open(staging / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
            print(f"  -> running abTEM multislice (slow)...")
            try:
                subprocess.run(cmd, check=True)
                # Move outputs into sample_dir/abtem/
                src = Path("/data/aet-bench/data/simulated") / f"{sample_id}_abtem"
                if src.exists():
                    abtem_dir.mkdir(parents=True, exist_ok=True)
                    for f in src.iterdir():
                        f.rename(abtem_dir / f.name)
                meta["backends"].append("abtem")
            except subprocess.CalledProcessError as e:
                print(f"  abTEM failed for {sample_id}: {e}")

        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        manifest.append(meta)

    with open(out_root / "manifest.json", "w") as f:
        json.dump({"n_samples": len(manifest), "samples": manifest}, f, indent=2)

    print(f"\nDone. Manifest: {out_root}/manifest.json ({len(manifest)} samples)")


if __name__ == "__main__":
    main()
