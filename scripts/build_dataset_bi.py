#!/usr/bin/env python
"""
AET-Bench: build the AET-Sim-Bi subset (binary alloy nanoparticles).

Generates ~30 binary alloy samples spanning three systems:
  - CuPt  (random solid solution, 50/50)
  - FePt  (L10 ordered, alternating layers)
  - AuAg  (random solid solution, 50/50)

For each system, sweeps the size ladder. Always emits analytical HAADF
tilt series (fast). abTEM backend opt-in via --backend both.

Output layout matches Mono:
    data/aet_bench/bi/<sample_id>/
        atoms_gt.npy
        metadata.json
        analytical/data.pkl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.cluster import FaceCenteredCubic
from ase.data import atomic_numbers as Z_TABLE

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset_mono import generate_analytical, LATTICE_CONST  # noqa: E402


SIZE_LADDER = [201, 309, 459, 711, 923, 1157, 1415, 1697, 2057, 309]
SYSTEMS = [
    ("CuPt", "Cu", "Pt", "random"),
    ("FePt", "Fe", "Pt", "L10"),
    ("AuAg", "Au", "Ag", "random"),
]
LATTICE_CONST_BI = {**LATTICE_CONST, "Fe": 2.866}


def build_alloy(elem_a, elem_b, ordering, target_n, seed=0):
    """Build a binary FCC nanoparticle.

    `ordering`:
      - "random": 50/50 random substitution
      - "L10":    alternating elem_a / elem_b layers along z (FePt-style)
    """
    rng = np.random.default_rng(seed)
    a = LATTICE_CONST_BI.get(elem_a, 3.8)
    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    for n in range(2, 18):
        atoms = FaceCenteredCubic(elem_a, surfaces, [n, n, n], latticeconstant=a)
        if len(atoms) >= target_n:
            break
    atoms.positions -= atoms.positions.mean(axis=0)

    Z_a, Z_b = Z_TABLE[elem_a], Z_TABLE[elem_b]
    nums = np.full(len(atoms), Z_a)
    if ordering == "random":
        idx = rng.choice(len(atoms), len(atoms) // 2, replace=False)
        nums[idx] = Z_b
    elif ordering == "L10":
        z = atoms.positions[:, 2]
        layer = np.round((z - z.min()) / (a / 2)).astype(int)
        nums[layer % 2 == 1] = Z_b
    else:
        raise ValueError(ordering)
    atoms.set_atomic_numbers(nums)
    return atoms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", type=str,
                   default="/data/aet-bench/data/aet_bench/bi")
    p.add_argument("--n_samples", type=int, default=30)
    p.add_argument("--n_angles", type=int, default=71)
    p.add_argument("--tilt_range", type=float, default=70.0)
    p.add_argument("--detector_size", type=int, default=128)
    p.add_argument("--vol_size", type=int, default=128)
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    samples = []
    for sys_name, a_el, b_el, ordering in SYSTEMS:
        for n in SIZE_LADDER:
            samples.append((sys_name, a_el, b_el, ordering, n))
            if len(samples) >= args.n_samples:
                break
        if len(samples) >= args.n_samples:
            break

    print(f"Building {len(samples)} Bi samples → {out_root}")

    manifest = []
    for idx, (sys_name, a_el, b_el, ordering, target_n) in enumerate(samples):
        atoms = build_alloy(a_el, b_el, ordering, target_n, seed=idx)
        positions = atoms.positions.copy()
        atomic_nums = atoms.get_atomic_numbers().copy()
        n_atoms = len(atoms)

        sample_id = f"{sys_name}-{n_atoms:04d}-{idx:02d}"
        sample_dir = out_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        atom_data = np.column_stack([positions, atomic_nums]).astype(np.float32)
        np.save(sample_dir / "atoms_gt.npy", atom_data)

        r_max = float(np.linalg.norm(positions, axis=1).max())
        cell_size = 2 * r_max + 10.0

        unique = {int(z): int((atomic_nums == z).sum())
                  for z in np.unique(atomic_nums)}
        meta = {
            "sample_id": sample_id,
            "system": sys_name,
            "elements": [a_el, b_el],
            "ordering": ordering,
            "n_atoms": int(n_atoms),
            "element_counts": unique,
            "morphology": "octahedral_fcc",
            "radius_angstrom": r_max,
            "cell_size_angstrom": cell_size,
            "n_angles": args.n_angles,
            "tilt_range_deg": args.tilt_range,
            "detector_size": args.detector_size,
            "vol_size": args.vol_size,
            "backends": ["analytical"],
        }
        print(f"\n[{idx+1}/{len(samples)}] {sample_id}: {n_atoms} atoms, "
              f"{unique}, r={r_max:.1f} Å")

        generate_analytical(
            positions, atomic_nums, cell_size, sample_dir / "analytical",
            tilt_range=args.tilt_range, n_angles=args.n_angles,
            detector_size=args.detector_size, vol_size=args.vol_size,
        )
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        manifest.append(meta)

    with open(out_root / "manifest.json", "w") as f:
        json.dump({"n_samples": len(manifest), "samples": manifest}, f, indent=2)
    print(f"\nDone. {len(manifest)} samples → {out_root}")


if __name__ == "__main__":
    main()
