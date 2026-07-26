#!/usr/bin/env python
"""
Build nanoparticle structures for AET-4DGS benchmarking.

Structures (per plan Phase 1b.1):
  Cu-309   (small, fast iteration)
  Cu-923   (medium, main benchmark)
  Pt-923   (different Z, test Z-contrast)
  FePt-923 (binary alloy, test element discrimination)
  Cu-2057  (large, stress test)
"""

import numpy as np
from pathlib import Path
from ase.cluster import FaceCenteredCubic
from ase.data import atomic_numbers as Z_TABLE
from ase import Atoms
import json


# FCC lattice constants (Angstroms)
LATTICE = {"Cu": 3.615, "Pt": 3.924, "Au": 4.078, "Fe": 2.866, "Ag": 4.085}


def build_fcc_cluster(element, target_atoms, a=None):
    """Build FCC nanoparticle with approximately target_atoms atoms."""
    if a is None:
        a = LATTICE[element]

    # Empirical: n_atoms ~ 10*n_layers^3 for octahedral FCC
    # Try increasing layers until we exceed target
    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    for n in range(2, 20):
        atoms = FaceCenteredCubic(element, surfaces, [n, n, n], latticeconstant=a)
        if len(atoms) >= target_atoms * 0.8:
            break

    # Center
    atoms.positions -= atoms.positions.mean(axis=0)
    return atoms


def build_fept_cluster(target_atoms):
    """Build L10-ordered FePt nanoparticle."""
    # Use Pt lattice constant (compromise for alloy)
    a = 3.86  # approximate FePt L10 lattice constant

    atoms = build_fcc_cluster("Pt", target_atoms, a=a)
    positions = atoms.positions.copy()

    # L10 ordering: alternate Fe and Pt layers along z
    z_vals = positions[:, 2]
    layer_spacing = a / 2
    z_min = z_vals.min()

    numbers = np.full(len(atoms), Z_TABLE["Pt"])
    for i, z in enumerate(z_vals):
        layer_idx = int(round((z - z_min) / layer_spacing))
        if layer_idx % 2 == 0:
            numbers[i] = Z_TABLE["Fe"]

    atoms.set_atomic_numbers(numbers)
    return atoms


def save_structure(atoms, name, out_dir):
    """Save atomic structure in multiple formats."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    positions = atoms.positions.copy()
    atomic_nums = atoms.get_atomic_numbers()

    # Save as numpy (our format)
    atom_data = np.column_stack([positions, atomic_nums])
    np.save(out_dir / "atoms_gt.npy", atom_data.astype(np.float32))

    # Save as ASE atoms object
    from ase.io import write
    write(str(out_dir / "structure.xyz"), atoms)

    # Save metadata
    unique_z = np.unique(atomic_nums)
    element_counts = {int(z): int((atomic_nums == z).sum()) for z in unique_z}
    r_max = np.linalg.norm(positions, axis=1).max()

    metadata = {
        "name": name,
        "n_atoms": len(atoms),
        "elements": element_counts,
        "radius_angstrom": float(r_max),
        "cell_size_angstrom": float(2 * r_max + 10),
        "lattice_constant": None,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  {name}: {len(atoms)} atoms, radius={r_max:.1f} Å, "
          f"elements={element_counts}")
    return metadata


def main():
    base_dir = Path("/data/aet-bench/data/simulated")

    structures = [
        ("Cu-309",   "Cu",   309),
        ("Cu-923",   "Cu",   923),
        ("Pt-923",   "Pt",   923),
        ("Cu-2057",  "Cu",   2057),
    ]

    print("Building nanoparticle structures...")
    print("=" * 60)

    all_meta = {}
    for name, element, target in structures:
        atoms = build_fcc_cluster(element, target)
        meta = save_structure(atoms, name, base_dir / name)
        all_meta[name] = meta

    # FePt special case
    atoms_fept = build_fept_cluster(923)
    meta = save_structure(atoms_fept, "FePt-923", base_dir / "FePt-923")
    all_meta["FePt-923"] = meta

    print("\n" + "=" * 60)
    print("Summary:")
    for name, meta in all_meta.items():
        print(f"  {name}: {meta['n_atoms']} atoms, r={meta['radius_angstrom']:.1f} Å")
    print(f"\nAll saved to {base_dir}")


if __name__ == "__main__":
    main()
