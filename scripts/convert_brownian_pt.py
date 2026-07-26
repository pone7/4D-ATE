#!/usr/bin/env python
"""
Convert the Brownian-tomography Pt nanocrystal dataset (Zenodo 15531575)
into NAF pickle format for AET-Bench.

Source: 66 published 3D reconstructions of Pt nanocrystals (time-resolved
Brownian tomography, Zhou group). For each sample we have:
  - ./mrcs/<id>.mrc  : reconstructed 3D volume (160³, ~0.36 Å voxel)
  - ./pdbs/<id>.pdb  : atomic model extracted from the reconstruction

We don't have raw tilt series, so we create a "hybrid" data tier:
**real atomic structure + synthetic linear projections** obtained by forward
projecting the published volumes via Radon transform. This lets every
reconstruction method in our benchmark run end-to-end while keeping the
ground truth tied to a real physical structure.

Output:
  data/aet_bench/brownian_pt/<sample_id>/
      atoms_gt.npy                 # (N, 4) xyz + Z
      metadata.json
      analytical/data.pkl          # NAF-format synthetic tilt series
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import mrcfile
import numpy as np

RAW = Path("/data/aet-bench/data/real_raw/brownian_pt")
OUT = Path("/data/aet-bench/data/aet_bench/brownian_pt")

TARGET_VOL = 128
N_ANGLES = 60
TILT_RANGE_DEG = 70.0


def load_pdb_atoms(pdb_path: Path) -> np.ndarray:
    """Return (N, 4) atomic positions (x,y,z in Å) + atomic number (Pt=78)."""
    lines = pdb_path.read_text().splitlines()
    atoms = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
    coords = np.array(
        [[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in atoms]
    )
    return np.column_stack([coords, np.full(len(coords), 78.0)]).astype(np.float32)


def load_mrc_volume(mrc_path: Path):
    """Return (volume, voxel_size_angstrom)."""
    with mrcfile.open(str(mrc_path), permissive=True) as m:
        vol = np.asarray(m.data, dtype=np.float32).copy()
        cell_x = float(m.header.cella.x)
    voxel = cell_x / vol.shape[0]
    return vol, voxel


def radon_project(vol: np.ndarray, angles_rad: np.ndarray) -> np.ndarray:
    """Forward project the volume along y-axis rotations.

    Uses ASTRA's parallel3d projector. Returns (n_angles, det_rows, det_cols)
    with det = vol.shape[0].
    """
    import astra
    V = vol.shape[0]
    vol_geom = astra.create_vol_geom(V, V, V)
    proj_geom = astra.create_proj_geom("parallel3d", 1.0, 1.0, V, V, angles_rad)
    vol_id = astra.data3d.create("-vol", vol_geom, vol.astype(np.float32))
    proj_id = astra.data3d.create("-proj3d", proj_geom, 0.0)
    cfg = astra.astra_dict("FP3D_CUDA")
    cfg["ProjectionDataId"] = proj_id
    cfg["VolumeDataId"] = vol_id
    alg = astra.algorithm.create(cfg)
    astra.algorithm.run(alg)
    astra.algorithm.delete(alg)
    projs_vna = astra.data3d.get(proj_id).astype(np.float32)
    astra.data3d.delete(vol_id)
    astra.data3d.delete(proj_id)
    # convert (det_rows, n_angles, det_cols) → (n_angles, det_rows, det_cols)
    return np.transpose(projs_vna, (1, 0, 2)).copy()


def _default_scanner(vol: int):
    return {
        "DSD": 10000.0, "DSO": 5000.0,
        "nVoxel": [vol] * 3, "dVoxel": [2.0] * 3,
        "nDetector": [vol, vol], "dDetector": [1.0, 1.0],
        "offOrigin": [0.0, 0.0, 0.0], "offDetector": [0.0, 0.0],
        "mode": "parallel",
    }


def convert_one(mrc_path: Path, pdb_path: Path, out_dir: Path):
    vol_orig, voxel_orig = load_mrc_volume(mrc_path)
    atoms_raw = load_pdb_atoms(pdb_path)

    # Center atoms at volume center (PDB uses cell coordinates, not origin)
    V_orig = vol_orig.shape[0]
    cell_size_orig = V_orig * voxel_orig
    atoms_centered = atoms_raw.copy()
    atoms_centered[:, :3] -= cell_size_orig / 2.0

    # Downsample volume to TARGET_VOL for benchmark consistency
    from skimage.transform import resize
    vol = resize(vol_orig, (TARGET_VOL,) * 3, anti_aliasing=True,
                  preserve_range=True).astype(np.float32)
    # Clip negatives (brownian reconstructions have mild negative values)
    vol = np.clip(vol, 0, None)

    # Forward project via ASTRA at TARGET_VOL resolution
    angles_rad = np.radians(np.linspace(-TILT_RANGE_DEG, TILT_RANGE_DEG, N_ANGLES))
    projs = radon_project(vol, angles_rad)
    projs_nhw = projs.astype(np.float32)

    # Build NAF-format dict
    out_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(angles_rad)
    n_test = max(1, n_total // 8)
    test_step = max(1, n_total // n_test)
    test_idx = sorted(list(range(0, n_total, test_step))[:n_test])
    train_idx = sorted(set(range(n_total)) - set(test_idx))

    image_stack = np.stack([vol] * 10, axis=0)
    data = {
        **_default_scanner(TARGET_VOL),
        "totalAngle": float(angles_rad.max() - angles_rad.min()),
        "startAngle": float(angles_rad.min()),
        "accuracy": 0.5,
        "numTrain": len(train_idx), "numVal": len(test_idx),
        "image": image_stack,
        "train": {
            "angles": angles_rad[train_idx].astype(np.float64),
            "projections": projs_nhw[train_idx],
            "time": np.zeros(len(train_idx), dtype=np.float32),
            "phase": np.zeros(len(train_idx), dtype=np.int32),
        },
        "val": {
            "angles": angles_rad[test_idx].astype(np.float64),
            "projections": projs_nhw[test_idx],
            "time": np.zeros(len(test_idx), dtype=np.float32),
            "phase": np.zeros(len(test_idx), dtype=np.int32),
        },
    }
    (out_dir / "analytical").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "analytical" / "data.pkl", "wb") as f:
        pickle.dump(data, f, protocol=4)

    np.save(out_dir / "atoms_gt.npy", atoms_centered.astype(np.float32))

    meta = {
        "sample_id": out_dir.name,
        "source": "Zhou group 2024 Brownian Pt (Zenodo 15531575)",
        "kind": "real_atoms_synthetic_projections",
        "element": "Pt",
        "n_atoms": int(atoms_centered.shape[0]),
        "n_angles": int(len(angles_rad)),
        "tilt_range_deg": float(TILT_RANGE_DEG * 2),
        "cell_size_angstrom": float(TARGET_VOL),  # downsampled cell in voxel units
        "voxel_size_angstrom_original": float(voxel_orig),
        "original_vol_size": int(V_orig),
        "target_vol_size": int(TARGET_VOL),
        "has_atom_gt": True,
        "note": "Volume from published reconstruction, projections computed via ASTRA FP3D_CUDA Radon",
        "backends": ["analytical"],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta, vol, atoms_centered


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mrc_paths = sorted((RAW / "mrcs").glob("*.mrc"))
    pdb_paths = {p.stem: p for p in (RAW / "pdbs").glob("*.pdb")}

    manifest = []
    for mrc_path in mrc_paths:
        stem = mrc_path.stem
        pdb_path = pdb_paths.get(stem)
        if pdb_path is None:
            print(f"  skip {stem}: no matching PDB")
            continue
        # Correct cell_size_angstrom after we know the physical scale
        sample_dir = OUT / f"Pt-{stem}"
        try:
            meta, vol, atoms = convert_one(mrc_path, pdb_path, sample_dir)
            # Rewrite cell_size_angstrom using physical coordinates — atoms
            # already express Å. The downsampled cell is V_target * voxel_new.
            # voxel_new = voxel_orig * V_orig / V_target, so cell = V_orig * voxel_orig
            cell_phys = meta["original_vol_size"] * meta["voxel_size_angstrom_original"]
            meta["cell_size_angstrom"] = float(cell_phys)
            meta["voxel_size_angstrom_downsampled"] = float(cell_phys / TARGET_VOL)
            with open(sample_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
            manifest.append(meta)
            print(f"  Pt-{stem}: {meta['n_atoms']:4d} atoms, cell={cell_phys:.1f} Å, "
                  f"vox={cell_phys/TARGET_VOL:.3f} Å")
        except Exception as e:
            print(f"  FAILED Pt-{stem}: {e}")

    with open(OUT / "manifest.json", "w") as f:
        json.dump({"n_samples": len(manifest), "samples": manifest}, f, indent=2)
    print(f"\nDone. {len(manifest)} brownian Pt samples → {OUT}")


if __name__ == "__main__":
    main()
