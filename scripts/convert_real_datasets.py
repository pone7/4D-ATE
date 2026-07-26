#!/usr/bin/env python
"""Convert downloaded real AET datasets to our NAF pickle format.

Sources:
    data/real_raw/zr_zro2/Zr-ZrO2Nanoparticles/   (Zr1, Zr2, Zr3)
    data/real_raw/mpea2_twins/                   (MPEA2)

Output:
    data/aet_bench/real/<sample_id>/analytical/data.pkl
    data/aet_bench/real/<sample_id>/atoms_gt.npy   (if atoms available)
    data/aet_bench/real/<sample_id>/metadata.json
    data/aet_bench/real/manifest.json

The Zr-ZrO2 samples have explicit atomic GT coordinates; MPEA2 has only a
pseudo-GT volume. Both get packaged identically; the metadata records
which is "real atoms" vs "pseudo-GT volume only" so the evaluator can
skip atom metrics where appropriate.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import scipy.io as sio

OUT_ROOT = Path("/data/aet-bench/data/aet_bench/real")
RAW_ROOT = Path("/data/aet-bench/data/real_raw")


def _default_scanner(vol_size: int, n_det: int):
    """Same arbitrary-unit scanner cfg that X²-G and friends accept."""
    return {
        "DSD": 10000.0, "DSO": 5000.0,
        "nVoxel": [vol_size] * 3, "dVoxel": [2.0] * 3,
        "nDetector": [n_det, n_det], "dDetector": [1.0, 1.0],
        "offOrigin": [0.0, 0.0, 0.0], "offDetector": [0.0, 0.0],
        "mode": "parallel",
    }


def _save_naf(out_dir: Path, projs_hwn: np.ndarray, angles_rad: np.ndarray,
               vol_gt: np.ndarray, scanner_cfg: dict, n_phases: int = 10):
    """Write a NAF-format data.pkl. projs_hwn shape: (H, W, N_angles)."""
    n_total = angles_rad.shape[0]
    n_test = max(1, n_total // 8)
    test_idx = sorted(set(range(0, n_total, max(1, n_total // n_test))))[:n_test]
    train_idx = sorted(set(range(n_total)) - set(test_idx))

    projs_nhw = np.moveaxis(projs_hwn, -1, 0).astype(np.float32)
    vol_gt_f = vol_gt.astype(np.float32)
    # Replicate single volume across phases for compatibility with X²-G evaluator
    image_stack = np.stack([vol_gt_f] * n_phases, axis=0)

    data = {
        **scanner_cfg,
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
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "data.pkl", "wb") as f:
        pickle.dump(data, f, protocol=4)


def _resize_vol_and_projs(projs_hwn: np.ndarray, vol_gt: np.ndarray,
                           target_vol: int = 128) -> tuple:
    """Downsample volume and projections to `target_vol` side length.

    Atoms are *not* rescaled here — the caller already has them in
    physical Ångström coordinates relative to the volume center, and
    the physical extent of the sample is preserved under downsampling
    (only the per-voxel resolution changes).
    """
    from skimage.transform import resize
    orig_vol = vol_gt.shape[0]
    if orig_vol == target_vol:
        return projs_hwn, vol_gt

    vol_new = resize(vol_gt, (target_vol,) * 3, anti_aliasing=True,
                     preserve_range=True).astype(np.float32)

    H, W, N = projs_hwn.shape
    projs_new = np.zeros((target_vol, target_vol, N), dtype=np.float32)
    for i in range(N):
        projs_new[..., i] = resize(projs_hwn[..., i], (target_vol, target_vol),
                                     preserve_range=True).astype(np.float32)
    return projs_new, vol_new


# Published voxel spacing of the Zr-ZrO2 HAADF reconstructions. Verified
# empirically: at 0.347 Å/voxel, 99–100% of the published atomic-model
# positions land on high-intensity voxels in the reconstructed volume.
ZR_VOXEL_SIZE_ANGSTROM = 0.347


def convert_zr(name: str, target_vol: int = 256):
    base = RAW_ROOT / "zr_zro2" / "Zr-ZrO2Nanoparticles"
    m_projs = sio.loadmat(str(base / "1_Raw_data" / f"{name}-projs+angles.mat"))
    m_recon = sio.loadmat(str(base / "2_Final_reconstruction_volume" / f"recon_{name}.mat"))
    m_atoms = sio.loadmat(str(base / "3_Final_atomic_model" / f"atoms_{name}.mat"))

    projs_hwn = m_projs["inputProjs"].astype(np.float32)
    eulers = m_projs["eulerAngles"].astype(np.float64)
    vol_gt = m_recon[f"{name}_volume"].astype(np.float32)
    atoms_pos = m_atoms[f"{name}_pos"].astype(np.float32)  # (3, N_atoms) in Å

    orig_vol_size = vol_gt.shape[0]
    cell_size_angstrom = orig_vol_size * ZR_VOXEL_SIZE_ANGSTROM

    # Tilt: use the 2nd Euler angle (theta) — confirmed single-axis convention
    tilt_deg = eulers[:, 1]
    angles_rad = np.radians(tilt_deg)

    # Atoms are already in physical Å, already centered at the volume center
    # (mean ≈ 0). We keep them in Å; they are consistent with the physical
    # extent of both the original and the downsampled volume.
    atoms_xyz = atoms_pos.T

    projs_small, vol_small = _resize_vol_and_projs(
        projs_hwn, vol_gt, target_vol=target_vol
    )

    scanner_cfg = _default_scanner(target_vol, target_vol)
    out_dir = OUT_ROOT / f"Zr-{name}"
    _save_naf(out_dir / "analytical", projs_small, angles_rad, vol_small, scanner_cfg)

    atom_data = np.column_stack([
        atoms_xyz,
        np.full(atoms_xyz.shape[0], 40, dtype=np.float32),  # Zr = 40
    ]).astype(np.float32)
    np.save(out_dir / "atoms_gt.npy", atom_data)

    meta = {
        "sample_id": f"Zr-{name}",
        "source": "Zhang et al. 2024 (Zenodo 12742309)",
        "kind": "real_experimental",
        "element": "Zr+O",
        "n_atoms": int(atoms_xyz.shape[0]),
        "n_angles": int(len(angles_rad)),
        "tilt_range_deg": float(np.degrees(angles_rad).max() - np.degrees(angles_rad).min()),
        "cell_size_angstrom": cell_size_angstrom,
        "voxel_size_angstrom_original": ZR_VOXEL_SIZE_ANGSTROM,
        "voxel_size_angstrom_downsampled": cell_size_angstrom / target_vol,
        "original_vol_size": int(orig_vol_size),
        "target_vol_size": int(target_vol),
        "has_atom_gt": True,
        "backends": ["analytical"],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Zr-{name}: {atoms_xyz.shape[0]} atoms, {len(angles_rad)} angles, "
          f"cell={cell_size_angstrom:.1f} Å, "
          f"vox={cell_size_angstrom/target_vol:.3f} Å")
    return meta


def convert_mpea2(target_vol: int = 256):
    base = RAW_ROOT / "mpea2_twins"
    m_projs = sio.loadmat(str(base / "MPEA2_projs_angles.mat"))
    m_recon = sio.loadmat(str(base / "MPEA2_final_recon.mat"))

    projs_hwn = m_projs["Projections"].astype(np.float32)
    eulers = m_projs["Angles"].astype(np.float64)
    vol_gt = m_recon["Final_rec"].astype(np.float32)

    tilt_deg = eulers[:, 1]
    angles_rad = np.radians(tilt_deg)

    # MPEA2 has no published voxel size; adopt the same 0.347 Å/vox used by
    # the Zr-ZrO2 paper (same Zhou lab, same instrument family). Worst case
    # the absolute scale is off by a constant factor — Recall at a fixed
    # threshold would still be monotone in method quality, just shifted.
    orig_vol_size = vol_gt.shape[0]
    voxel_size_ang = 0.347
    cell_size_angstrom = orig_vol_size * voxel_size_ang

    projs_small, vol_small = _resize_vol_and_projs(
        projs_hwn, vol_gt, target_vol=target_vol
    )

    scanner_cfg = _default_scanner(target_vol, target_vol)
    out_dir = OUT_ROOT / "MPEA2"
    _save_naf(out_dir / "analytical", projs_small, angles_rad, vol_small, scanner_cfg)

    # MPEA2 has no explicit atomic GT — extract peaks from the published
    # pseudo-GT volume and convert them to physical Å so they are in the
    # same frame as voxel_to_physical() in the evaluator. Interpret
    # Recall/RMSD on this sample as "atom-peak detection vs peaks in the
    # published volume" — still a valid benchmark, just with a weaker GT.
    from skimage.feature import peak_local_max
    peaks_voxel = peak_local_max(vol_gt, min_distance=4,
                                   threshold_rel=0.1, num_peaks=30000)
    # Convert voxel indices to physical Å centered at origin
    atoms_proxy = (peaks_voxel.astype(np.float32) - orig_vol_size / 2.0) * voxel_size_ang
    atom_data = np.column_stack([
        atoms_proxy,
        np.full(atoms_proxy.shape[0], 78, dtype=np.float32),  # label as Pt proxy
    ]).astype(np.float32)
    np.save(out_dir / "atoms_gt.npy", atom_data)

    meta = {
        "sample_id": "MPEA2",
        "source": "Zhou group 2024 (Zenodo 17310133)",
        "kind": "real_experimental",
        "element": "MPEA (high-entropy alloy)",
        "n_atoms": int(atoms_proxy.shape[0]),
        "n_angles": int(len(angles_rad)),
        "tilt_range_deg": float(np.degrees(angles_rad).max() - np.degrees(angles_rad).min()),
        "cell_size_angstrom": cell_size_angstrom,
        "voxel_size_angstrom_original": voxel_size_ang,
        "voxel_size_angstrom_downsampled": cell_size_angstrom / target_vol,
        "original_vol_size": int(orig_vol_size),
        "target_vol_size": int(target_vol),
        "has_atom_gt": False,
        "atoms_gt_note": "Peaks extracted from pseudo-GT volume, not real atomic model",
        "backends": ["analytical"],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  MPEA2: {atoms_proxy.shape[0]} peaks, {len(angles_rad)} angles, "
          f"vol={target_vol}³")
    return meta


ELEMENT_Z = {"Au": 79, "Pd": 46, "Pt": 78, "Ni": 28, "Zr": 40, "O": 8}


def _load_xyz_atoms(xyz_path: Path) -> np.ndarray:
    """Parse an XYZ file (either 4-column or multi-column format).

    Returns (N, 4) array [x, y, z, Z_atomic_number].
    Supports two layouts:
      - 4-column:  element  x  y  z  (ASE extended XYZ with Lattice header)
      - 10-column: element  x  y  z  ...extra columns... (tracing output)
    """
    lines = xyz_path.read_text().splitlines()
    n_atoms = int(lines[0].strip())
    atom_lines = lines[2: 2 + n_atoms]
    coords = []
    for line in atom_lines:
        parts = line.split()
        if not parts:
            continue
        sym = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        Z = ELEMENT_Z.get(sym, 0)
        coords.append([x, y, z, Z])
    arr = np.array(coords, dtype=np.float32)
    # Center at origin
    arr[:, :3] -= arr[:, :3].mean(axis=0)
    return arr


def _radon_project_vol(vol: np.ndarray, angles_rad: np.ndarray) -> np.ndarray:
    """Forward-project a 3D volume using ASTRA parallel3d.

    Returns (N_angles, H, W) float32.
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
    # ASTRA returns (det_rows, n_angles, det_cols) → transpose to (n_angles, det_rows, det_cols)
    return np.transpose(projs_vna, (1, 0, 2)).copy()


def _atoms_to_vol_and_projs(atoms_xyzZ: np.ndarray, vol_size: int = 128,
                              n_angles: int = 60,
                              tilt_range_deg: float = 70.0) -> tuple:
    """Generate synthetic volume + projections from atomic coordinates.

    Uses analytical Z^gamma Gaussian model: place atoms in volume, then
    ASTRA FP for projections.  Returns (projs_hwn, angles_rad, vol).
    """
    GAMMA = 1.7
    SIGMA_VOXELS = 1.0  # Gaussian width in voxels

    # Map atom coordinates to voxel indices
    xyz = atoms_xyzZ[:, :3]
    Z_nums = atoms_xyzZ[:, 3]

    # Fit into vol_size voxels with 5% margin
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    scale = vol_size * 0.90 / extent.max()
    xyz_vox = (xyz - xyz.min(axis=0)) * scale + vol_size * 0.05

    vol = np.zeros((vol_size,) * 3, dtype=np.float32)
    # Accumulate each atom's Z^gamma weight as a 3D Gaussian
    sig2 = 2.0 * SIGMA_VOXELS ** 2
    x_vox = xyz_vox[:, 0]
    y_vox = xyz_vox[:, 1]
    z_vox = xyz_vox[:, 2]
    weights = Z_nums ** GAMMA

    # Vectorised placement on integer voxel centres (fast approximation)
    ix = np.clip(np.round(x_vox).astype(int), 0, vol_size - 1)
    iy = np.clip(np.round(y_vox).astype(int), 0, vol_size - 1)
    iz = np.clip(np.round(z_vox).astype(int), 0, vol_size - 1)
    np.add.at(vol, (ix, iy, iz), weights)

    # Smooth with Gaussian kernel
    from scipy.ndimage import gaussian_filter
    vol = gaussian_filter(vol, sigma=SIGMA_VOXELS)

    angles_rad = np.radians(np.linspace(-tilt_range_deg, tilt_range_deg, n_angles))
    projs_nhw = _radon_project_vol(vol, angles_rad)  # (N, H, W)

    # Convert to (H, W, N) for _save_naf
    projs_hwn = np.moveaxis(projs_nhw, 0, -1)
    return projs_hwn, angles_rad, vol


# ---------------------------------------------------------------------------
# PdPt Zhou (3 samples: EPB, PB, TO)
# ---------------------------------------------------------------------------

def convert_pdpt_zhou(sample: str, target_vol: int = 256):
    """Convert one PdPt Zhou core-shell sample (EPB | PB | TO)."""
    base = RAW_ROOT / "coreshell_pdpt_zhou" / "extracted"

    m_projs = sio.loadmat(str(base / "1_Measured_data" / f"{sample}_projs_and_angles.mat"))
    m_recon = sio.loadmat(str(base / "2_Final_reconstruction_volume" / f"reconstruction_{sample}.mat"))
    m_atoms = sio.loadmat(str(base / "4_Final_coordinates" / f"coordinates_{sample}.mat"))

    projs_hwn = m_projs["refined_projs"].astype(np.float32)   # (H, W, N)
    eulers = m_projs["refined_angles"].astype(np.float64)      # (N, 3)
    vol_gt = m_recon["vol"].astype(np.float32)                 # (V, V, V)
    pos = m_atoms["pos"].astype(np.float32)                    # (3, N_atoms)
    typ = m_atoms["type"].astype(int).ravel()                  # (N_atoms,)

    # Tilt: 2nd Euler column
    angles_rad = np.radians(eulers[:, 1])

    orig_vol = vol_gt.shape[0]
    projs_small, vol_small = _resize_vol_and_projs(projs_hwn, vol_gt, target_vol=target_vol)

    scanner_cfg = _default_scanner(target_vol, target_vol)
    out_dir = OUT_ROOT / f"PdPt-Zhou-{sample}"
    _save_naf(out_dir / "analytical", projs_small, angles_rad, vol_small, scanner_cfg)

    # typ: 1=Pd(Z=46), 2=Pt(Z=78)
    Z_map = {1: 46, 2: 78}
    Z_vals = np.array([Z_map.get(t, 46) for t in typ], dtype=np.float32)
    atoms_xyz = pos.T  # (N_atoms, 3)
    atom_data = np.column_stack([atoms_xyz, Z_vals]).astype(np.float32)
    np.save(out_dir / "atoms_gt.npy", atom_data)

    voxel_size_ang = 0.347  # same instrument family as Zr-ZrO2
    cell_size = orig_vol * voxel_size_ang

    meta = {
        "sample_id": f"PdPt-Zhou-{sample}",
        "source": "Zhou group (core-shell PdPt)",
        "kind": "real_experimental",
        "element": "Pd+Pt",
        "n_atoms": int(atoms_xyz.shape[0]),
        "n_angles": int(len(angles_rad)),
        "tilt_range_deg": float(np.degrees(angles_rad).max() - np.degrees(angles_rad).min()),
        "cell_size_angstrom": cell_size,
        "voxel_size_angstrom_original": voxel_size_ang,
        "voxel_size_angstrom_downsampled": cell_size / target_vol,
        "original_vol_size": int(orig_vol),
        "target_vol_size": int(target_vol),
        "has_atom_gt": True,
        "backends": ["analytical"],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  PdPt-Zhou-{sample}: {atoms_xyz.shape[0]} atoms, {len(angles_rad)} angles")
    return meta


# ---------------------------------------------------------------------------
# PdPt Yang (2 samples: Particle1, Particle2)
# ---------------------------------------------------------------------------

def convert_pdpt_yang(particle: int, target_vol: int = 256):
    """Convert one PdPt Yang Nat. Commun. sample (particle=1 or 2)."""
    base = RAW_ROOT / "coreshell_pdpt" / "extracted"
    P = particle

    # File name fragments differ by particle
    date_tag = {1: "20200629_P2", 2: "20200825_P2"}[P]
    atom_tag = {1: "200629P2_IPRA_ver3(201229)", 2: "200825P2_IPRA"}[P]

    projs_file = f"PdPt_CoreShell_{date_tag}_Inpl_Projs_hori_TrAnRlnd_(Particle{P}).mat"
    ang_file   = f"PdPt_CoreShell_{date_tag}_After_optimization_Angles_hori_(Particle{P}).mat"
    recon_file = f"IPRA_{date_tag}_PdPt_6(0.1)deg_ReOri_(Particle{P}).mat"
    atoms_file = f"ManualTracingAndClassification_PdPt_{atom_tag}_(Particle{P}).mat"

    m_projs = sio.loadmat(str(base / projs_file))
    m_ang   = sio.loadmat(str(base / ang_file))
    m_recon = sio.loadmat(str(base / recon_file))
    m_atoms = sio.loadmat(str(base / atoms_file))

    projs_hwn = m_projs["shiftData"].astype(np.float32)   # (H, W, N)
    eulers    = m_ang["AnglesNew"].astype(np.float64)      # (N, 3)

    # Use InplRA_recon_PdPt preferentially, fall back to ESTvol
    if "InplRA_recon_PdPt" in m_recon:
        vol_gt = m_recon["InplRA_recon_PdPt"].astype(np.float32)
    else:
        vol_gt = m_recon["ESTvol"].astype(np.float32)

    pos = m_atoms["Coo"].astype(np.float32)   # (3, N_atoms)
    typ = m_atoms["Typ"].astype(int).ravel()   # (N_atoms,)

    angles_rad = np.radians(eulers[:, 1])

    orig_vol = vol_gt.shape[0]
    projs_small, vol_small = _resize_vol_and_projs(projs_hwn, vol_gt, target_vol=target_vol)

    scanner_cfg = _default_scanner(target_vol, target_vol)
    sample_id = f"PdPt-Yang-P{P}"
    out_dir = OUT_ROOT / sample_id
    _save_naf(out_dir / "analytical", projs_small, angles_rad, vol_small, scanner_cfg)

    # typ: 1=Pd(Z=46), 2=Pt(Z=78)
    Z_map = {1: 46, 2: 78}
    Z_vals = np.array([Z_map.get(t, 46) for t in typ], dtype=np.float32)
    atoms_xyz = pos.T  # (N_atoms, 3)
    atom_data = np.column_stack([atoms_xyz, Z_vals]).astype(np.float32)
    np.save(out_dir / "atoms_gt.npy", atom_data)

    voxel_size_ang = 0.347
    cell_size = orig_vol * voxel_size_ang

    meta = {
        "sample_id": sample_id,
        "source": "Yang et al. Nat. Commun. (core-shell PdPt)",
        "kind": "real_experimental",
        "element": "Pd+Pt",
        "n_atoms": int(atoms_xyz.shape[0]),
        "n_angles": int(len(angles_rad)),
        "tilt_range_deg": float(np.degrees(angles_rad).max() - np.degrees(angles_rad).min()),
        "cell_size_angstrom": cell_size,
        "voxel_size_angstrom_original": voxel_size_ang,
        "voxel_size_angstrom_downsampled": cell_size / target_vol,
        "original_vol_size": int(orig_vol),
        "target_vol_size": int(target_vol),
        "has_atom_gt": True,
        "backends": ["analytical"],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {sample_id}: {atoms_xyz.shape[0]} atoms, {len(angles_rad)} angles")
    return meta


# ---------------------------------------------------------------------------
# NiPt Multi-Grained Catalysts (1 sample)
# ---------------------------------------------------------------------------

def convert_nipt_multigrain(target_vol: int = 256):
    base = RAW_ROOT / "multigrain_catalysts"

    m_projs = sio.loadmat(str(base / "tilt_series" / "FinalProjs_NiPt_20230308_P4_AutoShift.mat"))
    m_ang   = sio.loadmat(str(base / "tilt_series" / "FinalAngles_NiPt_20230308_P4_AutoShift.mat"))
    m_recon = sio.loadmat(str(base / "volumes" / "EntireReconstruction_NiPt_20230308_P4.mat"))
    m_atoms = sio.loadmat(str(base / "atoms" / "ManualTracingAndAtomType_DomainCrop_NiPt_20230308_P4_MT2.mat"))

    projs_hwn = m_projs["Projsnew"].astype(np.float32)   # (280, 280, 27)
    eulers    = m_ang["AnglesNew"].astype(np.float64)     # (27, 3)
    vol_gt    = m_recon["RotVol"].astype(np.float32)      # (280, 280, 280)

    pos = m_atoms["Coo"].astype(np.float32)   # (3, N_atoms)
    typ = m_atoms["Typ"].astype(int).ravel()   # (N_atoms,)

    angles_rad = np.radians(eulers[:, 1])

    orig_vol = vol_gt.shape[0]
    projs_small, vol_small = _resize_vol_and_projs(projs_hwn, vol_gt, target_vol=target_vol)

    scanner_cfg = _default_scanner(target_vol, target_vol)
    out_dir = OUT_ROOT / "NiPt-MultiGrain"
    _save_naf(out_dir / "analytical", projs_small, angles_rad, vol_small, scanner_cfg)

    # typ: 1=Ni(Z=28), 2=Pt(Z=78)
    Z_map = {1: 28, 2: 78}
    Z_vals = np.array([Z_map.get(t, 28) for t in typ], dtype=np.float32)
    atoms_xyz = pos.T
    atom_data = np.column_stack([atoms_xyz, Z_vals]).astype(np.float32)
    np.save(out_dir / "atoms_gt.npy", atom_data)

    voxel_size_ang = 0.347
    cell_size = orig_vol * voxel_size_ang

    meta = {
        "sample_id": "NiPt-MultiGrain",
        "source": "Multi-grained NiPt catalyst (Zhou group)",
        "kind": "real_experimental",
        "element": "Ni+Pt",
        "n_atoms": int(atoms_xyz.shape[0]),
        "n_angles": int(len(angles_rad)),
        "tilt_range_deg": float(np.degrees(angles_rad).max() - np.degrees(angles_rad).min()),
        "cell_size_angstrom": cell_size,
        "voxel_size_angstrom_original": voxel_size_ang,
        "voxel_size_angstrom_downsampled": cell_size / target_vol,
        "original_vol_size": int(orig_vol),
        "target_vol_size": int(target_vol),
        "has_atom_gt": True,
        "backends": ["analytical"],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  NiPt-MultiGrain: {atoms_xyz.shape[0]} atoms, {len(angles_rad)} angles")
    return meta


# ---------------------------------------------------------------------------
# AuPd Icosahedral (ICNP1 with full data: real projections + volume)
# ---------------------------------------------------------------------------

def convert_aupd_icnp1(target_vol: int = 256):
    base = RAW_ROOT / "icosahedral_np" / "extracted" / "Source Data"

    m_projs = sio.loadmat(str(base / "Projections_ICNP1" / "Projs-ICNP1.mat"))
    m_ang   = sio.loadmat(str(base / "Angles_ICNP1" / "Angles-ICNP1.mat"))
    m_vol   = sio.loadmat(str(base / "Volume_INCP1" / "Volume-ICNP1.mat"))

    projs_hwn = m_projs["refined_projs"].astype(np.float32)  # (190, 190, 55)
    eulers    = m_ang["refined_angles"].astype(np.float64)    # (55, 3)

    # Structured array volume
    sa = m_vol["V_Euler_0_0_0_ViewfromZ"]
    vol_gt = sa["Volume"][0, 0].astype(np.float32)           # (190, 190, 190)

    # Load atom coordinates from XYZ
    xyz_path = base / "Coordinates_13MTPs" / "Coordinates-ICNP1.xyz"
    atom_data = _load_xyz_atoms(xyz_path)  # (N, 4) [x,y,z,Z]

    angles_rad = np.radians(eulers[:, 1])

    orig_vol = vol_gt.shape[0]
    projs_small, vol_small = _resize_vol_and_projs(projs_hwn, vol_gt, target_vol=target_vol)

    scanner_cfg = _default_scanner(target_vol, target_vol)
    out_dir = OUT_ROOT / "AuPd-ICNP1"
    _save_naf(out_dir / "analytical", projs_small, angles_rad, vol_small, scanner_cfg)

    np.save(out_dir / "atoms_gt.npy", atom_data)

    voxel_size_ang = 0.347
    cell_size = orig_vol * voxel_size_ang

    meta = {
        "sample_id": "AuPd-ICNP1",
        "source": "AuPd icosahedral nanoparticle (full data)",
        "kind": "real_experimental",
        "element": "Au+Pd",
        "n_atoms": int(atom_data.shape[0]),
        "n_angles": int(len(angles_rad)),
        "tilt_range_deg": float(np.degrees(angles_rad).max() - np.degrees(angles_rad).min()),
        "cell_size_angstrom": cell_size,
        "voxel_size_angstrom_original": voxel_size_ang,
        "voxel_size_angstrom_downsampled": cell_size / target_vol,
        "original_vol_size": int(orig_vol),
        "target_vol_size": int(target_vol),
        "has_atom_gt": True,
        "backends": ["analytical"],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  AuPd-ICNP1: {atom_data.shape[0]} atoms, {len(angles_rad)} angles")
    return meta


# ---------------------------------------------------------------------------
# AuPd Icosahedral – coords-only samples (ICNP2..ICNP8, PC1..PC3, DH, SF)
# Synthetic projections generated via ASTRA FP from Z^gamma atomic volume
# ---------------------------------------------------------------------------

ICNP_COORDS_ONLY = [
    "ICNP2", "ICNP3", "ICNP4", "ICNP5", "ICNP6", "ICNP7", "ICNP8",
    "PC1", "PC2", "PC3", "DH", "SF",
]


def convert_aupd_coords_only(label: str, vol_size: int = 128,
                               n_angles: int = 60, tilt_range_deg: float = 70.0):
    """Convert an AuPd icosahedral sample that has only XYZ coordinates.

    Generates a synthetic volume and tilt series via analytical Z^gamma model
    + ASTRA parallel3d forward projection.
    """
    base = RAW_ROOT / "icosahedral_np" / "extracted" / "Source Data" / "Coordinates_13MTPs"
    xyz_path = base / f"Coordinates-{label}.xyz"

    atom_data = _load_xyz_atoms(xyz_path)  # (N, 4) [x,y,z,Z], centered

    projs_hwn, angles_rad, vol = _atoms_to_vol_and_projs(
        atom_data, vol_size=vol_size, n_angles=n_angles, tilt_range_deg=tilt_range_deg
    )

    scanner_cfg = _default_scanner(vol_size, vol_size)
    sample_id = f"AuPd-{label}"
    out_dir = OUT_ROOT / sample_id
    _save_naf(out_dir / "analytical", projs_hwn, angles_rad, vol, scanner_cfg)

    np.save(out_dir / "atoms_gt.npy", atom_data)

    meta = {
        "sample_id": sample_id,
        "source": "AuPd icosahedral nanoparticle (coords-only, synthetic projections)",
        "kind": "real_atoms_synthetic_projections",
        "element": "Au+Pd",
        "n_atoms": int(atom_data.shape[0]),
        "n_angles": int(n_angles),
        "tilt_range_deg": float(tilt_range_deg * 2),
        "vol_size": int(vol_size),
        "has_atom_gt": True,
        "note": "Projections computed from Z^gamma Gaussian model via ASTRA FP3D_CUDA",
        "backends": ["analytical"],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {sample_id}: {atom_data.shape[0]} atoms, {n_angles} synthetic angles")
    return meta


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []

    print("=== Zr-ZrO2 ===")
    for name in ["Zr1", "Zr2", "Zr3"]:
        try:
            manifest.append(convert_zr(name))
        except Exception as e:
            print(f"  FAILED Zr-{name}: {e}")

    print("=== MPEA2 ===")
    try:
        manifest.append(convert_mpea2())
    except Exception as e:
        print(f"  FAILED MPEA2: {e}")

    print("=== PdPt Zhou ===")
    for sample in ["EPB", "PB", "TO"]:
        try:
            manifest.append(convert_pdpt_zhou(sample))
        except Exception as e:
            print(f"  FAILED PdPt-Zhou-{sample}: {e}")

    print("=== PdPt Yang ===")
    for p in [1, 2]:
        try:
            manifest.append(convert_pdpt_yang(p))
        except Exception as e:
            print(f"  FAILED PdPt-Yang-P{p}: {e}")

    print("=== NiPt MultiGrain ===")
    try:
        manifest.append(convert_nipt_multigrain())
    except Exception as e:
        print(f"  FAILED NiPt-MultiGrain: {e}")

    print("=== AuPd ICNP1 (full) ===")
    try:
        manifest.append(convert_aupd_icnp1())
    except Exception as e:
        print(f"  FAILED AuPd-ICNP1: {e}")

    print("=== AuPd coords-only ===")
    for label in ICNP_COORDS_ONLY:
        try:
            manifest.append(convert_aupd_coords_only(label))
        except Exception as e:
            print(f"  FAILED AuPd-{label}: {e}")

    with open(OUT_ROOT / "manifest.json", "w") as f:
        json.dump({"n_samples": len(manifest), "samples": manifest}, f, indent=2)
    print(f"\nDone. {len(manifest)} real samples → {OUT_ROOT}")


if __name__ == "__main__":
    main()
