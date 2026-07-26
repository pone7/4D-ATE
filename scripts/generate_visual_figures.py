#!/usr/bin/env python
"""Generate fancy visualization figures for the paper:
  1. Volume slice comparison: GT vs FBP vs SIRT vs MLEM vs X²-G (one sample)
  2. 4D dynamic sequence: 10-phase volume slices showing atom motion
  3. Real data: Zr-ZrO2 reconstruction slice comparison
  4. Brownian Pt: 3D atom scatter plot
"""
import os, pickle, json, sys
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path

FIG_DIR = Path("/data/aet-bench/paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ═══════════════════════════════════════════════════════════════
# FIG: Volume slice comparison on Cu-309 analytical
# ═══════════════════════════════════════════════════════════════
def fig_slice_comparison():
    """Show central Z-slice of GT vs 5 methods on Cu-0165-00."""
    base = Path("/data/aet-bench")
    sample = "Cu-0165-00"
    data = load_pkl(base / f"data/aet_bench/mono/{sample}/analytical/data.pkl")
    gt = data["image"][0]  # (V, V, V)
    V = gt.shape[0]
    mid = V // 2

    # Load method reconstructions by re-running fast methods inline
    sys.path.insert(0, str(base / "scripts"))
    from run_benchmark import method_fbp, method_sirt
    from baselines.run_astra import run_astra
    from baselines.run_tomopy import run_tomopy

    vol_fbp = method_fbp(data)
    vol_cgls = run_astra(data, sample, Path("/tmp/vis_cgls"), algorithm="CGLS", n_iter=50)
    vol_mlem = run_tomopy(data, sample, Path("/tmp/vis_mlem"), algorithm="mlem")

    # Try loading X²-G from existing output
    x2g_dir = base / f"output/aet_bench/{sample}/model_static/point_cloud"
    x2g_vol = None
    if x2g_dir.exists():
        cands = sorted(x2g_dir.glob("iteration_*/vol_pred_T0.npy"))
        if cands:
            x2g_vol = np.load(cands[-1])

    # Normalize all to GT scale
    def norm(v):
        if v is None:
            return np.zeros((V, V))
        s = gt.max() / max(v.max(), 1e-8)
        return (v * s)[mid]

    vols = [
        ("Ground Truth", gt[mid]),
        ("FBP", norm(vol_fbp)),
        ("ASTRA-CGLS", norm(vol_cgls)),
        ("TomoPy-MLEM", norm(vol_mlem)),
    ]
    if x2g_vol is not None:
        vols.append(("X²-Gaussian", norm(x2g_vol)))

    n = len(vols)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 2.5))
    vmin, vmax = 0, gt[mid].max()
    for ax, (title, slc) in zip(axes, vols):
        ax.imshow(slc, cmap="hot", vmin=vmin, vmax=vmax,
                  interpolation="bilinear")
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.axis("off")

    fig.suptitle(f"Central Z-slice: {sample} (analytical)", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_slice_comparison.pdf", dpi=300)
    fig.savefig(FIG_DIR / "fig_slice_comparison.png", dpi=300)
    plt.close(fig)
    print("  fig_slice_comparison.pdf")


# ═══════════════════════════════════════════════════════════════
# FIG: 4D dynamic sequence — 10 phases of Cu annealing
# ═══════════════════════════════════════════════════════════════
def fig_4d_sequence():
    """Show GT volume slices across 10 phases of a 4D trajectory."""
    base = Path("/data/aet-bench")
    sample = "Cu4D-08A-00"
    data = load_pkl(base / f"data/aet_bench/sim4d/{sample}/analytical/data.pkl")
    gt_4d = data["image"]  # (10, V, V, V)
    n_phases = gt_4d.shape[0]
    V = gt_4d.shape[1]
    mid = V // 2

    # Show every other phase (5 panels)
    phases_show = [0, 2, 4, 6, 9]
    fig, axes = plt.subplots(1, len(phases_show), figsize=(len(phases_show) * 2, 2.2))
    vmin, vmax = 0, gt_4d.max()
    for ax, t in zip(axes, phases_show):
        ax.imshow(gt_4d[t, mid], cmap="inferno", vmin=vmin, vmax=vmax,
                  interpolation="bilinear")
        ax.set_title(f"t={t}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"4D Cu annealing: GT central slice across time ({sample})",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_4d_sequence.pdf", dpi=300)
    fig.savefig(FIG_DIR / "fig_4d_sequence.png", dpi=300)
    plt.close(fig)
    print("  fig_4d_sequence.pdf")


# ═══════════════════════════════════════════════════════════════
# FIG: Real data — Zr-ZrO2 reconstruction comparison
# ═══════════════════════════════════════════════════════════════
def fig_real_recon():
    """Show the published GT volume vs our best reconstruction for Zr-Zr1."""
    base = Path("/data/aet-bench")
    sample = "Zr-Zr1"
    data = load_pkl(base / f"data/aet_bench/real/{sample}/analytical/data.pkl")
    gt = data["image"][0]
    V = gt.shape[0]
    mid = V // 2

    # Load atoms for overlay
    atoms = np.load(base / f"data/aet_bench/real/{sample}/atoms_gt.npy")
    with open(base / f"data/aet_bench/real/{sample}/metadata.json") as f:
        meta = json.load(f)
    cell = meta["cell_size_angstrom"]

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    # Panel 1: GT volume slice
    axes[0].imshow(gt[mid], cmap="hot", interpolation="bilinear")
    axes[0].set_title("Published recon (GT)", fontsize=8, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: GT with atom overlay
    axes[1].imshow(gt[mid], cmap="hot", interpolation="bilinear")
    # Project atoms near the mid slice onto the image
    voxel = cell / V
    z_phys = (mid - V / 2) * voxel
    mask = np.abs(atoms[:, 2] - z_phys) < voxel * 3
    if mask.sum() > 0:
        ax_x = (atoms[mask, 0] / cell + 0.5) * V
        ax_y = (atoms[mask, 1] / cell + 0.5) * V
        axes[1].scatter(ax_x, ax_y, s=3, c="cyan", alpha=0.6, linewidths=0)
    axes[1].set_title(f"GT + atoms ({mask.sum()} near slice)", fontsize=8,
                       fontweight="bold")
    axes[1].axis("off")

    # Panel 3: 3D atom scatter (isometric projection)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    # Subsample for visibility
    n_show = min(3000, len(atoms))
    idx = np.random.default_rng(0).choice(len(atoms), n_show, replace=False)
    ax3.scatter(atoms[idx, 0], atoms[idx, 1], atoms[idx, 2],
                s=1, c=atoms[idx, 2], cmap="viridis", alpha=0.5)
    ax3.set_xlabel("x (Å)", fontsize=6)
    ax3.set_ylabel("y (Å)", fontsize=6)
    ax3.set_zlabel("z (Å)", fontsize=6)
    ax3.set_title(f"Zr-Zr1: {len(atoms):,} atoms", fontsize=8, fontweight="bold")
    ax3.tick_params(labelsize=5)

    fig.suptitle("Real experimental data: Zr-ZrO₂ nanoparticle", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_real_recon.pdf", dpi=300)
    fig.savefig(FIG_DIR / "fig_real_recon.png", dpi=300)
    plt.close(fig)
    print("  fig_real_recon.pdf")


# ═══════════════════════════════════════════════════════════════
# FIG: Brownian Pt — 3D atom positions across time
# ═══════════════════════════════════════════════════════════════
def fig_brownian_atoms():
    """Show 3D atom scatter for 3 time points of one Pt trajectory."""
    base = Path("/data/aet-bench")
    samples = ["Pt-ECT1_time04s", "Pt-ECT1_time13s", "Pt-ECT1_time25s"]
    labels = ["t = 4s", "t = 13s", "t = 25s"]

    fig = plt.figure(figsize=(9, 3))
    for i, (sid, lab) in enumerate(zip(samples, labels)):
        atoms_path = base / f"data/aet_bench/brownian_pt/{sid}/atoms_gt.npy"
        if not atoms_path.exists():
            continue
        atoms = np.load(atoms_path)
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(atoms[:, 0], atoms[:, 1], atoms[:, 2],
                   s=8, c="#8172B2", alpha=0.7, edgecolors="black",
                   linewidth=0.2)
        ax.set_title(f"{lab}\n({len(atoms)} atoms)", fontsize=8,
                     fontweight="bold")
        ax.set_xlabel("x (Å)", fontsize=6)
        ax.set_ylabel("y (Å)", fontsize=6)
        ax.set_zlabel("z (Å)", fontsize=6)
        ax.tick_params(labelsize=5)
        # Same view angle
        ax.view_init(elev=20, azim=45)

    fig.suptitle("Time-resolved Pt nanocrystal (Brownian tomography, ECT1)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_brownian_atoms.pdf", dpi=300)
    fig.savefig(FIG_DIR / "fig_brownian_atoms.png", dpi=300)
    plt.close(fig)
    print("  fig_brownian_atoms.pdf")


# ═══════════════════════════════════════════════════════════════
# FIG: Multislice vs Analytical — same sample, different physics
# ═══════════════════════════════════════════════════════════════
def fig_multislice_gap():
    """Side-by-side of analytical vs multislice GT for Cu-309."""
    base = Path("/data/aet-bench")

    # Analytical
    d_ana = load_pkl(base / "data/aet_bench/mono/Cu-0279-01/analytical/data.pkl")
    gt_ana = d_ana["image"][0]

    # Multislice
    d_ms = load_pkl(base / "data/aet_bench/multislice/Cu-309/analytical/data.pkl")
    gt_ms = d_ms["image"][0]

    V_ana, V_ms = gt_ana.shape[0], gt_ms.shape[0]

    fig, axes = plt.subplots(1, 4, figsize=(9, 2.5))

    # Analytical GT + projection
    axes[0].imshow(gt_ana[V_ana // 2], cmap="hot", interpolation="bilinear")
    axes[0].set_title("Analytical GT\n(Z^γ model)", fontsize=8)
    axes[0].axis("off")

    ana_projs = d_ana["train"]["projections"]
    axes[1].imshow(ana_projs[len(ana_projs)//2], cmap="gray",
                   interpolation="bilinear")
    axes[1].set_title("Analytical proj\n(0° tilt)", fontsize=8)
    axes[1].axis("off")

    # Multislice GT + projection
    axes[2].imshow(gt_ms[V_ms // 2], cmap="hot", interpolation="bilinear")
    axes[2].set_title("Multislice GT\n(abTEM)", fontsize=8)
    axes[2].axis("off")

    ms_projs = d_ms["train"]["projections"]
    axes[3].imshow(ms_projs[len(ms_projs)//2], cmap="gray",
                   interpolation="bilinear")
    axes[3].set_title("Multislice proj\n(abTEM, 0° tilt)", fontsize=8)
    axes[3].axis("off")

    fig.suptitle("Physics gap: analytical vs. multislice simulation (Cu nanoparticle)",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_multislice_gap.pdf", dpi=300)
    fig.savefig(FIG_DIR / "fig_multislice_gap.png", dpi=300)
    plt.close(fig)
    print("  fig_multislice_gap.pdf")


def main():
    print("Generating visual figures...")
    fig_slice_comparison()
    fig_4d_sequence()
    fig_real_recon()
    fig_brownian_atoms()
    fig_multislice_gap()
    print("Done.")


if __name__ == "__main__":
    main()
