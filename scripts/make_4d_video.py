#!/usr/bin/env python
"""
Generate 4D reconstruction videos showing atomic dynamics over time.

Creates:
1. Side-by-side GT vs Pred volume slice animation (GIF)
2. Rotating 3D view at different time points
3. Atom displacement trajectory visualization
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path


def load_4d_volumes(output_dir, iteration=15000):
    """Load all 10 phase volumes."""
    pc_dir = Path(output_dir) / f"point_cloud/iteration_{iteration}"
    gt_vols, pred_vols = [], []
    for t in range(10):
        gt_vols.append(np.load(pc_dir / f"vol_gt_T{t}.npy"))
        pred_vols.append(np.load(pc_dir / f"vol_pred_T{t}.npy"))
    return np.stack(gt_vols), np.stack(pred_vols)


def make_phase_sweep_gif(gt_vols, pred_vols, save_path, axis="z"):
    """Animated GIF sweeping through 10 phases, GT vs Pred side by side."""
    n_phases = gt_vols.shape[0]
    n = gt_vols.shape[1]
    mid = n // 2
    gt_max = gt_vols.max() * 0.8
    pred_max = pred_vols.max() * 0.8
    scale = gt_max / (pred_max + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def update(t):
        for ax in axes:
            ax.clear()

        if axis == "z":
            gt_slice = gt_vols[t, :, :, mid]
            pred_slice = pred_vols[t, :, :, mid] * scale
            diff_slice = gt_slice - pred_slice
        elif axis == "y":
            gt_slice = gt_vols[t, :, mid, :]
            pred_slice = pred_vols[t, :, mid, :] * scale
            diff_slice = gt_slice - pred_slice
        else:
            gt_slice = gt_vols[t, mid, :, :]
            pred_slice = pred_vols[t, mid, :, :] * scale
            diff_slice = gt_slice - pred_slice

        axes[0].imshow(gt_slice, cmap="hot", origin="lower", vmin=0, vmax=gt_max)
        axes[0].set_title(f"Ground Truth — Phase {t}", fontsize=13)
        axes[0].axis("off")

        axes[1].imshow(pred_slice, cmap="hot", origin="lower", vmin=0, vmax=gt_max)
        axes[1].set_title(f"AET-4DGS Reconstruction — Phase {t}", fontsize=13)
        axes[1].axis("off")

        axes[2].imshow(diff_slice, cmap="bwr", origin="lower",
                       vmin=-gt_max*0.2, vmax=gt_max*0.2)
        axes[2].set_title(f"Difference — Phase {t}", fontsize=13)
        axes[2].axis("off")

        fig.suptitle("4D Atomic Reconstruction: Cu Nanoparticle Annealing",
                     fontsize=15, fontweight="bold")

    anim = FuncAnimation(fig, update, frames=n_phases, interval=800)
    anim.save(save_path, writer=PillowWriter(fps=2))
    plt.close()
    print(f"Saved: {save_path}")


def make_multislice_gif(gt_vols, pred_vols, save_path):
    """Animated GIF sweeping through z-slices at phase 0 and phase 9."""
    n = gt_vols.shape[1]
    gt_max = gt_vols.max() * 0.8
    scale = gt_max / (pred_vols.max() + 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    def update(z):
        for row in axes:
            for ax in row:
                ax.clear()

        axes[0, 0].imshow(gt_vols[0, :, :, z], cmap="hot", origin="lower", vmin=0, vmax=gt_max)
        axes[0, 0].set_title(f"GT Phase 0 — z={z}", fontsize=11)
        axes[0, 0].axis("off")

        axes[0, 1].imshow(gt_vols[9, :, :, z], cmap="hot", origin="lower", vmin=0, vmax=gt_max)
        axes[0, 1].set_title(f"GT Phase 9 — z={z}", fontsize=11)
        axes[0, 1].axis("off")

        axes[1, 0].imshow(pred_vols[0, :, :, z] * scale, cmap="hot", origin="lower", vmin=0, vmax=gt_max)
        axes[1, 0].set_title(f"Pred Phase 0 — z={z}", fontsize=11)
        axes[1, 0].axis("off")

        axes[1, 1].imshow(pred_vols[9, :, :, z] * scale, cmap="hot", origin="lower", vmin=0, vmax=gt_max)
        axes[1, 1].set_title(f"Pred Phase 9 — z={z}", fontsize=11)
        axes[1, 1].axis("off")

        fig.suptitle("Z-slice Scan: Phase 0 vs Phase 9", fontsize=14, fontweight="bold")

    # Sweep from z=n//4 to z=3n//4
    z_range = list(range(n//4, 3*n//4, 2))
    anim = FuncAnimation(fig, update, frames=z_range, interval=200)
    anim.save(save_path, writer=PillowWriter(fps=5))
    plt.close()
    print(f"Saved: {save_path}")


def make_temporal_difference_gif(gt_vols, pred_vols, save_path):
    """Show how the structure changes between consecutive phases."""
    n_phases = gt_vols.shape[0]
    mid = gt_vols.shape[1] // 2
    gt_max = gt_vols.max() * 0.15
    scale = gt_vols.max() / (pred_vols.max() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def update(t):
        for ax in axes:
            ax.clear()

        t_next = (t + 1) % n_phases
        gt_diff = gt_vols[t_next, :, :, mid] - gt_vols[t, :, :, mid]
        pred_diff = (pred_vols[t_next, :, :, mid] - pred_vols[t, :, :, mid]) * scale

        axes[0].imshow(gt_diff, cmap="bwr", origin="lower", vmin=-gt_max, vmax=gt_max)
        axes[0].set_title(f"GT: Phase {t} → {t_next}", fontsize=13)
        axes[0].axis("off")

        axes[1].imshow(pred_diff, cmap="bwr", origin="lower", vmin=-gt_max, vmax=gt_max)
        axes[1].set_title(f"Pred: Phase {t} → {t_next}", fontsize=13)
        axes[1].axis("off")

        fig.suptitle("Temporal Changes Between Phases (red=increase, blue=decrease)",
                     fontsize=13, fontweight="bold")

    anim = FuncAnimation(fig, update, frames=n_phases - 1, interval=1000)
    anim.save(save_path, writer=PillowWriter(fps=1.5))
    plt.close()
    print(f"Saved: {save_path}")


def make_projection_comparison_gif(output_dir, iteration=15000):
    """Show rendered projections vs GT at different angles."""
    import yaml
    eval_dir = Path(output_dir) / f"eval/iter_{iteration:06d}"
    # Check if 2D render images exist
    if not eval_dir.exists():
        print(f"  No eval dir at {eval_dir}, skipping projection GIF")
        return

    # Just use the volume data to create a rotating view
    gt_vols, pred_vols = load_4d_volumes(output_dir, iteration)
    mid = gt_vols.shape[1] // 2
    gt_max = gt_vols[0].max() * 0.8
    scale = gt_max / (pred_vols[0].max() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    save_path = Path(output_dir) / "4d_rotating_view.gif"

    def update(angle_idx):
        for ax in axes:
            ax.clear()

        # Simulate rotation by taking different axis slices
        n = gt_vols.shape[1]
        vol_gt = gt_vols[0]
        vol_pred = pred_vols[0] * scale

        # Create a maximum intensity projection (MIP) along a rotated axis
        angle = angle_idx * 5  # degrees
        rad = np.radians(angle)
        c, s = np.cos(rad), np.sin(rad)

        # Simple: just show XZ slice at different Y positions (simulates rotation)
        y = int(n * 0.3 + n * 0.4 * (np.sin(rad) + 1) / 2)

        axes[0].imshow(vol_gt[:, y, :], cmap="hot", origin="lower", vmin=0, vmax=gt_max)
        axes[0].set_title(f"GT — Slice y={y}", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(vol_pred[:, y, :], cmap="hot", origin="lower", vmin=0, vmax=gt_max)
        axes[1].set_title(f"Pred — Slice y={y}", fontsize=12)
        axes[1].axis("off")

        fig.suptitle("Rotating Cross-Section View (Phase 0)", fontsize=14, fontweight="bold")

    anim = FuncAnimation(fig, update, frames=36, interval=200)
    anim.save(save_path, writer=PillowWriter(fps=5))
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    output_dir = "/data/aet-bench/output/cu_anneal_4d"
    save_dir = Path("/data/aet-bench/paper_figures")

    print("Loading 4D volumes...")
    gt_vols, pred_vols = load_4d_volumes(output_dir, 15000)
    print(f"  GT: {gt_vols.shape}, Pred: {pred_vols.shape}")

    print("\n1. Phase sweep animation...")
    make_phase_sweep_gif(gt_vols, pred_vols, save_dir / "4d_phase_sweep.gif", axis="z")

    print("\n2. Z-slice scan animation...")
    make_multislice_gif(gt_vols, pred_vols, save_dir / "4d_zslice_scan.gif")

    print("\n3. Temporal difference animation...")
    make_temporal_difference_gif(gt_vols, pred_vols, save_dir / "4d_temporal_changes.gif")

    print("\n4. Rotating view...")
    make_projection_comparison_gif(output_dir)

    print("\nAll videos generated!")
