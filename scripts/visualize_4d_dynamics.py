#!/usr/bin/env python
"""
Visualize 4D reconstruction dynamics:
1. GT vs Pred volume slices across 10 phases (animated-style grid)
2. Atomic displacement heatmap over time
3. Per-phase difference maps
4. Sinogram comparison
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml


def main(output_dir, iteration=15000):
    output_dir = Path(output_dir)
    pc_dir = output_dir / f"point_cloud/iteration_{iteration}"

    # Load all 10 phase volumes
    gt_vols = []
    pred_vols = []
    for t in range(10):
        gt = np.load(pc_dir / f"vol_gt_T{t}.npy")
        pred = np.load(pc_dir / f"vol_pred_T{t}.npy")
        gt_vols.append(gt)
        pred_vols.append(pred)

    gt_vols = np.stack(gt_vols)   # (10, Nx, Ny, Nz)
    pred_vols = np.stack(pred_vols)

    print(f"GT: {gt_vols.shape}, range [{gt_vols.min():.2f}, {gt_vols.max():.2f}]")
    print(f"Pred: {pred_vols.shape}, range [{pred_vols.min():.4f}, {pred_vols.max():.4f}]")

    mid = gt_vols.shape[1] // 2
    gt_max = gt_vols.max()
    pred_max = pred_vols.max()
    scale = gt_max / (pred_max + 1e-8)

    # ========== Figure 1: Phase evolution (GT vs Pred) ==========
    fig1, axes1 = plt.subplots(4, 10, figsize=(24, 10))

    for t in range(10):
        # Row 0: GT XY slice at mid-z
        axes1[0, t].imshow(gt_vols[t, :, :, mid], cmap="hot", origin="lower",
                           vmin=0, vmax=gt_max * 0.8)
        axes1[0, t].set_title(f"T{t}", fontsize=9)
        axes1[0, t].axis("off")

        # Row 1: Pred XY slice (scaled)
        axes1[1, t].imshow(pred_vols[t, :, :, mid] * scale, cmap="hot", origin="lower",
                           vmin=0, vmax=gt_max * 0.8)
        axes1[1, t].axis("off")

        # Row 2: GT XZ slice at mid-y
        axes1[2, t].imshow(gt_vols[t, :, mid, :], cmap="hot", origin="lower",
                           vmin=0, vmax=gt_max * 0.8)
        axes1[2, t].axis("off")

        # Row 3: Pred XZ slice (scaled)
        axes1[3, t].imshow(pred_vols[t, :, mid, :] * scale, cmap="hot", origin="lower",
                           vmin=0, vmax=gt_max * 0.8)
        axes1[3, t].axis("off")

    axes1[0, 0].set_ylabel("GT (XY)", fontsize=11)
    axes1[1, 0].set_ylabel("Pred (XY)", fontsize=11)
    axes1[2, 0].set_ylabel("GT (XZ)", fontsize=11)
    axes1[3, 0].set_ylabel("Pred (XZ)", fontsize=11)
    fig1.suptitle("4D Phase Evolution: 10 Time Points", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save1 = output_dir / "4d_phase_evolution.png"
    fig1.savefig(save1, dpi=150, bbox_inches="tight")
    print(f"Saved: {save1}")
    plt.close(fig1)

    # ========== Figure 2: Temporal difference maps ==========
    fig2, axes2 = plt.subplots(3, 9, figsize=(22, 8))

    for i in range(9):
        # GT difference between consecutive phases
        gt_diff = gt_vols[i+1, :, :, mid] - gt_vols[i, :, :, mid]
        axes2[0, i].imshow(gt_diff, cmap="bwr", origin="lower",
                           vmin=-gt_max*0.15, vmax=gt_max*0.15)
        axes2[0, i].set_title(f"T{i}→T{i+1}", fontsize=8)
        axes2[0, i].axis("off")

        # Pred difference
        pred_diff = (pred_vols[i+1, :, :, mid] - pred_vols[i, :, :, mid]) * scale
        axes2[1, i].imshow(pred_diff, cmap="bwr", origin="lower",
                           vmin=-gt_max*0.15, vmax=gt_max*0.15)
        axes2[1, i].axis("off")

        # Temporal variance map
        gt_var = gt_vols[:, :, :, mid].std(axis=0)
        pred_var = (pred_vols[:, :, :, mid] * scale).std(axis=0)

    # Last column: temporal std maps
    axes2[2, 0].imshow(gt_var, cmap="magma", origin="lower")
    axes2[2, 0].set_title("GT temporal std", fontsize=9)
    axes2[2, 0].axis("off")
    axes2[2, 1].imshow(pred_var, cmap="magma", origin="lower")
    axes2[2, 1].set_title("Pred temporal std", fontsize=9)
    axes2[2, 1].axis("off")

    # Per-phase SSIM bar chart
    eval_path = output_dir / f"eval/iter_{iteration:06d}/eval3d.yml"
    if eval_path.exists():
        with open(eval_path) as f:
            eval3d = yaml.safe_load(f)
        ax = axes2[2, 3]
        ssim_vals = eval3d.get("ssim_3d", [])
        ax.bar(range(len(ssim_vals)), ssim_vals, color="steelblue", alpha=0.8)
        ax.set_title("3D SSIM per phase", fontsize=9)
        ax.set_xlabel("Phase")
        ax.set_ylim(min(ssim_vals) - 0.01, max(ssim_vals) + 0.01)

        ax = axes2[2, 5]
        psnr_vals = eval3d.get("psnr_3d", [])
        ax.bar(range(len(psnr_vals)), psnr_vals, color="coral", alpha=0.8)
        ax.set_title("3D PSNR per phase", fontsize=9)
        ax.set_xlabel("Phase")

    # Hide unused axes
    for j in [2, 4, 6, 7, 8]:
        axes2[2, j].axis("off")

    axes2[0, 0].set_ylabel("GT diff", fontsize=11)
    axes2[1, 0].set_ylabel("Pred diff", fontsize=11)
    fig2.suptitle("Temporal Dynamics: Phase-to-Phase Changes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save2 = output_dir / "4d_temporal_dynamics.png"
    fig2.savefig(save2, dpi=150, bbox_inches="tight")
    print(f"Saved: {save2}")
    plt.close(fig2)

    # ========== Figure 3: Line profile evolution ==========
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))

    # Center line through volume at each phase
    ax = axes3[0]
    for t in range(10):
        line = gt_vols[t, mid, :, mid]
        ax.plot(line, alpha=0.3 + 0.07*t, color=plt.cm.viridis(t/9), label=f"T{t}")
    ax.set_title("GT center line (Y) across phases")
    ax.set_xlabel("Voxel")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=6, ncol=2)

    ax = axes3[1]
    for t in range(10):
        line = pred_vols[t, mid, :, mid] * scale
        ax.plot(line, alpha=0.3 + 0.07*t, color=plt.cm.viridis(t/9), label=f"T{t}")
    ax.set_title("Pred center line (Y) across phases")
    ax.set_xlabel("Voxel")

    # Phase 0 vs Phase 9 comparison
    ax = axes3[2]
    ax.plot(gt_vols[0, mid, :, mid], 'b-', label="GT T0", linewidth=1.5)
    ax.plot(gt_vols[9, mid, :, mid], 'b--', label="GT T9", linewidth=1.5)
    ax.plot(pred_vols[0, mid, :, mid] * scale, 'r-', label="Pred T0", linewidth=1.5)
    ax.plot(pred_vols[9, mid, :, mid] * scale, 'r--', label="Pred T9", linewidth=1.5)
    ax.set_title("T0 vs T9: GT and Pred")
    ax.set_xlabel("Voxel")
    ax.legend(fontsize=8)

    fig3.suptitle("Temporal Line Profiles", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save3 = output_dir / "4d_line_profiles.png"
    fig3.savefig(save3, dpi=150, bbox_inches="tight")
    print(f"Saved: {save3}")
    plt.close(fig3)

    # ========== Figure 4: Animated GIF of phase evolution ==========
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig4, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(10, 5))

        def update(t):
            ax_gt.clear()
            ax_pred.clear()
            ax_gt.imshow(gt_vols[t, :, :, mid], cmap="hot", origin="lower",
                         vmin=0, vmax=gt_max * 0.8)
            ax_gt.set_title(f"GT Phase {t}", fontsize=12)
            ax_gt.axis("off")
            ax_pred.imshow(pred_vols[t, :, :, mid] * scale, cmap="hot", origin="lower",
                           vmin=0, vmax=gt_max * 0.8)
            ax_pred.set_title(f"Pred Phase {t}", fontsize=12)
            ax_pred.axis("off")
            fig4.suptitle("4D Reconstruction: Phase Evolution", fontsize=14, fontweight="bold")

        anim = FuncAnimation(fig4, update, frames=10, interval=500)
        gif_path = output_dir / "4d_animation.gif"
        anim.save(gif_path, writer=PillowWriter(fps=2))
        print(f"Saved animation: {gif_path}")
        plt.close(fig4)
    except Exception as e:
        print(f"GIF generation skipped: {e}")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/aet-bench/output/cu_anneal_4d"
    iteration = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
    main(output_dir, iteration)
