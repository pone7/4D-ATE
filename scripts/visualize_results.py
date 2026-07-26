#!/usr/bin/env python
"""Visualize X2-Gaussian reconstruction results vs ground truth."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path


def main(output_dir, iteration=10000):
    output_dir = Path(output_dir)
    pc_dir = output_dir / f"point_cloud/iteration_{iteration}"

    vol_gt = np.load(pc_dir / "vol_gt_T0.npy")
    vol_pred = np.load(pc_dir / "vol_pred_T0.npy")

    print(f"GT volume: {vol_gt.shape}, range=[{vol_gt.min():.2f}, {vol_gt.max():.2f}]")
    print(f"Pred volume: {vol_pred.shape}, range=[{vol_pred.min():.4f}, {vol_pred.max():.4f}]")

    # Normalize for visualization
    gt_max = vol_gt.max()
    pred_max = vol_pred.max()

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    # 5 slices through volume
    n = vol_gt.shape[0]
    slice_indices = [n//4, 3*n//8, n//2, 5*n//8, 3*n//4]

    # Row 1: GT slices (XY plane)
    for i, idx in enumerate(slice_indices):
        ax = axes[0, i]
        ax.imshow(vol_gt[:, :, idx], cmap="hot", origin="lower")
        ax.set_title(f"GT z={idx}", fontsize=10)
        ax.axis("off")

    # Row 2: Predicted slices (XY plane)
    for i, idx in enumerate(slice_indices):
        ax = axes[1, i]
        ax.imshow(vol_pred[:, :, idx], cmap="hot", origin="lower")
        ax.set_title(f"Pred z={idx}", fontsize=10)
        ax.axis("off")

    # Row 3: Difference + profiles + stats
    for i, idx in enumerate(slice_indices[:3]):
        ax = axes[2, i]
        gt_slice = vol_gt[:, :, idx]
        pred_slice = vol_pred[:, :, idx]
        # Normalize pred to GT scale for comparison
        if pred_max > 0:
            pred_scaled = pred_slice * (gt_max / pred_max)
        else:
            pred_scaled = pred_slice
        diff = gt_slice - pred_scaled
        ax.imshow(diff, cmap="bwr", origin="lower",
                  vmin=-gt_max*0.3, vmax=gt_max*0.3)
        ax.set_title(f"Diff z={idx}", fontsize=10)
        ax.axis("off")

    # Central line profile comparison
    ax = axes[2, 3]
    mid = n // 2
    gt_line = vol_gt[mid, :, mid]
    pred_line = vol_pred[mid, :, mid]
    if pred_max > 0:
        pred_line_scaled = pred_line * (gt_max / pred_max)
    else:
        pred_line_scaled = pred_line
    ax.plot(gt_line, "b-", label="GT", linewidth=1.5)
    ax.plot(pred_line_scaled, "r--", label="Pred (scaled)", linewidth=1.5)
    ax.set_title("Center line profile", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlabel("Voxel")

    # Stats
    ax = axes[2, 4]
    ax.axis("off")

    # Load eval metrics
    eval_path = output_dir / f"eval/iter_{iteration:06d}/eval3d.yml"
    eval2d_path = output_dir / f"eval/iter_{iteration:06d}/eval2d_render_test.yml"
    metrics_text = f"Iteration: {iteration}\n\n"

    if eval_path.exists():
        import yaml
        with open(eval_path) as f:
            eval3d = yaml.safe_load(f)
        metrics_text += f"3D PSNR: {eval3d['psnr_3d_mean']:.2f} dB\n"
        metrics_text += f"3D SSIM: {eval3d['ssim_3d_mean']:.4f}\n\n"

    if eval2d_path.exists():
        with open(eval2d_path) as f:
            eval2d = yaml.safe_load(f)
        metrics_text += f"2D PSNR: {eval2d['psnr_2d']:.2f} dB\n"
        metrics_text += f"2D SSIM: {eval2d['ssim_2d']:.4f}\n\n"

    metrics_text += f"GT range: [{vol_gt.min():.1f}, {vol_gt.max():.1f}]\n"
    metrics_text += f"Pred range: [{vol_pred.min():.4f}, {vol_pred.max():.4f}]\n"
    metrics_text += f"Scale ratio: {gt_max/pred_max:.1f}x" if pred_max > 0 else ""

    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("X2-Gaussian Reconstruction: Cu Nanoparticle", fontsize=14, fontweight="bold")
    axes[0, 0].set_ylabel("Ground Truth", fontsize=12)
    axes[1, 0].set_ylabel("Predicted", fontsize=12)
    axes[2, 0].set_ylabel("Difference", fontsize=12)

    plt.tight_layout()
    save_path = output_dir / "reconstruction_results.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/aet-bench/output/cu_nano_static"
    iteration = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    main(output_dir, iteration)
