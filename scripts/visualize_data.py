#!/usr/bin/env python
"""Visualize generated HAADF-STEM tilt series data."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys


def main(data_dir):
    data_dir = Path(data_dir)
    out_path = data_dir / "visualization.png"

    # Load metadata
    with open(data_dir / "meta_data.json") as f:
        meta = json.load(f)

    # Load ground truth
    gt_vol = np.load(data_dir / "gt_volume.npy")
    atoms = np.load(data_dir / "atoms_gt.npy")  # (N, 4): x, y, z, Z
    positions = atoms[:, :3]
    atomic_nums = atoms[:, 3]

    # Load a few projections at representative angles
    all_projs = meta["proj_train"] + meta["proj_test"]
    all_projs.sort(key=lambda p: p["angle"])
    angles_deg = [np.degrees(p["angle"]) for p in all_projs]

    # Pick 5 representative angles
    pick_angles = [-60, -30, 0, 30, 60]
    picked = []
    for target in pick_angles:
        idx = np.argmin([abs(a - target) for a in angles_deg])
        proj = np.load(data_dir / all_projs[idx]["file_path"])
        picked.append((angles_deg[idx], proj))

    # Create figure: 3 rows
    # Row 1: 5 projections
    # Row 2: 3 volume slices (XY, XZ, YZ at center)
    # Row 3: atomic structure (XY and XZ views)
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    # Row 1: Projections
    for i, (angle, proj) in enumerate(picked):
        ax = axes[0, i]
        im = ax.imshow(proj, cmap="inferno", origin="lower")
        ax.set_title(f"Tilt = {angle:.0f} deg", fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2: Volume slices
    mid = gt_vol.shape[0] // 2
    slice_data = [
        (gt_vol[mid, :, :], "XY slice (z=mid)"),
        (gt_vol[:, mid, :], "XZ slice (y=mid)"),
        (gt_vol[:, :, mid], "YZ slice (x=mid)"),
    ]
    for i, (slc, title) in enumerate(slice_data):
        ax = axes[1, i]
        im = ax.imshow(slc, cmap="hot", origin="lower")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Volume histogram
    ax = axes[1, 3]
    ax.hist(gt_vol[gt_vol > 0.01].ravel(), bins=80, color="steelblue", alpha=0.8)
    ax.set_title("Volume intensity histogram", fontsize=11)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")

    # Volume stats
    ax = axes[1, 4]
    ax.axis("off")
    stats_text = (
        f"Volume shape: {gt_vol.shape}\n"
        f"Range: [{gt_vol.min():.2f}, {gt_vol.max():.2f}]\n"
        f"Non-zero voxels: {(gt_vol > 0.01).sum()}\n"
        f"Total voxels: {gt_vol.size}\n"
        f"Sparsity: {(gt_vol > 0.01).sum() / gt_vol.size:.1%}\n\n"
        f"Atoms: {len(positions)}\n"
        f"Z values: {np.unique(atomic_nums).astype(int)}\n"
        f"Projections: {len(all_projs)}\n"
        f"Detector: {picked[0][1].shape}\n"
        f"Angle range: [{min(angles_deg):.0f}, {max(angles_deg):.0f}] deg"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Row 3: Atomic structure
    # XY view (top)
    ax = axes[2, 0]
    colors = atomic_nums / atomic_nums.max()
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c=colors,
                         cmap="coolwarm", s=15, alpha=0.7, edgecolors="k", linewidths=0.3)
    ax.set_title("Atoms: XY view (top)", fontsize=11)
    ax.set_xlabel("X (A)")
    ax.set_ylabel("Y (A)")
    ax.set_aspect("equal")

    # XZ view (side)
    ax = axes[2, 1]
    ax.scatter(positions[:, 0], positions[:, 2], c=colors,
               cmap="coolwarm", s=15, alpha=0.7, edgecolors="k", linewidths=0.3)
    ax.set_title("Atoms: XZ view (side)", fontsize=11)
    ax.set_xlabel("X (A)")
    ax.set_ylabel("Z (A)")
    ax.set_aspect("equal")

    # YZ view (front)
    ax = axes[2, 2]
    ax.scatter(positions[:, 1], positions[:, 2], c=colors,
               cmap="coolwarm", s=15, alpha=0.7, edgecolors="k", linewidths=0.3)
    ax.set_title("Atoms: YZ view (front)", fontsize=11)
    ax.set_xlabel("Y (A)")
    ax.set_ylabel("Z (A)")
    ax.set_aspect("equal")

    # Projection sinogram (center row of each projection)
    ax = axes[2, 3]
    center_row = picked[0][1].shape[0] // 2
    sinogram = []
    for p in all_projs:
        proj = np.load(data_dir / p["file_path"])
        sinogram.append(proj[center_row, :])
    sinogram = np.array(sinogram)
    ax.imshow(sinogram, cmap="inferno", aspect="auto", origin="lower",
              extent=[0, sinogram.shape[1], min(angles_deg), max(angles_deg)])
    ax.set_title("Sinogram (center row)", fontsize=11)
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Angle (deg)")

    # Projection intensity vs angle
    ax = axes[2, 4]
    mean_intensities = []
    max_intensities = []
    for p in all_projs:
        proj = np.load(data_dir / p["file_path"])
        mean_intensities.append(proj.mean())
        max_intensities.append(proj.max())
    ax.plot(angles_deg, mean_intensities, "b.-", label="mean", markersize=3)
    ax2 = ax.twinx()
    ax2.plot(angles_deg, max_intensities, "r.-", label="max", markersize=3)
    ax.set_title("Projection intensity", fontsize=11)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Mean intensity", color="b")
    ax2.set_ylabel("Max intensity", color="r")

    fig.suptitle(f"HAADF-STEM Data: {data_dir.name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/aet-bench/data/cu_nano"
    main(data_dir)
