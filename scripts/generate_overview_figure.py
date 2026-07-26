#!/usr/bin/env python
"""Generate Figure 1: AET-Bench overview diagram.

Layout (inspired by RealPDEBench Figure 1):
  Top row: 6 data tiers with one representative image each
  Bottom: pipeline arrow showing Data → Methods → Metrics → Findings

This creates a single high-quality PDF/PNG figure.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import pickle
from pathlib import Path

FIG_DIR = Path("/data/aet-bench/paper/figures")
BASE = Path("/data/aet-bench")


def load_slice(pkl_path, phase=0):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    img = d["image"]
    if img.ndim == 4:
        img = img[phase]
    return img[img.shape[0] // 2]


def main():
    fig = plt.figure(figsize=(7.5, 4.5))

    # ── Top row: 6 data tier thumbnails ──
    tiers = [
        ("AET-Sim\nMono", "data/aet_bench/mono/Cu-0279-01/analytical/data.pkl",
         "410 samples\n5 elements", "#4C72B0"),
        ("AET-Sim\nBi", "data/aet_bench/bi/CuPt-0617-02/analytical/data.pkl",
         "240 samples\n3 alloy systems", "#55A868"),
        ("AET-Sim\n4D", "data/aet_bench/sim4d/Cu4D-08A-00/analytical/data.pkl",
         "10×10 phases\nCu annealing", "#C44E52"),
        ("AET\nMulti", "data/aet_bench/multislice/Cu-309/analytical/data.pkl",
         "3 samples\nabTEM", "#8172B2"),
        ("AET\nReal", "data/aet_bench/real/Zr-Zr1/analytical/data.pkl",
         "26 samples\n7 studies", "#CCB974"),
        ("AET\nBrownian", "data/aet_bench/brownian_pt/Pt-ECT1_time04s/analytical/data.pkl",
         "66 samples\nPt nanocrystals", "#64B5CD"),
    ]

    for i, (name, pkl, desc, color) in enumerate(tiers):
        ax = fig.add_axes([0.02 + i * 0.16, 0.52, 0.14, 0.42])
        try:
            slc = load_slice(BASE / pkl)
            ax.imshow(slc, cmap="hot", interpolation="bilinear")
        except Exception:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="white")
            ax.set_facecolor("black")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color); spine.set_linewidth(2)
        ax.set_title(name, fontsize=7, fontweight="bold", color=color, pad=3)
        ax.text(0.5, -0.08, desc, transform=ax.transAxes, fontsize=5,
                ha="center", va="top", color="#444444")

    # ── Bottom: pipeline flow ──
    ax_bot = fig.add_axes([0.0, 0.0, 1.0, 0.45])
    ax_bot.set_xlim(0, 10); ax_bot.set_ylim(0, 3)
    ax_bot.axis("off")

    # Boxes
    boxes = [
        (0.5, 1.5, "6 Data Tiers\n755 samples\n7 real studies", "#E8F0FE", "#4C72B0"),
        (3.0, 1.5, "14 Methods\n5 families\nClassical / Post\nINR / Hash / GS", "#E8F5E9", "#2E7D32"),
        (5.5, 1.5, "9 Metrics\nPSNR / SSIM / FSC\nRecall / RMSD\nTraj RMSD / Persist", "#FFF3E0", "#E65100"),
        (8.0, 1.5, "6 Key Findings\nGENFIRE gap\n4D paradox\nSim-to-real chasm", "#FCE4EC", "#C62828"),
    ]
    for x, y, text, facecolor, edgecolor in boxes:
        rect = mpatches.FancyBboxPatch(
            (x - 0.9, y - 0.9), 1.8, 1.8,
            boxstyle="round,pad=0.1",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5
        )
        ax_bot.add_patch(rect)
        ax_bot.text(x, y, text, ha="center", va="center", fontsize=6,
                    fontweight="bold", color=edgecolor, linespacing=1.4)

    # Arrows
    for x1, x2 in [(1.5, 2.0), (4.0, 4.5), (6.5, 7.0)]:
        ax_bot.annotate("", xy=(x2, 1.5), xytext=(x1, 1.5),
                        arrowprops=dict(arrowstyle="->", color="#666666",
                                       lw=2, connectionstyle="arc3"))

    # Title
    fig.text(0.5, 0.97, "AET-Bench: 755 samples · 6 tiers · 14 methods · 4 tasks · 14,000+ results",
             ha="center", va="top", fontsize=10, fontweight="bold")

    fig.savefig(FIG_DIR / "fig_overview.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig_overview.pdf")


if __name__ == "__main__":
    main()
