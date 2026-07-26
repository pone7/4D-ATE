#!/usr/bin/env python
"""Concept teaser for AET-Bench: pixel-accurate is not atom-accurate.

Builds a clean, self-contained schematic (no external assets) from a real
ground-truth volume: a visually faithful density slice on the left, the same
field with the atom graph overlaid on the right showing the failure modes
(missed / merged / hallucinated / displaced), and a compact SSIM-vs-recall
bar underneath. All labels sit *inside* their panels (addressing the Fig. 1
label-overflow comment).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

INK = "#1a1a2e"
GOOD = "#2f8f5b"
BAD = "#c1443c"
WARN = "#d19a1d"
MUTE = "#6b7280"


def _load_slice(sample="Cu-0279-01"):
    p = ROOT / "data" / "aet_bench" / "mono" / sample / "analytical" / "data.pkl"
    with p.open("rb") as f:
        d = pickle.load(f)
    img = d["image"]
    img = img[0] if img.ndim == 4 else img
    z = img.shape[2] // 2
    sl = img[:, :, z - 2 : z + 3].mean(axis=2)
    sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-9)
    return sl


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sl = _load_slice()
    n = sl.shape[0]

    fig = plt.figure(figsize=(11.0, 4.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.82], wspace=0.16)

    # Panel 1: density looks accurate
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(sl, cmap="magma", origin="lower")
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.text(0.5, 1.06, "Reconstructed density", transform=ax0.transAxes,
             ha="center", va="bottom", fontsize=12, weight="bold", color=INK)
    ax0.text(0.03, 0.04, "SSIM 0.94  —  looks correct", transform=ax0.transAxes,
             ha="left", va="bottom", fontsize=10.5, color="white",
             bbox=dict(boxstyle="round,pad=0.3", fc=GOOD, ec="none", alpha=0.9))

    # Panel 2: atom graph overlay with failure modes
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(sl, cmap="gray", origin="lower", alpha=0.55)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.text(0.5, 1.06, "Extracted atoms", transform=ax1.transAxes,
             ha="center", va="bottom", fontsize=12, weight="bold", color=INK)

    rng = np.random.default_rng(3)
    # place a few illustrative markers over bright regions
    ys, xs = np.where(sl > 0.55)
    idx = rng.choice(len(xs), size=min(26, len(xs)), replace=False)
    pts = np.column_stack([xs[idx], ys[idx]]).astype(float)
    for (x, y) in pts:
        ax1.add_patch(Circle((x, y), 1.6, fill=False, ec=GOOD, lw=1.4))
    # failure-mode annotations (schematic positions inside the panel)
    def mark(x, y, color, label, dx, dy):
        ax1.add_patch(Circle((x, y), 2.4, fill=False, ec=color, lw=2.2))
        ax1.annotate(label, xy=(x, y), xytext=(x + dx, y + dy),
                     fontsize=9.5, color="white", ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none"),
                     arrowprops=dict(arrowstyle="-", color=color, lw=1.4))
    cx = n * 0.5
    mark(n * 0.30, n * 0.30, BAD, "missed", -0.0, -n * 0.16)
    mark(n * 0.68, n * 0.34, WARN, "merged", n * 0.0, -n * 0.17)
    mark(n * 0.72, n * 0.68, BAD, "hallucinated", 0, n * 0.16)
    mark(n * 0.34, n * 0.70, WARN, "displaced", 0, n * 0.16)
    ax1.set_xlim(0, n); ax1.set_ylim(0, n)

    # big not-equal between panels
    fig.text(0.505, 0.52, r"$\neq$", ha="center", va="center",
             fontsize=30, color=INK, weight="bold")

    # Panel 3: SSIM vs recall bars
    ax2 = fig.add_subplot(gs[0, 2])
    methods = ["4D-GS", "FISTA-TV"]
    ssim = [0.94, 0.62]
    recall = [0.44, 0.95]
    x = np.arange(len(methods))
    w = 0.36
    ax2.bar(x - w / 2, ssim, w, color=MUTE, label="SSIM (voxel)")
    ax2.bar(x + w / 2, recall, w, color=INK, label="atom recall")
    ax2.set_xticks(x); ax2.set_xticklabels(methods, fontsize=10.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("score", fontsize=10.5)
    ax2.set_title("The inversion", fontsize=12, weight="bold", color=INK, pad=8)
    ax2.legend(fontsize=8.5, frameon=False, loc="upper center", ncol=1)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.annotate("high pixels,\nfew atoms", xy=(0 + w / 2, 0.44),
                 xytext=(0.15, 0.72), textcoords="data", fontsize=8.5,
                 color=BAD, ha="left",
                 arrowprops=dict(arrowstyle="->", color=BAD, lw=1.3))

    fig.suptitle("Pixel-accurate atomic tomography is not atom-accurate",
                 fontsize=13.5, weight="bold", color=INK, y=1.005)
    fig.savefig(OUT / "fig_concept.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_concept.png", dpi=150, bbox_inches="tight")
    print("wrote", OUT / "fig_concept.pdf")


if __name__ == "__main__":
    main()
