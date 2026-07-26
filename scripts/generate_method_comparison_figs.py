#!/usr/bin/env python
"""Generate method-comparison visualization figures:
  1. Mono: GT vs 6 representative methods (central slice + zoomed inset)
  2. Multislice: GT vs 4 methods showing physics gap
  3. 4D: GT vs SIRT vs X²-G across 3 time points
  4. Real: GT vs best 4 methods on Zr-Zr1
"""
import os, pickle, json, sys
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path("/data/aet-bench/paper/figures")
BASE = Path("/data/aet-bench")
sys.path.insert(0, str(BASE / "scripts"))

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def reconstruct(data, method, sample_id="sample"):
    """Run a method and return the volume."""
    if method == "fbp":
        from run_benchmark import method_fbp
        return method_fbp(data)
    elif method == "astra_cgls":
        from baselines.run_astra import run_astra
        return run_astra(data, sample_id, Path(f"/tmp/vis_{method}_{sample_id}"),
                          algorithm="CGLS", n_iter=50)
    elif method == "tomopy_mlem":
        from baselines.run_tomopy import run_tomopy
        return run_tomopy(data, sample_id, Path(f"/tmp/vis_{method}_{sample_id}"),
                           algorithm="mlem")
    elif method == "sirt":
        from run_benchmark import method_sirt
        return method_sirt(data)
    elif method == "genfire":
        from baselines.run_genfire import run_genfire
        return run_genfire(data, n_iter=50, oversampling=2)
    elif method == "isonet":
        from baselines.run_isonet import run_isonet
        return run_isonet(data, sample_id, Path(f"/tmp/vis_{method}_{sample_id}"))
    elif method == "dart":
        from baselines.run_classical_extras import run_dart
        return run_dart(data, sample_id, Path(f"/tmp/vis_{method}_{sample_id}"))
    elif method == "neat":
        from baselines.run_neat import run_neat
        return run_neat(data, sample_id, Path(f"/tmp/vis_{method}_{sample_id}"),
                         iterations=1500)
    elif method == "x2gaussian":
        # Load from existing output if available
        out = BASE / f"output/aet_bench/{sample_id}/model_static/point_cloud"
        if not out.exists():
            out = BASE / f"output/aet_bench/{sample_id}/model/point_cloud"
        cands = sorted(out.glob("iteration_*/vol_pred_T0.npy")) if out.exists() else []
        if cands:
            return np.load(cands[-1])
        return None
    return None


def norm_to_gt(vol, gt):
    """Rescale vol to match GT intensity range."""
    if vol is None:
        return np.zeros_like(gt)
    s = gt.max() / max(vol.max(), 1e-8)
    return vol * s


def plot_comparison_grid(slices, titles, suptitle, outname,
                          zoom_region=None, cmap="hot"):
    """Plot a row of 2D slices with optional zoom inset."""
    n = len(slices)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.0 + 0.5, 2.8))
    if n == 1:
        axes = [axes]

    vmin = 0
    vmax = max(s.max() for s in slices if s is not None) * 0.95

    for ax, slc, title in zip(axes, slices, titles):
        if slc is None:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="white")
            ax.set_facecolor("black")
        else:
            ax.imshow(slc, cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="bilinear")
            # Add zoom inset if requested
            if zoom_region is not None:
                r = zoom_region  # (y0, y1, x0, x1)
                rect = plt.Rectangle((r[2], r[0]), r[3]-r[2], r[1]-r[0],
                                      linewidth=1, edgecolor="cyan",
                                      facecolor="none")
                ax.add_patch(rect)
                # Inset axes
                axins = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
                axins.imshow(slc[r[0]:r[1], r[2]:r[3]], cmap=cmap,
                             vmin=vmin, vmax=vmax, interpolation="bilinear")
                axins.set_xticks([]); axins.set_yticks([])
                for spine in axins.spines.values():
                    spine.set_edgecolor("cyan"); spine.set_linewidth(1)

        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{outname}.pdf", dpi=300)
    fig.savefig(FIG_DIR / f"{outname}.png", dpi=300)
    plt.close(fig)
    print(f"  {outname}.pdf")


# ═══════════════════════════════════════════════════════════════
# FIG: Mono — GT vs 6 methods
# ═══════════════════════════════════════════════════════════════
def fig_mono_method_comparison():
    """GT vs 8 representative methods (one per family + best-in-class)."""
    sample = "Cu-0279-01"  # 279 atoms, nice FCC pattern
    data = load_pkl(BASE / f"data/aet_bench/mono/{sample}/analytical/data.pkl")
    gt = data["image"][0]
    V = gt.shape[0]
    mid = V // 2

    # Cover all 4 families: Classical(FBP, CGLS), Regularized(IsoNet, FISTA-TV),
    # INR(NeAT), Gaussian(X²-G) + SIRT as the workhorse
    methods = ["fbp", "astra_cgls", "sirt", "tomopy_mlem",
               "dart", "isonet", "neat", "x2gaussian"]
    titles = ["GT", "FBP", "CGLS", "SIRT", "MLEM",
              "DART", "IsoNet", "NeAT", "X²-G"]

    slices = [gt[mid]]
    for m in methods:
        print(f"    running {m} on {sample}...")
        vol = reconstruct(data, m, sample)
        slices.append(norm_to_gt(vol, gt)[mid] if vol is not None else None)

    zoom = (40, 88, 40, 88)
    plot_comparison_grid(slices, titles,
                          f"Method comparison: {sample} (analytical, 279 atoms)",
                          "fig_method_compare_mono", zoom_region=zoom)


# ═══════════════════════════════════════════════════════════════
# FIG: Multislice — GT vs 4 methods showing degradation
# ═══════════════════════════════════════════════════════════════
def fig_multislice_method_comparison():
    """Multislice: GT vs 6 methods showing physics gap."""
    sample = "Cu-309"
    data = load_pkl(BASE / f"data/aet_bench/multislice/{sample}/analytical/data.pkl")
    gt = data["image"][0]
    V = gt.shape[0]
    mid = V // 2

    methods = ["fbp", "astra_cgls", "sirt", "tomopy_mlem", "isonet", "neat"]
    titles = ["GT (abTEM)", "FBP", "CGLS", "SIRT", "MLEM", "IsoNet", "NeAT"]

    slices = [gt[mid]]
    for m in methods:
        print(f"    running {m} on {sample} (multislice)...")
        vol = reconstruct(data, m, sample)
        slices.append(norm_to_gt(vol, gt)[mid] if vol is not None else None)

    plot_comparison_grid(slices, titles,
                          f"Multislice: all methods degrade ({sample}, abTEM)",
                          "fig_method_compare_multislice")


# ═══════════════════════════════════════════════════════════════
# FIG: 4D — GT vs SIRT vs X²-G across 3 time points
# ═══════════════════════════════════════════════════════════════
def fig_4d_method_comparison():
    sample = "Cu4D-08A-00"
    data = load_pkl(BASE / f"data/aet_bench/sim4d/{sample}/analytical/data.pkl")
    gt_4d = data["image"]  # (10, V, V, V)
    V = gt_4d.shape[1]
    mid = V // 2

    # FBP — static baseline
    from run_benchmark import method_fbp, method_sirt
    vol_fbp = method_fbp(data)
    vol_sirt = method_sirt(data)

    # ASTRA-CGLS
    from baselines.run_astra import run_astra
    vol_cgls = run_astra(data, sample, Path(f"/tmp/vis_cgls_{sample}"),
                          algorithm="CGLS", n_iter=50)

    # X²-G 4D — load from existing output
    x2g_dir = BASE / f"output/aet_bench/{sample}/model_4d/point_cloud"
    x2g_vols = {}
    if x2g_dir.exists():
        for p in sorted(x2g_dir.glob("iteration_*/vol_pred_T*.npy")):
            t = int(p.stem.split("T")[-1])
            x2g_vols[t] = np.load(p)

    phases = [0, 2, 4, 7, 9]
    n_cols = len(phases)
    n_rows = 4  # GT, FBP, SIRT, X²-G
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.0))

    vmin, vmax = 0, gt_4d.max() * 0.95
    row_labels = ["GT", "FBP (static)", "SIRT (static)", "X²-G (4D)"]
    static_vols = [None, vol_fbp, vol_sirt, None]  # None = dynamic

    for j, t in enumerate(phases):
        # GT
        axes[0, j].imshow(gt_4d[t, mid], cmap="inferno", vmin=vmin, vmax=vmax,
                           interpolation="bilinear")
        axes[0, j].set_title(f"t={t}", fontsize=9)

        # FBP (static — same for all t)
        fbp_norm = norm_to_gt(vol_fbp, gt_4d[t])
        axes[1, j].imshow(fbp_norm[mid], cmap="inferno", vmin=vmin, vmax=vmax,
                           interpolation="bilinear")

        # SIRT (static)
        sirt_norm = norm_to_gt(vol_sirt, gt_4d[t])
        axes[2, j].imshow(sirt_norm[mid], cmap="inferno", vmin=vmin, vmax=vmax,
                           interpolation="bilinear")

        # X²-G 4D (time-varying)
        if t in x2g_vols:
            x2g_norm = norm_to_gt(x2g_vols[t], gt_4d[t])
            axes[3, j].imshow(x2g_norm[mid], cmap="inferno", vmin=vmin, vmax=vmax,
                               interpolation="bilinear")
        else:
            axes[3, j].text(0.5, 0.5, "N/A", transform=axes[3, j].transAxes,
                             ha="center", color="white")
            axes[3, j].set_facecolor("black")

    for i, label in enumerate(row_labels):
        axes[i, 0].set_ylabel(label, fontsize=8, fontweight="bold", rotation=90,
                               labelpad=10)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"4D reconstruction comparison ({sample})", fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_method_compare_4d.pdf", dpi=300)
    fig.savefig(FIG_DIR / "fig_method_compare_4d.png", dpi=300)
    plt.close(fig)
    print("  fig_method_compare_4d.pdf")


# ═══════════════════════════════════════════════════════════════
# FIG: Real — Zr-Zr1 GT vs 4 best methods
# ═══════════════════════════════════════════════════════════════
def fig_real_method_comparison():
    """Real data: GT vs 5 representative methods."""
    sample = "Zr-Zr1"
    data = load_pkl(BASE / f"data/aet_bench/real/{sample}/analytical/data.pkl")
    gt = data["image"][0]
    V = gt.shape[0]
    mid = V // 2

    methods = ["fbp", "astra_cgls", "sirt", "isonet", "neat"]
    titles = ["Published GT", "FBP", "CGLS", "SIRT", "IsoNet", "NeAT"]

    slices = [gt[mid]]
    for m in methods:
        print(f"    running {m} on {sample} (real)...")
        vol = reconstruct(data, m, sample)
        slices.append(norm_to_gt(vol, gt)[mid] if vol is not None else None)

    plot_comparison_grid(slices, titles,
                          f"Real data: {sample} (Zr-ZrO₂, 15k atoms)",
                          "fig_method_compare_real", cmap="hot")


def main():
    print("Generating method comparison figures...")
    fig_mono_method_comparison()
    fig_multislice_method_comparison()
    fig_4d_method_comparison()
    fig_real_method_comparison()
    print("Done.")


if __name__ == "__main__":
    main()
