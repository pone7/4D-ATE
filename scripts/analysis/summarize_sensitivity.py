#!/usr/bin/env python
"""Figures + LaTeX tables for the extraction-fairness analyses (NeurIPS 3869).

Reads:
  data/aet_bench/analysis/extractor_ceiling.json      (recall vs min-distance)
  data/aet_bench/analysis/native_extraction.jsonl     (shared/subvoxel/native)

Writes:
  paper/figures/fig_extractor_ceiling.{pdf,png}
  paper/figures/fig_native_extraction.{pdf,png}
  paper/tables/tab_native_extraction.tex
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ANA = ROOT / "data" / "aet_bench" / "analysis"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"

INK = "#1a1a2e"
PALETTE = {
    "GT": "#111827",
    "4dgs": "#c1443c",
    "r2gaussian": "#d19a1d",
    "fista_tv": "#2f8f5b",
    "tomopy_mlem": "#3b6ea5",
    "astra_cgls": "#7a5195",
}
LABEL = {
    "GT": "Ground truth (ceiling)",
    "4dgs": "4D-GS",
    "r2gaussian": "R$^2$-Gaussian",
    "fista_tv": "FISTA-TV",
    "tomopy_mlem": "TomoPy-MLEM",
    "astra_cgls": "ASTRA-CGLS",
}


def _boot_ci(vals, n=1000, seed=0):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if len(vals) == 0:
        return None, None, None
    if len(vals) == 1:
        return float(vals[0]), float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    bm = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))


def fig_ceiling():
    path = ANA / "extractor_ceiling.json"
    if not path.exists():
        print("skip ceiling: no data")
        return
    d = json.loads(path.read_text())
    grid = [float(x) for x in d["min_dist_grid"]]
    mr = d["mean_recall"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name in ["GT", "fista_tv", "tomopy_mlem", "astra_cgls", "4dgs", "r2gaussian"]:
        if name not in mr:
            continue
        ys = [mr[name].get(str(md)) for md in grid]
        if all(y is None for y in ys):
            continue
        ys = [np.nan if y is None else y for y in ys]
        style = dict(marker="o", lw=2.2, ms=6)
        if name == "GT":
            style.update(lw=3.0, ls="--", zorder=5)
        ax.plot(grid, ys, color=PALETTE.get(name, "#888"), label=LABEL.get(name, name), **style)
    ax.axvspan(0.8, 1.3, color="#2f8f5b", alpha=0.08)
    ax.axvline(1.0, color="#2f8f5b", ls=":", lw=1.4)
    ax.text(1.0, 0.02, " pinned\n 1.0 Å", color="#2f8f5b", fontsize=8.5, va="bottom", ha="left")
    ax.set_xlabel("NMS min-distance $d_\\mathrm{min}$ (Å)", fontsize=11)
    ax.set_ylabel(f"atom recall (τ=0.5 Å), n={d['n_samples']}", fontsize=11)
    ax.set_title("Extraction ceiling: the harness, not the method,\ncaps recall when the window merges atoms",
                 fontsize=11.5, weight="bold", color=INK)
    ax.set_ylim(0, 1.02)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_extractor_ceiling.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_extractor_ceiling.png", dpi=150, bbox_inches="tight")
    print("wrote fig_extractor_ceiling")


def _native_table_data():
    rows = [json.loads(l) for l in (ANA / "native_extraction.jsonl").open() if l.strip()]
    agg = defaultdict(list)
    for r in rows:
        if r["mode"] == "oracle_count" and r["tau_angstrom"] == 0.5:
            agg[(r["method"], r["extractor"])].append(r["metrics"]["recall"])
    return agg


def fig_native():
    path = ANA / "native_extraction.jsonl"
    if not path.exists():
        print("skip native: no data")
        return
    agg = _native_table_data()
    methods = sorted({k[0] for k in agg})
    extractors = ["shared", "shared_subvoxel", "native"]
    exlabel = {"shared": "shared peak", "shared_subvoxel": "shared+sub-voxel", "native": "native (primitives)"}
    excol = {"shared": "#3b6ea5", "shared_subvoxel": "#7a5195", "native": "#c1443c"}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(methods))
    w = 0.26
    for i, ex in enumerate(extractors):
        means, los, his = [], [], []
        for m in methods:
            mean, lo, hi = _boot_ci(agg.get((m, ex), []))
            means.append(mean or 0); los.append((mean or 0) - (lo or 0)); his.append((hi or 0) - (mean or 0))
        ax.bar(x + (i - 1) * w, means, w, yerr=[los, his], capsize=3,
               color=excol[ex], label=exlabel[ex])
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(m, m) for m in methods], fontsize=10.5)
    ax.set_ylabel("atom recall (τ=0.5 Å)", fontsize=11)
    ax.set_title("Representation-native extraction does not rescue\nGaussian methods — the inversion is real",
                 fontsize=11.5, weight="bold", color=INK)
    ax.set_ylim(0, 1.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_native_extraction.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_native_extraction.png", dpi=150, bbox_inches="tight")
    print("wrote fig_native_extraction")


def tab_native():
    path = ANA / "native_extraction.jsonl"
    if not path.exists():
        return
    agg = _native_table_data()
    methods = sorted({k[0] for k in agg})
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{\textbf{Extractor comparison for Gaussian methods} (AET-Sim-Mono, oracle-count mode, $\tau{=}0.5$\,\AA, mean $\pm$ 95\% bootstrap CI). The shared extractor is at the corrected operating point ($d_\mathrm{min}{=}1.0$\,\AA). Sub-voxel refinement and representation-native primitive clustering do not lift recall above the shared extractor, so the voxel-vs-atom inversion is a property of the reconstructions.}",
        r"\label{tab:native_extraction}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{shared peak} & \textbf{shared+sub-voxel} & \textbf{native (primitives)} \\",
        r"\midrule",
    ]
    for m in methods:
        cells = []
        for ex in ["shared", "shared_subvoxel", "native"]:
            mean, lo, hi = _boot_ci(agg.get((m, ex), []))
            if mean is None:
                cells.append("--")
            else:
                cells.append(f"{mean:.3f}\\,$\\pm$\\,{(hi-lo)/2:.3f}")
        name = LABEL.get(m, m)
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "tab_native_extraction.tex").write_text("\n".join(lines) + "\n")
    print("wrote tab_native_extraction.tex")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_ceiling()
    fig_native()
    tab_native()


if __name__ == "__main__":
    sys.exit(main())
