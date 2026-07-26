#!/usr/bin/env python3
"""Summarize the dose/noise pilot tier into a paper figure + table.

Reads data/aet_bench/results_dose.jsonl and emits:
    paper/figures/fig_dose_recall.pdf  — per-method curves: dose vs (PSNR, recall)
    paper/tables/tab_dose_t1.tex       — per-(dose, method) metrics with 95% CI
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/data/aet-bench")
RF = ROOT / "data/aet_bench/results_dose.jsonl"
FIG = ROOT / "paper/figures"
TAB = ROOT / "paper/tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
from bench_bootstrap import bootstrap_ci  # noqa: E402

METHODS = ["wbp", "astra_cgls", "tomopy_mlem", "isonet", "x2gaussian"]
LABELS = {
    "wbp": "WBP", "astra_cgls": "ASTRA-CGLS", "tomopy_mlem": "MLEM",
    "isonet": "IsoNet(GP)", "x2gaussian": "X²-G",
}
COLORS = {
    "wbp": "#4C72B0", "astra_cgls": "#55A868",
    "tomopy_mlem": "#C44E52", "isonet": "#8172B2", "x2gaussian": "#CCB974",
}
MARKERS = {
    "wbp": "o", "astra_cgls": "s", "tomopy_mlem": "D",
    "isonet": "^", "x2gaussian": "P",
}


def _load():
    rows = []
    if not RF.exists():
        print(f"no results at {RF}")
        return rows
    for line in RF.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in r:
            continue
        rows.append(r)
    return rows


def main():
    rows = _load()
    if not rows:
        return
    print(f"Loaded {len(rows)} dose rows")

    # Group: (dose, method) → list of metric rows
    grp = defaultdict(list)
    for r in rows:
        grp[(r.get("dose"), r["method"])].append(r["metrics"])
    doses = sorted({k[0] for k in grp})
    # ── Figure: dose vs (PSNR, Recall) ─────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.2), dpi=200)
    for m in METHODS:
        xs, psnr_mu, psnr_lo, psnr_hi = [], [], [], []
        rec_mu, rec_lo, rec_hi = [], [], []
        for d in doses:
            ms = grp.get((d, m), [])
            if not ms:
                continue
            xs.append(d)
            p_vals = [v.get("vol_psnr") for v in ms]
            mu, lo, hi = bootstrap_ci(p_vals)
            psnr_mu.append(mu); psnr_lo.append(lo); psnr_hi.append(hi)
            r_vals = [v.get("recall") for v in ms]
            mu, lo, hi = bootstrap_ci(r_vals)
            rec_mu.append(mu); rec_lo.append(lo); rec_hi.append(hi)
        if not xs:
            continue
        xs = np.asarray(xs)
        psnr_mu = np.asarray(psnr_mu)
        psnr_lo = np.asarray(psnr_lo); psnr_hi = np.asarray(psnr_hi)
        rec_mu = np.asarray(rec_mu)
        rec_lo = np.asarray(rec_lo); rec_hi = np.asarray(rec_hi)
        ax1.fill_between(xs, psnr_lo, psnr_hi, color=COLORS[m], alpha=0.18)
        ax1.plot(xs, psnr_mu, marker=MARKERS[m], color=COLORS[m],
                 label=LABELS[m], lw=1.5, ms=6)
        ax2.fill_between(xs, np.asarray(rec_lo)*100, np.asarray(rec_hi)*100,
                         color=COLORS[m], alpha=0.18)
        ax2.plot(xs, rec_mu*100, marker=MARKERS[m], color=COLORS[m],
                 label=LABELS[m], lw=1.5, ms=6)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("Dose (e/Å²)")
        ax.grid(alpha=0.3)
    ax1.set_ylabel("PSNR (dB) ↑")
    ax1.set_title("(a) Volume quality")
    ax2.set_ylabel("Atom recall (%) ↑")
    ax2.set_title("(b) Atom localization")
    ax1.legend(fontsize=7, loc="lower right")
    ax2.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_dose_recall.pdf")
    fig.savefig(FIG / "fig_dose_recall.png")
    plt.close(fig)
    print(f"  → {FIG / 'fig_dose_recall.pdf'}")

    # ── LaTeX table ─────────────────────────────────────────
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{T1 results on AET-Sim-Mono-Dose (20 Cu samples × 3 Poisson "
        "electron doses × 5 methods). Subscripts are 95\\% bootstrap CI half-widths.}",
        "\\label{tab:dose}",
        "\\small",
        "\\begin{tabular}{l" + "cc" * len(doses) + "}",
        "\\toprule",
        "Method & " + " & ".join([f"\\multicolumn{{2}}{{c}}{{{int(d)}\\,e/\\AA$^2$}}" for d in doses]) + " \\\\",
        "& " + " & ".join(["PSNR & Recall"] * len(doses)) + " \\\\",
        "\\midrule",
    ]
    for m in METHODS:
        parts = [LABELS[m]]
        for d in doses:
            ms = grp.get((d, m), [])
            if not ms:
                parts += ["--", "--"]
                continue
            p_mu, p_lo, p_hi = bootstrap_ci([v.get("vol_psnr") for v in ms])
            r_mu, r_lo, r_hi = bootstrap_ci([v.get("recall") for v in ms])
            if p_mu is None or math.isnan(p_mu):
                parts.append("--")
            else:
                p_hw = (p_hi - p_lo) / 2.0 if p_hi is not None else 0.0
                parts.append(f"{p_mu:.1f}{{\\scriptsize$_{{\\pm{p_hw:.1f}}}$}}")
            if r_mu is None or math.isnan(r_mu):
                parts.append("--")
            else:
                r_hw = (r_hi - r_lo) / 2.0 if r_hi is not None else 0.0
                parts.append(f"{r_mu*100:.1f}\\%{{\\scriptsize$_{{\\pm{r_hw*100:.1f}}}$}}")
        lines.append(" & ".join(parts) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "tab_dose_t1.tex").write_text("\n".join(lines))
    print(f"  → {TAB / 'tab_dose_t1.tex'}")


if __name__ == "__main__":
    main()
