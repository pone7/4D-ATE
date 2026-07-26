#!/usr/bin/env python3
"""Analyze the 4D tracking paradox: why 4DGS hits SSIM 0.94 but tracks <20% of atoms.

Uses per-phase Gaussian XYZ arrays saved by 4DGS during evaluation
(output/aet_bench_4dgs/<sid>/model_4d/point_cloud/iteration_<k>/gauss_xyz_T{0..9}.npy)
and ground-truth atom trajectories (data/aet_bench/sim4d/<sid>/atoms_gt_4d.npy).

Produces paper/figures/fig_4d_paradox_mechanism.pdf + tab_4d_mechanism.tex.

Story:
  (a) 4DGS over-parameterizes atoms by 400-2800× and only ~15% of GT atoms
      keep a stable Gaussian "home" across all 10 phases.
  (b) The bulk of Gaussians are either static decorative clutter or drift
      to re-approximate density at each phase — the deformation field does
      not track atoms.
  (c) Atoms with larger motion are disproportionately lost, consistent with
      adaptive density control preferring re-initialization over deformation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path("/data/aet-bench")
OUT_4DGS = ROOT / "output/aet_bench_4dgs"
SIM4D = ROOT / "data/aet_bench/sim4d"
FIG_DIR = ROOT / "paper/figures"
TAB_DIR = ROOT / "paper/tables"
FIG_DIR.mkdir(exist_ok=True, parents=True)
TAB_DIR.mkdir(exist_ok=True, parents=True)

MATCH_R = 0.5           # Å tolerance for atom-Gaussian match
STATIC_DISP = 0.05      # Gaussian T0→T9 displacement < this ⇒ "static"
DRIFT_DISP = 1.5        # Gaussian displacement > this ⇒ "drift/re-init"


def _load_sample(sid: str) -> dict | None:
    pc_root = OUT_4DGS / sid / "model_4d/point_cloud"
    if not pc_root.exists():
        return None
    iters = sorted([p.name for p in pc_root.iterdir() if p.is_dir()])
    if not iters:
        return None
    latest = pc_root / iters[-1]
    meta = json.load(open(SIM4D / sid / "metadata.json"))
    cell = float(meta["cell_size_angstrom"])
    gauss = []
    for t in range(10):
        p = latest / f"gauss_xyz_T{t}.npy"
        if not p.exists():
            return None
        gauss.append(np.load(p))
    if len(set(g.shape[0] for g in gauss)) != 1:
        return None
    gauss = np.stack(gauss, axis=0) * cell  # Å
    atoms = np.load(SIM4D / sid / "atoms_gt_4d.npy")[:, :, :3]
    return {"gauss": gauss, "atoms": atoms, "sample_id": sid,
            "iter": iters[-1], "cell": cell}


def _atom_persistence(d: dict) -> dict:
    """For each GT atom at every phase, is it within MATCH_R of any Gaussian.

    Returns an array `has_match` of shape (T, N) and the per-atom
    consecutive-persistence length (max run of True phases starting from T0).
    """
    atoms = d["atoms"]          # (T, N, 3)
    gauss = d["gauss"]          # (T, G, 3)
    T, N, _ = atoms.shape
    has_match = np.zeros((T, N), dtype=bool)
    for t in range(T):
        tree = cKDTree(gauss[t])
        dd, _ = tree.query(atoms[t], k=1)
        has_match[t] = dd < MATCH_R
    # phases persisted from T0 onward = first False index; if all True → T
    persist_len = np.zeros(N, dtype=int)
    for n in range(N):
        p = 0
        for t in range(T):
            if has_match[t, n]:
                p += 1
            else:
                break
        persist_len[n] = p
    return {"has_match": has_match, "persist_len": persist_len, "T": T, "N": N}


def _gaussian_taxonomy(d: dict) -> dict:
    gauss = d["gauss"]
    atoms = d["atoms"]
    T, G, _ = gauss.shape
    disp = np.linalg.norm(gauss[-1] - gauss[0], axis=-1)
    tree0 = cKDTree(atoms[0])
    dd0, _ = tree0.query(gauss[0], k=1)
    near_t0 = dd0 < MATCH_R
    static = disp < STATIC_DISP
    drift = disp > DRIFT_DISP
    moderate = (~static) & (~drift)
    return {
        "disp": disp, "near_t0": near_t0,
        "static_n": int(static.sum()),
        "drift_n": int(drift.sum()),
        "moderate_n": int(moderate.sum()),
        "near_atom_n": int(near_t0.sum()),
        "G": int(G),
    }


def main():
    samples = sorted([p.name for p in SIM4D.iterdir() if p.is_dir()])
    data = []
    for sid in samples:
        d = _load_sample(sid)
        if d is None:
            print(f"  skip {sid}")
            continue
        ap = _atom_persistence(d)
        gt = _gaussian_taxonomy(d)
        persistent_atoms = int((ap["persist_len"] == ap["T"]).sum())
        data.append({**d, "ap": ap, "gt": gt, "persistent_atoms": persistent_atoms})
        print(
            f"  {sid}: G={gt['G']} N={ap['N']} ratio={gt['G']//ap['N']}× "
            f"persistent_atoms={persistent_atoms}/{ap['N']} "
            f"({persistent_atoms/ap['N']*100:.0f}%)"
        )
    if not data:
        return

    # Pooled statistics for figure
    all_persist = np.concatenate([d["ap"]["persist_len"] for d in data])
    all_atom_disp = np.concatenate([
        np.linalg.norm(d["atoms"][-1] - d["atoms"][0], axis=-1) for d in data
    ])
    all_ratios = np.array([d["gt"]["G"] / d["ap"]["N"] for d in data])

    # ── Figure 3 panels ───────────────────────────────────────
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    })
    fig = plt.figure(figsize=(11.5, 3.6), dpi=200)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # (a) Atom-identity persistence length histogram (pooled)
    bins = np.arange(0, 12)
    counts, _ = np.histogram(all_persist, bins=bins)
    frac = counts / counts.sum() * 100
    cmap = plt.cm.RdYlGn(np.linspace(0.05, 0.95, 11))
    ax1.bar(np.arange(11), frac, color=cmap, edgecolor="k", linewidth=0.3)
    ax1.set_xlabel("Consecutive phases with a nearby Gaussian")
    ax1.set_ylabel("Atoms (%)")
    ax1.set_title("(a) Atom-identity persistence")
    ax1.set_xticks(np.arange(11))
    ax1.axvline(10, color="#2ca02c", ls="--", lw=1, alpha=0.6)
    persist_frac = float((all_persist == 10).mean())
    ax1.text(
        10, max(frac) * 0.95,
        f" full 10/10:\n {persist_frac*100:.1f}%",
        fontsize=7, va="top", color="#2ca02c",
    )

    # (b) Gaussian taxonomy per sample (stacked bars)
    labels = ["static", "moderate (fitting)", "drift (re-init)"]
    colors = ["#888888", "#4C72B0", "#d62728"]
    N_s = len(data)
    fracs = np.zeros((N_s, 3))
    for i, d in enumerate(data):
        gt = d["gt"]
        G = gt["G"]
        fracs[i] = [gt["static_n"] / G, gt["moderate_n"] / G, gt["drift_n"] / G]
    idx = np.arange(N_s)
    bottom = np.zeros(N_s)
    for j, lab in enumerate(labels):
        ax2.bar(idx, fracs[:, j], bottom=bottom, color=colors[j],
                label=lab, width=0.7, edgecolor="white", linewidth=0.25)
        bottom += fracs[:, j]
    ax2.set_xticks(idx)
    ax2.set_xticklabels([d["sample_id"].replace("Cu4D-", "") for d in data],
                         rotation=70, fontsize=6)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Fraction of Gaussians")
    ax2.set_title("(b) Where Gaussians spend the 10 phases")
    ax2.legend(fontsize=7, loc="upper center",
                bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)

    # (c) Scatter: per-atom displacement vs persistence length
    ax3.scatter(all_atom_disp, all_persist,
                c=all_persist, cmap="RdYlGn", alpha=0.65, s=14,
                edgecolors="none", vmin=0, vmax=10)
    # Pearson r
    if np.std(all_atom_disp) > 0:
        r = float(np.corrcoef(all_atom_disp, all_persist)[0, 1])
        ax3.text(0.95, 0.95, f"r = {r:.2f}", transform=ax3.transAxes,
                  ha="right", va="top", fontsize=8,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    # Median persistence per motion bin
    bin_edges = np.linspace(all_atom_disp.min(), all_atom_disp.max(), 6)
    mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    med = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (all_atom_disp >= lo) & (all_atom_disp < hi)
        med.append(np.median(all_persist[m]) if m.any() else np.nan)
    ax3.plot(mids, med, "k-", lw=1.4, label="binned median")
    ax3.set_xlabel("GT atom displacement T0→T9 (Å)")
    ax3.set_ylabel("Persistence length (phases)")
    ax3.set_title("(c) Persistence loss is not explained by motion")
    ax3.set_ylim(-0.5, 10.5)
    ax3.legend(fontsize=7, loc="lower left")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_4d_paradox_mechanism.pdf")
    fig.savefig(FIG_DIR / "fig_4d_paradox_mechanism.png")
    plt.close(fig)
    print(f"  → {FIG_DIR / 'fig_4d_paradox_mechanism.pdf'}")

    # ── Appendix table ──────────────────────────────────────────
    G_tot = sum(d["gt"]["G"] for d in data)
    N_tot = sum(d["ap"]["N"] for d in data)
    persistent_tot = sum(d["persistent_atoms"] for d in data)

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{4D-GS Gaussian population and atom persistence on AET-Sim-4D (10 trajectories). "
        "`Persistent atoms' = GT atoms that keep a Gaussian within 0.5\\,\\AA\\ at every one of the 10 phases (a loose occupancy test). "
        "The stricter \\emph{atom identity persistence} metric in Table~\\ref{tab:4d_results} (15\\%) "
        "requires the \\emph{same} Gaussian to remain nearest at every phase, which is why it is lower than the 25\\% pooled here. "
        "Gaussian categories use the total trajectory displacement $\\|g_{T_9}-g_{T_0}\\|$ with thresholds "
        f"{STATIC_DISP:g}\\,\\AA\\ (static) and {DRIFT_DISP:g}\\,\\AA\\ (drift).}}",
        "\\label{tab:4d_mechanism}",
        "\\small",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Sample & \\#Gauss & \\#Atoms & Ratio & Persistent atoms & Static & Moderate & Drift \\\\",
        "\\midrule",
    ]
    for d in data:
        g = d["gt"]
        a = d["ap"]
        G = g["G"]
        N = a["N"]
        per_atom_pct = d["persistent_atoms"] / N * 100
        lines.append(
            f"{d['sample_id']} & {G} & {N} & {G//N}$\\times$ & "
            f"{d['persistent_atoms']}/{N}\\,({per_atom_pct:.0f}\\%) & "
            f"{g['static_n']/G*100:.1f}\\% & "
            f"{g['moderate_n']/G*100:.1f}\\% & "
            f"{g['drift_n']/G*100:.1f}\\% \\\\"
        )
    per_tot_pct = persistent_tot / N_tot * 100
    st_tot = sum(d["gt"]["static_n"] for d in data) / G_tot * 100
    mo_tot = sum(d["gt"]["moderate_n"] for d in data) / G_tot * 100
    dr_tot = sum(d["gt"]["drift_n"] for d in data) / G_tot * 100
    lines.append("\\midrule")
    lines.append(
        f"\\textbf{{pooled}} & {G_tot} & {N_tot} & {G_tot//N_tot}$\\times$ & "
        f"\\textbf{{{persistent_tot}/{N_tot}\\,({per_tot_pct:.0f}\\%)}} & "
        f"{st_tot:.1f}\\% & {mo_tot:.1f}\\% & {dr_tot:.1f}\\% \\\\"
    )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB_DIR / "tab_4d_mechanism.tex").write_text("\n".join(lines))
    print(f"  → {TAB_DIR / 'tab_4d_mechanism.tex'}")

    # Machine-readable numbers for paper cross-check
    payload = {
        "n_samples": len(data),
        "pooled_gauss": G_tot,
        "pooled_atoms": N_tot,
        "pooled_ratio": G_tot / N_tot,
        "pooled_persistent_atoms_frac": per_tot_pct / 100,
        "pooled_static_gaussian_frac": st_tot / 100,
        "pooled_moderate_gaussian_frac": mo_tot / 100,
        "pooled_drift_gaussian_frac": dr_tot / 100,
        "motion_persistence_corr": float(np.corrcoef(all_atom_disp, all_persist)[0, 1]),
        "per_sample": [
            {
                "sample_id": d["sample_id"],
                "G": int(d["gt"]["G"]),
                "N": int(d["ap"]["N"]),
                "persistent_atoms": int(d["persistent_atoms"]),
                "static_frac": float(d["gt"]["static_n"] / d["gt"]["G"]),
                "moderate_frac": float(d["gt"]["moderate_n"] / d["gt"]["G"]),
                "drift_frac": float(d["gt"]["drift_n"] / d["gt"]["G"]),
            }
            for d in data
        ],
    }
    (ROOT / "data/aet_bench/4d_mechanism.json").write_text(
        json.dumps(payload, indent=2)
    )
    print("  → data/aet_bench/4d_mechanism.json")


if __name__ == "__main__":
    main()
