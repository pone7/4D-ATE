#!/usr/bin/env python
"""
Generate all paper figures + LaTeX tables from results.jsonl.

Outputs:
    paper/figures/fig_*.pdf      — matplotlib figures
    paper/tables/tab_*.tex       — auto-generated LaTeX table fragments
"""
from __future__ import annotations

import json
import math
import os
import argparse
import hashlib
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_bootstrap import bootstrap_ci  # noqa: E402

# Lazy imports for matplotlib (set non-interactive backend)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aet_bench_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS = Path(
    os.environ.get(
        "AET_BENCH_RESULTS",
        "/data/aet-bench/data/aet_bench/results.jsonl",
    )
)
FIG_DIR = Path("/data/aet-bench/paper/figures")
TAB_DIR = Path("/data/aet-bench/paper/tables")
DATA_ROOT = Path("/data/aet-bench/data/aet_bench")
ANALYSIS_DIR = DATA_ROOT / "analysis"
SENSITIVITY_JSONL = ANALYSIS_DIR / "extraction_sensitivity.jsonl"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
PANEL_FIGSIZE = (3.65, 2.55)

# ─── Style ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

METHOD_ORDER = [
    "fbp", "wbp", "astra_sirt", "astra_sart", "astra_cgls", "astra_ossart",
    "tomopy_gridrec", "tomopy_mlem", "dart", "fista_tv", "genfire",
    "sirt", "isonet", "neat", "implicit_et", "r2gaussian", "x2gaussian",
]
METHOD_LABELS = {
    "fbp": "FBP", "wbp": "WBP",
    "astra_sirt": "ASTRA-SIRT", "astra_sart": "ASTRA-SART",
    "astra_cgls": "ASTRA-CGLS", "astra_ossart": "ASTRA-OSSART",
    "tomopy_gridrec": "gridrec", "tomopy_mlem": "MLEM",
    "dart": "DART", "fista_tv": "FISTA-TV", "genfire": "GENFIRE",
    "sirt": "SIRT", "isonet": "IsoNet(GP)",
    "neat": "NeAT", "implicit_et": "Impl-ET",
    "r2gaussian": "R²-G", "x2gaussian": "X²-G",
    "fourdgs": "4D-GS-compat",
    # Display names are also LaTeX-safe: a missing entry falls through to the
    # raw registry key, whose underscore breaks text-mode LaTeX.
    "sax_nerf": "SAX-NeRF", "dif_gaussian": "DIF-G",
    "instamap": "InstaMap", "x_gaussian": "X-G",
}
FAMILY_COLORS = {
    "Classical": "#4C72B0",
    "Regularized": "#55A868",
    "INR": "#C44E52",
    "Gaussian": "#8172B2",
}
def _family(m):
    if m in ("fbp","wbp","astra_sirt","astra_sart","astra_cgls","astra_ossart",
             "tomopy_gridrec","tomopy_mlem","dart","genfire","sirt"):
        return "Classical"
    if m in ("fista_tv","isonet"):
        return "Regularized"
    if m in ("neat","implicit_et"):
        return "INR"
    return "Gaussian"

# ─── Load data ───────────────────────────────────────────────
def load_results():
    rows = []
    with open(RESULTS) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(rows):
    by_subset_task = defaultdict(lambda: {"rows": 0, "samples": set(), "methods": set()})
    for r in rows:
        key = f"{r.get('subset')}/{r.get('task', 'T1')}"
        by_subset_task[key]["rows"] += 1
        by_subset_task[key]["samples"].add(r.get("sample_id"))
        by_subset_task[key]["methods"].add(r.get("method"))

    provenance = {
        "results_path": str(RESULTS),
        "results_sha256": _sha256(RESULTS),
        "rows": len(rows),
        "coverage": {
            key: {
                "rows": val["rows"],
                "samples": len(val["samples"]),
                "methods": len(val["methods"]),
            }
            for key, val in sorted(by_subset_task.items())
        },
    }
    out = TAB_DIR / "results_provenance.json"
    out.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"  {out.name}")

def group_by(rows, subset, task="T1"):
    groups = defaultdict(list)
    for r in rows:
        if r["subset"] == subset and r.get("task", "T1") == task:
            m = r["method"]
            groups[m].append(r["metrics"])
    return groups


def _clean_float(v):
    if v in (None, "nan", "inf", "-inf"):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _mean_metric_from_rows(rows, key):
    vals = []
    for r in rows:
        val = _clean_float(r.get("metrics", {}).get(key))
        if val is not None:
            vals.append(val)
    return float(np.mean(vals)) if vals else None


def _latex_escape(text):
    text = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


# Atom-level metrics computed by the fast_eval pipeline. Rows written by the
# earlier run_benchmark.py pipeline do not carry "n_pred"; those values are
# trustworthy. Rows written by the broken fast_eval (pre-v2) hardcoded
# "n_pred": 0, poisoning averages. atom_eval == "v2" marks the fixed pipeline.
_ATOM_KEYS = {"recall", "precision", "f1", "rmsd", "n_pred", "n_matched"}


def _atom_valid(m):
    if "n_pred" not in m:
        return True
    return m.get("atom_eval") == "v2"


def _metric_vals(ms, key):
    """Extract a metric column, filtering atom metrics through the eval guard."""
    if key in _ATOM_KEYS:
        return [x.get(key) for x in ms if _atom_valid(x)]
    return [x.get(key) for x in ms]

def mean_std(vals):
    if not vals:
        return None, None
    a = [v for v in vals if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    if not a:
        return None, None
    m = np.mean(a)
    s = np.std(a) if len(a) > 1 else 0.0
    return float(m), float(s)


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Metric inversion on analytical tiers
# ═══════════════════════════════════════════════════════════════
def fig_mono_bars(rows):
    """Finding 1 panel: voxel-level and atom-level rankings diverge on Mono."""
    subset = "mono"
    methods = [
        "tomopy_mlem", "fourdgs", "isonet", "sirt", "x2gaussian",
        "r2gaussian", "implicit_et", "neat", "fista_tv",
    ]
    g = group_by(rows, subset)
    short_labels = {
        "isonet": "IsoNet",
        "tomopy_mlem": "MLEM",
        "fista_tv": "FISTA-TV",
        "implicit_et": "Impl-ET",
        "r2gaussian": "R²-G",
        "x2gaussian": "X²-G",
        "fourdgs": "4D-GS-compat",
    }
    pts = []
    for m in methods:
        if m not in g:
            continue
        ss, _ = mean_std(_metric_vals(g[m], "vol_ssim"))
        rc, _ = mean_std(_metric_vals(g[m], "recall"))
        if ss is None or rc is None:
            continue
        pts.append((m, ss, rc * 100.0))

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    for m, ss, rc in pts:
        fam = _family(m)
        marker = "s" if fam == "Gaussian" else "o"
        ax.scatter(
            ss, rc, s=48, c=FAMILY_COLORS[fam], marker=marker,
            edgecolors="white", linewidth=0.65, zorder=4,
        )

    label_offsets = {
        "tomopy_mlem": (-0.010, -8.0),
        "fourdgs": (-0.030, 4.2),
        "isonet": (0.007, 3.8),
        "sirt": (-0.006, -7.0),
        "x2gaussian": (0.008, -8.0),
        "r2gaussian": (0.008, 1.5),
        "implicit_et": (0.008, 1.5),
        "neat": (0.008, 3.0),
        "fista_tv": (0.008, -7.0),
    }
    for m, ss, rc in pts:
        dx, dy = label_offsets.get(m, (0.006, 1.5))
        ax.annotate(
            short_labels.get(m, METHOD_LABELS.get(m, m)),
            xy=(ss, rc), xytext=(ss + dx, rc + dy),
            fontsize=6.2, ha="left" if dx >= 0 else "right", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.35, color="0.50"),
            zorder=5,
        )

    best_ss = max(pts, key=lambda x: x[1])
    best_rc = max(pts, key=lambda x: x[2])
    ax.axvline(best_ss[1], color="0.82", lw=0.75, ls="--", zorder=1)
    ax.axhline(best_rc[2], color="0.82", lw=0.75, ls="--", zorder=1)
    ax.text(
        0.02, 0.98,
        f"SSIM leader: {short_labels.get(best_ss[0], METHOD_LABELS.get(best_ss[0], best_ss[0]))}\n"
        f"Recall leader: {short_labels.get(best_rc[0], METHOD_LABELS.get(best_rc[0], best_rc[0]))}",
        transform=ax.transAxes, va="top", ha="left", fontsize=6.2,
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="0.84", lw=0.55),
    )
    ax.set_title("Finding 1: metric inversion on Mono", fontsize=8.5)
    ax.set_xlabel("Voxel SSIM ↑")
    ax.set_ylabel("Atom recall (%) ↑")
    ax.set_xlim(0.695, 0.922)
    ax.set_ylim(-2, 96)
    ax.grid(alpha=0.24)
    fig.tight_layout(pad=0.35)
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(FIG_DIR / "fig_mono_bars.pdf")
        fig.savefig(FIG_DIR / "fig_mono_bars.png")
    plt.close(fig)
    print("  fig_mono_bars.pdf")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: X²-Gaussian recall vs nanoparticle size
# ═══════════════════════════════════════════════════════════════
def fig_size_scaling(rows):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for method, marker, color, label in [
        ("x2gaussian", "o", FAMILY_COLORS["Gaussian"], "X²-Gaussian"),
        ("r2gaussian", "s", "#8172B2", "R²-Gaussian"),
        ("fbp", "^", FAMILY_COLORS["Classical"], "FBP"),
        ("astra_cgls", "D", "#4C72B0", "ASTRA-CGLS"),
    ]:
        pts = []
        for r in rows:
            if r["subset"] == "mono" and r["method"] == method and r.get("task", "T1") == "T1":
                sid = r["sample_id"]
                # Extract atom count from sample_id (e.g. Cu-0617-03 → 617)
                parts = sid.split("-")
                if len(parts) >= 2:
                    try:
                        n_atoms = int(parts[1])
                    except ValueError:
                        continue
                    if not _atom_valid(r["metrics"]):
                        continue
                    recall = r["metrics"].get("recall", 0) or 0
                    pts.append((n_atoms, recall * 100))
        if not pts:
            continue
        pts.sort()
        # Bin by n_atoms (dedup)
        bins = defaultdict(list)
        for n, rec in pts:
            bins[n].append(rec)
        ns = sorted(bins)
        means = [np.mean(bins[n]) for n in ns]
        ax.plot(ns, means, marker=marker, color=color, label=label,
                markersize=4, linewidth=1.2, alpha=0.85)

    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Recall@0.5Å (%)")
    ax.set_title("Atom localization recall vs. nanoparticle size")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_ylim(-5, 105)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_size_scaling.pdf")
    fig.savefig(FIG_DIR / "fig_size_scaling.png")
    plt.close(fig)
    print("  fig_size_scaling.pdf")


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Realism ladder — voxel score vs atom recovery
# ═══════════════════════════════════════════════════════════════
def fig_tier_comparison(rows):
    """Finding 2 panel: realism lowers atom recovery despite usable voxel scores."""
    tiers = [
        ("mono", "T1", "Mono"),
        ("bi", "T1", "Bi"),
        ("multislice", "T1", "Multi."),
        ("real", "T1", "Real"),
        ("brownian_pt", "T1", "Brownian"),
    ]
    methods = [
        "fbp", "wbp", "astra_sirt", "astra_sart", "astra_cgls", "astra_ossart",
        "tomopy_gridrec", "tomopy_mlem", "dart", "fista_tv", "genfire",
        "sirt", "isonet", "neat", "implicit_et", "r2gaussian", "x2gaussian", "fourdgs",
    ]

    best_ssim, best_recall = [], []
    for subset, task, _ in tiers:
        g = group_by(rows, subset, task=task)
        ssim_candidates = []
        recall_candidates = []
        for m in methods:
            if m not in g:
                continue
            ss, _ = mean_std(_metric_vals(g[m], "vol_ssim"))
            rc, _ = mean_std(_metric_vals(g[m], "recall"))
            if ss is not None:
                ssim_candidates.append((ss * 100.0, m))
            if rc is not None:
                recall_candidates.append((rc * 100.0, m))
        best_ssim.append(max(ssim_candidates, default=(0, "")))
        best_recall.append(max(recall_candidates, default=(0, "")))

    short_labels = {
        "isonet": "IsoNet",
        "tomopy_mlem": "MLEM",
        "fourdgs": "4D-GS-compat",
        "astra_cgls": "ASTRA-CGLS",
        "genfire": "GENFIRE",
    }
    labels = [x[2] for x in tiers]
    x = np.arange(len(tiers))
    ssim_vals = [v for v, _ in best_ssim]
    recall_vals = [v for v, _ in best_recall]

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    ax.plot(x, ssim_vals, marker="o", ms=4.2, lw=1.55, color="#4C72B0", label="Best SSIM ×100")
    ax.plot(x, recall_vals, marker="s", ms=4.2, lw=1.55, color="#C44E52", label="Best atom recall")
    ax.fill_between(x, recall_vals, ssim_vals, color="0.78", alpha=0.16, zorder=0)

    for i, (val, (_, method)) in enumerate(zip(ssim_vals, best_ssim)):
        ha = "right" if i == len(ssim_vals) - 1 else "center"
        x_text = i - 0.02 if i == len(ssim_vals) - 1 else i
        ax.text(x_text, min(val + 4.0, 99.0), f"{val:.1f}\n{short_labels.get(method, METHOD_LABELS.get(method, method))}",
                ha=ha, va="bottom", fontsize=5.7, color="#2F4B7C")
    for i, (val, (_, method)) in enumerate(zip(recall_vals, best_recall)):
        dy = -7.5 if val > 20 else 3.0
        va = "top" if val > 20 else "bottom"
        ha = "left" if i == 0 else ("right" if i == len(recall_vals) - 1 else "center")
        x_text = i + 0.06 if i == 0 else (i - 0.02 if i == len(recall_vals) - 1 else i)
        ax.text(x_text, val + dy, f"{val:.1f}\n{short_labels.get(method, METHOD_LABELS.get(method, method))}",
                ha=ha, va=va, fontsize=5.7, color="#8F3035")

    ax.set_title("Finding 2: realism opens a voxel-atom gap", fontsize=8.5)
    ax.set_ylabel("Score on common 0-100 scale ↑")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 106)
    ax.grid(axis="y", alpha=0.24)
    ax.legend(loc="lower left", fontsize=5.8, framealpha=0.92)
    fig.tight_layout(pad=0.35)
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(FIG_DIR / "fig_tier_comparison.pdf")
        fig.savefig(FIG_DIR / "fig_tier_comparison.png")
    plt.close(fig)
    print("  fig_tier_comparison.pdf")

# ═══════════════════════════════════════════════════════════════
# FIGURE 4: 4D T7 — smooth density vs atom identity
# ═══════════════════════════════════════════════════════════════
def fig_4d_paradox(rows):
    """Finding 3 panel: high 4D volume similarity does not preserve atom identity."""
    g = group_by(rows, "4d", task="T7")
    methods_4d = ["x2gaussian", "fourdgs"]
    labels_4d = [METHOD_LABELS.get(m, m) for m in methods_4d]
    ssim = []
    persist = []
    traj = []
    for m in methods_4d:
        ss, _ = mean_std(_metric_vals(g[m], "vol_ssim"))
        pe, _ = mean_std([r.get("atom_identity_persistence") for r in g[m]])
        tr, _ = mean_std([r.get("trajectory_rmsd") for r in g[m]])
        ssim.append((ss or 0) * 100.0)
        persist.append((pe or 0) * 100.0)
        traj.append(tr or 0)

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    x = np.arange(len(methods_4d))
    w = 0.30
    bars1 = ax.bar(x - w/2, ssim, w, label="Frame SSIM ×100", color="#8172B2",
                   edgecolor="white", linewidth=0.65)
    bars2 = ax.bar(x + w/2, persist, w, label="Atom ID persistence", color="#C44E52",
                   edgecolor="white", linewidth=0.65)

    for bar, val in zip(bars1, ssim):
        ax.text(bar.get_x() + bar.get_width()/2, val - 4.2, f"{val:.1f}",
                ha="center", va="top", fontsize=7.0, color="white", fontweight="bold")
    for bar, val, tr in zip(bars2, persist, traj):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2.2, f"{val:.1f}\nRMSD {tr:.2f}Å",
                ha="center", va="bottom", fontsize=6.2, color="#7A252A")

    ax.axhspan(90, 100, color="#8172B2", alpha=0.08, zorder=0)
    ax.set_title("Finding 3: smooth 4D density, weak identity", fontsize=8.5)
    ax.set_ylabel("Score on common 0-100 scale ↑")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_4d)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 106)
    ax.grid(axis="y", alpha=0.24)
    ax.legend(loc="upper center", ncol=2, fontsize=5.8, framealpha=0.92)
    fig.tight_layout(pad=0.35)
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(FIG_DIR / "fig_4d_paradox.pdf")
        fig.savefig(FIG_DIR / "fig_4d_paradox.png")
    plt.close(fig)
    print("  fig_4d_paradox.pdf")


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Dataset realism ladder
# ═══════════════════════════════════════════════════════════════
def fig_realism_ladder(rows=None):
    """Dataset-section ladder built from representative AET-Bench samples.

    Visual language matches Figure 1 (numbered rounded panels, coloured borders,
    thick connector arrows, shared species colours: A = blue, B = orange).
    """
    import pickle
    import matplotlib.patches as mpatches

    root = Path("/data/aet-bench/data/aet_bench")

    # Shared with Figure 1 so the two teasers read as one system.
    SPECIES_A = "#2E75B6"   # blue
    SPECIES_B = "#E8890C"   # orange
    INK = "#1A1A2E"
    SUBTLE = "#5B6472"
    ARROW = "#5A5A5A"

    # Cool -> warm as realism increases.
    tiers = [
        ("Mono", "localization", "#1F4E79"),
        ("Bi", "chemistry", "#2E75B6"),
        ("4D", "trajectories", "#17868A"),
        ("Multislice", "scattering", "#C98A00"),
        ("Real", "reference agreement", "#ED7D31"),
        ("Brownian-Pt", "dynamic AET", "#C0504D"),
    ]

    def _norm_img(img):
        img = np.asarray(img, dtype=float)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = np.percentile(img, [2, 99.6])
        if hi <= lo:
            lo, hi = float(np.min(img)), float(np.max(img) + 1e-6)
        return np.clip((img - lo) / (hi - lo), 0, 1)

    def _crop_nonempty(img, pad=8):
        arr = np.asarray(img)
        mask = arr > np.percentile(arr, 72)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return arr
        y0, y1 = max(0, ys.min() - pad), min(arr.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(arr.shape[1], xs.max() + pad + 1)
        return arr[y0:y1, x0:x1]

    def _projection(path, index=None):
        with Path(path).open("rb") as f:
            d = pickle.load(f)
        proj = d["train"]["projections"]
        if index is None:
            index = len(proj) // 2
        return _norm_img(_crop_nonempty(proj[index]))

    def _load_atoms(path):
        return np.load(path, allow_pickle=True).astype(float)

    def _rot(pts, angle_deg):
        th = np.deg2rad(angle_deg)
        c, s = np.cos(th), np.sin(th)
        return pts @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).T

    def _frame(pts_ref, angle_deg, sx=0.68, sy=0.58, cy=0.50):
        """Return a projector mapping physical xyz -> axes coords, fixed by pts_ref."""
        ref = _rot(np.asarray(pts_ref, float)[:, :3] - np.asarray(pts_ref, float)[:, :3].mean(0), angle_deg)
        cx_, cy_ = ref[:, 0].mean(), ref[:, 1].mean()
        span = max(np.ptp(ref[:, 0]), np.ptp(ref[:, 1]), 1e-6)
        centre = np.asarray(pts_ref, float)[:, :3].mean(0)

        def proj(xyz):
            q = _rot(np.asarray(xyz, float)[:, :3] - centre, angle_deg)
            return (q[:, 0] - cx_) / span * sx + 0.5, (q[:, 1] - cy_) / span * sy + cy, q[:, 2]
        return proj

    def _content_axes(fig, rect):
        ax = fig.add_axes(rect)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("none")
        for sp in ax.spines.values():
            sp.set_visible(False)
        return ax

    fig = plt.figure(figsize=(7.2, 2.0))
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")

    m_x, gap = 0.007, 0.030
    pw = (1 - 2 * m_x - 5 * gap) / 6
    pb, ph = 0.035, 0.925

    panels = []
    for i, (name, sub, colour) in enumerate(tiers):
        left = m_x + i * (pw + gap)
        panels.append(left)
        bg.add_patch(mpatches.FancyBboxPatch(
            (left, pb), pw, ph,
            boxstyle="round,pad=0,rounding_size=0.010",
            facecolor="white", edgecolor=colour, linewidth=1.15, zorder=1,
            mutation_aspect=fig.get_figwidth() / fig.get_figheight()))
        # numbered badge + title (Figure 1 language)
        bx, by = left + 0.017, pb + ph - 0.085
        bg.plot([bx], [by], marker="o", ms=7.4, color=colour, zorder=4, clip_on=False)
        bg.text(bx, by, str(i + 1), ha="center", va="center", fontsize=4.6,
                fontweight="bold", color="white", zorder=5)
        bg.text(bx + 0.016, by, name, ha="left", va="center", fontsize=6.9,
                fontweight="bold", color=colour, zorder=5)
        bg.text(left + pw / 2, pb + 0.052, sub, ha="center", va="center",
                fontsize=5.9, color=SUBTLE, zorder=5)
        if i < 5:
            xa = left + pw
            bg.annotate("", xy=(xa + gap - 0.006, 0.5), xytext=(xa + 0.006, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.5,
                                        mutation_scale=9, shrinkA=0, shrinkB=0), zorder=6)

    def _rect(i, pad_l=0.010, top=0.175, bot=0.125):
        return [panels[i] + pad_l, pb + bot, pw - 2 * pad_l, ph - top - bot]

    # 1. Mono — one species.
    ax = _content_axes(fig, _rect(0))
    a = _load_atoms(root / "mono/Cu-0279-01/atoms_gt.npy")
    pr = _frame(a, -30)
    x, y, z = pr(a)
    o = np.argsort(z)
    ax.scatter(x[o], y[o], s=8.5, c=SPECIES_A, alpha=0.95, linewidths=0.18, edgecolors="white", zorder=3)

    # 2. Bi — two species, colours shared with Figure 1.
    ax = _content_axes(fig, _rect(1))
    a = _load_atoms(root / "bi/CuPt-0279-00/atoms_gt.npy")
    pr = _frame(a, -30)
    x, y, z = pr(a)
    zi = a[:, 3].astype(int)
    lo_el = np.sort(np.unique(zi))[0]
    cols = np.where(zi == lo_el, SPECIES_A, SPECIES_B)
    o = np.argsort(z)
    ax.scatter(x[o], y[o], s=8.5, c=cols[o], alpha=0.95, linewidths=0.18, edgecolors="white", zorder=3)
    # inline identity key (never colour-alone)
    for dx, cc, lb in [(0.0, SPECIES_A, "Cu"), (0.30, SPECIES_B, "Pt")]:
        ax.scatter([0.20 + dx], [0.055], s=8.0, c=cc, linewidths=0.18, edgecolors="white",
                   zorder=5, clip_on=False)
        ax.text(0.255 + dx, 0.055, lb, fontsize=4.9, color=SUBTLE, ha="left", va="center",
                zorder=5, clip_on=False)

    # 3. 4D — real per-atom trajectories across the 10 phases.
    ax = _content_axes(fig, _rect(2))
    a4 = np.load(root / "sim4d/Cu4D-14A-07/atoms_gt_4d.npy", allow_pickle=True).astype(float)
    disp = np.linalg.norm(a4[-1, :, :3] - a4[0, :, :3], axis=1)
    # Zoom onto one compact neighbourhood around the most mobile atom (a true
    # zoom, no exaggeration) so the real displacements are legible at tile size.
    hot = int(np.argmax(disp))
    d0 = np.linalg.norm(a4[0, :, :3] - a4[0, hot, :3], axis=1)
    near = np.where(d0 < 7.0)[0]
    mob = near[np.argsort(disp[near])[::-1][:7]]                 # movers in that window
    pr = _frame(a4[0][mob], -25, sx=0.46, sy=0.40)               # centre on the movers
    xs = np.stack([pr(a4[t])[0] for t in range(a4.shape[0])])    # (T, N)
    ys = np.stack([pr(a4[t])[1] for t in range(a4.shape[0])])
    inview = np.where((xs[0] > 0.02) & (xs[0] < 0.98) & (ys[0] > 0.10) & (ys[0] < 0.98))[0]
    ctx = np.setdiff1d(inview, mob)
    ax.scatter(xs[-1][ctx], ys[-1][ctx], s=9.0, c="#D2D8DF", alpha=0.8,
               linewidths=0, zorder=2)                            # static context
    for k in mob:
        ax.plot(xs[:, k], ys[:, k], lw=1.7, color="#17868A", alpha=0.95,
                solid_capstyle="round", zorder=4)                 # trajectory
    ax.scatter(xs[0][mob], ys[0][mob], s=10.0, c="white", alpha=1.0,
               linewidths=0.75, edgecolors="#8E99A6", zorder=5)    # t0
    ax.scatter(xs[-1][mob], ys[-1][mob], s=15.0, c=SPECIES_A, alpha=1.0,
               linewidths=0.3, edgecolors="white", zorder=6)       # t9
    ax.text(0.5, 0.045, r"$t_0 \rightarrow t_9$ (zoom)", fontsize=4.9, color=SUBTLE,
            ha="center", va="center", zorder=7)

    # 4. Multislice — scattering-aware projection.
    ax = _content_axes(fig, _rect(3))
    multi = _projection(root / "multislice/Cu-309/analytical/data.pkl", index=0)
    ax.imshow(multi, cmap="magma", origin="lower", extent=(0.10, 0.90, 0.222, 0.778),
              interpolation="bilinear", aspect="auto", zorder=2)

    # 5. Real — experimental projection + sparse reference atom overlay.
    ax = _content_axes(fig, _rect(4))
    real_img = _projection(root / "real/AuPd-ICNP1/analytical/data.pkl", index=24)
    ax.imshow(real_img, cmap="gray", origin="lower", extent=(0.10, 0.90, 0.222, 0.778),
              interpolation="bilinear", aspect="auto", zorder=2)
    ra = _load_atoms(root / "real/AuPd-ICNP1/atoms_gt.npy")
    keep = np.linspace(0, len(ra) - 1, min(230, len(ra))).astype(int)
    pr = _frame(ra[keep], -18, sx=0.44, sy=0.40, cy=0.52)
    x, y, _ = pr(ra[keep])
    ax.scatter(x, y, s=1.9, c="#3FA45B", alpha=0.75, linewidths=0, zorder=4)

    # 6. Brownian-Pt — experimental time points (no atom correspondence -> ghosted).
    ax = _content_axes(fig, _rect(5))
    bp = ["Pt-EFC2_time05s", "Pt-EFC2_time21s", "Pt-EFC2_time39s"]
    ref = _load_atoms(root / f"brownian_pt/{bp[1]}/atoms_gt.npy")
    pr = _frame(ref, -33, sx=0.56, sy=0.50)
    for name, cx, lb in zip(bp, [0.175, 0.50, 0.825], ["5 s", "21 s", "39 s"]):
        atoms = _load_atoms(root / f"brownian_pt/{name}/atoms_gt.npy")
        x, y, _ = pr(atoms)
        ax.scatter((x - 0.5) * 0.52 + cx, (y - 0.5) * 0.52 + 0.56, s=1.5,
                   c=SPECIES_B, alpha=0.92, linewidths=0, zorder=3)
        ax.text(cx, 0.135, lb, fontsize=4.6, color=SUBTLE, ha="center", va="center", zorder=5)
    for x0, x1 in [(0.295, 0.375), (0.620, 0.700)]:
        ax.annotate("", xy=(x1, 0.56), xytext=(x0, 0.56),
                    arrowprops=dict(arrowstyle="-|>", color="#9AA6B2", lw=0.6,
                                    mutation_scale=4.5), zorder=5)

    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(FIG_DIR / "fig_realism_ladder.pdf")
        fig.savefig(FIG_DIR / "fig_realism_ladder.png", dpi=300)
    plt.close(fig)
    print("  fig_realism_ladder.pdf")

# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Runtime vs PSNR Pareto frontier
# ═══════════════════════════════════════════════════════════════
def fig_pareto(rows):
    g = group_by(rows, "mono")
    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Skip WBP (overlaps FBP exactly) and gridrec (outlier, PSNR broken)
    skip = {"wbp", "tomopy_gridrec"}
    # Manual nudges: (dx, dy) in points from data position
    nudge = {
        "fbp":          (5, -8),
        "astra_sirt":   (5, 4),
        "astra_sart":   (5, -6),
        "astra_cgls":   (5, 4),
        "astra_ossart": (5, -6),
        "tomopy_mlem":  (5, 4),
        "dart":         (5, -6),
        "fista_tv":     (-40, 4),
        "genfire":      (5, 5),
        "sirt":         (-28, -8),
        "isonet":       (-38, 5),
        "neat":         (5, 5),
        "implicit_et":  (5, -8),
        "r2gaussian":   (5, -10),
        "x2gaussian":   (5, 6),
    }

    for m in METHOD_ORDER:
        if m not in g or m in skip:
            continue
        p, _ = mean_std([r.get("vol_psnr") for r in g[m]])
        runtimes = [r2.get("runtime_sec", 1) for r2 in rows
                     if r2["subset"] == "mono" and r2["method"] == m
                     and r2.get("task", "T1") == "T1"]
        t = np.mean(runtimes) if runtimes else 1
        if p is None:
            continue
        color = FAMILY_COLORS[_family(m)]
        marker = "D" if "gaussian" in m else ("*" if m in ("neat","implicit_et") else "o")
        ms = 80 if marker == "*" else (70 if marker == "D" else 50)
        ax.scatter(t, p, s=ms, c=color, marker=marker,
                    edgecolors="black", linewidth=0.5, zorder=5)
        dx, dy = nudge.get(m, (5, 3))
        ax.annotate(METHOD_LABELS.get(m, m), (t, p),
                     fontsize=6, ha="left", va="center",
                     xytext=(dx, dy), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Runtime per sample (seconds, log scale)")
    ax.set_ylabel("PSNR (dB) ↑")
    ax.set_title("Quality--speed tradeoff (AET-Sim-Mono)")
    ax.grid(alpha=0.3)
    ax.set_ylim(12, 33)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=handles, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_pareto.pdf")
    fig.savefig(FIG_DIR / "fig_pareto.png")
    plt.close(fig)
    print("  fig_pareto.pdf")



# ═══════════════════════════════════════════════════════════════
# REVIEWER ADDENDA: sensitivity, coverage, real-reference metadata
# ═══════════════════════════════════════════════════════════════
def _case_key(row):
    return (
        row.get("subset"), row.get("sample_id"), row.get("phase"),
        row.get("volume_source"), row.get("mode"),
    )


def _setting_key(row):
    return (
        row.get("mode"),
        float(row.get("tau_angstrom", 0.0)),
        float(row.get("nms_radius_angstrom", 0.0)),
        float(row.get("threshold_rel", 0.0)),
    )


def _sensitivity_rank_correlations(rows):
    try:
        from scipy.stats import kendalltau, spearmanr
    except Exception:
        return {}

    by_setting = defaultdict(dict)
    for row in rows:
        f1 = _clean_float(row.get("metrics", {}).get("f1"))
        if f1 is None:
            continue
        by_setting[_setting_key(row)][_case_key(row)] = f1

    out = {}
    for mode in sorted({r.get("mode") for r in rows}):
        default_key = (mode, 0.5, 2.0, 0.05)
        default = by_setting.get(default_key, {})
        if len(default) < 2:
            continue
        for setting, vals in by_setting.items():
            if setting[0] != mode:
                continue
            common = sorted(set(default) & set(vals))
            if len(common) < 2:
                continue
            base_vec = [default[k] for k in common]
            val_vec = [vals[k] for k in common]
            sp = spearmanr(base_vec, val_vec).correlation
            kt = kendalltau(base_vec, val_vec).correlation
            out[setting] = {
                "spearman": None if sp is None or math.isnan(sp) else float(sp),
                "kendall": None if kt is None or math.isnan(kt) else float(kt),
            }
    return out


def fig_threshold_sweep(rows):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    modes = ["blind_fixed", "oracle_count"]
    colors = {"blind_fixed": "#4C72B0", "oracle_count": "#C44E52"}
    labels = {"blind_fixed": "Blind fixed", "oracle_count": "Oracle count"}
    for mode in modes:
        xs, ys, lo_hi = [], [], []
        for tau in sorted({float(r.get("tau_angstrom", 0.0)) for r in rows}):
            vals = [
                _clean_float(r.get("metrics", {}).get("f1"))
                for r in rows
                if r.get("mode") == mode and float(r.get("tau_angstrom", 0.0)) == tau
            ]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            xs.append(tau)
            ys.append(float(np.mean(vals)) * 100.0)
            lo_hi.append((float(np.percentile(vals, 5)) * 100.0, float(np.percentile(vals, 95)) * 100.0))
        if xs:
            lows = [y - lo for y, (lo, _) in zip(ys, lo_hi)]
            highs = [hi - y for y, (_, hi) in zip(ys, lo_hi)]
            ax.errorbar(xs, ys, yerr=[lows, highs], marker="o", ms=4.0, lw=1.35,
                        capsize=2.5, color=colors[mode], label=labels[mode])
    ax.set_title("Extraction sensitivity to match radius", fontsize=8.5)
    ax.set_xlabel("Matching radius $\\tau$ (\\AA)")
    ax.set_ylabel("Mean F1 (%)")
    ax.set_xticks([0.3, 0.5, 0.7, 1.0])
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.24)
    ax.legend(fontsize=6.2, framealpha=0.92)
    fig.tight_layout(pad=0.35)
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(FIG_DIR / "fig_threshold_sweep.pdf")
        fig.savefig(FIG_DIR / "fig_threshold_sweep.png")
    plt.close(fig)
    print("  fig_threshold_sweep.pdf")


def fig_extraction_sensitivity(rows):
    if not rows:
        return
    subset = [
        r for r in rows
        if r.get("mode") == "blind_fixed" and abs(float(r.get("tau_angstrom", 0.0)) - 0.5) < 1e-9
    ] or rows
    nms_vals = sorted({float(r.get("nms_radius_angstrom", 0.0)) for r in subset})
    thr_vals = sorted({float(r.get("threshold_rel", 0.0)) for r in subset})
    heat = np.full((len(nms_vals), len(thr_vals)), np.nan)
    for i, nms in enumerate(nms_vals):
        for j, thr in enumerate(thr_vals):
            vals = [
                _clean_float(r.get("metrics", {}).get("recall"))
                for r in subset
                if abs(float(r.get("nms_radius_angstrom", 0.0)) - nms) < 1e-9
                and abs(float(r.get("threshold_rel", 0.0)) - thr) < 1e-9
            ]
            vals = [v for v in vals if v is not None]
            if vals:
                heat[i, j] = float(np.mean(vals)) * 100.0

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    im = ax.imshow(heat, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_title("Blind extraction recall at $\\tau=0.5$\\AA", fontsize=8.5)
    ax.set_xlabel("Peak threshold")
    ax.set_ylabel("NMS radius (\\AA)")
    ax.set_xticks(np.arange(len(thr_vals)))
    ax.set_xticklabels([f"{v:.2f}" for v in thr_vals])
    ax.set_yticks(np.arange(len(nms_vals)))
    ax.set_yticklabels([f"{v:.1f}" for v in nms_vals])
    for i in range(len(nms_vals)):
        for j in range(len(thr_vals)):
            if not np.isnan(heat[i, j]):
                ax.text(j, i, f"{heat[i, j]:.0f}", ha="center", va="center",
                        fontsize=6.2, color="white" if heat[i, j] < 55 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (%)", fontsize=7)
    fig.tight_layout(pad=0.35)
    with plt.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(FIG_DIR / "fig_extraction_sensitivity.pdf")
        fig.savefig(FIG_DIR / "fig_extraction_sensitivity.png")
    plt.close(fig)
    print("  fig_extraction_sensitivity.pdf")


def write_sensitivity_table(rows):
    if not rows:
        return
    correlations = _sensitivity_rank_correlations(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("mode"), float(row.get("tau_angstrom", 0.0)))].append(row)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Extraction-protocol sensitivity sweep. Rows aggregate audited volume cases over NMS radii $\\{1.5,2.0,2.5\\}$~\\AA{} and peak thresholds $\\{0.03,0.05,0.10\\}$. Rank correlations compare F1 rankings with the default setting ($\\tau=0.5$~\\AA, NMS 2.0~\\AA, threshold 0.05) within each extraction mode.}",
        "\\label{tab:sensitivity}",
        "\\small",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        '\\textbf{Mode} & $\\boldsymbol{\\tau}$ & \\textbf{Rows} & \\textbf{Recall} & \\textbf{Precision} & \\textbf{F1} & $\\boldsymbol{\\rho}$ & $\\boldsymbol{\\tau_K}$ \\\\',
        "\\midrule",
    ]
    for (mode, tau), vals in sorted(grouped.items()):
        recall = _mean_metric_from_rows(vals, "recall")
        precision = _mean_metric_from_rows(vals, "precision")
        f1 = _mean_metric_from_rows(vals, "f1")
        corr_vals = [
            correlations.get(_setting_key(v), {})
            for v in vals
        ]
        spearman_vals = [c.get("spearman") for c in corr_vals if c.get("spearman") is not None]
        kendall_vals = [c.get("kendall") for c in corr_vals if c.get("kendall") is not None]
        spearman = float(np.mean(spearman_vals)) if spearman_vals else None
        kendall = float(np.mean(kendall_vals)) if kendall_vals else None
        def pct(x):
            return "--" if x is None else f"{x*100:.1f}\\%"
        def num(x):
            return "--" if x is None else f"{x:.2f}"
        label = "oracle-count" if mode == "oracle_count" else "blind-fixed"
        lines.append(
            f'{label} & {tau:.1f} & {len(vals)} & {pct(recall)} & {pct(precision)} & {pct(f1)} & {num(spearman)} & {num(kendall)} \\\\')
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    out = TAB_DIR / "tab_sensitivity.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  {out.name}")


def collect_real_reference_metadata(data_root: Path = DATA_ROOT):
    rows = []
    real_root = data_root / "real"
    for meta_path in sorted(real_root.glob("*/metadata.json")):
        sample_dir = meta_path.parent
        meta = json.loads(meta_path.read_text())
        atoms_path = sample_dir / "atoms_gt.npy"
        atom_count = int(meta.get("n_atoms") or 0)
        if atoms_path.exists():
            try:
                atom_count = int(len(np.load(atoms_path, allow_pickle=True)))
            except Exception:
                pass
        has_true = bool(meta.get("has_atom_gt")) and atom_count > 0
        if has_true:
            reference_type = "true_reference"
            coord_note = "published coordinates"
        elif sample_dir.name == "MPEA2":
            reference_type = "pseudo_peaks"
            coord_note = "pseudo peak list"
        else:
            reference_type = "no_atom_list"
            coord_note = "volume only"

        cell_size = _clean_float(meta.get("cell_size_angstrom"))
        voxel_size = _clean_float(meta.get("voxel_size_angstrom"))
        # Keep this metadata-only: real-data pickles can be hundreds of MB and
        # are unnecessary for the reviewer-facing reference taxonomy.
        rows.append({
            "sample_id": sample_dir.name,
            "source": meta.get("source", "unknown"),
            "atom_count": atom_count,
            "reference_type": reference_type,
            "voxel_size_angstrom": voxel_size,
            "cell_size_angstrom": cell_size,
            "coord_note": coord_note,
        })
    return rows


def write_real_reference_metadata_table():
    rows = collect_real_reference_metadata()
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{AET-Real reference metadata. Atom-level real-data metrics are reported as reference agreement, not direct physical ground truth.}",
        "\\label{tab:real_reference_metadata}",
        "\\scriptsize",
        "\\begin{tabular}{llrlll}",
        "\\toprule",
        '\\textbf{Sample} & \\textbf{Source} & \\textbf{Atoms} & \\textbf{Reference} & \\textbf{Voxel (\\AA)} & \\textbf{Frame note} \\\\',
        "\\midrule",
    ]
    for row in rows:
        voxel = row["voxel_size_angstrom"]
        voxel_s = "--" if voxel is None else f"{voxel:.3f}"
        lines.append(
            f"{_latex_escape(row['sample_id'])} & {_latex_escape(row['source'])} & {row['atom_count']} & "
            f"{_latex_escape(row['reference_type'].replace('_', ' '))} & {voxel_s} & {_latex_escape(row['coord_note'])} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    out = TAB_DIR / "tab_real_reference_metadata.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  {out.name}")


def write_baseline_coverage_matrix(rows):
    subsets = [
        ("mono", "Mono"), ("bi", "Bi"), ("4d", "4D"),
        ("multislice", "Multi."), ("real", "Real"), ("brownian_pt", "Brownian"),
    ]
    methods = [m for m in METHOD_ORDER + ["sax_nerf", "x_gaussian", "dif_gaussian", "instamap", "fourdgs"]
               if any(r.get("method") == m for r in rows)]
    sample_counts = defaultdict(set)
    for r in rows:
        sample_counts[(r.get("method"), r.get("subset"), r.get("task", "T1"))].add(r.get("sample_id"))

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Implementation-level coverage matrix. Entries give the number of evaluated samples in the audited result log; zeros denote rows that are not claimed for that tier. Compatibility and fallback implementations are labelled in Table~\\ref{tab:methods}.}",
        "\\label{tab:baseline_coverage}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "\\textbf{Method} & " + " & ".join(f"\\textbf{{{label}}}" for _, label in subsets) + ' \\\\',
        "\\midrule",
    ]
    for method in methods:
        cells = []
        for subset, _ in subsets:
            task = "T7" if subset == "4d" and method in {"x2gaussian", "fourdgs"} else "T1"
            n = len(sample_counts.get((method, subset, task), set()))
            cells.append(str(n) if n else "--")
        lines.append(f"{METHOD_LABELS.get(method, method)} & " + " & ".join(cells) + ' \\\\')
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    out = TAB_DIR / "tab_baseline_coverage.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  {out.name}")

# ═══════════════════════════════════════════════════════════════
# TABLES: Auto-generate LaTeX for each subset
# ═══════════════════════════════════════════════════════════════
def _rank_color(rank, n_total):
    """Map rank (1=best) to color index 1-8 for LaTeX \\rc command."""
    if n_total <= 1:
        return 4
    # Map linearly: rank 1 → color 1, rank n → color 8
    idx = 1 + int((rank - 1) / max(1, n_total - 1) * 7)
    return min(8, max(1, idx))


def generate_latex_table(rows, subset, task="T1", caption="", label="",
                          colored=True, with_ci=True, n_bootstrap=1000):
    g = group_by(rows, subset, task)
    methods = [m for m in METHOD_ORDER + ["fourdgs"] if m in g]
    # higher_better: True if higher values are better for this metric
    metrics_keys = [
        ("vol_psnr", "PSNR$\\uparrow$", "{:.1f}", True),
        ("vol_ssim", "SSIM$\\uparrow$", "{:.3f}", True),
        ("recall", "Recall$\\uparrow$", "{:.1\\%}", True),
        ("rmsd", "RMSD$\\downarrow$", "{:.3f}", False),
    ]
    if task == "T7":
        metrics_keys += [
            ("trajectory_rmsd", "Traj.RMSD$\\downarrow$", "{:.2f}", False),
            ("atom_identity_persistence", "Persist.$\\uparrow$", "{:.1\\%}", True),
        ]

    # Precompute means + bootstrap 95% CI per method per metric
    method_means = {}
    method_ci = {}
    for m in methods:
        ms = g[m]
        method_means[m] = {}
        method_ci[m] = {}
        for key, _, _, _ in metrics_keys:
            vals = _metric_vals(ms, key)
            if with_ci:
                mu, lo, hi = bootstrap_ci(vals, n_resample=n_bootstrap)
            else:
                mu, _ = mean_std(vals)
                lo = hi = None
            method_means[m][key] = mu
            method_ci[m][key] = (lo, hi)

    # Compute ranks per metric
    method_ranks = {m: {} for m in methods}
    for key, _, _, higher_better in metrics_keys:
        valid = [(m, method_means[m][key]) for m in methods
                 if method_means[m][key] is not None
                 and not math.isinf(method_means[m][key])]
        valid.sort(key=lambda x: x[1], reverse=higher_better)
        for rank, (m, _) in enumerate(valid, 1):
            method_ranks[m][key] = rank
        n_valid = len(valid)
        for m in methods:
            if m not in dict(valid):
                method_ranks[m][key] = None
            method_ranks[m][key + "_n"] = n_valid

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\small")
    cols = "l" + "c" * (len(metrics_keys) + 1)
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")
    header = "\\textbf{Method} & \\textbf{n}"
    for _, h, _, _ in metrics_keys:
        header += f" & \\textbf{{{h}}}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for m in methods:
        ms = g[m]
        n = len(set(r.get("sample_id", str(i)) for i, r in enumerate(
            [rr for rr in rows if rr["subset"] == subset and rr["method"] == m
             and rr.get("task", "T1") == task]
        )))
        row = f"{METHOD_LABELS.get(m, m)} & {n}"
        for key, _, fmt, _ in metrics_keys:
            mu = method_means[m][key]
            rank = method_ranks[m].get(key)
            n_valid = method_ranks[m].get(key + "_n", 1)
            lo, hi = method_ci[m].get(key, (None, None))

            def _wrap(vstr: str) -> str:
                if colored and rank is not None:
                    c = _rank_color(rank, n_valid)
                    return f"\\rc{{{c}}}{{{vstr}}}"
                return vstr

            def _ci_sub(lo, hi, mu, is_pct: bool) -> str:
                # Tight 95% CI half-width as subscript. Skip if n=1 (lo==hi==mu)
                # or CI undefined.
                if lo is None or hi is None or mu is None:
                    return ""
                hw = (hi - lo) / 2.0
                if is_pct:
                    hw_str = f"{hw*100:.1f}" if hw >= 0.001 else "0.0"
                else:
                    # Use same precision as value format
                    if "{:.3f}" in fmt:
                        hw_str = f"{hw:.3f}"
                    elif "{:.2f}" in fmt:
                        hw_str = f"{hw:.2f}"
                    else:
                        hw_str = f"{hw:.1f}"
                return f"{{\\scriptsize\\,$\\pm${hw_str}}}"

            if mu is None or math.isinf(mu):
                row += " & --"
            elif "%" in fmt:
                val_str = f"{mu*100:.1f}\\%"
                cell = _wrap(val_str) + _ci_sub(lo, hi, mu, True)
                row += f" & {cell}"
            else:
                val_str = fmt.format(mu)
                cell = _wrap(val_str) + _ci_sub(lo, hi, mu, False)
                row += f" & {cell}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def write_all_tables(rows):
    # Count actual samples per subset
    from collections import defaultdict
    subset_n = defaultdict(set)
    for r in rows:
        subset_n[r["subset"]].add(r.get("sample_id", ""))
    ci_note = (
        "Subscripts are 95\\% bootstrap CI half-widths (1000 resamples); "
        "rank colors shade best (blue) to worst (red) per column. "
        "Recall/RMSD are aggregated over the subset of runs with validated "
        "atom detection (canonical samples plus \\texttt{atom\\_eval=v2}); "
        "PSNR/SSIM span all runs in the column labelled~$n$."
    )
    configs = [
        ("mono", "T1", f"T1 results on AET-Sim-Mono ({len(subset_n.get('mono',set()))} samples). {ci_note}", "tab:full_mono"),
        ("bi", "T1", f"T1 results on AET-Sim-Bi ({len(subset_n.get('bi',set()))} samples). {ci_note}", "tab:full_bi"),
        ("brownian_pt", "T1", f"T1 results on AET-Brownian ({len(subset_n.get('brownian_pt',set()))} samples). {ci_note}", "tab:full_brownian"),
        ("multislice", "T1", f"T1 results on AET-Multi ({len(subset_n.get('multislice',set()))} samples). {ci_note}", "tab:full_multi"),
        ("real", "T1", f"T1 results on AET-Real ({len(subset_n.get('real',set()))} samples). {ci_note}", "tab:full_real"),
        ("4d", "T7", f"T7 results on AET-Sim-4D ({len(subset_n.get('4d',set()))} trajectories). {ci_note}", "tab:full_4d"),
    ]
    for subset, task, caption, label in configs:
        tex = generate_latex_table(rows, subset, task, caption, label)
        out = TAB_DIR / f"tab_{subset}_{task.lower()}.tex"
        out.write_text(tex)
        print(f"  {out.name}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS,
        help="JSONL result file used for every generated figure/table",
    )
    args = parser.parse_args()
    RESULTS = args.results
    rows = load_results()
    sensitivity_rows = load_jsonl(SENSITIVITY_JSONL)
    print(f"Loaded {len(rows)} result rows")
    if sensitivity_rows:
        print(f"Loaded {len(sensitivity_rows)} extraction-sensitivity rows")
    else:
        print(f"No extraction-sensitivity rows found at {SENSITIVITY_JSONL}")

    print("\nGenerating figures...")
    fig_mono_bars(rows)
    fig_realism_ladder(rows)
    fig_size_scaling(rows)
    fig_tier_comparison(rows)
    fig_4d_paradox(rows)
    fig_pareto(rows)
    fig_threshold_sweep(sensitivity_rows)
    fig_extraction_sensitivity(sensitivity_rows)

    print("\nGenerating tables...")
    write_all_tables(rows)
    write_sensitivity_table(sensitivity_rows)
    write_baseline_coverage_matrix(rows)
    write_real_reference_metadata_table()
    print("\nWriting provenance...")
    write_provenance(rows)

    print(f"\nDone. Figures: {FIG_DIR}/, Tables: {TAB_DIR}/")


if __name__ == "__main__":
    main()
