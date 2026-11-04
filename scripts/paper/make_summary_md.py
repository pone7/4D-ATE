#!/usr/bin/env python3
"""Write the dataset's SUMMARY.md leaderboards from aggregate_results.py output.

The numbers follow the paper's aggregation rules (see aggregate_results.py):
canonical tier sets, one result per sample, v2-audited atom metrics, and
95% bootstrap confidence intervals.

Usage:
    python scripts/paper/make_summary_md.py --agg paper_results.json --out SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TIERS = [
    ("mono", "Mono", "T1", "50 canonical base samples"),
    ("bi", "Bi", "T1", "30 canonical base samples"),
    ("4d", "4D", "T7", "10 trajectories"),
    ("multislice", "Multislice (pilot)", "T1", "3 samples; listed per sample below, no ranking"),
    ("real", "Real (reference agreement)", "T1", "26 samples; scored against reference atom models"),
    ("brownian_pt", "Brownian-Pt", "T1", "66 dynamic frames"),
]
LABELS = {
    "tomopy_mlem": "TomoPy-MLEM", "fourdgs": "4D-GS (X²-G compat.)", "isonet": "IsoNet (GP)",
    "x2gaussian": "X²-Gaussian", "r2gaussian": "R²-Gaussian", "sirt": "SIRT", "implicit_et": "Implicit-ET",
    "neat": "NeAT", "fista_tv": "FISTA-TV", "astra_cgls": "ASTRA-CGLS", "astra_sirt": "ASTRA-SIRT",
    "astra_sart": "ASTRA-SART", "astra_ossart": "ASTRA-OS-SART", "dart": "DART", "genfire": "GENFIRE",
    "fbp": "FBP", "wbp": "WBP", "tomopy_gridrec": "TomoPy-Gridrec", "dif_gaussian": "DIF-Gaussian",
    "instamap": "InstaMap", "sax_nerf": "SAX-NeRF",
}


def fmt(x, d: int = 3, pct: bool = False) -> str:
    if x is None:
        return "—"
    if pct:
        return f"{100 * x[0]:.1f}% ± {100 * x[1]:.1f}"
    return f"{x[0]:.{d}f} ± {x[1]:.{d}f}"


def n_str(entry: dict) -> str:
    n = entry["n"]
    rec = entry.get("recall")
    n_atom = rec[2] if rec else 0
    return str(n) if n_atom in (n, 0) else f"{n} / {n_atom}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agg", required=True, help="JSON written by aggregate_results.py")
    ap.add_argument("--out", default="SUMMARY.md")
    args = ap.parse_args()
    D = json.loads(Path(args.agg).read_text())

    lines = [
        "# AET-Bench v1.0.0 leaderboards",
        "",
        "These tables match the paper (Table 4 and Appendix E). Each cell is the mean over samples",
        "± the half-width of a 95% percentile-bootstrap interval (1,000 resamples). Rules:",
        "",
        "* Mono and Bi use their canonical base samples (sample ids with an `-sNN` augmentation",
        "  suffix are excluded).",
        "* One result per (method, sample): the most recently logged run in `results.jsonl`.",
        "* Atom metrics come only from rows whose atom evaluation passed the v2 audit; `—` means",
        "  no audited atom metrics exist for that method. `n = a / b` means a samples have voxel",
        "  metrics and b have audited atom metrics.",
        "",
        "Regenerate with `scripts/paper/aggregate_results.py` and `scripts/paper/make_summary_md.py`",
        "from https://github.com/pone7/4D-ATE.",
        "",
    ]
    for key, name, task, note in TIERS:
        if key == "multislice":
            continue
        rows = sorted(D[key].items(), key=lambda kv: -(kv[1]["vol_ssim"] or [0])[0])
        lines += [f"## {name} — task {task} ({note})", ""]
        head = "| Method | n | SSIM ↑ | Atom recall ↑ | Precision ↑ | RMSD (Å) ↓ |"
        sep = "|---|---|---|---|---|---|"
        if key == "4d":
            head += " Traj. RMSD (Å) ↓ | ID persistence ↑ |"
            sep += "---|---|"
        lines += [head, sep]
        for m, e in rows:
            row = (
                f"| {LABELS.get(m, m)} | {n_str(e)} | {fmt(e['vol_ssim'])} | {fmt(e['recall'], pct=True)} "
                f"| {fmt(e['precision'], pct=True)} | {fmt(e['rmsd'])} |"
            )
            if key == "4d":
                row += f" {fmt(e['trajectory_rmsd'], 2)} | {fmt(e['atom_identity_persistence'], pct=True)} |"
            lines.append(row)
        lines.append("")

    lines += [
        "## Multislice (pilot) — per sample",
        "",
        "With n = 3 the tier supports a direction-of-effect claim only, so no aggregate ranking is given.",
        "",
        "| Sample | Method | SSIM ↑ | Atom recall ↑ | RMSD (Å) ↓ |",
        "|---|---|---|---|---|",
    ]
    for sample in sorted(D["multislice_per_sample"]):
        for m in ["fourdgs", "x2gaussian", "r2gaussian"]:
            ssim, rec, rmsd = D["multislice_per_sample"][sample][m]
            lines.append(f"| {sample} | {LABELS[m]} | {ssim:.3f} | {100 * rec:.1f}% | {rmsd:.3f} |")
    lines.append("")
    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
