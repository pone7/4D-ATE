#!/usr/bin/env python3
"""Produce the ranking-differential appendix table.

Question: does AET-Bench re-rank methods relative to prior narrow-tier
evaluations? For each (subset, metric) pair we compare:
  - local rank: rank of method inside that single subset
  - overall rank: rank of the same method when averaged across all
    simulation tiers (mono + bi + multislice + 4d, which is the union
    of "clean" conditions previously treated independently).

Output: paper/tables/tab_ranking_differential.tex — a compact matrix that
highlights how rank changes between the classic narrow-tier evaluation
style and the aggregated AET-Bench view. Large absolute rank deltas
(|Δ| ≥ 4) flag methods that have been over- or underrated by prior
single-tier protocols.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/aet-bench")
RF = ROOT / "data/aet_bench/results.jsonl"
TAB = ROOT / "paper/tables"
TAB.mkdir(parents=True, exist_ok=True)

SUBSETS = ["mono", "bi", "multislice"]
LABELS = {
    "wbp": "WBP", "astra_sirt": "ASTRA-SIRT", "astra_sart": "ASTRA-SART",
    "astra_cgls": "ASTRA-CGLS", "astra_ossart": "ASTRA-OSSART",
    "tomopy_gridrec": "gridrec", "tomopy_mlem": "MLEM",
    "dart": "DART", "fista_tv": "FISTA-TV", "genfire": "GENFIRE",
    "sirt": "SIRT", "isonet": "IsoNet(GP)",
    "neat": "NeAT", "implicit_et": "Impl-ET",
    "r2gaussian": "R²-G", "x2gaussian": "X²-G",
}
METRIC = "recall"  # atom-level recall is the headline metric


def _load():
    rows = []
    for line in RF.open():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(r)
    return rows


def _rank(values: dict[str, float]) -> dict[str, int]:
    """Assign ranks (1 = best) for higher-is-better metric."""
    items = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    return {m: r + 1 for r, (m, _) in enumerate(items)}


def main():
    rows = _load()
    # Per-subset mean recall per method (T1 only; only methods with ≥3 samples)
    per_sub = {s: defaultdict(list) for s in SUBSETS}
    for r in rows:
        if r.get("task") != "T1":
            continue
        sub = r["subset"]
        if sub not in per_sub:
            continue
        v = r["metrics"].get(METRIC)
        if v is None:
            continue
        per_sub[sub][r["method"]].append(v)

    sub_means = {s: {m: float(np.mean(v)) for m, v in d.items() if len(v) >= 3}
                 for s, d in per_sub.items()}

    # Methods that appear in ≥2 subsets
    methods = sorted({m for sm in sub_means.values() for m in sm})
    methods = [m for m in methods if sum(1 for sm in sub_means.values() if m in sm) >= 2]

    # Overall (AET-Bench aggregate): equal-weight mean across available subsets
    overall = {}
    for m in methods:
        vals = [sub_means[s][m] for s in SUBSETS if m in sub_means[s]]
        if vals:
            overall[m] = float(np.mean(vals))

    # Ranks per (metric, subset) and overall
    sub_ranks = {s: _rank({m: sub_means[s][m] for m in methods if m in sub_means[s]})
                 for s in SUBSETS}
    overall_rank = _rank(overall)

    # Emit table — rows per method, cols: each subset rank + overall rank + largest delta
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{How AET-Bench re-ranks methods versus prior narrow-tier protocols. "
        "Ranks by atom-level \\emph{recall} on three simulated tiers (mono/bi/multislice), "
        "then the aggregate AET-Bench rank (mean of per-tier means). "
        "$\\Delta$ column shows $\\max_s |\\textit{rank}_s - \\textit{rank}_\\text{AET-Bench}|$: "
        "values $\\geq$\\,4 flag methods whose prior tier-specific evaluation "
        "disagrees with AET-Bench's aggregated view.}",
        "\\label{tab:ranking_diff}",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & rank$_\\text{mono}$ & rank$_\\text{bi}$ & rank$_\\text{multi}$ & rank$_\\text{AET-Bench}$ & $\\Delta$ \\\\",
        "\\midrule",
    ]
    # Sort methods by overall rank
    methods_sorted = sorted(methods, key=lambda m: overall_rank.get(m, 99))
    for m in methods_sorted:
        cells = []
        per_sub_rank = []
        for s in SUBSETS:
            r = sub_ranks[s].get(m)
            cells.append(str(r) if r else "--")
            if r is not None:
                per_sub_rank.append(r)
        or_ = overall_rank.get(m)
        cells.append(str(or_) if or_ else "--")
        if or_ and per_sub_rank:
            delta = max(abs(rs - or_) for rs in per_sub_rank)
            flag = "\\textbf{" + str(delta) + "}" if delta >= 4 else str(delta)
            cells.append(flag)
        else:
            cells.append("--")
        lines.append(f"{LABELS.get(m, m)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "tab_ranking_differential.tex").write_text("\n".join(lines))
    print(f"→ {TAB / 'tab_ranking_differential.tex'}")

    # CLI dump for sanity
    print("Overall rank (AET-Bench):")
    for m, r in sorted(overall_rank.items(), key=lambda kv: kv[1]):
        print(f"  {r:2d}. {LABELS.get(m, m):15s}  recall_mean={overall[m]*100:.1f}%")


if __name__ == "__main__":
    main()
