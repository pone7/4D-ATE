#!/usr/bin/env python
"""
Generate AET-Bench summary tables from results.jsonl.

Reads each row produced by run_benchmark.py and emits, for every (subset,
backend), a markdown table aggregating metrics per method (mean ± std
across samples). Writes a single combined report to stdout or to --out.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

METRICS = [
    ("vol_psnr", "PSNR ↑", "{:.2f}"),
    ("vol_ssim", "SSIM ↑", "{:.3f}"),
    ("fsc_0.143_angstrom", "FSC@0.143 Å ↓", "{:.2f}"),
    ("rmsd", "RMSD Å ↓", "{:.3f}"),
    ("recall", "Recall ↑", "{:.1%}"),
    ("precision", "Precision ↑", "{:.1%}"),
    ("f1", "F1 ↑", "{:.3f}"),
    ("trajectory_rmsd", "Traj RMSD Å ↓", "{:.3f}"),
    ("atom_identity_persistence", "Atom ID persist ↑", "{:.1%}"),
]


def mean_std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None, None
    if len(xs) == 1:
        return xs[0], 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def fmt(val, std, fmtstr):
    if val is None:
        return "—"
    if val == float("inf"):
        return "∞"
    if std is None or std == 0:
        return fmtstr.format(val)
    if "%" in fmtstr:
        return f"{fmtstr.format(val)} ± {fmtstr.format(std)}"
    return f"{fmtstr.format(val)} ± {fmtstr.format(std)}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="/data/aet-bench/data/aet_bench/results.jsonl")
    p.add_argument("--out", default=None, help="Output markdown file (default: stdout)")
    args = p.parse_args()

    rows = []
    with open(args.results) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # group: (subset, backend, method) -> {metric: [vals]}
    groups = defaultdict(lambda: defaultdict(list))
    n_samples = defaultdict(set)
    for r in rows:
        key = (r["subset"], r["backend"], r["method"], r.get("task", "T1"))
        for k, v in r.get("metrics", {}).items():
            if k == "per_phase":
                continue  # already aggregated into top-level keys
            groups[key][k].append(v)
        groups[key]["runtime_sec"].append(r.get("runtime_sec", 0.0))
        n_samples[key].add(r["sample_id"])

    out_lines = ["# AET-Bench results", ""]
    # group by (subset, backend, task), list methods as rows
    sub_back = defaultdict(list)
    for (subset, backend, method, task) in groups:
        sub_back[(subset, backend, task)].append(method)

    for (subset, backend, task), methods in sorted(sub_back.items()):
        out_lines.append(f"## subset={subset}  backend={backend}  task={task}")
        out_lines.append("")
        header = ["Method", "n"] + [m[1] for m in METRICS] + ["Runtime (s)"]
        out_lines.append("| " + " | ".join(header) + " |")
        out_lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for method in sorted(methods):
            key = (subset, backend, method, task)
            row = [method, str(len(n_samples[key]))]
            for mkey, _, fmtstr in METRICS:
                m, s = mean_std(groups[key].get(mkey, []))
                row.append(fmt(m, s, fmtstr))
            rt_m, rt_s = mean_std(groups[key].get("runtime_sec", []))
            row.append(fmt(rt_m, rt_s, "{:.1f}"))
            out_lines.append("| " + " | ".join(row) + " |")
        out_lines.append("")

    text = "\n".join(out_lines)
    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
