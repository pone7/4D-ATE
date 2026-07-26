#!/usr/bin/env python
"""Validate and audit AET-Bench JSONL result files.

This intentionally checks paper-facing hygiene rather than scientific merit:
schema consistency, duplicate runs, non-finite metrics, method/sample coverage,
and presence of known buggy sidecar files in the same directory.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CORE_FIELDS = ("subset", "sample_id", "backend", "method", "task", "metrics")
ALLOWED_NONFINITE = {
    "vol_psnr",
    "fsc_0.143",
    "fsc_0.5",
    "fsc_0.143_angstrom",
    "fsc_0.5_angstrom",
    "resolution_0.143",
    "resolution_0.5",
    "fsc_curve",
    "rmsd",
    "mean_dist",
    "median_dist",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_metrics(prefix: str, obj: Any):
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else key
            yield from walk_metrics(name, value)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            yield from walk_metrics(f"{prefix}[{idx}]", value)
    else:
        yield prefix, obj


def metric_base_name(name: str) -> str:
    # Metric names themselves contain dots (fsc_0.143_angstrom), while nested
    # paths use "." separators (per_phase[0].rmsd). Only strip known nesting.
    if "]." in name:
        name = name.rsplit("].", 1)[-1]
    return name.split("[", 1)[0]


def load_rows(path: Path):
    rows = []
    errors = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {lineno}: invalid JSON: {exc}")
                continue
            rows.append((lineno, row))
    return rows, errors


def validate(path: Path, strict: bool = False) -> int:
    rows, errors = load_rows(path)
    warnings = []
    by_subset_task = collections.Counter()
    samples = collections.defaultdict(set)
    methods = collections.defaultdict(set)
    duplicates = collections.defaultdict(list)
    nonfinite = collections.Counter()
    missing_metrics = 0
    atom_eval_versions = collections.Counter()

    for lineno, row in rows:
        if not isinstance(row, dict):
            errors.append(f"line {lineno}: row is not an object")
            continue
        if "error" in row:
            warnings.append(f"line {lineno}: error row present: {row.get('error')}")
            continue
        for field in CORE_FIELDS:
            if field not in row:
                errors.append(f"line {lineno}: missing core field {field!r}")

        subset = row.get("subset")
        task = row.get("task", "T1")
        method = row.get("method")
        sample = row.get("sample_id")
        backend = row.get("backend")
        key = (subset, task)
        run_key = (subset, task, method, sample, backend)
        by_subset_task[key] += 1
        samples[key].add(sample)
        methods[key].add(method)
        duplicates[run_key].append(lineno)

        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            missing_metrics += 1
            errors.append(f"line {lineno}: metrics is not an object")
            continue
        if "atom_eval" in metrics:
            atom_eval_versions[str(metrics.get("atom_eval"))] += 1
        for name, value in walk_metrics("", metrics):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                base = metric_base_name(name)
                nonfinite[base] += 1
                if base not in ALLOWED_NONFINITE:
                    msg = (
                        f"line {lineno}: non-finite metric {name}={value!r} "
                        f"is not allowlisted"
                    )
                    if strict:
                        errors.append(msg)
                    else:
                        warnings.append(msg)

    duplicate_runs = {k: v for k, v in duplicates.items() if len(v) > 1}
    if duplicate_runs:
        msg = f"{len(duplicate_runs)} duplicate subset/task/method/sample/backend runs"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)

    buggy_files = sorted(path.parent.glob("*.v1_buggy*.jsonl"))
    if buggy_files:
        warnings.append(
            "buggy sidecar result files present; do not merge into paper "
            f"results: {', '.join(p.name for p in buggy_files)}"
        )

    print(f"Result file: {path}")
    print(f"SHA256: {sha256_file(path)}")
    print(f"Rows parsed: {len(rows)}")
    print("\nCoverage by subset/task:")
    for key in sorted(by_subset_task):
        print(
            f"  {key[0]}/{key[1]}: rows={by_subset_task[key]} "
            f"samples={len(samples[key])} methods={len(methods[key])}"
        )
    if nonfinite:
        print("\nNon-finite metrics:")
        for name, count in sorted(nonfinite.items()):
            print(f"  {name}: {count}")
    if atom_eval_versions:
        print("\nAtom eval versions:")
        for version, count in sorted(atom_eval_versions.items()):
            print(f"  {version}: {count}")
    if missing_metrics:
        print(f"\nRows with missing metrics: {missing_metrics}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings[:50]:
            print(f"  WARN: {warning}")
        if len(warnings) > 50:
            print(f"  ... {len(warnings) - 50} more warnings")
    if errors:
        print("\nErrors:")
        for error in errors[:50]:
            print(f"  ERROR: {error}")
        if len(errors) > 50:
            print(f"  ... {len(errors) - 50} more errors")
        return 1

    print("\nValidation passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat duplicate run rows as errors instead of warnings",
    )
    args = parser.parse_args()
    return validate(args.results, strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
