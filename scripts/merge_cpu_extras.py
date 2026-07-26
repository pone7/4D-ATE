#!/usr/bin/env python3
"""Merge results_cpu_extras.jsonl into results.jsonl.

Deduplicates on (subset, sample_id, method) — keeps the first successful
occurrence and drops error rows. Results already present in the main file
are preserved (not overwritten).

Usage: python scripts/merge_cpu_extras.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path("/data/aet-bench")
RF_MAIN = ROOT / "data/aet_bench/results.jsonl"
RF_EXTRAS = ROOT / "data/aet_bench/results_cpu_extras.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not RF_EXTRAS.exists():
        print(f"No extras file at {RF_EXTRAS}")
        return
    # Load main file's keys to skip duplicates
    main_keys: set = set()
    with open(RF_MAIN) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            main_keys.add((r["subset"], r["sample_id"], r["method"]))
    # Collect new rows from extras (dedup within extras too)
    new_rows = []
    seen_extras: set = set()
    err_count = 0
    dup_count = 0
    with open(RF_EXTRAS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r:
                err_count += 1
                continue
            key = (r["subset"], r["sample_id"], r["method"])
            if key in main_keys or key in seen_extras:
                dup_count += 1
                continue
            seen_extras.add(key)
            new_rows.append(line)

    print(f"Extras file: {RF_EXTRAS}")
    print(f"  error rows:      {err_count}")
    print(f"  already in main: {dup_count}")
    print(f"  new to append:   {len(new_rows)}")
    from collections import Counter
    by_method = Counter()
    by_subset = Counter()
    for line in new_rows:
        r = json.loads(line)
        by_method[r["method"]] += 1
        by_subset[r["subset"]] += 1
    print("  by method:", dict(by_method))
    print("  by subset:", dict(by_subset))

    if args.dry_run:
        return
    if not new_rows:
        print("Nothing to merge.")
        return

    # Backup main before append (safety)
    bk = RF_MAIN.with_suffix(RF_MAIN.suffix + f".bak.{int(time.time())}")
    shutil.copy2(RF_MAIN, bk)
    print(f"Backup of main -> {bk}")

    with open(RF_MAIN, "a") as f:
        for line in new_rows:
            f.write(line + "\n")
    print(f"Appended {len(new_rows)} rows to {RF_MAIN}")

    # Archive the extras file so we don't re-merge
    archived = RF_EXTRAS.with_suffix(f".merged.{int(time.time())}.jsonl")
    RF_EXTRAS.rename(archived)
    print(f"Archived extras -> {archived}")


if __name__ == "__main__":
    main()
