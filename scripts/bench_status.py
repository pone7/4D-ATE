#!/usr/bin/env python3
"""AET-Bench progress snapshot.

Usage: python scripts/bench_status.py [--json]

Reads:
  - /proc/<pid> to check worker PIDs 484660/2273284/2273285/2273286/2273287
  - /tmp/gpu{0,1,2,3}_full.txt for per-worker progress
  - data/aet_bench/results.jsonl for completion counts
  - data/aet_bench/{subset}/ for sample lists

Output: table of remaining job counts per (subset, method) + ETA.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path("/data/aet-bench")
RF = ROOT / "data/aet_bench/results.jsonl"
BENCH = ROOT / "data/aet_bench"
WORKER_PIDS = [484660, 2273284, 2273285, 2273286, 2273287]
WORKER_LOGS = {
    484660: "/tmp/gpu0_full.txt",
    2273284: "/tmp/gpu0_full.txt",
    2273285: "/tmp/gpu1_full.txt",
    2273286: "/tmp/gpu2_full.txt",
    2273287: "/tmp/gpu3_full.txt",
}
METHODS_CLASSICAL = ["wbp", "astra_sirt", "astra_cgls", "tomopy_mlem"]
METHODS_NEURAL = ["neat", "implicit_et", "isonet", "sax_nerf", "instamap", "dif_gaussian"]
SUBSETS = ["mono", "bi", "multislice", "real", "brownian_pt"]

PROG_RE = re.compile(r"\[GPU(\d+)\]\[(\d+)/(\d+)\]\s+(\S+)\s+(\S+)\s+.*\((\d+)s\)")


def proc_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def proc_cpu_minutes(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
        utime = int(parts[13])
        stime = int(parts[14])
        hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        return (utime + stime) / hz / 60.0
    except Exception:
        return None


def last_log_progress(log_path: str) -> dict | None:
    """Parse last '[GPUN][i/total]' line and return progress dict."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            chunk = min(size, 20000)
            f.seek(size - chunk)
            tail = f.read().decode("utf-8", errors="ignore")
    except FileNotFoundError:
        return None
    matches = PROG_RE.findall(tail)
    if not matches:
        return None
    gpu, i, total, method, sid, secs = matches[-1]
    mtime = os.path.getmtime(log_path)
    return {
        "gpu": int(gpu),
        "progress": int(i),
        "total": int(total),
        "last_method": method,
        "last_sample": sid,
        "last_runtime_sec": int(secs),
        "log_mtime": mtime,
        "log_age_sec": time.time() - mtime,
    }


def load_done() -> tuple[set[tuple[str, str, str]], int, int]:
    """Return (done_set, n_main_rows, n_extras_rows)."""
    done: set[tuple[str, str, str]] = set()
    n_main = 0
    n_extras = 0
    for path, is_extras in [(RF, False),
                             (BENCH / "results_cpu_extras.jsonl", True),
                             (BENCH / "results_gpu_extras.jsonl", True)]:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in r:
                    continue
                done.add((r["subset"], r["sample_id"], r["method"]))
                if is_extras:
                    n_extras += 1
                else:
                    n_main += 1
    return done, n_main, n_extras


def remaining_table(done: set) -> tuple[dict, int]:
    result = {}
    total = 0
    for sub in SUBSETS:
        d = BENCH / sub
        if not d.exists():
            continue
        samples = sorted([p.name for p in d.iterdir() if p.is_dir()])
        row = {"n_samples": len(samples)}
        for m in METHODS_CLASSICAL + METHODS_NEURAL:
            missing = sum(1 for s in samples if (sub, s, m) not in done)
            row[m] = missing
            total += missing
        result[sub] = row
    return result, total


def rate_last_minutes(minutes: int = 30) -> tuple[int, float]:
    """Count rows appended in the last N minutes using file mtime heuristics.

    Since jsonl has no per-row timestamp, we estimate by scanning output/
    directory modifications.
    """
    out = ROOT / "output"
    cutoff = time.time() - minutes * 60
    count = 0
    if out.exists():
        for d in out.iterdir():
            if not d.is_dir():
                continue
            for sub in d.iterdir():
                if sub.is_dir() and sub.stat().st_mtime > cutoff:
                    count += 1
    return count, minutes * 60.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    done, n_main, n_extras = load_done()
    table, remaining = remaining_table(done)
    rate_n, rate_win = rate_last_minutes(30)
    per_sec = rate_n / rate_win if rate_win > 0 else 0.0
    per_hour = per_sec * 3600
    eta_hours = remaining / per_hour if per_hour > 0 else float("inf")

    workers = []
    for pid in WORKER_PIDS:
        alive = proc_alive(pid)
        prog = last_log_progress(WORKER_LOGS[pid]) if alive else None
        workers.append({
            "pid": pid,
            "alive": alive,
            "cpu_minutes": proc_cpu_minutes(pid) if alive else None,
            "log": WORKER_LOGS[pid],
            **(prog or {}),
        })

    snapshot = {
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "total_rows": len(done),
        "main_rows": n_main,
        "extras_rows": n_extras,
        "remaining_jobs": remaining,
        "rate_jobs_per_hour_30m": per_hour,
        "eta_hours": eta_hours,
        "remaining_by_subset": table,
        "workers": workers,
    }

    if args.json:
        print(json.dumps(snapshot, default=str))
        return

    # Text report
    print(f"=== AET-Bench status  {snapshot['timestamp_iso']} ===")
    print(f"Total completed rows: {snapshot['total_rows']}  "
          f"(main={n_main}, extras={n_extras})")
    print(f"Remaining jobs: {remaining}")
    print(f"Rate (last 30m): {per_hour:.1f} jobs/h")
    print(f"ETA: {eta_hours:.1f} h ({eta_hours/24:.1f} d)")
    print()
    print("Remaining by (subset, method):")
    hdr_methods = METHODS_CLASSICAL + METHODS_NEURAL
    print(f"  {'subset':12s} {'n':>4s}  " + " ".join(f"{m:>12s}" for m in hdr_methods))
    for sub, row in table.items():
        line = f"  {sub:12s} {row['n_samples']:>4d}  "
        line += " ".join(f"{row[m]:>12d}" for m in hdr_methods)
        print(line)
    print()
    print("Workers:")
    for w in workers:
        status = "ALIVE" if w["alive"] else "DEAD "
        if "progress" in w:
            frac = w["progress"] / w["total"] if w["total"] else 0
            age = w.get("log_age_sec", float("nan"))
            cpu_m = w.get("cpu_minutes") or 0
            print(
                f"  PID {w['pid']:>8d} [{status}] cpu={cpu_m:7.1f}m  "
                f"slice-progress={w['progress']}/{w['total']} ({frac*100:5.1f}%)  "
                f"last={w.get('last_method','?'):12s} {w.get('last_sample','?'):22s}  "
                f"log-age={age:5.0f}s"
            )
        else:
            print(f"  PID {w['pid']:>8d} [{status}]  (no recent log parse)")


if __name__ == "__main__":
    main()
