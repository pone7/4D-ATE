#!/usr/bin/env python3
"""Generate the dose/noise pilot tier AET-Sim-Mono-Dose.

For 20 Cu monometallic samples drawn from the mono tier, re-emit the analytical
HAADF projections with Poisson shot noise at three electron doses
({100, 1000, 10000} e/Å²). Ground-truth volumes and atom coordinates are
copied verbatim so that the reconstruction problem differs from the clean
mono samples only in the projection noise level.

Output structure mirrors other tiers:
    data/aet_bench/dose/<sample>/
        analytical/data.pkl    (noisy projections + clean image + noise meta)
        atoms_gt.npy           (copy of mono)
        metadata.json          (copy of mono + dose fields)

Sample naming: Cu-<atoms>-<idx>-d<dose_code> where dose_code is d2, d3, d4.
Usage:
    python scripts/generate_dose_tier.py
    python scripts/generate_dose_tier.py --doses 100,1000 --seed 42
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

ROOT = Path("/data/aet-bench")
MONO = ROOT / "data/aet_bench/mono"
DOSE = ROOT / "data/aet_bench/dose"

DEFAULT_DOSES = [100, 1000, 10000]
DOSE_CODE = {100: "d2", 1000: "d3", 10000: "d4"}


def _pick_samples(n: int, seed: int = 0) -> list[str]:
    cu_samples = sorted(
        p.name for p in MONO.iterdir() if p.is_dir() and p.name.startswith("Cu-")
    )
    if not cu_samples:
        raise RuntimeError("no Cu samples found in mono tier")
    rng = np.random.default_rng(seed)
    n = min(n, len(cu_samples))
    return sorted(rng.choice(cu_samples, size=n, replace=False).tolist())


def _apply_poisson(projs: np.ndarray, dose: float, cell_size_ang: float,
                    detector_size: int, rng: np.random.Generator) -> np.ndarray:
    """Poisson-noise a stack of HAADF projections at given electron dose (e/Å²).

    Model: total expected electron count per projection =
        dose [e/Å²] * projected_area [Å²] = dose * cell_size²
    We rescale each projection to that electron count, sample Poisson,
    and rescale back to original dynamic range so downstream methods see
    the same intensity units as clean data (only the noise statistics change).
    """
    out = np.empty_like(projs)
    total_area = cell_size_ang ** 2
    target_count = dose * total_area
    for t in range(projs.shape[0]):
        p = projs[t].astype(np.float64)
        s = p.sum()
        if s <= 0:
            out[t] = p
            continue
        expected = p * (target_count / s)
        noisy = rng.poisson(expected).astype(np.float64)
        # Rescale back so the peak matches the clean peak (reconstruction-friendly)
        out[t] = noisy * (s / max(noisy.sum(), 1.0))
    return out.astype(np.float32)


def _process_one(src_dir: Path, dst_dir: Path, dose: float, seed: int) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "analytical").mkdir(exist_ok=True)

    # Copy ground truth
    shutil.copy2(src_dir / "atoms_gt.npy", dst_dir / "atoms_gt.npy")

    # Load + modify pickle
    with open(src_dir / "analytical/data.pkl", "rb") as f:
        data = pickle.load(f)
    meta_src = json.load(open(src_dir / "metadata.json"))
    cell_size = float(meta_src["cell_size_angstrom"])
    det = int(meta_src.get("detector_size", 128))
    rng = np.random.default_rng(seed)

    for split in ("train", "val"):
        if split in data and "projections" in data[split]:
            data[split] = {**data[split]}
            data[split]["projections"] = _apply_poisson(
                data[split]["projections"], dose, cell_size, det, rng
            )

    with open(dst_dir / "analytical/data.pkl", "wb") as f:
        pickle.dump(data, f)

    meta = copy.deepcopy(meta_src)
    meta["sample_id"] = dst_dir.name
    meta["parent_sample"] = src_dir.name
    meta["dose_electrons_per_ang2"] = float(dose)
    meta["noise_model"] = "poisson_shot_noise"
    meta["dose_seed"] = int(seed)
    meta["backends"] = ["analytical"]
    with open(dst_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--doses", default=",".join(str(d) for d in DEFAULT_DOSES))
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    doses = [int(x) for x in args.doses.split(",")]
    samples = _pick_samples(args.n, seed=args.seed)
    print(f"Selected {len(samples)} mono samples:")
    for s in samples:
        print(f"  {s}")

    manifest = []
    for dose in doses:
        code = DOSE_CODE.get(dose, f"d{int(np.log10(dose))}")
        for i, sid in enumerate(samples):
            # Per-(sample, dose) seed derived from base seed
            seed = args.seed + dose * 7919 + i
            new_id = f"{sid}-{code}"
            dst = DOSE / new_id
            print(f"  generating {new_id} (dose={dose} e/Å²)", flush=True)
            _process_one(MONO / sid, dst, dose, seed)
            manifest.append({
                "sample_id": new_id, "parent": sid,
                "dose_electrons_per_ang2": dose, "seed": seed,
            })

    with open(DOSE / "manifest.json", "w") as f:
        json.dump({"n_parent_samples": len(samples), "doses": doses,
                    "samples": manifest}, f, indent=2)
    print(f"\nDone. {len(manifest)} dose samples written to {DOSE}/")


if __name__ == "__main__":
    main()
