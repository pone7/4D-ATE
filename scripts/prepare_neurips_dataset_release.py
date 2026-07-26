#!/usr/bin/env python3
"""Prepare AET-Bench dataset release metadata for NeurIPS D&B submission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RELEASE_SUBSETS = ("mono", "bi", "sim4d", "multislice", "real", "brownian_pt")

SAMPLE_COLUMNS = (
    "sample_id",
    "subset",
    "element",
    "n_atoms",
    "n_angles",
    "has_atom_gt",
    "metadata_path",
    "data_pickle_path",
    "atoms_gt_path",
    "atoms_gt_4d_path",
    "tier_archive",
    "archive_member_prefix",
    "source",
)

RESULT_COLUMNS = (
    "result_row_id",
    "sample_id",
    "subset",
    "method",
    "task",
    "vol_psnr",
    "vol_ssim",
    "fsc_0.143_angstrom",
    "recall",
    "precision",
    "f1",
    "rmsd",
    "trajectory_rmsd",
    "atom_identity_persistence",
    "runtime_sec",
)


def rel(path: Path) -> str:
    return path.as_posix()


def clean_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256_cached(path: Path, cache: dict[str, str]) -> str:
    key = str(path)
    if key not in cache:
        cache[key] = file_sha256(path)
    return cache[key]


def content_url(base_url: str, filename: str) -> str:
    if not base_url:
        return filename
    return f"{base_url.rstrip('/')}/{filename}"


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def sample_rows(data_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset in RELEASE_SUBSETS:
        subset_dir = data_root / subset
        if not subset_dir.exists():
            continue
        for sample_dir in sorted(p for p in subset_dir.iterdir() if p.is_dir()):
            metadata_path = sample_dir / "metadata.json"
            data_pickle_path = sample_dir / "analytical" / "data.pkl"
            atoms_gt_path = sample_dir / "atoms_gt.npy"
            atoms_gt_4d_path = sample_dir / "atoms_gt_4d.npy"
            meta = load_metadata(metadata_path)
            elements = meta.get("elements")
            element = meta.get("element") or ("+".join(elements) if isinstance(elements, list) else "")
            row = {
                "sample_id": meta.get("sample_id") or sample_dir.name,
                "subset": subset,
                "element": element,
                "n_atoms": meta.get("n_atoms", ""),
                "n_angles": meta.get("n_angles", ""),
                "has_atom_gt": bool(meta.get("has_atom_gt", atoms_gt_path.exists())),
                "metadata_path": rel(metadata_path) if metadata_path.exists() else "",
                "data_pickle_path": rel(data_pickle_path) if data_pickle_path.exists() else "",
                "atoms_gt_path": rel(atoms_gt_path) if atoms_gt_path.exists() else "",
                "atoms_gt_4d_path": rel(atoms_gt_4d_path) if atoms_gt_4d_path.exists() else "",
                "tier_archive": f"aet-bench-{subset}.tar",
                "archive_member_prefix": rel(sample_dir),
                "source": meta.get("source", ""),
            }
            rows.append(row)
    return rows


def result_rows(results_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with results_path.open() as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            row = {"result_row_id": i}
            for column in RESULT_COLUMNS[1:]:
                row[column] = clean_scalar(obj.get(column, ""))
            rows.append(row)
    return rows


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def release_files(output: Path) -> list[str]:
    names = [
        "aet-bench-samples.csv",
        "aet-bench-results.csv",
        "aet-bench-release-manifest.json",
        "results.jsonl",
        "README.md",
        "SUMMARY.md",
        "croissant.json",
    ]
    names.extend(f"aet-bench-{subset}.tar" for subset in RELEASE_SUBSETS)
    return names


def describe_file(output: Path, filename: str, base_url: str, hash_cache: dict[str, str]) -> dict[str, Any]:
    path = output / filename
    entry: dict[str, Any] = {
        "filename": filename,
        "contentUrl": content_url(base_url, filename),
        "exists": path.exists(),
    }
    if path.exists():
        entry["contentSize"] = f"{path.stat().st_size} B"
        entry["sha256"] = file_sha256_cached(path, hash_cache)
    else:
        entry["sha256"] = "pending"
    return entry


def write_release_manifest(
    path: Path,
    output: Path,
    base_url: str,
    samples: list[dict[str, Any]],
    results_count: int,
    hash_cache: dict[str, str],
) -> None:
    subset_counts = {subset: 0 for subset in RELEASE_SUBSETS}
    for row in samples:
        subset_counts[row["subset"]] += 1
    manifest = {
        "name": "AET-Bench",
        "version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(samples),
        "result_rows": results_count,
        "subsets": subset_counts,
        "files": [describe_file(output, name, base_url, hash_cache) for name in release_files(output)],
    }
    with path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def write_sha256sums(output: Path, hash_cache: dict[str, str]) -> None:
    manifest_path = output / "aet-bench-release-manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)

    by_name = {
        entry["filename"]: entry["sha256"]
        for entry in manifest["files"]
        if entry.get("exists") and entry.get("sha256") != "pending"
    }
    for filename in ("aet-bench-release-manifest.json", "croissant.json"):
        path = output / filename
        if path.exists():
            by_name[filename] = file_sha256_cached(path, hash_cache)

    ordered = [name for name in release_files(output) if name in by_name]
    with (output / "SHA256SUMS").open("w") as f:
        for name in ordered:
            f.write(f"{by_name[name]}  {name}\n")


def seed_archive_hashes(output: Path, sha256sums: Path, hash_cache: dict[str, str]) -> None:
    if not sha256sums.exists():
        return
    with sha256sums.open() as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            digest, filename = parts
            filename = filename.strip()
            if filename.endswith(".tar"):
                hash_cache[str(output / filename)] = digest


def croissant_file_object(
    filename: str,
    output: Path,
    base_url: str,
    encoding: str,
    description: str,
    hash_cache: dict[str, str],
) -> dict[str, Any]:
    path = output / filename
    obj: dict[str, Any] = {
        "@type": "cr:FileObject",
        "@id": filename,
        "name": filename,
        "description": description,
        "contentUrl": content_url(base_url, filename),
        "encodingFormat": encoding,
    }
    if path.exists():
        obj["contentSize"] = f"{path.stat().st_size} B"
        obj["sha256"] = file_sha256_cached(path, hash_cache)
    else:
        obj["sha256"] = "pending"
    return obj


def croissant_field(record_id: str, name: str, dtype: str, table_id: str, description: str = "") -> dict[str, Any]:
    field: dict[str, Any] = {
        "@type": "cr:Field",
        "@id": f"{record_id}/{name}",
        "name": name,
        "dataType": dtype,
        "source": {
            "fileObject": {"@id": table_id},
            "extract": {"column": name},
        },
    }
    if description:
        field["description"] = description
    return field


def update_croissant(croissant_path: Path, output: Path, base_url: str, hash_cache: dict[str, str]) -> None:
    with croissant_path.open() as f:
        doc = json.load(f)

    context = doc.setdefault("@context", {})
    context.update(
        {
            "@vocab": "https://schema.org/",
            "sc": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "ml": "http://mlcommons.org/croissant/",
            "dct": "http://purl.org/dc/terms/",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "prov": "http://www.w3.org/ns/prov#",
            "conformsTo": "dct:conformsTo",
            "recordSet": "cr:recordSet",
            "field": "cr:field",
            "dataType": "cr:dataType",
            "source": "cr:source",
            "fileObject": "cr:fileObject",
            "fileSet": "cr:fileSet",
            "extract": "cr:extract",
            "column": "cr:column",
            "containedIn": "cr:containedIn",
            "includes": "cr:includes",
            "isLiveDataset": "cr:isLiveDataset",
            "key": "cr:key",
        }
    )
    doc["conformsTo"] = "http://mlcommons.org/croissant/1.0"
    doc["isLiveDataset"] = False
    doc.pop("ml:isLiveDataset", None)
    doc.pop("ml:croissantVersion", None)

    distribution: list[dict[str, Any]] = [
        croissant_file_object(
            "aet-bench-samples.csv",
            output,
            base_url,
            "text/csv",
            "One row per released AET-Bench sample, including paths to sample metadata, NAF pickle data, and atom coordinates.",
            hash_cache,
        ),
        croissant_file_object(
            "aet-bench-results.csv",
            output,
            base_url,
            "text/csv",
            "Columnar view of the benchmark result log for Croissant validators and reviewer inspection.",
            hash_cache,
        ),
        croissant_file_object(
            "results.jsonl",
            output,
            base_url,
            "application/jsonlines",
            "Canonical benchmark result log used by paper table and figure generation.",
            hash_cache,
        ),
    ]
    for subset in RELEASE_SUBSETS:
        archive = f"aet-bench-{subset}.tar"
        distribution.append(
            croissant_file_object(
                archive,
                output,
                base_url,
                "application/x-tar",
                f"AET-Bench {subset} tier archive.",
                hash_cache,
            )
        )
        distribution.append(
            {
                "@type": "cr:FileSet",
                "@id": f"{subset}-files",
                "name": f"{subset} tier files",
                "containedIn": {"@id": archive},
                "encodingFormat": "application/octet-stream",
                "includes": f"data/aet_bench/{subset}/**/*",
            }
        )
    doc["distribution"] = distribution

    sample_fields = [
        croissant_field("samples", "sample_id", "sc:Text", "aet-bench-samples.csv", "Unique sample identifier."),
        croissant_field("samples", "subset", "sc:Text", "aet-bench-samples.csv", "Data tier."),
        croissant_field("samples", "element", "sc:Text", "aet-bench-samples.csv", "Chemical element or element system."),
        croissant_field("samples", "n_atoms", "sc:Integer", "aet-bench-samples.csv", "Number of reference atoms, if available."),
        croissant_field("samples", "n_angles", "sc:Integer", "aet-bench-samples.csv", "Number of projection angles."),
        croissant_field("samples", "has_atom_gt", "sc:Boolean", "aet-bench-samples.csv", "Whether atom-level reference coordinates are included."),
        croissant_field("samples", "metadata_path", "sc:Text", "aet-bench-samples.csv", "Path to per-sample metadata JSON inside the release archives."),
        croissant_field("samples", "data_pickle_path", "sc:Text", "aet-bench-samples.csv", "Path to the NAF-format pickle inside the release archives."),
        croissant_field("samples", "atoms_gt_path", "sc:Text", "aet-bench-samples.csv", "Path to atom-coordinate NumPy file, if available."),
        croissant_field("samples", "atoms_gt_4d_path", "sc:Text", "aet-bench-samples.csv", "Path to 4D atom-coordinate NumPy file, if available."),
        croissant_field("samples", "tier_archive", "sc:Text", "aet-bench-samples.csv", "Archive containing the sample files."),
        croissant_field("samples", "archive_member_prefix", "sc:Text", "aet-bench-samples.csv", "Directory prefix for the sample inside the tier archive."),
        croissant_field("samples", "source", "sc:Text", "aet-bench-samples.csv", "Original source for real-data samples when applicable."),
    ]
    result_fields = [
        croissant_field("results", "result_row_id", "sc:Integer", "aet-bench-results.csv", "Stable row number in the release CSV."),
        croissant_field("results", "sample_id", "sc:Text", "aet-bench-results.csv"),
        croissant_field("results", "subset", "sc:Text", "aet-bench-results.csv"),
        croissant_field("results", "method", "sc:Text", "aet-bench-results.csv", "Reconstruction method identifier."),
        croissant_field("results", "task", "sc:Text", "aet-bench-results.csv", "Evaluation task identifier."),
        croissant_field("results", "vol_psnr", "sc:Float", "aet-bench-results.csv", "Peak signal-to-noise ratio."),
        croissant_field("results", "vol_ssim", "sc:Float", "aet-bench-results.csv", "Structural similarity index."),
        croissant_field("results", "fsc_0.143_angstrom", "sc:Float", "aet-bench-results.csv", "Fourier shell correlation resolution at threshold 0.143."),
        croissant_field("results", "recall", "sc:Float", "aet-bench-results.csv", "Atom localization recall."),
        croissant_field("results", "precision", "sc:Float", "aet-bench-results.csv", "Atom localization precision."),
        croissant_field("results", "f1", "sc:Float", "aet-bench-results.csv", "Atom localization F1 score."),
        croissant_field("results", "rmsd", "sc:Float", "aet-bench-results.csv", "Root mean square deviation of matched atoms."),
        croissant_field("results", "trajectory_rmsd", "sc:Float", "aet-bench-results.csv", "Trajectory RMSD for 4D methods."),
        croissant_field("results", "atom_identity_persistence", "sc:Float", "aet-bench-results.csv", "Fraction of atoms tracked consistently across phases."),
        croissant_field("results", "runtime_sec", "sc:Float", "aet-bench-results.csv", "Wall-clock runtime in seconds."),
    ]
    doc["recordSet"] = [
        {
            "@type": "cr:RecordSet",
            "@id": "samples",
            "name": "samples",
            "description": "One row per released AET-Bench sample.",
            "key": {"@id": "samples/sample_id"},
            "field": sample_fields,
        },
        {
            "@type": "cr:RecordSet",
            "@id": "results",
            "name": "results",
            "description": "Benchmark evaluation results.",
            "key": {"@id": "results/result_row_id"},
            "field": result_fields,
        },
    ]

    with croissant_path.open("w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=True)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/aet_bench"))
    parser.add_argument("--results", type=Path, default=Path("data/aet_bench/results.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("output/aet_bench_release"))
    parser.add_argument("--base-url", default="https://zenodo.org/records/XXXXXXX/files")
    parser.add_argument("--croissant", type=Path, default=Path("paper/croissant.json"))
    parser.add_argument("--update-croissant", action="store_true")
    parser.add_argument("--write-sha256sums", action="store_true")
    parser.add_argument("--reuse-archive-sha256sums", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    hash_cache: dict[str, str] = {}
    if args.reuse_archive_sha256sums:
        seed_archive_hashes(args.output, args.reuse_archive_sha256sums, hash_cache)
    samples = sample_rows(args.data_root)
    results = result_rows(args.results)

    samples_csv = args.output / "aet-bench-samples.csv"
    results_csv = args.output / "aet-bench-results.csv"
    write_csv(samples_csv, SAMPLE_COLUMNS, samples)
    write_csv(results_csv, RESULT_COLUMNS, results)
    shutil.copy2(args.results, args.output / "results.jsonl")
    shutil.copy2("README.md", args.output / "README.md")
    summary = args.data_root / "SUMMARY.md"
    if summary.exists():
        shutil.copy2(summary, args.output / "SUMMARY.md")

    release_manifest = args.output / "aet-bench-release-manifest.json"
    write_release_manifest(release_manifest, args.output, args.base_url, samples, len(results), hash_cache)

    if args.update_croissant:
        update_croissant(args.croissant, args.output, args.base_url, hash_cache)
        shutil.copy2(args.croissant, args.output / "croissant.json")
        write_release_manifest(release_manifest, args.output, args.base_url, samples, len(results), hash_cache)
    if args.write_sha256sums:
        write_sha256sums(args.output, hash_cache)

    print(f"samples={len(samples)}")
    print(f"results={len(results)}")
    print(f"output={args.output}")
    print(f"samples_csv={samples_csv}")
    print(f"results_csv={results_csv}")
    print(f"release_manifest={release_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
