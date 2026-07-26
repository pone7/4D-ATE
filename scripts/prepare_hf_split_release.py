#!/usr/bin/env python3
"""Prepare a Hugging Face-friendly AET-Bench release directory.

The canonical release can contain tar files larger than the Hugging Face Hub's
recommended single-file size. This script creates a sibling upload directory
where oversized tar archives are split into deterministic parts, then rewrites
the release manifest, Croissant file, README, and checksums to describe the
actual uploaded files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RELEASE_PREFIX = "aet-bench-"
MANIFEST = "aet-bench-release-manifest.json"
ORIGINAL_SHA256SUMS = "ORIGINAL_ARCHIVE_SHA256SUMS"
UPLOAD_SHA256SUMS = "SHA256SUMS"
SKIP_NAMES = {MANIFEST, "croissant.json", UPLOAD_SHA256SUMS, ORIGINAL_SHA256SUMS}
COPY_NAMES = {"README.md", "aet-bench-samples.csv"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def content_url(base_url: str, filename: str) -> str:
    return f"{base_url.rstrip('/')}/{filename}" if base_url else filename


def read_sha256sums(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                out[parts[1].strip()] = parts[0]
    return out


def hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_mutable_or_link(src: Path, dst: Path) -> None:
    if src.name in COPY_NAMES:
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            dst.unlink()
        shutil.copy2(src, dst)
        return
    hardlink_or_copy(src, dst)


def split_file(src: Path, output: Path, part_size: int) -> list[dict[str, Any]]:
    part_count = (src.stat().st_size + part_size - 1) // part_size
    parts: list[dict[str, Any]] = []

    for index in range(part_count):
        part_name = f"{src.name}.part-{index:03d}"
        part_path = output / part_name
        expected_size = min(part_size, src.stat().st_size - index * part_size)
        if part_path.exists() and part_path.stat().st_size == expected_size:
            parts.append({"filename": part_name, "path": part_path, "part_index": index})
            continue
        part_path.unlink(missing_ok=True)
        with src.open("rb") as f_in, part_path.open("wb") as f_out:
            f_in.seek(index * part_size)
            remaining = expected_size
            while remaining:
                chunk = f_in.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise IOError(f"unexpected EOF while splitting {src}")
                f_out.write(chunk)
                remaining -= len(chunk)
        parts.append({"filename": part_name, "path": part_path, "part_index": index})

    return parts


def describe_upload_file(path: Path, base_url: str, digest: str | None = None) -> dict[str, Any]:
    return {
        "filename": path.name,
        "contentUrl": content_url(base_url, path.name),
        "contentSize": f"{path.stat().st_size} B",
        "exists": True,
        "sha256": digest or sha256_file(path),
    }


def update_samples_csv(output: Path, split_by_archive: dict[str, list[str]]) -> None:
    csv_path = output / "aet-bench-samples.csv"
    if not csv_path.exists():
        return
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    for row in rows:
        archive = row.get("tier_archive", "")
        if archive in split_by_archive:
            row["tier_archive"] = json.dumps(split_by_archive[archive], separators=(",", ":"))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_existing_manifest(source: Path) -> dict[str, Any]:
    manifest_path = source / MANIFEST
    if not manifest_path.exists():
        return {}
    with manifest_path.open() as f:
        return json.load(f)


def build_manifest(
    source: Path,
    output: Path,
    base_url: str,
    upload_entries: list[dict[str, Any]],
    split_archives: list[dict[str, Any]],
) -> dict[str, Any]:
    original = load_existing_manifest(source)
    manifest = {
        "name": original.get("name", "AET-Bench"),
        "version": original.get("version", "1.0.0"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": original.get("sample_count"),
        "result_rows": original.get("result_rows"),
        "subsets": original.get("subsets", {}),
        "huggingface_split_release": True,
        "split_part_size": split_archives[0]["part_size"] if split_archives else None,
        "reconstruction_note": (
            "Archives listed in split_archives must be reconstructed by concatenating "
            "their parts in lexical order before extracting the tar archive."
        ),
        "files": upload_entries,
        "split_archives": split_archives,
    }
    return manifest


def croissant_file_object(path: Path, base_url: str, description: str, encoding: str) -> dict[str, Any]:
    return {
        "@type": "cr:FileObject",
        "@id": path.name,
        "name": path.name,
        "description": description,
        "contentUrl": content_url(base_url, path.name),
        "encodingFormat": encoding,
        "contentSize": f"{path.stat().st_size} B",
        "sha256": sha256_file(path),
    }


def update_croissant(
    source: Path,
    output: Path,
    base_url: str,
    split_by_archive: dict[str, list[dict[str, Any]]],
) -> None:
    with (source / "croissant.json").open() as f:
        doc = json.load(f)

    new_distribution: list[dict[str, Any]] = []
    for item in doc.get("distribution", []):
        item_id = item.get("@id")
        item_type = item.get("@type")
        if item_type == "cr:FileObject":
            filename = item.get("name") or item_id
            if filename in split_by_archive:
                original_digest = split_by_archive[filename][0]["original_sha256"]
                for part in split_by_archive[filename]:
                    part_path = output / part["filename"]
                    new_distribution.append(
                        croissant_file_object(
                            part_path,
                            base_url,
                            (
                                f"Part {part['part_index'] + 1} of {part['part_count']} for {filename}. "
                                f"Concatenate all parts in lexical order to reconstruct the original tar; "
                                f"original SHA256: {original_digest}."
                            ),
                            "application/octet-stream",
                        )
                    )
                continue
            path = output / filename
            if path.exists():
                item["contentUrl"] = content_url(base_url, filename)
                item["contentSize"] = f"{path.stat().st_size} B"
                item["sha256"] = sha256_file(path)
        elif item_type == "cr:FileSet":
            contained = item.get("containedIn")
            contained_id = contained.get("@id") if isinstance(contained, dict) else None
            if contained_id in split_by_archive:
                item["containedIn"] = [{"@id": part["filename"]} for part in split_by_archive[contained_id]]
                item["description"] = (
                    f"Files contained in the reconstructed {contained_id} archive. "
                    "Concatenate the listed parts before extracting."
                )
        new_distribution.append(item)

    for extra in (ORIGINAL_SHA256SUMS,):
        path = output / extra
        if path.exists():
            new_distribution.append(
                croissant_file_object(path, base_url, f"AET-Bench release metadata file {extra}.", "text/plain")
            )

    doc["distribution"] = new_distribution
    doc["url"] = base_url.rsplit("/resolve/main", 1)[0] if "/resolve/main" in base_url else base_url
    with (output / "croissant.json").open("w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=True)
        f.write("\n")


def append_readme_split_notes(output: Path, split_archives: list[dict[str, Any]]) -> None:
    readme = output / "README.md"
    if not readme.exists() or not split_archives:
        return
    note_lines = [
        "",
        "## Hugging Face Large-File Parts",
        "",
        "Some tier archives are split into parts to keep individual Hub files below the recommended size limit.",
        "Concatenate parts in lexical order before extracting the reconstructed tar archive:",
        "",
    ]
    for archive in split_archives:
        original = archive["original_filename"]
        pattern = f"{original}.part-*"
        note_lines.append(f"```bash")
        note_lines.append(f"cat {pattern} > {original}")
        note_lines.append(f"sha256sum {original}")
        note_lines.append(f"```")
        note_lines.append(f"Expected SHA256 for `{original}`: `{archive['original_sha256']}`.")
        note_lines.append("")
    note_lines.append("Uploaded-file checksums are in `SHA256SUMS`; reconstructed archive checksums are in `ORIGINAL_ARCHIVE_SHA256SUMS`.")
    note_lines.append("")

    text = readme.read_text()
    marker = "## Hugging Face Large-File Parts"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    readme.write_text(text.rstrip() + "\n" + "\n".join(note_lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output/aet_bench_release"))
    parser.add_argument("--output", type=Path, default=Path("output/aet_bench_release_hf"))
    parser.add_argument("--base-url", default="https://huggingface.co/datasets/ANONYMIZED/aet-bench/resolve/main")
    parser.add_argument("--part-size-gib", type=float, default=18.0)
    parser.add_argument("--croissant-copy", type=Path, default=Path("paper/croissant.json"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    part_size = int(args.part_size_gib * 1024**3)
    original_hashes = read_sha256sums(args.source / UPLOAD_SHA256SUMS)

    split_by_archive: dict[str, list[dict[str, Any]]] = {}
    split_archives: list[dict[str, Any]] = []

    for src in sorted(p for p in args.source.iterdir() if p.is_file()):
        if src.name in SKIP_NAMES:
            continue
        if src.suffix == ".tar" and src.stat().st_size > part_size:
            original_sha = original_hashes.get(src.name) or sha256_file(src)
            part_records = split_file(src, args.output, part_size)
            for record in part_records:
                path = record["path"]
                record.update(
                    {
                        "contentSize": f"{path.stat().st_size} B",
                        "contentUrl": content_url(args.base_url, record["filename"]),
                        "part_count": len(part_records),
                        "original_filename": src.name,
                        "original_sha256": original_sha,
                        "sha256": sha256_file(path),
                    }
                )
                record.pop("path", None)
            split_by_archive[src.name] = part_records
            split_archives.append(
                {
                    "original_filename": src.name,
                    "original_contentSize": f"{src.stat().st_size} B",
                    "original_sha256": original_sha,
                    "part_size": f"{part_size} B",
                    "parts": part_records,
                    "reconstruct_command": f"cat {src.name}.part-* > {src.name}",
                }
            )
        else:
            copy_mutable_or_link(src, args.output / src.name)

    update_samples_csv(args.output, {name: [part["filename"] for part in parts] for name, parts in split_by_archive.items()})

    original_sha_path = args.output / ORIGINAL_SHA256SUMS
    with original_sha_path.open("w") as f:
        for name, digest in sorted(original_hashes.items()):
            if name.endswith(".tar"):
                f.write(f"{digest}  {name}\n")

    append_readme_split_notes(args.output, split_archives)
    update_croissant(args.source, args.output, args.base_url, split_by_archive)

    upload_entries = [
        describe_upload_file(path, args.base_url)
        for path in sorted(p for p in args.output.iterdir() if p.is_file() and p.name not in {MANIFEST, UPLOAD_SHA256SUMS})
    ]
    manifest = build_manifest(args.source, args.output, args.base_url, upload_entries, split_archives)
    with (args.output / MANIFEST).open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    final_entries = [
        describe_upload_file(path, args.base_url)
        for path in sorted(p for p in args.output.iterdir() if p.is_file() and p.name != UPLOAD_SHA256SUMS)
    ]
    with (args.output / UPLOAD_SHA256SUMS).open("w") as f:
        for entry in final_entries:
            f.write(f"{entry['sha256']}  {entry['filename']}\n")

    if args.croissant_copy:
        shutil.copy2(args.output / "croissant.json", args.croissant_copy)

    print(f"source={args.source}")
    print(f"output={args.output}")
    print(f"split_archives={len(split_archives)}")
    print(f"upload_files={len(list(p for p in args.output.iterdir() if p.is_file()))}")
    for archive in split_archives:
        print(f"split {archive['original_filename']} -> {len(archive['parts'])} parts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
