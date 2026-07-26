#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-output/aet_bench_release}"
BASE_URL="${BASE_URL:-https://zenodo.org/records/XXXXXXX/files}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/aet4dgs/bin/python}"

mkdir -p "${OUT}"

"${PYTHON_BIN}" scripts/prepare_neurips_dataset_release.py \
  --output "${OUT}" \
  --base-url "${BASE_URL}" \
  --update-croissant

for subset in mono bi sim4d multislice real brownian_pt; do
  archive="${OUT}/aet-bench-${subset}.tar"
  if [[ -f "${archive}" ]]; then
    echo "skip existing ${archive}"
    continue
  fi
  echo "packing ${archive}"
  tar \
    --sort=name \
    --mtime="UTC 2026-05-05" \
    --owner=0 \
    --group=0 \
    --numeric-owner \
    -cf "${archive}" \
    "data/aet_bench/${subset}"
done

reuse_args=()
if [[ -f "${OUT}/SHA256SUMS" ]]; then
  reuse_args=(--reuse-archive-sha256sums "${OUT}/SHA256SUMS")
fi

"${PYTHON_BIN}" scripts/prepare_neurips_dataset_release.py \
  --output "${OUT}" \
  --base-url "${BASE_URL}" \
  "${reuse_args[@]}" \
  --update-croissant \
  --write-sha256sums

echo "release_dir=${OUT}"
echo "sha256_file=${OUT}/SHA256SUMS"
