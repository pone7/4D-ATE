#!/usr/bin/env bash
# Package the AET-Bench evaluation + method code for reviewer release.
# Produces a deterministic tarball with a SHA256 manifest and verifies that a
# clean unpack exposes a working `run_benchmark.py --help`. Addresses the
# reviewers' request (AC, Yz2i, WQbd) for the evaluation code, method
# implementations, and reproduction scripts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${ROOT}/output/aet_bench_code_release}"
PY="${PYTHON_BIN:-/opt/conda/envs/aet4dgs/bin/python}"
STAMP="UTC 2026-07-24"

mkdir -p "${OUT}"
ARCHIVE="${OUT}/aet-bench-code.tar.gz"

# What ships: the full harness, metrics, method wrappers, analyses, tests,
# configs, environment, and docs — everything needed to rerun the scoring
# pipeline and reproduce the paper's tables/figures. Excludes large data and
# training outputs (shipped separately as the dataset release).
cd "${ROOT}"
tar \
  --sort=name \
  --mtime="${STAMP}" \
  --owner=0 --group=0 --numeric-owner \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  -czf "${ARCHIVE}" \
  scripts \
  tests \
  configs \
  environment.yml \
  README.md \
  CONTRIBUTING.md \
  AGENTS.md \
  CLAUDE.md \
  paper/croissant.json \
  docs/neurips_2026_response.md

# Deterministic manifest.
( cd "${OUT}" && sha256sum "$(basename "${ARCHIVE}")" > SHA256SUMS.code )
echo "wrote ${ARCHIVE}"
cat "${OUT}/SHA256SUMS.code"

# Clean-unpack smoke test: unpack into a scratch dir and confirm the CLI loads.
SMOKE="$(mktemp -d)"
trap 'rm -rf "${SMOKE}"' EXIT
tar -xzf "${ARCHIVE}" -C "${SMOKE}"
if ( cd "${SMOKE}/scripts" && "${PY}" run_benchmark.py --help >/dev/null 2>&1 ); then
  echo "SMOKE OK: run_benchmark.py --help works from clean unpack"
else
  echo "SMOKE FAILED: run_benchmark.py --help did not run from clean unpack" >&2
  exit 1
fi
