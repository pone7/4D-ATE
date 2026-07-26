#!/usr/bin/env bash
set -u

ROOT="/data/aet-bench"
PY="/opt/conda/envs/aet4dgs/bin/python"
RESULTS="$ROOT/data/aet_bench/results_real_splatting.jsonl"

cd "$ROOT"
unset LD_LIBRARY_PATH

samples=(
  AlSiYb-HAADF
  AuPd-DH
  AuPd-ICNP1
  AuPd-ICNP2
  AuPd-ICNP3
  AuPd-ICNP4
  AuPd-ICNP5
  AuPd-ICNP6
  AuPd-ICNP7
  AuPd-ICNP8
  AuPd-PC1
  AuPd-PC2
  AuPd-PC3
  AuPd-SF
  Coral-dragonhead
  NiPt-MultiGrain
  PdPt-Yang-P1
  PdPt-Yang-P2
  PdPt-Zhou-EPB
  PdPt-Zhou-PB
  PdPt-Zhou-TO
  Ta-thinfilm
)

for sample in "${samples[@]}"; do
  echo "[real-r2] START $sample"
  "$PY" -u scripts/run_benchmark.py \
    --subset real \
    --sample "$sample" \
    --method r2gaussian \
    --task T1 \
    --results "$RESULTS"
  rc=$?
  echo "[real-r2] END $sample rc=$rc"
  if [ "$rc" -ne 0 ]; then
    exit "$rc"
  fi
done

echo "[real-r2] DONE"
