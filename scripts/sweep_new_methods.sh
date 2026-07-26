#!/bin/bash
# AET-Bench: sweep the newly-added baselines across all subsets (incl. real).
# Uses LD_LIBRARY_PATH to handle scipy/libstdc++ version drift in aet4dgs.

set -u
cd /data/aet-bench
export LD_LIBRARY_PATH=/opt/conda/envs/aet4dgs/lib:${LD_LIBRARY_PATH:-}
PY=/opt/conda/envs/aet4dgs/bin/python

# Fast (GPU) methods: ASTRA × 4 + TomoPy × 2 + FISTA-TV + DART + WBP + NeAT
fast_methods=(astra_sirt astra_sart astra_cgls astra_ossart
              tomopy_gridrec tomopy_mlem wbp fista_tv dart)
inr_methods=(neat)

log() { echo "[$(date +%H:%M:%S)] $*"; }

for subset in mono bi 4d real; do
  log "=== subset=$subset: classical ==="
  for m in "${fast_methods[@]}"; do
    CUDA_VISIBLE_DEVICES=0 $PY scripts/run_benchmark.py --subset $subset --method $m --task T1 2>&1 | grep -E "FAILED|/T1|/T7"
  done
  log "=== subset=$subset: INR ==="
  for m in "${inr_methods[@]}"; do
    CUDA_VISIBLE_DEVICES=1 $PY scripts/run_benchmark.py --subset $subset --method $m --task T1 2>&1 | grep -E "FAILED|/T1|/T7"
  done
done
log "=== sweep done ==="
