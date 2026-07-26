#!/bin/bash
# Fill fast methods on brownian_pt subset (66 samples).
# Skip heavy GPU methods (x2gaussian, r2gaussian) by default.
set -u
cd /data/aet-bench
export LD_LIBRARY_PATH=/opt/conda/envs/aet4dgs/lib:${LD_LIBRARY_PATH:-}
PY=/opt/conda/envs/aet4dgs/bin/python

methods="${METHODS:-fbp wbp astra_sirt astra_sart astra_cgls astra_ossart tomopy_gridrec tomopy_mlem dart fista_tv neat}"

for m in $methods; do
    echo "[$(date +%H:%M:%S)] $m full sweep"
    $PY scripts/run_benchmark.py --subset brownian_pt --method $m --task T1 \
        2>&1 | grep -E "/T1|FAILED" | tail -4
done
echo done
