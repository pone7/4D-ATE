#!/bin/bash
# Run all non-4D methods on the multislice subset.
set -u
cd /data/aet-bench
export LD_LIBRARY_PATH=/opt/conda/envs/aet4dgs/lib:${LD_LIBRARY_PATH:-}
PY=/opt/conda/envs/aet4dgs/bin/python

methods=(fbp wbp sirt isonet genfire fista_tv dart
         astra_sirt astra_sart astra_cgls astra_ossart
         tomopy_gridrec tomopy_mlem
         neat implicit_et
         r2gaussian x2gaussian)

# List multislice samples dynamically
samples=$(ls data/aet_bench/multislice/ 2>/dev/null | grep -v json)

for m in "${methods[@]}"; do
    for s in $samples; do
        echo "[$(date +%H:%M:%S)] $m on $s"
        $PY scripts/run_benchmark.py --subset multislice --sample $s \
            --method $m --task T1 2>&1 | grep -E "/T1|FAILED" | tail -1
    done
done
echo done
