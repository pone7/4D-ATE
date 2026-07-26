#!/bin/bash
# Fill heavy GPU methods on the real subset (x2gaussian, r2gaussian).
set -u
cd /data/aet-bench
export LD_LIBRARY_PATH=/opt/conda/envs/aet4dgs/lib:${LD_LIBRARY_PATH:-}
PY=/opt/conda/envs/aet4dgs/bin/python

for m in x2gaussian r2gaussian; do
    for s in Zr-Zr1 Zr-Zr2 Zr-Zr3 MPEA2; do
        echo "[$(date +%H:%M:%S)] $m on $s"
        $PY scripts/run_benchmark.py --subset real --sample $s --method $m --task T1 \
            2>&1 | grep -E "/T1|FAILED" | tail -1 || echo "  (no output)"
    done
done
echo done
