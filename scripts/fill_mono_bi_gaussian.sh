#!/bin/bash
# Fill missing samples for X²-Gaussian and R²-Gaussian on mono + bi.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 METHOD=x2gaussian scripts/fill_mono_bi_gaussian.sh
#   CUDA_VISIBLE_DEVICES=1 METHOD=r2gaussian scripts/fill_mono_bi_gaussian.sh
set -u
cd /data/aet-bench
export LD_LIBRARY_PATH=/opt/conda/envs/aet4dgs/lib:${LD_LIBRARY_PATH:-}
PY=/opt/conda/envs/aet4dgs/bin/python
METHOD=${METHOD:-x2gaussian}

# Collect list of missing samples via python.
MISSING=$($PY - <<PYEOF
import json, os
done = {('mono','$METHOD'): set(), ('bi','$METHOD'): set()}
with open('data/aet_bench/results.jsonl') as f:
    for line in f:
        r = json.loads(line)
        k = (r['subset'], r['method'])
        if k in done:
            done[k].add(r['sample_id'])
for sub in ('mono','bi'):
    samples = sorted(s for s in os.listdir(f'data/aet_bench/{sub}')
                      if not s.endswith('.json'))
    for s in samples:
        if s not in done[(sub, '$METHOD')]:
            print(f'{sub} {s}')
PYEOF
)

echo "$MISSING" | while read -r sub sample; do
    [ -z "$sub" ] && continue
    echo "[$(date +%H:%M:%S)] $METHOD $sub/$sample"
    $PY scripts/run_benchmark.py --subset $sub --sample $sample \
        --method $METHOD --task T1 2>&1 | grep -E "/T1|FAILED" | tail -1
done
echo done
