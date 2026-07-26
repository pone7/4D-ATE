#!/usr/bin/env bash
# Launch phase-2 GPU-based 补齐 (run only after main queue finishes!).
#
# Guard: refuses to start if any of the 5 known main-queue worker PIDs are
# still alive. If you intentionally want to co-locate with the main queue,
# set FORCE=1.
# Defaults to one worker per visible GPU. Override with N_GPUS=... if needed.
set -u

ROOT=/data/aet-bench
DEFAULT_PY=/opt/conda/envs/aet4dgs/bin/python
if [ -x "$DEFAULT_PY" ]; then
    PY=${PYTHON_BIN:-$DEFAULT_PY}
else
    PY=${PYTHON_BIN:-$(command -v python)}
fi
MAIN_PIDS=(484660 2273284 2273285 2273286 2273287)
LOG_DIR=$ROOT/logs
mkdir -p "$LOG_DIR"

if [ "${FORCE:-0}" != "1" ]; then
    for pid in "${MAIN_PIDS[@]}"; do
        if [ -d "/proc/$pid" ]; then
            echo "Main-queue PID $pid still running — refusing to start GPU extras." >&2
            echo "Rerun with FORCE=1 if you really want to overlap." >&2
            exit 1
        fi
    done
fi

cd "$ROOT"
# The host image may export a global torch/CUDA LD_LIBRARY_PATH that is
# incompatible with the benchmark conda env. Let each wrapper set only the
# libraries it needs.
unset LD_LIBRARY_PATH
if [ -z "${N_GPUS:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    else
        N_GPUS=1
    fi
fi

if [ "$N_GPUS" -lt 1 ]; then
    echo "N_GPUS must be >= 1" >&2
    exit 1
fi

echo "Using python: $PY"
echo "Starting $N_GPUS GPU worker(s)"
if [ "${SKIP_CUDA_PREFLIGHT:-0}" != "1" ]; then
    "$PY" - <<'PY'
import sys
try:
    import torch
    ok = torch.cuda.is_available()
    count = torch.cuda.device_count() if ok else 0
except Exception as exc:
    print(f"CUDA preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)
if not ok or count < 1:
    print("CUDA preflight failed: torch cannot initialize any CUDA device", file=sys.stderr)
    raise SystemExit(1)
print(f"CUDA preflight ok: {count} device(s)")
PY
fi
for i in $(seq 0 $((N_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$i GPU_SLICE=$i N_GPUS=$N_GPUS nohup "$PY" -u scripts/run_gpu_extras.py \
        > "$LOG_DIR/gpu_extras.slice${i}.log" 2>&1 &
    echo "started slice $i  cuda=$i  pid=$!"
done
wait
