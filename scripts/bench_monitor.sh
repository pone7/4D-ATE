#!/usr/bin/env bash
# AET-Bench background monitor.
# Appends a JSON snapshot every INTERVAL seconds to LOG.
# Also flags if all 5 workers look stalled (log age > STALL_SEC).
#
# Run as:  nohup bash scripts/bench_monitor.sh >> logs/bench_monitor.daemon.log 2>&1 &
set -u

ROOT=/data/aet-bench
INTERVAL=${INTERVAL:-1800}   # 30 min
STALL_SEC=${STALL_SEC:-900}  # 15 min per worker log
LOG_DIR=${ROOT}/logs
mkdir -p "$LOG_DIR"
LOG=${LOG:-$LOG_DIR/bench_monitor.jsonl}
EVENT_LOG=${EVENT_LOG:-$LOG_DIR/bench_monitor.events.log}

echo "[$(date -Iseconds)] monitor start pid=$$ interval=${INTERVAL}s log=$LOG" >> "$EVENT_LOG"

while true; do
    snap=$(python3 "$ROOT/scripts/bench_status.py" --json 2>/dev/null)
    if [ -z "$snap" ]; then
        echo "[$(date -Iseconds)] ERR: status script returned empty" >> "$EVENT_LOG"
    else
        echo "$snap" >> "$LOG"
        # stall detection
        stalled=$(python3 -c "
import json,sys
s=json.loads(sys.stdin.read())
stall=${STALL_SEC}
bad=[w['pid'] for w in s['workers'] if not w['alive'] or w.get('log_age_sec',0)>stall]
print(','.join(str(p) for p in bad))
" <<< "$snap")
        if [ -n "$stalled" ]; then
            echo "[$(date -Iseconds)] WARN stalled/dead workers: $stalled" >> "$EVENT_LOG"
        fi
        # short human line
        remaining=$(python3 -c "import json,sys;print(json.loads(sys.stdin.read())['remaining_jobs'])" <<< "$snap")
        rate=$(python3 -c "import json,sys;print(f\"{json.loads(sys.stdin.read())['rate_jobs_per_hour_30m']:.1f}\")" <<< "$snap")
        eta=$(python3 -c "import json,sys;print(f\"{json.loads(sys.stdin.read())['eta_hours']:.1f}\")" <<< "$snap")
        echo "[$(date -Iseconds)] remaining=$remaining rate=${rate}/h eta=${eta}h stalled=${stalled:-none}" >> "$EVENT_LOG"
    fi
    sleep "$INTERVAL"
done
