#!/usr/bin/env bash
# Monitor V1 jobs and start V2 jobs after they finish.
# Usage: nohup bash monitor_and_run_v2.sh > monitor_v2.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

DS_V1_KEYWORD="${DS_V1_KEYWORD:-modeling/event_ds_fix.py}"
AP_V1_KEYWORD="${AP_V1_KEYWORD:-modeling/event_ap_fix.py}"
DS_V2_SCRIPT="$SCRIPT_DIR/run_event_ds_fix_v2.sh"
AP_V2_SCRIPT="$SCRIPT_DIR/run_event_ap_fix_v2.sh"
MODEL="${1:-mistral}"
SETTING="${2:-gt}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-600}"
START_DELAY_SECONDS="${START_DELAY_SECONDS:-60}"

mkdir -p "$LOG_DIR"

if [ ! -f "$DS_V2_SCRIPT" ]; then
    echo "Error: DS V2 runner not found: $DS_V2_SCRIPT" >&2
    exit 1
fi

if [ ! -f "$AP_V2_SCRIPT" ]; then
    echo "Error: AP V2 runner not found: $AP_V2_SCRIPT" >&2
    exit 1
fi

find_pids() {
    local keyword="$1"
    pgrep -f "$keyword" || true
}

echo "===== monitor_and_run_v2 started: $(date) ====="
echo "Project: $SCRIPT_DIR"
echo "Monitoring: DS V1='$DS_V1_KEYWORD', AP V1='$AP_V1_KEYWORD'"
echo "Check interval: ${CHECK_INTERVAL_SECONDS}s"
echo "V2 model: $MODEL, AP setting: $SETTING"

while true; do
    DS_PID="$(find_pids "$DS_V1_KEYWORD")"
    AP_PID="$(find_pids "$AP_V1_KEYWORD")"

    if [ -n "$DS_PID" ]; then
        echo "[$(date)] DS V1 is running (PID: ${DS_PID//$'\n'/,})"
    else
        echo "[$(date)] DS V1 has finished"
    fi

    if [ -n "$AP_PID" ]; then
        echo "[$(date)] AP V1 is running (PID: ${AP_PID//$'\n'/,})"
    else
        echo "[$(date)] AP V1 has finished"
    fi

    if [ -z "$DS_PID" ] && [ -z "$AP_PID" ]; then
        echo ""
        echo "===== All V1 jobs finished. Waiting ${START_DELAY_SECONDS}s before starting V2... ====="
        sleep "$START_DELAY_SECONDS"

        echo "===== Starting DS V2... ====="
        bash "$DS_V2_SCRIPT" "$MODEL"

        sleep 10

        echo "===== Starting AP V2... ====="
        bash "$AP_V2_SCRIPT" "$MODEL" "$SETTING"

        echo "===== V2 jobs started: $(date) ====="
        echo "DS V2 log: $LOG_DIR/event_ds_fix_v2_${MODEL}.log"
        echo "AP V2 log: $LOG_DIR/event_ap_fix_v2_${MODEL}_${SETTING}.log"
        exit 0
    fi

    sleep "$CHECK_INTERVAL_SECONDS"
done
