#!/usr/bin/env bash
# monitor_and_run_v2.sh — 自动监测 V1 跑完→启动 V2
# 用法: nohup bash monitor_and_run_v2.sh > monitor_v2.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

DS_V1_KEYWORD="event_ds_fix.py"
AP_V1_KEYWORD="event_ap_fix.py"
DS_V2_SCRIPT="$SCRIPT_DIR/run_event_ds_fix_v2.sh"
AP_V2_SCRIPT="$SCRIPT_DIR/run_event_ap_fix_v2.sh"

echo "===== 自动监测脚本启动: $(date) ====="
echo "监测目标: DS V1 + AP V1"
echo "监测间隔: 60 秒"

while true; do
    DS_PID=$(ps aux | grep "$DS_V1_KEYWORD" | grep -v grep | grep -v v2 | awk '{print $2}' || true)
    AP_PID=$(ps aux | grep "$AP_V1_KEYWORD" | grep -v grep | grep -v v2 | awk '{print $2}' || true)

    [ -n "$DS_PID" ] && echo "[$(date)] DS V1 运行中 (PID: $DS_PID)" || echo "[$(date)] DS V1 已结束"
    [ -n "$AP_PID" ] && echo "[$(date)] AP V1 运行中 (PID: $AP_PID)" || echo "[$(date)] AP V1 已结束"

    if [ -z "$DS_PID" ] && [ -z "$AP_PID" ]; then
        echo ""
        echo "===== V1 全部结束！等待 60 秒释放显存... ====="
        sleep 600
        echo "===== 启动 DS V2... ====="
        bash "$DS_V2_SCRIPT" mistral
        sleep 10
        echo "===== 启动 AP V2... ====="
        bash "$AP_V2_SCRIPT" mistral gt
        echo "===== V2 已启动: $(date) ====="
        echo "DS V2 日志: $LOG_DIR/event_ds_fix_v2_mistral.log"
        echo "AP V2 日志: $LOG_DIR/event_ap_fix_v2_mistral_gt.log"
        exit 0
    fi

    sleep 600
done
