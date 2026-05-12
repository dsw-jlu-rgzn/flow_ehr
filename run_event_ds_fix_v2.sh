#!/usr/bin/env bash
# run_event_ds_fix_v2.sh — Run event_ds_fix_v2.py
set -euo pipefail
MODEL="${1:-mistral}"
HF_TOKEN=""
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/modeling/event_ds_fix_v2.py"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/event_ds_fix_v2_${MODEL}.log"
mkdir -p "$LOG_DIR"
echo "=== event_ds_fix_v2.py ==="
echo "Model: $MODEL | GPU: 6,7"
echo "Output: data/DS/generated/EE/${MODEL}_v2/"
echo "Log: $LOG_FILE"
nohup bash -c "
  export HF_TOKEN='$HF_TOKEN'
  export PYTHONUNBUFFERED=1
  source /home/csuvla/miniconda3/etc/profile.d/conda.sh
  conda activate safevla
  cd '$PROJECT_DIR'
  python -u '$SCRIPT_PATH' --inputdir data/DS/full_input --outputdir data/DS/generated --model '$MODEL'
" > "$LOG_FILE" 2>&1 &
PID=$!
echo "PID: $PID"
echo "tail -f $LOG_FILE"
