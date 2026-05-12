#!/usr/bin/env bash
# Run modeling/event_ds_fix_v2.py.
# Usage: bash run_event_ds_fix_v2.sh [model]

set -euo pipefail

MODEL="${1:-mistral}"
GPU_DEVICES="${GPU_DEVICES:-6,7}"
HF_TOKEN="${HF_TOKEN:-}"
CONDA_SH="${CONDA_SH:-/home/csuvla/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-safevla}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
SCRIPT_PATH="$PROJECT_DIR/modeling/event_ds_fix_v2.py"
INPUT_DIR="$PROJECT_DIR/data/DS/full_input"
OUTPUT_DIR="$PROJECT_DIR/data/DS/generated"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/event_ds_fix_v2_${MODEL}.log"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: script not found: $SCRIPT_PATH" >&2
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: input directory not found: $INPUT_DIR" >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

echo "=== event_ds_fix_v2.py ==="
echo "Project: $PROJECT_DIR"
echo "Model: $MODEL"
echo "GPU: $GPU_DEVICES"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR/EE/${MODEL}_v2/"
echo "Log: $LOG_FILE"

nohup bash -c "
  set -euo pipefail
  export HF_TOKEN='$HF_TOKEN'
  export CUDA_VISIBLE_DEVICES='$GPU_DEVICES'
  export PYTHONUNBUFFERED=1
  if [ -f '$CONDA_SH' ]; then
    source '$CONDA_SH'
    conda activate '$CONDA_ENV'
  fi
  cd '$PROJECT_DIR'
  python -u '$SCRIPT_PATH' --inputdir '$INPUT_DIR' --outputdir '$OUTPUT_DIR' --model '$MODEL'
" > "$LOG_FILE" 2>&1 &

PID=$!
echo "PID: $PID"
echo "tail -f $LOG_FILE"
