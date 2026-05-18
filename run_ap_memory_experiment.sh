#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
INPUT_DIR="${INPUT_DIR:-data/AP/input}"
GOLD_DIR="${GOLD_DIR:-data/AP/gold}"
OUT_DIR="${OUT_DIR:-experiments/problemflow_ap/outputs_ap_memory}"
METHOD="${METHOD:-all}"
LLM="${LLM:-mock}"
LIMIT="${LIMIT:-0}"
AUTOREGRESSIVE_HISTORY="${AUTOREGRESSIVE_HISTORY:-0}"

args=(
  experiments/problemflow_ap/ap_memory_experiment.py
  run-all
  --inputdir "$INPUT_DIR"
  --golddir "$GOLD_DIR"
  --outdir "$OUT_DIR"
  --method "$METHOD"
  --llm "$LLM"
  --limit "$LIMIT"
)

if [[ "$AUTOREGRESSIVE_HISTORY" == "1" ]]; then
  args+=(--autoregressive-history)
fi

"$PYTHON_BIN" "${args[@]}"
