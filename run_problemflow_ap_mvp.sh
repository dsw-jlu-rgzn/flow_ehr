#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${INPUT_DIR:-data/AP/input}"
GOLD_DIR="${GOLD_DIR:-data/AP/gold}"
OUT_DIR="${OUT_DIR:-experiments/problemflow_ap/outputs}"
METHOD="${METHOD:-all}"
LLM="${LLM:-mock}"
LIMIT="${LIMIT:-0}"

"$PYTHON_BIN" experiments/problemflow_ap/problemflow_ap.py run-all \
  --inputdir "$INPUT_DIR" \
  --golddir "$GOLD_DIR" \
  --outdir "$OUT_DIR" \
  --method "$METHOD" \
  --llm "$LLM" \
  --limit "$LIMIT"
