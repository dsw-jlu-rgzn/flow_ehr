#!/usr/bin/env bash
# Compare baseline AP generations with a prefilter AP generation directory.
# Usage:
#   bash run_compare_ap_generation.sh [baseline_dir] [prefilter_dir] [prefilter_name]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
CONDA_SH="${CONDA_SH:-/home/csuvla/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-safevla}"

BASELINE_DIR="${1:-$PROJECT_DIR/data/AP/generated/EE/mistral/gt_v2}"
PREFILTER_DIR="${2:-$PROJECT_DIR/data/AP/generated/EE/mistral/embedding_previous_note}"
PREFILTER_NAME="${3:-embedding_previous_note}"
GT_DIR="$PROJECT_DIR/data/AP/gold"
OUTPUT_DIR="$PROJECT_DIR/outputs"

if [ ! -d "$BASELINE_DIR" ]; then
    echo "Error: baseline generated directory not found: $BASELINE_DIR" >&2
    exit 1
fi

if [ ! -d "$PREFILTER_DIR" ]; then
    echo "Error: prefilter generated directory not found: $PREFILTER_DIR" >&2
    echo "Run event_ap_fix_v2.py on the prefiltered input first." >&2
    exit 1
fi

if [ ! -d "$GT_DIR" ]; then
    echo "Error: AP gold directory not found: $GT_DIR" >&2
    exit 1
fi

if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi

cd "$PROJECT_DIR"
python -u evaluation/compare_ap_generation.py \
    --gt-dir "$GT_DIR" \
    --run "baseline=$BASELINE_DIR" \
    --run "$PREFILTER_NAME=$PREFILTER_DIR" \
    --baseline baseline \
    --output-csv "$OUTPUT_DIR/ap_compare_${PREFILTER_NAME}.csv" \
    --summary-csv "$OUTPUT_DIR/ap_compare_${PREFILTER_NAME}_summary.csv"
