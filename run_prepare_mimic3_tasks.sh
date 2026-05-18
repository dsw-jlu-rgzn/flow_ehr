#!/usr/bin/env bash
# Prepare MIMIC-III raw data and generate original AP/DS tasks.
# Usage:
#   bash run_prepare_mimic3_tasks.sh [sample_size] [raw_dir]

set -euo pipefail

SAMPLE_SIZE="${1:-100}"
RAW_DIR="${2:-C:/Users/dsw54/Desktop/MIMIC_related/mimic-iii-20260513T124356Z-3-001/mimic-iii}"
CONDA_SH="${CONDA_SH:-/home/csuvla/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-safevla}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi

cd "$PROJECT_DIR"
python -u processing/prepare_mimic3_tasks.py \
    --raw-dir "$RAW_DIR" \
    --output-root data \
    --sample-size "$SAMPLE_SIZE" \
    --make-tasks

python -u processing/validate_mimic3_tasks.py --data-root data
