#!/usr/bin/env bash
# Export AP inputs filtered by frozen old-model embeddings, with no training.
# Usage: bash run_embedding_prefilter_no_train.sh [top_k] [previous_note|day_context|oracle_gt] [model]

set -euo pipefail

TOP_K="${1:-40}"
QUERY_MODE="${2:-previous_note}"
MODEL="${3:-mistral}"
GPU_DEVICES="${GPU_DEVICES:-7}"
CONDA_SH="${CONDA_SH:-/home/csuvla/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-safevla}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
INPUT_DIR="$PROJECT_DIR/data/AP/input"
OUTPUT_DIR="$PROJECT_DIR/data/AP/input_embedding_topk_${QUERY_MODE}"
SCRIPT_PATH="$PROJECT_DIR/modeling/embedding_prefilter_no_train.py"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: script not found: $SCRIPT_PATH" >&2
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: AP input directory not found: $INPUT_DIR" >&2
    exit 1
fi

case "$QUERY_MODE" in
    previous_note|day_context|oracle_gt) ;;
    *)
        echo "Error: query mode must be previous_note, day_context, or oracle_gt; got: $QUERY_MODE" >&2
        exit 1
        ;;
esac

if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
python -u "$SCRIPT_PATH" \
    --inputdir "$INPUT_DIR" \
    --outputdir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --query-mode "$QUERY_MODE" \
    --top-k "$TOP_K"

echo ""
echo "Condensed AP input written to: $OUTPUT_DIR"
echo "Next LLM run:"
echo "python -u modeling/event_ap_fix_v2.py --inputdir '$OUTPUT_DIR' --outputdir data/AP/generated --setting gt --model '$MODEL' --run_name embedding_${QUERY_MODE}"
