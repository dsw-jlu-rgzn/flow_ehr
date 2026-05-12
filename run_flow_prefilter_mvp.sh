#!/usr/bin/env bash
# Train the minimal flow prefilter and export condensed AP inputs.
# Usage: bash run_flow_prefilter_mvp.sh [top_k] [flow|mlp]

set -euo pipefail

TOP_K="${1:-40}"
MODEL_TYPE="${2:-flow}"
CONDA_SH="${CONDA_SH:-/home/csuvla/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-safevla}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
INPUT_DIR="$PROJECT_DIR/data/AP/input"
OUTPUT_DIR="$PROJECT_DIR/data/AP/input_flow_topk"
CHECKPOINT="$PROJECT_DIR/checkpoints/flow_prefilter_mvp.pt"
SCRIPT_PATH="$PROJECT_DIR/modeling/flow_prefilter_mvp.py"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: script not found: $SCRIPT_PATH" >&2
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: AP input directory not found: $INPUT_DIR" >&2
    exit 1
fi

if [ "$MODEL_TYPE" != "flow" ] && [ "$MODEL_TYPE" != "mlp" ]; then
    echo "Error: model type must be 'flow' or 'mlp', got: $MODEL_TYPE" >&2
    exit 1
fi

if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi

cd "$PROJECT_DIR"
python -u "$SCRIPT_PATH" all \
    --inputdir "$INPUT_DIR" \
    --outputdir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --model-type "$MODEL_TYPE" \
    --top-k "$TOP_K"

echo ""
echo "Condensed AP input written to: $OUTPUT_DIR"
echo "Next LLM run:"
echo "python -u modeling/event_ap_fix_v2.py --inputdir '$OUTPUT_DIR' --outputdir data/AP/generated --setting gt --model mistral"
