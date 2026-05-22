#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
DATA_DIR="${DATA_DIR:-outputs/ap_direct_sft_mimic3_age50_1w_exclude_first}"
OUT_DIR="${OUT_DIR:-outputs/ap_direct_sft_smoke_qwen25_05b_seq1024}"

accelerate launch --config_file configs/accelerate_4gpu_bf16.yaml scripts/train_ap_direct_sft_lora.py \
  --model-name-or-path "${MODEL_NAME}" \
  --train-file "${DATA_DIR}/train.jsonl" \
  --val-file "${DATA_DIR}/val.jsonl" \
  --output-dir "${OUT_DIR}" \
  --max-seq-len 1024 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-train-samples 32 \
  --max-eval-samples 16 \
  --max-steps 8 \
  --learning-rate 2e-4 \
  --warmup-ratio 0.03 \
  --logging-steps 1 \
  --eval-steps 4 \
  --save-steps 4 \
  --lora-r 4 \
  --lora-alpha 8 \
  --lora-dropout 0.05 \
  --gradient-checkpointing \
  --precision bf16 \
  --seed 13
