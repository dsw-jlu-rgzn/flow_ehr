#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
DATA_DIR="${DATA_DIR:-outputs/ap_direct_sft_mimic3_age50_1w_exclude_first}"
OUT_DIR="${OUT_DIR:-outputs/ap_direct_sft_mimic3_age50_1w_qwen25_7b_qlora_seq12288}"

accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 scripts/train_ap_direct_sft_lora.py \
  --model-name-or-path "${MODEL_NAME}" \
  --train-file "${DATA_DIR}/train.jsonl" \
  --val-file "${DATA_DIR}/val.jsonl" \
  --output-dir "${OUT_DIR}" \
  --max-seq-len 12288 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 2 \
  --learning-rate 2e-4 \
  --warmup-ratio 0.03 \
  --logging-steps 10 \
  --eval-steps 100 \
  --save-steps 100 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --qlora-4bit \
  --gradient-checkpointing \
  --precision bf16 \
  --seed 13
