# A&P Direct SFT Training Tutorial for 4x40GB GPUs

## Purpose

This tutorial trains the direct A&P SFT baseline:

```text
current-day EHR input -> current-day gold Assessment and Plan
```

It does not train the scaffold generator, verifier, judge, or reviser. The goal is to measure how far a plain finetuned model can go before adding the multi-agent components.

## Files

Training code:

- `scripts/train_ap_direct_sft_lora.py`
- `scripts/prepare_ap_direct_sft_dataset.py`
- `scripts/generate_ap_chronologies_safe.py`

4-GPU launch scripts:

- `scripts/run_ap_sft_smoke_4gpu.sh`
- `scripts/run_ap_sft_qwen25_7b_4gpu_seq8192.sh`
- `scripts/run_ap_sft_qwen3_8b_4gpu_seq8192.sh`
- `scripts/run_ap_sft_qwen25_7b_4gpu_seq12288.sh`

Config and dependencies:

- `configs/accelerate_4gpu_bf16.yaml`
- `requirements-ap-sft.txt`

Dataset report:

- `analysis/mimic3_age50_1w_ap_sft_dataset_and_4gpu_training.md`

## Dataset

The expected dataset directory is:

```text
outputs/ap_direct_sft_mimic3_age50_1w_exclude_first/
  train.jsonl
  val.jsonl
  test.jsonl
  manifest.json
  stats/
```

The current generated dataset has:

| Split | Examples | Admissions |
|---|---:|---:|
| train | 8,440 | 1,228 |
| val | 1,192 | 153 |
| test | 881 | 153 |
| total | 10,513 | 1,534 |

Selection:

```text
MIMIC-III
age >= 50
ICU LOS > 3 days
include deceased
require Physician Progress Note
exclude first gold A&P day per admission
```

Do not upload MIMIC-derived JSONL, raw CSV, filtered tables, or AP input/gold files to public GitHub. Keep them on the training machine or in approved private storage.

## Input and Output

Each JSONL row is a chat sample:

```json
{
  "id": "100123_day4",
  "messages": [
    {"role": "system", "content": "You are an experienced ICU clinician..."},
    {"role": "user", "content": "Current-day EHR input:\n```text\n...\n```\n\nWrite the current day's ICU Assessment and Plan."},
    {"role": "assistant", "content": "Assessment and Plan\n..."}
  ],
  "metadata": {
    "admission_id": "100123",
    "day": 4,
    "task": "ap_direct_input_to_gold"
  }
}
```

The training loss is standard causal language modeling cross entropy over assistant tokens only. System and user prompt tokens are masked with `-100`.

## Sequence Length

Token statistics with a Qwen2.5 tokenizer:

| Field | Mean | P50 | P75 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| prompt tokens | 3,578 | 3,288 | 4,601 | 6,077 | 7,052 | 8,753 | 10,534 |
| target tokens | 745 | 605 | 998 | 1,395 | 1,693 | 2,359 | 4,224 |
| full tokens | 4,325 | 4,034 | 5,440 | 6,979 | 8,023 | 9,763 | 13,112 |

Recommended settings:

| `max_seq_len` | Use |
|---:|---|
| 1024 | smoke test only |
| 8192 | first main experiment |
| 12288 | long-context ablation |
| 14336 or 16384 | upper-bound long-context run |

For 4 GPUs with 40GB VRAM each, start with QLoRA, bf16, and `max_seq_len=8192`.

## Environment

Use Python 3.10 or 3.11.

```bash
conda create -n ap-sft python=3.10 -y
conda activate ap-sft
pip install -U pip
pip install -r requirements-ap-sft.txt
```

If the machine needs a custom PyTorch/CUDA wheel, install PyTorch first, then install the remaining packages:

```bash
pip install transformers accelerate peft bitsandbytes datasets pandas tqdm sentencepiece safetensors
```

Check GPUs:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(i)
    print(i, torch.cuda.get_device_name(i), round(prop.total_memory / 1024**3, 1), "GB")
PY
```

## Model Preparation

If the training machine can access HuggingFace:

```bash
huggingface-cli login
```

If it cannot, download the model in advance and point `MODEL_NAME` to the local path:

```bash
MODEL_NAME=/models/Qwen2.5-7B-Instruct bash scripts/run_ap_sft_qwen25_7b_4gpu_seq8192.sh
```

## Runtime Environment Variables

The launch scripts set:

```bash
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

You can override GPU selection:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/run_ap_sft_qwen25_7b_4gpu_seq8192.sh
```

## Smoke Test

Run a tiny job first:

```bash
bash scripts/run_ap_sft_smoke_4gpu.sh
```

This uses Qwen2.5-0.5B, `max_seq_len=1024`, 32 train samples, 16 eval samples, and 8 steps. It only verifies that data loading, chat templating, loss masking, evaluation, and multi-GPU launch all work.

## Main Experiment: Qwen2.5-7B

```bash
bash scripts/run_ap_sft_qwen25_7b_4gpu_seq8192.sh
```

Default parameters:

```text
model: Qwen/Qwen2.5-7B-Instruct
training: QLoRA
precision: bf16
max_seq_len: 8192
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 4
num_gpus: 4
effective_batch_size: 16
epochs: 3
learning_rate: 2e-4
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
eval_steps: 100
save_steps: 100
```

Estimated optimizer steps:

```text
8440 train examples / 16 effective batch ~= 528 steps per epoch
3 epochs ~= 1583 optimizer steps
```

## Model Comparison: Qwen3-8B

```bash
bash scripts/run_ap_sft_qwen3_8b_4gpu_seq8192.sh
```

This uses the same training settings as Qwen2.5-7B. Qwen3 may have a different chat template, so rerun token statistics before final reporting if Qwen3 becomes the main model.

## Long-Context Experiment

```bash
bash scripts/run_ap_sft_qwen25_7b_4gpu_seq12288.sh
```

Default differences:

```text
max_seq_len: 12288
gradient_accumulation_steps: 8
effective_batch_size: 32
epochs: 2
```

This covers more than 99% of the current full sequences under the Qwen2.5 tokenizer.

## Overriding Paths

All launch scripts support environment overrides:

```bash
MODEL_NAME=/models/Qwen2.5-7B-Instruct \
DATA_DIR=/data/ap_direct_sft_mimic3_age50_1w_exclude_first \
OUT_DIR=/runs/ap_sft/qwen25_7b_seq8192 \
bash scripts/run_ap_sft_qwen25_7b_4gpu_seq8192.sh
```

## Expected Outputs

The output directory contains LoRA adapter checkpoints:

```text
outputs/ap_direct_sft_mimic3_age50_1w_qwen25_7b_qlora_seq8192/
  adapter_config.json
  adapter_model.safetensors
  checkpoint-*/
  trainer_state.json
  training_args.bin
```

The script saves adapters by default. Add `--merge-and-save` if you need a merged model, but the merged model is much larger.

## Troubleshooting

If CUDA OOM occurs, try in this order:

```text
1. keep per_device_train_batch_size = 1
2. lower max_seq_len from 8192 to 6144
3. lower lora_r from 16 to 8
4. keep QLoRA enabled
5. increase gradient_accumulation_steps instead of increasing batch size
```

If bitsandbytes fails:

```bash
python -m bitsandbytes
```

Then reinstall a version compatible with the local CUDA stack.

## Recommended First Run Order

```text
1. scripts/run_ap_sft_smoke_4gpu.sh
2. scripts/run_ap_sft_qwen25_7b_4gpu_seq8192.sh
3. scripts/run_ap_sft_qwen3_8b_4gpu_seq8192.sh
4. scripts/run_ap_sft_qwen25_7b_4gpu_seq12288.sh
```

After the first main run, evaluate generated test-set A&P with ROUGE-L, UMLS CUI-F1, and LLM pairwise judge against the old API base, V2, and V2+judge outputs.
