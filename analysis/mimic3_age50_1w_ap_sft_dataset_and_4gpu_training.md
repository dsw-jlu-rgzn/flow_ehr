# MIMIC-III Age>=50 1W A&P Direct SFT Dataset and 4-GPU Training Plan

## 1. 目标

本轮生成的是一个最简单的 direct A&P SFT baseline 数据集，用来评估“不给 scaffold/verifier/reviser，模型只看当日 EHR input，直接输出当日 gold A&P”的上限。

本数据集不是最终 multi-agent 训练集，也不是 verifier JSON 训练集。它的目标是先回答：

- 普通 finetune 能否学到 MIMIC-III ICU A&P 的格式；
- direct SFT 在同分布 held-out admission 上的上限大概在哪里；
- 后续 scaffold/verifier/reviser 是否仍然有必要。

## 2. Cohort 筛选口径

使用 MIMIC-III 原始数据，筛选条件如下：

| 条件 | 设置 |
|---|---|
| 数据源 | MIMIC-III |
| 年龄 | `age >= 50` |
| ICU LOS | `LOS > 3 days` |
| 死亡病例 | include deceased |
| A&P gold 来源 | `NOTEEVENTS` 中 `CATEGORY == Physician ` 且 `DESCRIPTION` 包含 `Progress Note` |
| SFT 样本 | 每个 admission 排除第一个 gold A&P day |
| 输入 | 当日 structured EHR rows，默认不包含 note rows |
| 输出 | 当日 gold progress note 中抽取的 A&P section |

排除第一个 gold day 的原因是为了更接近之前 base/V2 method2 的 longitudinal 设置：后续如果加入 previous generated A&P，需要从第二个 gold day 开始才有历史 A&P。

## 3. 生成命令

### 3.1 过滤 MIMIC-III 原始表

```powershell
python processing\prepare_mimic3_tasks.py `
  --raw-dir "C:\Users\dsw54\Desktop\MIMIC_related\mimic-iii-20260513T124356Z-3-001\mimic-iii" `
  --output-root data_mimic3_age50_1w `
  --sample-size 999999 `
  --min-age 50 `
  --min-los-days 3 `
  --include-deceased `
  --require-ap-progress-notes `
  --chunksize 500000
```

实际筛选结果：

| 项目 | 数量 |
|---|---:|
| selected admissions | 1,627 |
| filtered INPUTEVENTS_CV rows | 33,987 |
| filtered INPUTEVENTS_MV rows | 620,115 |
| filtered LABEVENTS rows | 1,249,745 |
| filtered CHARTEVENTS rows | 18,204,476 |
| filtered PRESCRIPTIONS rows | 227,257 |
| filtered NOTEEVENTS rows | 226,252 |

### 3.2 生成 AP input/gold

原始 `processing/get_chronologies_AP.py` 中途退出后，新增了一个可续跑 wrapper：

[scripts/generate_ap_chronologies_safe.py](C:/Users/dsw54/Desktop/codex_related/flow_ehr/scripts/generate_ap_chronologies_safe.py)

使用命令：

```powershell
python scripts\generate_ap_chronologies_safe.py `
  --target-path data_mimic3_age50_1w\target_population\filtered `
  --mimic-dir data_mimic3_age50_1w\MIMIC-III `
  --output-dir data_mimic3_age50_1w\AP\input `
  --gt-dir data_mimic3_age50_1w\AP\gold `
  --failure-log data_mimic3_age50_1w\AP\ap_chronology_failures.jsonl
```

实际结果：

| 项目 | 数量 |
|---|---:|
| AP input admissions | 1,627 |
| AP gold admissions | 1,627 |
| failures | 0 |
| AP input size | 590.44 MB |
| AP gold size | 81.67 MB |

### 3.3 生成 SFT JSONL

```powershell
python scripts\prepare_ap_direct_sft_dataset.py `
  --data-root data_mimic3_age50_1w\AP `
  --out-dir outputs\ap_direct_sft_mimic3_age50_1w_exclude_first `
  --target-mode ap_section `
  --exclude-first-gold-day `
  --max-input-chars 24000 `
  --min-target-chars 80 `
  --seed 13 `
  --val-ratio 0.1 `
  --test-ratio 0.1
```

输出路径：

| 文件 | 用途 |
|---|---|
| `outputs/ap_direct_sft_mimic3_age50_1w_exclude_first/train.jsonl` | train split |
| `outputs/ap_direct_sft_mimic3_age50_1w_exclude_first/val.jsonl` | validation split |
| `outputs/ap_direct_sft_mimic3_age50_1w_exclude_first/test.jsonl` | held-out test split |
| `outputs/ap_direct_sft_mimic3_age50_1w_exclude_first/manifest.json` | 数据生成参数 |
| `outputs/ap_direct_sft_mimic3_age50_1w_exclude_first/stats/dataset_summary.json` | 统计汇总 |
| `outputs/ap_direct_sft_mimic3_age50_1w_exclude_first/stats/sample_length_stats.csv` | 每条样本长度 |
| `outputs/ap_direct_sft_mimic3_age50_1w_exclude_first/stats/cohort_admission_metadata.csv` | admission 元数据 |
| `outputs/ap_direct_sft_mimic3_age50_1w_exclude_first/stats/top_50_icd9_codes.csv` | ICD-9 top 诊断 |
| `outputs/ap_direct_sft_mimic3_age50_1w_exclude_first/stats/diagnosis_category_counts.csv` | 诊断大类统计 |

## 4. SFT 数据格式

每条 JSONL 是 chat messages：

```json
{
  "id": "100123_day4",
  "messages": [
    {
      "role": "system",
      "content": "You are an experienced ICU clinician..."
    },
    {
      "role": "user",
      "content": "Current-day EHR input:\n```text\n...\n```\n\nWrite the current day's ICU Assessment and Plan."
    },
    {
      "role": "assistant",
      "content": "Assessment and Plan\n..."
    }
  ],
  "metadata": {
    "admission_id": "100123",
    "day": 4,
    "task": "ap_direct_input_to_gold"
  }
}
```

训练时只对 assistant 部分计算 loss，system/user prompt token 的 label 被 mask 为 `-100`。

## 5. 数据规模

| split | examples | admissions |
|---|---:|---:|
| train | 8,440 | 1,228 |
| val | 1,192 | 153 |
| test | 881 | 153 |
| total | 10,513 | 1,534 |

注意：原始筛选 admission 是 1,627 个；SFT 中保留 1,534 个 admission，因为排除首个 gold day 后，只有至少 2 个有效 gold A&P day 的 admission 才能贡献训练样本。

SFT JSONL 文件总大小约 `122.53 MB`。

## 6. 输入输出长度统计

token 统计使用本地缓存的 `Qwen/Qwen2.5-0.5B-Instruct` tokenizer；Qwen2.5-7B-Instruct 同系列 tokenizer 口径基本一致，Qwen3-8B 需要训练前再复核一次。

### 6.1 字符长度

| 字段 | mean | P50 | P75 | P90 | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| prompt chars | 9,243 | 8,540 | 11,970 | 15,610 | 18,320 | 24,099 | 24,100 |
| target chars | 2,414 | 1,974 | 3,268 | 4,644 | 5,556 | 7,476 | 12,948 |
| full chars | 11,657 | 10,955 | 14,789 | 18,618 | 21,293 | 26,356 | 36,020 |
| target words | 343 | 276 | 474 | 677 | 806 | 1,071 | 1,655 |

`--max-input-chars=24000` 下，约 `110 / 10513 = 1.05%` 的 prompt 触发了字符截断。

### 6.2 Token 长度

| 字段 | mean | P50 | P75 | P90 | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| prompt tokens | 3,578 | 3,288 | 4,601 | 6,077 | 7,052 | 8,753 | 10,534 |
| target tokens | 745 | 605 | 998 | 1,395 | 1,693 | 2,359 | 4,224 |
| full tokens | 4,325 | 4,034 | 5,440 | 6,979 | 8,023 | 9,763 | 13,112 |

训练长度建议：

| `max_seq_len` | 覆盖情况 | 建议用途 |
|---:|---|---|
| 8,192 | 覆盖约 95% full sequence | 第一版主实验，成本低 |
| 12,288 | 覆盖超过 99%，但仍可能截断最长样本 | 长上下文主实验 |
| 14,336 或 16,384 | 基本覆盖全部当前样本 | 长上下文上限实验 |

当前训练脚本在超过 `max_seq_len` 时保留 prompt 尾部和 assistant target 前部，并且仍然只对 assistant token 计算 loss。

## 7. 住院与人群统计

### 7.1 年龄

MIMIC-III 对 89 岁以上患者做日期脱敏，因此 raw age 会出现 300 岁左右。报告时建议使用 capped age，即 `min(age, 89)`。

| 年龄口径 | mean | P50 | P75 | P90 | P95 | max |
|---|---:|---:|---:|---:|---:|---:|
| raw age | 84.84 | 69.5 | 80.0 | 87.0 | 300.0 | 307.0 |
| capped age 89 | 70.07 | 69.5 | 80.0 | 87.0 | 89.0 | 89.0 |

### 7.2 ICU LOS

| 指标 | days |
|---|---:|
| mean | 8.25 |
| P50 | 5.36 |
| P75 | 9.62 |
| P90 | 16.57 |
| P95 | 22.79 |
| P99 | 37.58 |
| max | 60.94 |

### 7.3 Gold A&P days per admission

| 指标 | gold days | trainable days after excluding first |
|---|---:|---:|
| mean | 7.86 | 6.86 |
| P50 | 5 | 4 |
| P75 | 9 | 8 |
| P90 | 16 | 15 |
| P95 | 22 | 21 |
| P99 | 40 | 39 |
| max | 82 | 81 |

### 7.4 其他 cohort 信息

| 字段 | 结果 |
|---|---|
| hospital mortality | 304 / 1534 = 19.82% |
| gender | M 832, F 702 |
| admission type | Emergency 1452, Elective 56, Urgent 26 |
| top ethnicity | White 1198, Black/African American 145, Unable to obtain 39 |

## 8. 诊断分布

### 8.1 诊断大类

| category | admissions | percent |
|---|---:|---:|
| renal failure / CKD | 950 | 61.93% |
| respiratory failure | 806 | 52.54% |
| arrhythmia | 794 | 51.76% |
| heart failure | 682 | 44.46% |
| ischemic heart disease | 578 | 37.68% |
| diabetes | 562 | 36.64% |
| pneumonia | 481 | 31.36% |
| sepsis / shock | 469 | 30.57% |
| COPD | 335 | 21.84% |
| malignancy | 309 | 20.14% |
| liver disease | 244 | 15.91% |
| stroke / cerebrovascular | 177 | 11.54% |

这些类别是基于 ICD-9 code 前缀做的粗粒度统计，不是严格 Elixhauser/Charlson comorbidity mapping。正式论文统计可以后续换成标准 comorbidity package。

### 8.2 Top ICD-9 codes

| ICD-9 | short title | admissions |
|---|---|---:|
| 51881 | Acute respiratory failure | 710 |
| 4280 | CHF NOS | 672 |
| 5849 | Acute kidney failure NOS | 621 |
| 42731 | Atrial fibrillation | 593 |
| 4019 | Hypertension NOS | 547 |
| 2724 | Hyperlipidemia NEC/NOS | 398 |
| 99592 | Severe sepsis | 383 |
| 41401 | Coronary atherosclerosis native vessel | 354 |
| 25000 | Type II diabetes without complication | 348 |
| 5990 | UTI NOS | 337 |
| 486 | Pneumonia organism NOS | 300 |
| 2762 | Acidosis | 290 |
| 78552 | Septic shock | 277 |
| 5859 | CKD NOS | 273 |
| 40390 | Hypertensive CKD | 270 |
| 0389 | Septicemia NOS | 263 |
| 2859 | Anemia NOS | 259 |
| 5070 | Aspiration pneumonitis | 252 |
| 53081 | Esophageal reflux | 240 |
| 2760 | Hypernatremia / hyperosmolality | 240 |

## 9. 训练目标与 loss

训练方式是标准 causal language modeling SFT：

```text
input sequence = system + user + assistant
labels         = -100 for system/user tokens, assistant tokens for output
loss           = cross entropy over assistant tokens only
```

因此模型学习的是：

```text
Current-day EHR input -> current-day gold Assessment and Plan
```

不包含：

- previous generated A&P；
- previous gold A&P；
- scaffold；
- verifier JSON；
- revised A&P；
- LLM judge labels。

这是最干净的 direct baseline。

## 10. 单机 4 卡训练配置

假设 Linux 单机 4 卡，推荐优先用 QLoRA，尤其是 `seq_len >= 8192` 时。

### 10.1 环境变量

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 10.2 Accelerate 配置

可以先交互式配置：

```bash
accelerate config
```

建议选择：

```text
compute environment: This machine
distributed type: multi-GPU
num processes: 4
mixed precision: bf16
use DeepSpeed: no
```

如果显存紧张，再切换 DeepSpeed ZeRO-2/ZeRO-3；但第一版 QLoRA 通常不需要。

### 10.3 Qwen2.5-7B-Instruct 8192 主实验

```bash
accelerate launch --num_processes 4 scripts/train_ap_direct_sft_lora.py \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --train-file outputs/ap_direct_sft_mimic3_age50_1w_exclude_first/train.jsonl \
  --val-file outputs/ap_direct_sft_mimic3_age50_1w_exclude_first/val.jsonl \
  --output-dir outputs/ap_direct_sft_mimic3_age50_1w_qwen25_7b_qlora_seq8192 \
  --max-seq-len 8192 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --num-train-epochs 3 \
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
```

effective batch size:

```text
4 GPUs * per_device_batch 1 * grad_accum 4 = 16
```

steps:

```text
8440 train examples / 16 ≈ 528 optimizer steps per epoch
3 epochs ≈ 1583 optimizer steps
```

### 10.4 Qwen3-8B 8192 对照实验

```bash
accelerate launch --num_processes 4 scripts/train_ap_direct_sft_lora.py \
  --model-name-or-path Qwen/Qwen3-8B \
  --train-file outputs/ap_direct_sft_mimic3_age50_1w_exclude_first/train.jsonl \
  --val-file outputs/ap_direct_sft_mimic3_age50_1w_exclude_first/val.jsonl \
  --output-dir outputs/ap_direct_sft_mimic3_age50_1w_qwen3_8b_qlora_seq8192 \
  --max-seq-len 8192 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --num-train-epochs 3 \
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
```

Qwen3-8B 的 chat template 与 Qwen2.5 可能不同，训练前建议用同一 tokenizer 重新统计一次 token 长度。

### 10.5 长上下文实验

如果 4 卡显存充足，可以跑 `seq_len=12288`：

```bash
accelerate launch --num_processes 4 scripts/train_ap_direct_sft_lora.py \
  --model-name-or-path Qwen/Qwen2.5-7B-Instruct \
  --train-file outputs/ap_direct_sft_mimic3_age50_1w_exclude_first/train.jsonl \
  --val-file outputs/ap_direct_sft_mimic3_age50_1w_exclude_first/val.jsonl \
  --output-dir outputs/ap_direct_sft_mimic3_age50_1w_qwen25_7b_qlora_seq12288 \
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
```

effective batch size:

```text
4 * 1 * 8 = 32
```

如果显存仍然足够，再尝试 `max_seq_len=14336` 或 `16384`，但第一版没有必要直接上 16k。

## 11. 推荐实验矩阵

| 实验 | 模型 | seq_len | epoch | effective batch | 目的 |
|---|---|---:|---:|---:|---|
| D1 | Qwen2.5-7B-Instruct | 8192 | 3 | 16 | 主 baseline |
| D2 | Qwen3-8B | 8192 | 3 | 16 | 模型对照 |
| D3 | Qwen2.5-7B-Instruct | 12288 | 2 | 32 | 长上下文收益 |
| D4 | Qwen2.5-7B-Instruct | 8192 | 1 | 16 | 快速 sanity |
| D5 | Qwen2.5-7B-Instruct | 14336/16384 | 1-2 | 16-32 | 长上下文上限 |

第一轮建议先跑 D1 和 D2。若 D1/D2 validation loss 正常下降，再跑 D3。

## 12. 预期验证指标

训练脚本当前会输出：

- train loss；
- eval loss；
- eval runtime；
- eval samples/s。

后续建议补充生成式评估：

1. 在 test split 上生成 A&P；
2. 与 gold A&P 做 ROUGE-L / BERTScore / UMLS CUI-F1；
3. 抽样做 LLM pairwise judge：base direct SFT vs old base vs V2 vs V2+judge；
4. 单独看 late admission days，因为之前 base 的 trajectory drift 主要发生在后期。

## 13. 重要 caveat

这个 1W direct SFT 数据集不含 previous generated A&P，因此它和之前 API base method2 的输入不完全一致。

当前输入是：

```text
current-day structured EHR -> current-day gold A&P
```

之前 method2/base 输入更接近：

```text
current-day structured EHR + cumulative previous generated A&P -> current-day A&P
```

所以它适合做 direct generation baseline。如果后续要训练与 V2/base 完全同口径的模型，需要再生成一个 history-aware SFT 数据集，把 previous generated A&P 或 previous gold A&P 加入 user prompt。

