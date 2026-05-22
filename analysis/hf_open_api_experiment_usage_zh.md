# HuggingFace 生成 + HuggingFace 小模型评估实验脚本说明

生成日期：2026-05-21

## 1. 设计目标

后续实验将原先直接调用 DeepSeek / Qwen 等 API 的生成部分，替换为 HuggingFace 侧的开源模型调用；评估部分也采用 HuggingFace 中另一个较小的开源模型，用于 LLM-as-judge。

为避免泄露密钥，所有脚本只从环境变量读取 token，不把 token 写入代码或输出文件。

## 2. 新增文件

```text
modeling/hf_generation.py
scripts/run_ap_hf_experiment.py
scripts/run_ds_hf_experiment.py
scripts/run_hf_llm_evaluation.py
scripts/run_open_llm_evaluation.py
```

## 3. 环境变量

PowerShell 中设置 HuggingFace token：

```powershell
$env:HF_TOKEN = "your_huggingface_token"
```

评估同样默认读取 `HF_TOKEN`。如果希望生成和评估使用不同 HuggingFace token，可以在评估脚本里传入 `--hf-token-env HF_EVAL_TOKEN`，并设置 `$env:HF_EVAL_TOKEN`。

## 4. HuggingFace backend

脚本支持三种 backend：

| backend | 用途 |
|---|---|
| router | 使用 HuggingFace Inference Router，默认接口为 `https://router.huggingface.co/v1/chat/completions` |
| local | 使用本地 `transformers` 加载 HuggingFace 模型 |
| mock | 不调用模型，只验证数据、prompt、输出路径和脚本连通性 |

正式实验建议先用 `mock` 检查路径，再用 `router` 或 `local` 跑小样本。

## 5. A&P 实验示例

先做 smoke test：

```powershell
python scripts/run_ap_hf_experiment.py `
  --hf-backend mock `
  --config-name hf_ap_smoke `
  --data-root data_ap100_ap `
  --output-dir outputs/ap_hf_smoke `
  --baseline-run-name deepseek_api_full_gen `
  --baseline-setting gen `
  --baseline-method method2 `
  --memory-source baseline_method `
  --prompt-version v2 `
  --cases 105351:13
```

正式 HuggingFace Router 运行示例：

```powershell
python scripts/run_ap_hf_experiment.py `
  --hf-backend router `
  --model Qwen/Qwen2.5-72B-Instruct `
  --config-name hf_qwen_ap_v2 `
  --data-root data_ap100_ap `
  --output-dir outputs/ap_hf_qwen_v2 `
  --baseline-run-name deepseek_api_full_gen `
  --baseline-setting gen `
  --baseline-method method2 `
  --memory-source baseline_method `
  --prompt-version v2 `
  --use-judge-revise `
  --cases-file outputs/ap_memory_gated_scaffold_ap100/ap100_cases_120.txt
```

## 6. DS 实验示例

DS 阶段需要按顺序运行：

```powershell
python scripts/run_ds_hf_experiment.py minimal `
  --hf-backend mock `
  --output-dir outputs/ds_hf_smoke `
  --limit 1
```

正式运行时将 `mock` 替换为 `router`，并指定 HuggingFace 模型：

```powershell
python scripts/run_ds_hf_experiment.py minimal `
  --hf-backend router `
  --model Qwen/Qwen2.5-72B-Instruct `
  --output-dir outputs/ds_hf_qwen_minimal `
  --limit 10

python scripts/run_ds_hf_experiment.py variants `
  --hf-backend router `
  --model Qwen/Qwen2.5-72B-Instruct `
  --source-run outputs/ds_hf_qwen_minimal `
  --output-dir outputs/ds_hf_qwen_variants `
  --limit 10

python scripts/run_ds_hf_experiment.py dx2 `
  --hf-backend router `
  --model Qwen/Qwen2.5-72B-Instruct `
  --source-run outputs/ds_hf_qwen_minimal `
  --variant-run outputs/ds_hf_qwen_variants `
  --limit 10

python scripts/run_ds_hf_experiment.py dx3 `
  --hf-backend router `
  --model Qwen/Qwen2.5-72B-Instruct `
  --source-run outputs/ds_hf_qwen_minimal `
  --variant-run outputs/ds_hf_qwen_variants `
  --output-dir outputs/ds_hf_qwen_variants/method_outputs/ours2_v4_dx3_agent_diagnosis `
  --limit 10
```

## 7. HuggingFace 小模型 LLM-as-judge 评估示例

建议生成模型和评估模型分开。例如生成用较强模型，评估用较小 instruction model：

| 用途 | 示例模型 |
|---|---|
| AP/DS generation | `Qwen/Qwen2.5-72B-Instruct` |
| LLM judge evaluation | `Qwen/Qwen2.5-7B-Instruct` 或 `meta-llama/Llama-3.1-8B-Instruct` |

A&P judge：

```powershell
python scripts/run_hf_llm_evaluation.py ap `
  --hf-backend router `
  --eval-model Qwen/Qwen2.5-7B-Instruct `
  --detail-csv outputs/ap_hf_qwen_v2/hf_qwen_ap_v2_summary.csv `
  --augmented-dir outputs/ap_hf_qwen_v2 `
  --output-csv outputs/ap_hf_qwen_v2/hf_qwen_ap_v2_judge.csv
```

DS judge：

```powershell
python scripts/run_hf_llm_evaluation.py ds `
  --hf-backend router `
  --eval-model Qwen/Qwen2.5-7B-Instruct `
  --method-a-dir outputs/ds_hf_qwen_minimal/method_outputs/base1_full_context_direct `
  --method-b-dir outputs/ds_hf_qwen_variants/method_outputs/ours2_v4_dx3_agent_diagnosis `
  --method-a-name base `
  --method-b-name hf_ours_dx3 `
  --output-csv outputs/ds_hf_qwen_variants/llm_judge_base_vs_hf_dx3.csv `
  --limit 10
```

## 8. 注意事项

1. HuggingFace Router 是 OpenAI-compatible 接口，但可用模型和计费/provider 状态会变化，正式跑大规模实验前应先跑 1-2 个 case。
2. `local` backend 会直接加载模型，显存不足时可能失败；大模型建议使用 Router、TGI、vLLM 或其他 OpenAI-compatible serving。
3. AP/DS 生成和 LLM judge 应使用不同模型，避免 evaluator 与 generator 完全相同造成偏置。
4. 正式论文实验中需要记录生成模型、评估模型、endpoint、temperature、max tokens、case selection 和失败重试策略。
