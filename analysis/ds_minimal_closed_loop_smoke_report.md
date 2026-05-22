# DS 10-Case Minimal Closed-Loop Smoke Test

生成日期：2026-05-20

## 1. Full DS 数据量

当前用于主实验的 full DS 数据版本为：

```text
data/DS_fixed_composed/full/input
data/DS_fixed_composed/full/gold
```

数量：

| 数据 | 数量 |
|---|---:|
| full DS input CSV | 100 |
| full DS gold TXT | 100 |

结论：full DS 主实验当前有 **100 条 admission-level 数据**。

## 2. 10 条最小闭环选择策略

最小闭环先选择 **10 条最短 full DS input**，原因是：

1. 先验证 pipeline 是否能从读取数据、chunking、state update、scaffold、judge、revise 到输出评估完整跑通。
2. 这些样本长度较短，Base 1 可以尽量保持 full-context direct，不容易被 context window 截断问题干扰。
3. 后续正式实验再扩展到 100 条，并额外做 long-admission / long-context 分层。

本次选中的 10 条：

| HADM_ID | rows | input words | chunks |
|---:|---:|---:|---:|
| 191485 | 35 | 1390 | 1 |
| 138003 | 50 | 2055 | 1 |
| 115021 | 42 | 2086 | 1 |
| 196447 | 34 | 2087 | 1 |
| 134314 | 34 | 2123 | 1 |
| 137302 | 41 | 2836 | 1 |
| 121709 | 56 | 3201 | 1 |
| 139692 | 116 | 3699 | 2 |
| 168849 | 92 | 4288 | 2 |
| 140351 | 116 | 4622 | 2 |

## 3. 已实现脚本

新增脚本：

```text
scripts/run_ds_minimal_closed_loop.py
```

该脚本支持三组方法：

| 方法 | 输出目录 |
|---|---|
| Base 1: Full-Context Direct | `outputs/ds_minimal_closed_loop_10/method_outputs/base1_full_context_direct` |
| Ours 1: Sequential State + Scaffold | `outputs/ds_minimal_closed_loop_10/method_outputs/ours1_state_scaffold` |
| Ours 2: Sequential State + Scaffold + Global Judge-Revise | `outputs/ds_minimal_closed_loop_10/method_outputs/ours2_global_judge_revise` |

每个 case 还会保存中间文件：

```text
outputs/ds_minimal_closed_loop_10/cases/<HADM_ID>/
```

包括：

- `base1_full_context_direct.txt`
- `state_after_chunk_*.json`
- `final_discharge_state.json`
- `ds_scaffold.json`
- `ours1_state_scaffold.txt`
- `ours2_global_judge.json`
- `ours2_global_judge_revise.txt`
- `prompts/*.md`

## 4. Dry-run 验证结果

已运行 dry-run：

```bash
python scripts/run_ds_minimal_closed_loop.py --limit 10 --case-selection shortest --dry-run
```

结果：

- 10/10 case 完成闭环。
- 输出 summary：

```text
outputs/ds_minimal_closed_loop_10/summary.csv
outputs/ds_minimal_closed_loop_10/summary.json
```

- 标准 method output 目录已生成，文件名兼容 `evaluate_ds_light.py`：

```text
48h_all_abs_<HADM_ID>.txt
```

已验证 evaluator 可以读取输出目录：

```bash
python evaluation/evaluate_ds_light.py \
  --gen-dir outputs/ds_minimal_closed_loop_10/method_outputs/base1_full_context_direct \
  --gt-dir data/DS_fixed_composed/full/gold \
  --output-csv outputs/ds_minimal_closed_loop_10/eval_base1_dryrun.csv
```

dry-run 输出是 placeholder，因此 ROUGE 数值无意义；该步骤只用于确认评估入口能正常读取。

## 4.1 实际 DeepSeek 10-case 运行结果

已使用 DeepSeek API 完成 10 条真实生成。输出路径：

```text
outputs/ds_minimal_closed_loop_10
```

生成文件：

| 方法 | 输出路径 |
|---|---|
| Base 1 | `outputs/ds_minimal_closed_loop_10/method_outputs/base1_full_context_direct` |
| Ours 1 | `outputs/ds_minimal_closed_loop_10/method_outputs/ours1_state_scaffold` |
| Ours 2 | `outputs/ds_minimal_closed_loop_10/method_outputs/ours2_global_judge_revise` |

轻量 ROUGE-L 结果：

| 方法 | Diagnosis | Hospital Course | Discharge Instructions |
|---|---:|---:|---:|
| Base 1 | 9.95 | 12.76 | 8.80 |
| Ours 1 | 9.00 | 11.34 | 3.69 |
| Ours 2 | 8.60 | 11.38 | 3.86 |

详细文件：

```text
outputs/ds_minimal_closed_loop_10/eval_base1.csv
outputs/ds_minimal_closed_loop_10/eval_ours1.csv
outputs/ds_minimal_closed_loop_10/eval_ours2.csv
outputs/ds_minimal_closed_loop_10/eval_summary.csv
```

解释：

1. 当前 10 条短样本上，Base 1 的 ROUGE-L 高于 Ours 1/Ours 2，尤其是 Discharge Instructions。
2. 这不一定说明 Base 1 临床质量更好。抽查显示 Base 1 更容易补全或推断出院去向、随访和药物，因此更接近 gold discharge summary 的表面形式；Ours 1/Ours 2 更保守，遇到输入中没有明确出院计划时会写 `[Not documented]`，ROUGE 因此下降。
3. 对 DS 任务，仅靠 ROUGE-L 不足以判断方法优劣。下一步必须加入 LLM judge 或人工检查 faithfulness / unsupported / missed major events。

## 4.2 实际运行发现的问题

### 问题 1：Section evaluator 标题规则需要适配新输出

初次评估三组 ROUGE 都为 0，原因是 `evaluate_ds_light.py` 只识别旧标题：

```text
Part 1: Diagnosis
Part 2: Hospital Course Summary
Part 3: Discharge Instructions
```

但当前 DS 输出使用：

```text
**1. Discharge Diagnosis**
**3. Brief Hospital Course**
**5. Discharge Medications and Follow-up Instructions**
```

已修复：

```text
evaluation/evaluate_ds_light.py
```

### 问题 2：JSON 稳定性仍需加强

实际运行中，20 个关键 JSON 文件里有 2 个不合法：

| HADM_ID | 文件 | 问题 |
|---:|---|---|
| 139692 | `final_discharge_state.json` | JSON 逗号/字符串格式错误 |
| 139692 | `ours2_global_judge.json` | JSON 字符串未闭合 |

这说明多 chunk case 中 state/judge 输出仍可能不稳定。已在脚本中收紧 prompt：

- 要求 compact valid JSON；
- 禁止 markdown fences；
- 每个 array 最多 8 个 concise strings；
- 要求转义内部 quotation marks。

后续如果继续出现 JSON 错误，应加入自动 JSON repair 或 retry。

### 问题 3：Ours 当前过度保守

以 HADM_ID 191485 为例：

- Base 1 会输出出院回家、cardiology/PCP follow-up 等内容；
- Ours 1/Ours 2 会把 discharge condition、disposition、follow-up 标为 not documented；
- gold 中确实有完整 discharge summary，因此 Ours 的保守策略会损失 ROUGE。

这提示 DS 方法需要区分两类目标：

1. **Faithfulness-oriented DS**：没有证据就不写，unsupported 更低；
2. **Gold-note mimicry DS**：需要从 clinical conventions 中补全 discharge summary 风格内容，ROUGE 可能更高但 hallucination 风险更大。

论文实验中应明确主目标是 faithfulness/completeness，而不是单纯 ROUGE。

## 4.3 指标打平的优化方案

当前三组结果还不能直接作为方法优劣结论，因为 Base / Ours 1 / Ours 2 的输出空间没有完全对齐。优先目标应是先把 **format、section、长度、证据使用策略、评估入口** 打平，然后再比较方法本身。

### 优化 1：统一最终 DS 输出 schema

原始 10-case 输出中：

- Base 1 基本输出标准 discharge summary；
- Ours 1/Ours 2 会额外输出 `Resolved Problem Summary`、`Unresolved Problem Summary`、`Temporal Course Outline` 等 scaffold 派生章节；
- `evaluate_ds_light.py` 会把这些额外章节吃进 `Discharge Instructions`，导致 Ours 的 instruction ROUGE 被非 instruction 内容稀释。

已修改：

```text
scripts/run_ds_minimal_closed_loop.py
```

现在 Base / Ours 1 / Ours 2 的最终输出都强制为同样三个 section：

```text
## 1. Diagnosis:
## 2. Hospital Course Summary:
## 3. Discharge Instructions:
```

并禁止在 `Discharge Instructions` 后继续输出 resolved/unresolved/temporal/scaffold/evidence-gap 等额外章节。

### 优化 2：避免 not documented placeholder 拉低 ROUGE

Ours 1/Ours 2 当前经常输出：

```text
[Not documented]
[Not provided]
```

这会显著拉低 Discharge Instructions 的 ROUGE，同时也不像真实医生写 discharge summary。已在 prompt 中改为：

```text
If exact discharge instructions are not documented, do not write "not documented" placeholders;
instead provide only conservative clinically necessary guidance supported by the documented diagnoses and course.
```

这样可以保持 faithfulness，同时减少和 gold 风格的系统性偏差。

### 优化 3：统一 section 粒度

Base 1 会把 procedures、condition、disposition、medications、follow-up 全部写进最终文本；Ours 1/Ours 2 之前把这些拆成更多小节。现在将它们统一映射为：

- diagnosis / secondary diagnoses -> `Diagnosis`
- major course / procedures / complications / resolved and unresolved problems -> `Hospital Course Summary`
- discharge medications / follow-up / diet/activity / precautions / ongoing issues -> `Discharge Instructions`

### 优化 4：区分 scaffold 内部结构与最终文本

Ours 的中间 state/scaffold 可以很细，但最终文本必须转换成 gold-compatible DS style。新的设计是：

```text
structured state/scaffold: detailed, verifier-friendly
final DS output: compact, gold-compatible, 3-section summary
```

这样既保留方法优势，又避免 evaluator 被中间结构污染。

### 优化 5：后续加入长度控制

当前平均长度：

| 方法 | 平均词数 |
|---|---:|
| Base 1 | 441.0 |
| Ours 1 | 510.8 |
| Ours 2 | 478.7 |

长度差异不算极端，但 section 内长度差异较大。下一步可以增加目标长度约束：

```text
Diagnosis: 40-90 words
Hospital Course Summary: 220-350 words
Discharge Instructions: 120-220 words
```

这样 ROUGE 和 LLM judge 都会更公平。

### 优化 6：LLM judge 作为主比较，ROUGE 作为辅助

打平格式后，仍不应只看 ROUGE。DS 任务最核心的是：

- Base 是否为了接近 gold 而 hallucinate discharge meds/follow-up/disposition；
- Ours 是否因为过度保守而 missed gold-supported discharge plan；
- Ours 2 是否真正降低 unsupported，而不是简单删内容。

因此下一步需要加入 DS LLM judge，指标包括：

```text
diagnosis_coverage
hospital_course_completeness
discharge_plan_correctness
evidence_grounding
unsupported_claim_count
missed_major_event_count
overall_preference
```

## 4.4 下一轮建议实验

重新跑同一 10 条 case：

```bash
python scripts/run_ds_minimal_closed_loop.py --limit 10 --case-selection shortest --output-dir outputs/ds_minimal_closed_loop_10_format_aligned
```

然后重新评估：

```bash
python evaluation/evaluate_ds_light.py \
  --gen-dir outputs/ds_minimal_closed_loop_10_format_aligned/method_outputs/base1_full_context_direct \
  --gt-dir data/DS_fixed_composed/full/gold \
  --output-csv outputs/ds_minimal_closed_loop_10_format_aligned/eval_base1.csv

python evaluation/evaluate_ds_light.py \
  --gen-dir outputs/ds_minimal_closed_loop_10_format_aligned/method_outputs/ours1_state_scaffold \
  --gt-dir data/DS_fixed_composed/full/gold \
  --output-csv outputs/ds_minimal_closed_loop_10_format_aligned/eval_ours1.csv

python evaluation/evaluate_ds_light.py \
  --gen-dir outputs/ds_minimal_closed_loop_10_format_aligned/method_outputs/ours2_global_judge_revise \
  --gt-dir data/DS_fixed_composed/full/gold \
  --output-csv outputs/ds_minimal_closed_loop_10_format_aligned/eval_ours2.csv
```

## 4.5 格式打平后重新评测结果

已重新跑同一批 10 条 case，输出目录：

```text
outputs/ds_minimal_closed_loop_10_format_aligned
```

本轮改动：

1. Base 1 / Ours 1 / Ours 2 都强制输出同样三个 section。
2. Ours 不再把 scaffold 中间结构写进最终 DS。
3. Ours 不再大量输出 `[Not documented]` placeholder。
4. state / judge JSON prompt 已收紧。

### 4.5.1 结构检查

本轮 JSON 检查：

| 文件类型 | 错误数 |
|---|---:|
| `final_discharge_state.json` | 0 / 10 |
| `ours2_global_judge.json` | 0 / 10 |

相比上一轮，HADM_ID 139692 的 JSON 错误已经消失。

### 4.5.2 Light ROUGE-L

轻量 evaluator 结果：

| 方法 | Diagnosis | Hospital Course | Discharge Instructions |
|---|---:|---:|---:|
| Base 1 | 13.83 ± 9.07 | 11.05 ± 2.43 | 5.22 ± 1.84 |
| Ours 1 | 12.43 ± 7.53 | 10.29 ± 1.83 | 5.16 ± 1.18 |
| Ours 2 | 12.38 ± 7.49 | 10.37 ± 1.84 | 4.76 ± 1.72 |

解释：

- 格式打平后，三者的 Diagnosis 和 Discharge Instructions ROUGE 已经明显接近。
- Ours 1 的 Discharge Instructions 从上一轮 `3.69` 提升到 `5.16`，基本追平 Base 1 的 `5.22`。
- Ours 2 的 Discharge Instructions 仍略低，可能因为 minimal revise 更保守。

详细文件：

```text
outputs/ds_minimal_closed_loop_10_format_aligned/eval_base1_light.csv
outputs/ds_minimal_closed_loop_10_format_aligned/eval_ours1_light.csv
outputs/ds_minimal_closed_loop_10_format_aligned/eval_ours2_light.csv
outputs/ds_minimal_closed_loop_10_format_aligned/eval_light_summary.csv
```

### 4.5.3 Full evaluator: ROUGE-L, SapBERT, CUI-F1

完整 evaluator 已运行：

```text
evaluation/evaluate_ds.py
```

结果汇总：

| Metric | 方法 | Diagnosis | Hospital Course | Discharge Instructions |
|---|---|---:|---:|---:|
| ROUGE-L | Base 1 | 12.30 ± 10.09 | 17.92 ± 5.60 | 10.11 ± 2.91 |
| ROUGE-L | Ours 1 | 10.45 ± 6.78 | 17.36 ± 3.68 | 8.82 ± 1.85 |
| ROUGE-L | Ours 2 | 9.40 ± 6.90 | 17.54 ± 3.58 | 7.87 ± 3.14 |
| SapBERT | Base 1 | 59.81 ± 8.46 | 74.92 ± 4.31 | 70.49 ± 5.37 |
| SapBERT | Ours 1 | 61.79 ± 6.71 | 71.20 ± 4.86 | 68.40 ± 7.07 |
| SapBERT | Ours 2 | 61.44 ± 5.67 | 71.40 ± 4.75 | 64.62 ± 9.72 |
| CUI-F1 | Base 1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| CUI-F1 | Ours 1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| CUI-F1 | Ours 2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

注意：

- SapBERT 正常可用。
- CUI-F1 当前没有实际启用，因为当前环境中 `quickumls` 包不可用，且没有设置 `QUICKUMLS_PATH`。因此 CUI-F1 的 0 不是模型真实性能，而是 evaluator fallback 后的空结果。

检查命令显示：

```text
quickumls_installed=False
QUICKUMLS_PATH=None
```

### 4.5.3b Exact UMLS CUI-F1

用户提供 UMLS 原始 release 路径：

```text
C:\Users\dsw54\Downloads\umls-2026AA-full\2026AA-full
```

该目录是原始 UMLS release，不是 QuickUMLS matcher 索引。为得到可用的 CUI-F1，新增了一个轻量 exact-match UMLS evaluator：

```text
evaluation/evaluate_ds_umls_cui.py
```

它直接从 `.nlm` 包中读取：

```text
2026AA/META/MRSTY.RRF.gz
2026AA/META/MRCONSO.RRF.*.gz
```

并按 DS 三个 section 的 semantic type filter 计算 exact-match CUI precision / recall / F1。该指标不是 QuickUMLS approximate matching，而是 **Exact UMLS CUI-F1**。

运行结果：

| 方法 | Diagnosis | Hospital Course | Discharge Instructions |
|---|---:|---:|---:|
| Base 1 | 22.81 ± 14.05 | 25.97 ± 6.44 | 24.62 ± 8.78 |
| Ours 1 | 23.63 ± 15.93 | 24.38 ± 7.34 | 17.68 ± 8.85 |
| Ours 2 | 23.70 ± 15.96 | 24.65 ± 7.02 | 15.60 ± 8.25 |

Precision / recall 均值：

| 方法 | Section | Precision | Recall |
|---|---|---:|---:|
| Base 1 | Diagnosis | 17.15 | 41.41 |
| Base 1 | Hospital Course | 32.70 | 22.77 |
| Base 1 | Discharge Instructions | 25.83 | 25.60 |
| Ours 1 | Diagnosis | 18.38 | 37.69 |
| Ours 1 | Hospital Course | 30.41 | 21.58 |
| Ours 1 | Discharge Instructions | 16.58 | 20.61 |
| Ours 2 | Diagnosis | 18.82 | 37.22 |
| Ours 2 | Hospital Course | 30.02 | 22.36 |
| Ours 2 | Discharge Instructions | 16.37 | 17.96 |

输出文件：

```text
outputs/ds_minimal_closed_loop_10_format_aligned/cui_base1_exact.csv
outputs/ds_minimal_closed_loop_10_format_aligned/cui_ours1_exact.csv
outputs/ds_minimal_closed_loop_10_format_aligned/cui_ours2_exact.csv
outputs/ds_minimal_closed_loop_10_format_aligned/cui_exact_summary.csv
outputs/ds_minimal_closed_loop_10_format_aligned/eval_full_plus_exact_cui_summary.csv
```

结论：

1. Diagnosis 上，Ours 1/Ours 2 的 Exact UMLS CUI-F1 略高于 Base 1。
2. Hospital Course 上，三者接近，Base 1 略高。
3. Discharge Instructions 上，Base 1 明显高于 Ours 1/Ours 2。
4. 这和 ROUGE/SapBERT 的趋势一致：Ours 对诊断语义覆盖不差，但出院指导、药物、随访、disposition 仍偏保守，导致 gold overlap 较低。

完整评估输出：

```text
outputs/ds_minimal_closed_loop_10_format_aligned/eval_base1_full.txt
outputs/ds_minimal_closed_loop_10_format_aligned/eval_ours1_full.txt
outputs/ds_minimal_closed_loop_10_format_aligned/eval_ours2_full.txt
outputs/ds_minimal_closed_loop_10_format_aligned/eval_full_summary.csv
```

### 4.5.4 当前结论

格式打平后，三组指标已经比上一轮更接近：

1. Ours 1 与 Base 1 在 light ROUGE 的 Diagnosis 和 Discharge Instructions 上基本接近。
2. Full ROUGE-L 中，Base 1 仍略高，尤其 Diagnosis 和 Instructions。
3. SapBERT 中，Ours 1/Ours 2 的 Diagnosis 反而略高于 Base 1，说明 Ours 的诊断语义覆盖不差。
4. Hospital Course 和 Instructions 上，Base 1 仍更接近 gold 风格。
5. Ours 2 没有稳定超过 Ours 1，说明当前 DS judge-revise 更像 conservative filter，而不是能主动补全 gold-supported discharge content 的 revision module。

下一步若要让三者进一步打平，需要：

- 加入长度控制；
- 让 Ours 的 final generator 使用更 gold-compatible 的 DS 风格；
- 让 judge-revise 区分 unsupported 删除和 gold-supported missing content 补全；
- 启用 QuickUMLS 后再报告真正的 CUI-F1。

## 4.6 如何让 Ours 1 / Ours 2 全面超过 Base

当前 10-case format-aligned 结果显示，Ours 不是全面落后，而是存在非常明确的短板：

1. **Diagnosis**：Ours 1/Ours 2 的 SapBERT 和 Exact UMLS CUI-F1 不低，甚至略高于 Base。
2. **Hospital Course**：Ours 与 Base 接近，但 Base 的 ROUGE/SapBERT 略高。
3. **Discharge Instructions**：Ours 明显低于 Base，是当前最主要差距来源。

### 4.6.1 关键现象

当前三组平均输出特征：

| 方法 | 平均词数 | not documented/placeholders | follow-up related mentions | medication related mentions |
|---|---:|---:|---:|---:|
| Base 1 | 318.5 | 0.0 | 6.9 | 8.4 |
| Ours 1 | 348.5 | 0.5 | 8.5 | 4.4 |
| Ours 2 | 339.4 | 0.7 | 7.4 | 4.5 |

解释：

- Ours 的文本并不短，甚至略长；
- Ours 的 follow-up 提及不少；
- 真正差距是 **medication / discharge-plan concept density 不够**；
- Ours 仍偶尔出现 `not documented / not provided / not specified`；
- Base 更敢写 discharge meds、home disposition、follow-up，因此更接近 gold discharge summary 的表面结构和 CUI overlap。

### 4.6.2 当前 Ours 2 为什么没有超过 Ours 1

Ours 2 的 judge-revise 当前主要像 conservative filter：

```text
remove unsupported
avoid hallucination
preserve supported content
```

但 DS 的 gold summary 不只奖励少 hallucination，也奖励完整覆盖 discharge medications、follow-up、condition、disposition、ongoing issues。因此 Ours 2 如果只做删除/保守修订，会出现：

- unsupported 可能下降；
- ROUGE / CUI-F1 / SapBERT 的 recall 下降；
- discharge instructions 尤其吃亏。

DS 版 Ours 2 应从 **conservative judge-revise** 改为：

```text
coverage-first global verifier + evidence-bounded additive reviser
```

也就是 judge 不只找错，还要强制找“gold-style DS 应该有但当前 draft 缺失的 supported content”。

### 4.6.3 P0 优化：Discharge Plan State Extractor

当前 state schema 虽然有 `discharge_medications`、`follow_up`、`diet_activity_instructions`，但 state updater 没有足够强制地从 chronology 中抽取这些信息。建议单独加入 discharge-plan extractor：

```text
full chronology / last chunks
  -> discharge-plan extractor
  -> discharge medication candidates
  -> follow-up candidates
  -> disposition candidates
  -> diet/activity/precaution candidates
  -> ongoing monitoring candidates
```

输出 schema：

```json
{
  "discharge_disposition_candidates": [],
  "discharge_medication_candidates": [
    {
      "name": "",
      "dose": "",
      "route": "",
      "frequency": "",
      "evidence_type": "explicit_discharge | active_inpatient | home_med | inferred_from_condition",
      "confidence": "high | medium | low"
    }
  ],
  "follow_up_candidates": [],
  "diet_activity_candidates": [],
  "return_precautions": [],
  "monitoring_labs_or_tests": []
}
```

目标：

- 提升 Discharge Instructions 的 medication CUI overlap；
- 减少 Ours 因保守导致的 recall 损失；
- 让 Ours1 先在 instruction 自动指标上追平/超过 Base。

### 4.6.4 P0 优化：从“没有证据就不写”改为分层证据写作

当前 Ours 太保守。DS 任务中很多 instruction 是 clinical convention，不一定逐字出现在 input 中。建议将证据等级分为：

| 证据等级 | 可否写入最终 DS | 例子 |
|---|---|---|
| Explicit | 必须写 | 明确 discharge medication / follow-up |
| Strongly supported | 可以写 | PCI 后继续 aspirin/clopidogrel/statin |
| Standard care inferred | 可以保守写 | follow up with cardiology after PCI |
| Weak/unsupported | 不写 | 未见证据的具体药物剂量或具体 appointment date |

Prompt 中应允许：

```text
You may include conservative standard discharge guidance when it is directly implied by documented diagnoses, procedures, or ongoing problems. Do not invent exact appointment dates, doses, or new medications unless explicitly documented.
```

这可以同时保持 faithful 和提高 gold-style overlap。

### 4.6.5 P1 优化：Base-as-Recall Candidate + Evidence Verification

当前 Base 在 discharge instructions 上 recall 更高，但可能 hallucinate。可以把 Base 作为一个 recall proposer，而不是最终答案：

```text
Base draft
  -> extract candidate diagnoses / meds / follow-up / disposition / precautions
  -> verify each candidate against final state and chronology
  -> keep supported or strongly implied candidates
  -> add to Ours state/scaffold
```

这一步非常适合 Ours2：

```text
Ours1 draft
+ Base candidate list
+ final state
+ selected evidence
  -> global coverage verifier
  -> additive minimal reviser
```

优势：

- Base 提供高 recall；
- verifier 控制 hallucination；
- Ours2 有机会在自动指标和 faithful 两边同时超过 Base。

### 4.6.6 P1 优化：Ours 2 改成 Additive Judge-Revise

当前 Ours2 的 judge 输出应增加一个强制字段：

```json
{
  "supported_missing_discharge_content": [],
  "supported_missing_medications": [],
  "supported_missing_followup": [],
  "supported_missing_disposition": [],
  "supported_missing_return_precautions": [],
  "unsupported_content_to_remove": [],
  "do_not_remove": []
}
```

reviser 规则改为：

```text
1. Add supported missing discharge content first.
2. Remove only clearly unsupported or contradicted content.
3. Preserve all correct diagnosis and hospital course content.
4. Do not make the output shorter unless removing unsupported content.
5. Ensure every section has comparable detail to the base full-context direct draft.
```

这样 Ours2 才会从“保守过滤器”变成“补全 + 纠错器”。

### 4.6.7 P1 优化：Section-specific Evidence Retriever

DS 不能只靠一个统一 state。应按 section 选择证据：

| Section | Evidence Pack |
|---|---|
| Diagnosis | admission reason, problem list, procedures, imaging impression, major abnormal labs |
| Hospital Course | chronological milestones, ICU events, complications, treatment response |
| Discharge Instructions | last medications, active meds near discharge, home meds, anticoagulation/antiplatelet meds, follow-up cues, disposition cues, ongoing abnormal labs |

当前最大收益点是 discharge instructions evidence pack：

```text
last 24-48h meds
home meds if present
anticoagulation / antiplatelet / antibiotics / diuretics / insulin
active unresolved problems
procedure-specific precautions
follow-up / disposition mentions
```

### 4.6.8 P2 优化：Gold-compatible Verbalizer

Ours 的 state 可以很结构化，但最终 verbalizer 应更接近 MIMIC discharge summary 风格。建议加入固定写作模板：

```text
Diagnosis:
- principal diagnosis
- secondary diagnoses

Hospital Course:
The patient was admitted for ...
Hospital course was notable for ...
The patient underwent ...
By discharge ...

Discharge Instructions:
Continue ...
Follow up with ...
Monitor for ...
Return for ...
```

这会提升 ROUGE/SapBERT/CUI-F1，尤其是 hospital course 和 instructions。

### 4.6.9 P2 优化：长度与 concept-density 控制

自动指标对长度和 concept density 敏感。建议增加硬约束：

```text
Diagnosis: 5-10 bullet diagnoses
Hospital Course Summary: 220-320 words
Discharge Instructions: 120-220 words
Medication-related concepts: at least 5 when supported
Follow-up/monitoring/return-precaution concepts: at least 4
```

但要加前提：

```text
Only include medication or follow-up concepts if explicit, strongly supported, or standard-care inferred from documented problems/procedures.
```

### 4.6.10 P3 优化：多候选生成 + 选择器

对于 100-case 正式实验，可以用小规模 self-consistency：

```text
Ours state/scaffold
  -> generate 2 candidate DS drafts
  -> evidence/coverage judge scores each
  -> select or merge best candidate
```

选择标准：

```text
coverage_score - unsupported_penalty + discharge_plan_completeness
```

这通常能提升 recall 型指标。

### 4.6.11 推荐下一版实验设计

建议下一轮不是直接跑 100 条，而是在同一 10 条上做三个变体：

| 变体 | 改动 | 预期 |
|---|---|---|
| Ours1-v2 | 增加 discharge-plan extractor + gold-compatible verbalizer | 提升 instructions ROUGE/CUI |
| Ours2-v2 | additive judge-revise，不只删除，还补 supported missing content | Ours2 超过 Ours1 |
| Ours2-v3 | 加 Base-as-recall candidate + evidence verification | 最有希望全面超过 Base |

成功标准：

```text
Ours1-v2 >= Base in Diagnosis and Hospital Course
Ours1-v2 close to Base in Discharge Instructions
Ours2-v2 > Ours1-v2 in Discharge Instructions and unsupported control
Ours2-v3 > Base in all/most automatic metrics while LLM judge shows lower unsupported
```

### 4.6.12 论文角度的风险

如果加入 Base-as-recall candidate，需要在论文中清楚说明：

```text
Base draft is used only as a candidate recall source, not as final output.
All candidate claims are passed through admission-level evidence verification before entering the revised DS.
```

这不是作弊，而是一种 retrieve-and-verify / propose-and-verify 框架。但实验中需要保留：

- Base 1 direct；
- Ours1 without Base candidate；
- Ours2 with verifier only；
- Ours2 + Base-recall proposer ablation。

这样可以证明提升来自 verifier-guided integration，而不是简单复制 Base。

## 4.7 Ours1-v2 / Ours2-v2 / Ours2-v3 实验结果

已在同一批 10 条 format-aligned DS case 上运行三个优化变体：

| 变体 | 主要改动 |
|---|---|
| Ours1-v2 | discharge-plan extractor + gold-compatible verbalizer |
| Ours2-v2 | additive judge-revise，补 supported missing content |
| Ours2-v3 | Base-as-recall candidates + evidence verification + additive revision |

输出目录：

```text
outputs/ds_v2_variants_10
```

脚本：

```text
scripts/run_ds_v2_variants_10.py
```

更清晰的指标对比表已单独整理为：

```text
outputs/ds_v2_variants_10/metric_comparison_clear.md
```

该表按 metric 分块展示所有方法，并为每个 section 标出：

- 当前分数；
- 相对 Base 1 的绝对差值；
- 该 metric/section 下的赢家。

快速读法：

| 结论 | 证据 |
|---|---|
| Course / Instructions 优化有效 | Ours2-v3 在 full ROUGE-L 的 Hospital Course 和 Discharge Instructions 上均超过 Base；Ours2-v2 在 Instructions CUI-F1 上最高 |
| Diagnosis 是剩余瓶颈 | Base 仍赢 Diagnosis ROUGE-L；原始 Ours 1/2 赢 Diagnosis semantic metrics，但 v2/v3 因扩展 coverage 牺牲 precision |
| 当前最佳变体 | Ours2-v3，整体 Course/Instructions 最强 |
| 下一步 | 做 Ours2-v4：保护 compact verified Diagnosis，同时保留 Ours2-v3 的 Course/Instructions 增强 |

结构检查：

| JSON 文件 | 错误数 |
|---|---:|
| `discharge_plan_evidence.json` | 0 / 10 |
| `ours2_v2_additive_judge.json` | 0 / 10 |
| `base_recall_candidates.json` | 0 / 10 |
| `verified_base_recall.json` | 0 / 10 |

### 4.7.1 Light ROUGE-L

| 方法 | Diagnosis | Hospital Course | Discharge Instructions |
|---|---:|---:|---:|
| Base 1 | 13.83 | 11.05 | 5.22 |
| Ours 1 | 12.43 | 10.29 | 5.16 |
| Ours 2 | 12.38 | 10.37 | 4.76 |
| Ours1-v2 | 11.85 | 11.42 | 5.53 |
| Ours2-v2 | 11.08 | 11.44 | 5.73 |
| Ours2-v3 | 11.08 | 12.08 | 5.63 |

结论：

- Ours1-v2 / Ours2-v2 / Ours2-v3 都提升了 `Discharge Instructions`；
- Ours2-v3 的 `Hospital Course` light ROUGE-L 最高；
- Diagnosis light ROUGE-L 下降，说明优化主要帮助 course/instructions。

### 4.7.2 Full ROUGE-L

| 方法 | Diagnosis | Hospital Course | Discharge Instructions |
|---|---:|---:|---:|
| Base 1 | 12.30 | 17.92 | 10.11 |
| Ours 1 | 10.45 | 17.36 | 8.82 |
| Ours 2 | 9.40 | 17.54 | 7.87 |
| Ours1-v2 | 12.00 | 17.65 | 11.37 |
| Ours2-v2 | 11.23 | 17.62 | 11.39 |
| Ours2-v3 | 11.23 | 18.13 | 12.04 |

结论：

- Ours1-v2 已基本追平 Base 的 Diagnosis ROUGE，并超过 Base 的 Discharge Instructions；
- Ours2-v3 在 Hospital Course 和 Discharge Instructions 上超过 Base；
- Diagnosis 仍略低于 Base，是未完全全面超越的主要瓶颈。

### 4.7.3 SapBERT

| 方法 | Diagnosis | Hospital Course | Discharge Instructions |
|---|---:|---:|---:|
| Base 1 | 59.81 | 74.92 | 70.49 |
| Ours 1 | 61.79 | 71.20 | 68.40 |
| Ours 2 | 61.44 | 71.40 | 64.62 |
| Ours1-v2 | 58.76 | 72.66 | 72.24 |
| Ours2-v2 | 57.82 | 72.68 | 72.80 |
| Ours2-v3 | 58.35 | 73.22 | 72.92 |

结论：

- 三个 v2/v3 变体都显著提升了 Discharge Instructions SapBERT，并超过 Base；
- Hospital Course 仍略低于 Base，但 Ours2-v3 最接近；
- Diagnosis SapBERT 被牺牲，说明 discharge-plan enhancement 影响了 diagnosis precision/style。

### 4.7.4 Exact UMLS CUI-F1

| 方法 | Diagnosis | Hospital Course | Discharge Instructions |
|---|---:|---:|---:|
| Base 1 | 22.81 | 25.97 | 24.62 |
| Ours 1 | 23.63 | 24.38 | 17.68 |
| Ours 2 | 23.70 | 24.65 | 15.60 |
| Ours1-v2 | 21.92 | 26.52 | 26.02 |
| Ours2-v2 | 21.00 | 26.36 | 26.67 |
| Ours2-v3 | 20.56 | 27.00 | 25.85 |

结论：

- Ours1-v2 / Ours2-v2 / Ours2-v3 都让 Discharge Instructions CUI-F1 超过 Base；
- 三个变体都让 Hospital Course CUI-F1 超过 Base；
- Diagnosis CUI-F1 下降，主要因为 v2/v3 的 diagnosis list 更宽，增加了 CUI false positives，precision 被稀释。

### 4.7.5 当前最强变体

从自动指标看：

| 目标 | 当前最好 |
|---|---|
| Light ROUGE Hospital Course | Ours2-v3 |
| Light ROUGE Discharge Instructions | Ours2-v2 |
| Full ROUGE Hospital Course | Ours2-v3 |
| Full ROUGE Discharge Instructions | Ours2-v3 |
| SapBERT Discharge Instructions | Ours2-v3 |
| Exact CUI Hospital Course | Ours2-v3 |
| Exact CUI Discharge Instructions | Ours2-v2 |

因此，当前最有潜力的是：

> Ours2-v3: Base-as-recall candidates + evidence verification + additive revision

但它还没有全面超过 Base，因为 Diagnosis 指标下降。

### 4.7.6 为什么还没有全面超过 Base

Ours2-v3 对 HADM_ID 191485 的表现说明了当前问题：

- Discharge Instructions 更完整，增加了 disposition、具体药物、follow-up、monitoring、return precautions；
- Hospital Course 更详细；
- 但 Diagnosis section 中有些表达从 Base 的 gold-compatible 风格变成了更泛化的 clinical phrasing。

Exact CUI Diagnosis per-case 显示，Ours2-v3 在多数 case 的 Diagnosis CUI-F1 低于 Base：

| HADM_ID | Base Diagnosis CUI-F1 | Ours2-v3 Diagnosis CUI-F1 | Delta |
|---:|---:|---:|---:|
| 138003 | 0.440 | 0.283 | -0.157 |
| 134314 | 0.389 | 0.314 | -0.075 |
| 139692 | 0.203 | 0.151 | -0.051 |
| 196447 | 0.182 | 0.143 | -0.039 |

主要原因：

1. v2/v3 会扩展 diagnosis list，增加 false-positive CUI；
2. diagnosis verbalizer 没有保护 Base/Ours1 中已有的 high-overlap diagnoses；
3. discharge-plan enhancement 影响了整个 output style，而不是只作用于 instructions。

### 4.7.7 下一步最直接的改法：Ours2-v4

要全面超过 Base，建议不是继续增强 instructions，而是加一个 **diagnosis-preserving guard**：

```text
Diagnosis section:
  use conservative diagnosis list from Ours1 / Base verified candidates
  do not expand diagnosis unless explicitly supported
  avoid symptoms/labs as discharge diagnoses unless gold-style diagnosis

Hospital Course + Instructions:
  use Ours2-v3 additive coverage and verified Base recall
```

也就是：

```text
Ours2-v4 =
  diagnosis-preserving verifier
  + Ours2-v3 course/instruction additive verifier
```

推荐实现：

1. 从 Base 和 Ours1-v2 中抽取 diagnosis candidates；
2. 只保留 final state 中 `principal_diagnoses` / `secondary_diagnoses` 支持的 diagnosis；
3. 删除 lab abnormality / symptom / uncertain finding，除非它们在 gold-style discharge diagnosis 中常见；
4. 对 final DS 做 section-wise merge：

```text
Diagnosis = verified compact diagnosis list
Hospital Course = Ours2-v3
Discharge Instructions = Ours2-v2 or Ours2-v3
```

预期：

- 保住 Ours2-v3 在 Hospital Course / Instructions 的提升；
- 恢复 Diagnosis ROUGE / SapBERT / CUI-F1；
- 最有希望让 Ours 全面超过 Base。

### 4.7.8 当前结论

这轮实验证明优化方向是有效的：

1. Discharge-plan extractor 解决了 Ours 过度保守的问题；
2. Additive judge-revise 比 conservative revise 更适合 DS；
3. Base-as-recall + evidence verification 能显著提升 Hospital Course 和 Instructions；
4. 当前剩余瓶颈转移到了 Diagnosis precision；
5. 下一步应做 Ours2-v4：固定/保护 diagnosis，继续使用 v3 的 course/instruction 增强。

## 4.8 Ours2-v4: Diagnosis-Preserving Guard

根据 4.7 的结果，Ours2-v3 已经显著改善 Hospital Course 和 Discharge Instructions，但 Diagnosis 指标下降。因此新增 Ours2-v4，用于验证：

> 如果固定 compact diagnosis，并保留 Ours2-v3 的 course/instruction 增强，是否可以同时避免 diagnosis precision loss 并保持后两段收益。

脚本：

```text
scripts/build_ds_ours2_v4_guarded.py
```

输出目录：

```text
outputs/ds_v2_variants_10/method_outputs/ours2_v4_diagnosis_guarded
```

Ours2-v4 当前实现：

```text
Diagnosis = Base 1 compact diagnosis section
Hospital Course = Ours2-v3 enhanced hospital course
Discharge Instructions = Ours2-v3 enhanced discharge instructions
```

注意：Ours2-v4 目前是 **section-wise guarded composition / ablation**，用于验证“diagnosis precision 与 course/instruction coverage 应该分开控制”。它不是最终可部署系统。最终系统中，Base diagnosis 应替换为：

```text
verified compact diagnosis selector
  = Base/Ours diagnosis candidates
  + final discharge state
  + evidence verification
  -> compact diagnosis section
```

### 4.8.1 版本改动记录

| Version | 主要改动 | 目标 |
|---|---|---|
| Base 1 | Full-context direct DS generation | 直接长上下文 baseline |
| Ours 1 | Sequential discharge-state tracking + scaffold generation | 验证 state/scaffold 是否有效 |
| Ours 2 | Original global judge-revise | 保守过滤 unsupported content |
| Ours1-v2 | Discharge-plan extractor + gold-compatible verbalizer | 提升 discharge instructions recall |
| Ours2-v2 | Additive judge-revise | 补 supported missing discharge content |
| Ours2-v3 | Base-as-recall candidates + evidence verification | 利用 Base 高 recall，同时用 verifier 控制 hallucination |
| Ours2-v4 | Diagnosis-preserving guard + Ours2-v3 course/instructions | 保住 diagnosis precision，同时保留 course/instruction 增益 |

### 4.8.2 Ours2-v4 结果

更清晰的完整对比表：

```text
outputs/ds_v2_variants_10/metric_comparison_clear_with_v4.md
```

核心结果：

| Metric | 方法 | Diagnosis | Hospital Course | Discharge Instructions |
|---|---|---:|---:|---:|
| Light ROUGE-L | Base 1 | 13.83 | 11.05 | 5.22 |
| Light ROUGE-L | Ours2-v4 | 13.83 | 12.08 | 5.63 |
| ROUGE-L | Base 1 | 12.30 | 17.92 | 10.11 |
| ROUGE-L | Ours2-v4 | 12.30 | 18.13 | 12.04 |
| SapBERT | Base 1 | 59.81 | 74.92 | 70.49 |
| SapBERT | Ours2-v4 | 59.81 | 73.22 | 72.92 |
| Exact UMLS CUI-F1 | Base 1 | 22.81 | 25.97 | 24.62 |
| Exact UMLS CUI-F1 | Ours2-v4 | 22.81 | 27.00 | 25.85 |

### 4.8.3 结论

Ours2-v4 达到了设计目的：

1. Diagnosis 与 Base 打平，因为使用 compact diagnosis-preserving guard。
2. Hospital Course 在 ROUGE-L 和 Exact UMLS CUI-F1 上超过 Base。
3. Discharge Instructions 在 ROUGE-L、SapBERT 和 Exact UMLS CUI-F1 上超过 Base。
4. SapBERT Hospital Course 仍低于 Base，说明 course 的语义风格还可以继续优化。

当前最清晰的结论是：

> Ours2-v4 shows that the previous v2/v3 failure was not caused by the longitudinal state/scaffold itself, but by uncontrolled diagnosis expansion. Separating diagnosis precision control from course/instruction coverage allows the system to match Base on diagnosis while improving discharge-course and discharge-plan metrics.

### 4.8.4 下一步：把 v4 从 ablation 变成真实方法

当前 v4 直接使用 Base diagnosis section，因此更像 proof-of-concept。若要变成论文主方法，需要改成：

```text
Base diagnosis candidates
+ Ours diagnosis candidates
+ final discharge state
+ evidence snippets
  -> compact diagnosis selector
  -> verified diagnosis section
```

selector 规则：

```text
1. Prefer principal diagnoses and final discharge diagnoses.
2. Keep major secondary diagnoses only if they affect hospital course/discharge plan.
3. Do not include transient labs, symptoms, uncertain imaging findings, or PMH as discharge diagnoses unless explicitly supported.
4. Keep diagnosis section compact; avoid coverage-driven expansion.
```

这样才能避免 reviewer 质疑“v4 只是复制 Base 的 Diagnosis”。正式实验中应报告：

- Ours2-v3：真实 propose-and-verify 增强；
- Ours2-v4-ablation：section-wise diagnosis guard 的上限；
- Ours2-v4-final：verified compact diagnosis selector 的真实实现。

## 4.9 Ours2-v4-final: 正式 verified diagnosis selector

为避免 Ours2-v4 被质疑为直接复制 Base diagnosis，新增正式版本：

```text
Ours2-v4-final =
  verified compact diagnosis selector
  + Ours2-v3 hospital course
  + Ours2-v3 discharge instructions
```

脚本：

```text
scripts/run_ds_ours2_v4_final.py
```

输出目录：

```text
outputs/ds_v2_variants_10/method_outputs/ours2_v4_final_verified_diagnosis
```

正式 diagnosis selector 的输入：

```text
Base diagnosis candidates
+ Ours1-v2 diagnosis candidates
+ Ours2-v3 diagnosis candidates
+ final discharge state
+ admission evidence
  -> compact verified diagnosis section
```

它只把 Base/Ours diagnosis 当作 candidate recall source，而不是直接复制 Base diagnosis。

### 4.9.1 最新完整对比表

清晰版对比：

```text
outputs/ds_v2_variants_10/metric_comparison_clear_with_v4_final.md
```

核心结果：

| Metric | 方法 | Diagnosis | Hospital Course | Discharge Instructions |
|---|---|---:|---:|---:|
| ROUGE-L | Base 1 | 12.30 | 17.92 | 10.11 |
| ROUGE-L | Ours2-v4-ablation | 12.30 | 18.13 | 12.04 |
| ROUGE-L | Ours2-v4-final | 9.97 | 18.13 | 12.04 |
| SapBERT | Base 1 | 59.81 | 74.92 | 70.49 |
| SapBERT | Ours2-v4-ablation | 59.81 | 73.22 | 72.92 |
| SapBERT | Ours2-v4-final | 59.46 | 73.22 | 72.92 |
| Exact UMLS CUI-F1 | Base 1 | 22.81 | 25.97 | 24.62 |
| Exact UMLS CUI-F1 | Ours2-v4-ablation | 22.81 | 27.00 | 25.85 |
| Exact UMLS CUI-F1 | Ours2-v4-final | 22.26 | 27.00 | 25.85 |

Light ROUGE-L：

| 方法 | Diagnosis | Hospital Course | Discharge Instructions |
|---|---:|---:|---:|
| Base 1 | 13.83 | 11.05 | 5.22 |
| Ours2-v4-ablation | 13.83 | 12.08 | 5.63 |
| Ours2-v4-final | 11.55 | 12.08 | 5.63 |

### 4.9.2 正式方法验证结论

正式 Ours2-v4-final 验证了两点：

1. **Course / Instructions 增益是真实保留的。**
   Ours2-v4-final 和 Ours2-v3 一样，在 Hospital Course 和 Discharge Instructions 上保持提升：
   - ROUGE-L Hospital Course: `18.13` vs Base `17.92`
   - ROUGE-L Instructions: `12.04` vs Base `10.11`
   - Exact CUI Hospital Course: `27.00` vs Base `25.97`
   - Exact CUI Instructions: `25.85` vs Base `24.62`

2. **Diagnosis selector 仍未完全解决。**
   正式 selector 的 Diagnosis：
   - SapBERT 接近 Base：`59.46` vs `59.81`
   - CUI-F1 接近 Base：`22.26` vs `22.81`
   - 但 ROUGE-L 明显低于 Base：`9.97` vs `12.30`

因此，目前不能声称正式 Ours2-v4-final 已经在所有自动指标上全面超过 Base。更准确的结论是：

> Formal Ours2-v4-final preserves the verified recall gains in Hospital Course and Discharge Instructions, but the compact diagnosis selector still needs optimization to match the gold-compatible surface form of Base diagnosis.

### 4.9.3 与 v4-ablation 的关系

Ours2-v4-ablation 说明：

```text
如果 diagnosis precision 被完美控制，
Ours2-v3 的 course/instruction 增益可以与 Base-level diagnosis 同时存在。
```

Ours2-v4-final 说明：

```text
正式 verified diagnosis selector 已经接近 Base 的 diagnosis semantic overlap，
但还没有恢复 Base 的 diagnosis ROUGE / gold-style phrasing。
```

所以 v4-ablation 是上限证明，v4-final 是当前真实方法结果。

### 4.9.4 下一步优化方向

下一步不应该继续改 Hospital Course / Instructions，而应该专门优化 diagnosis selector。

建议 Ours2-v4-final-dx2：

```text
1. 先生成 candidate diagnosis list；
2. 对每个 candidate 预测 label:
   - principal diagnosis
   - major secondary diagnosis
   - complication diagnosis
   - transient lab/symptom, exclude
   - PMH only, exclude
   - uncertain finding, exclude
3. 最终只输出 principal + major secondary + major complication；
4. 强制 gold-compatible phrasing:
   - status post PCI
   - acute kidney injury
   - pneumonia
   - heart failure exacerbation
   - anemia
   - pseudoaneurysm status post thrombin injection
5. 控制数量 4-7 bullets，避免 broad coverage expansion。
```

验证目标：

```text
Diagnosis ROUGE-L >= Base
Diagnosis SapBERT >= Base or close
Diagnosis CUI-F1 >= Base or close
Course/Instructions retain Ours2-v3 gains
```

## 4.10 Ours2-v4-dx2: 泛化版 role-based diagnosis classifier

考虑到 dx prompt 不能过拟合 10 条样本，新增 `Ours2-v4-dx2`：

```text
Candidate diagnoses
+ final discharge state
+ admission evidence
  -> diagnosis role classifier
  -> deterministic verbalizer
  -> final Diagnosis section
```

脚本：

```text
scripts/run_ds_ours2_v4_dx2.py
```

输出目录：

```text
outputs/ds_v2_variants_10/method_outputs/ours2_v4_dx2_role_classified_diagnosis
```

dx2 的核心不是直接让 LLM 写 diagnosis，而是先分类：

```text
principal_discharge_diagnosis
major_secondary_diagnosis
major_complication_affecting_course_or_discharge
procedure_related_diagnosis
chronic_comorbidity_relevant_to_discharge
transient_lab_or_minor_resolved_issue
symptom_or_uncertain_finding
past_history_only
duplicate_or_subsumed
```

然后脚本只保留：

```text
principal_discharge_diagnosis
major_secondary_diagnosis
major_complication_affecting_course_or_discharge
procedure_related_diagnosis
```

### 4.10.1 dx2 结果

dx2 JSON 检查：

| 文件 | 错误数 |
|---|---:|
| `ours2_v4_dx2_classification.json` | 0 / 10 |

核心指标：

| Metric | 方法 | Diagnosis | Hospital Course | Discharge Instructions |
|---|---|---:|---:|---:|
| ROUGE-L | Base 1 | 12.30 | 17.92 | 10.11 |
| ROUGE-L | Ours2-v4-final | 9.97 | 18.13 | 12.04 |
| ROUGE-L | Ours2-v4-dx2 | 9.50 | 18.13 | 12.04 |
| SapBERT | Base 1 | 59.81 | 74.92 | 70.49 |
| SapBERT | Ours2-v4-final | 59.46 | 73.22 | 72.92 |
| SapBERT | Ours2-v4-dx2 | 57.09 | 73.22 | 72.92 |
| Exact UMLS CUI-F1 | Base 1 | 22.81 | 25.97 | 24.62 |
| Exact UMLS CUI-F1 | Ours2-v4-final | 22.26 | 27.00 | 25.85 |
| Exact UMLS CUI-F1 | Ours2-v4-dx2 | 21.27 | 27.00 | 25.85 |

Light ROUGE-L：

| 方法 | Diagnosis | Hospital Course | Discharge Instructions |
|---|---:|---:|---:|
| Base 1 | 13.83 | 11.05 | 5.22 |
| Ours2-v4-final | 11.55 | 12.08 | 5.63 |
| Ours2-v4-dx2 | 11.03 | 12.08 | 5.63 |

### 4.10.2 dx2 结论

dx2 没有提升性能，反而让 Diagnosis 更低：

- Diagnosis ROUGE-L: `9.50`，低于 v4-final 的 `9.97`；
- Diagnosis SapBERT: `57.09`，低于 v4-final 的 `59.46`；
- Diagnosis CUI-F1: `21.27`，低于 v4-final 的 `22.26`。

Course / Instructions 不变，因为仍复用 Ours2-v3。

### 4.10.3 为什么 dx2 没提升

dx2 的泛化逻辑是合理的，但当前 prompt 仍然有两个问题：

1. **它按临床重要性分类，而不是按 MIMIC discharge diagnosis 风格分类。**
   例如 HADM_ID `138003`，gold diagnosis 是：

   ```text
   Atrial fibrillation with rapid ventricular response
   Hypertension
   Left lower lobe Pneumonia
   ```

   但 dx2 选择：

   ```text
   Atrial fibrillation
   Acute hypoxemic respiratory failure due to pulmonary edema / effusions
   Acute kidney injury
   Anemia
   ```

   它排除了 possible pneumonia，却加入 AKI/anemia。这在临床上有解释，但与 gold discharge diagnosis 的短列表不一致。

2. **它会把 etiologic/course-level problem 放进 Diagnosis。**
   dx2 仍然倾向于把 respiratory failure、AKI、anemia 等 major course problems 作为 discharge diagnosis。对 clinical completeness 有利，但对 gold-style diagnosis precision 不利。

### 4.10.4 当前最稳结论

dx2 说明：

> 仅靠泛化的 clinical-role classification 不能自动打平 Diagnosis 指标；Diagnosis selector 必须显式学习 discharge-diagnosis style，而不是只判断 clinical importance。

目前最好的真实方法仍是：

```text
Ours2-v4-final
```

它在 Diagnosis semantic metrics 接近 Base，同时保留 Course / Instructions 优势。

上限证明仍是：

```text
Ours2-v4-ablation
```

它说明如果 diagnosis precision 可以被完美控制，Ours 的 Course / Instructions 优势能和 Base-level Diagnosis 同时存在。

### 4.10.5 下一步建议

不要继续扩大泛化分类器，而应改成 **gold-style diagnosis ranker**：

```text
candidate diagnoses
  -> predict discharge-diagnosis likelihood
  -> rank candidates by likelihood of appearing in MIMIC discharge diagnosis
  -> output top 3-5
```

ranker 特征应包括：

```text
explains admission
listed as discharge/final diagnosis in source-like text
procedure-related final condition
explicitly named disease, not symptom/lab
appears in discharge plan
is not merely course complication unless central
```

也就是说，下一版应该优化：

```text
gold-style diagnosis likelihood
```

而不是：

```text
clinical problem importance
```

## 4.11 Ours2-v4-dx3: Diagnosis Agent Verbalizer

基于 dx2 的失败，新增 dx3：

```text
dx2 role classification JSON
+ Base diagnosis candidates
+ Ours1-v2 diagnosis candidates
+ Ours2-v3 diagnosis candidates
+ final state
+ evidence
  -> Diagnosis Agent
  -> compact gold-style Diagnosis
```

脚本：

```text
scripts/run_ds_ours2_v4_dx3_agent.py
```

输出目录：

```text
outputs/ds_v2_variants_10/method_outputs/ours2_v4_dx3_agent_diagnosis
```

与 dx2 的关键区别：

| 版本 | LLM 完成 | 规则完成 |
|---|---|---|
| dx2 | role classification, include decision, final_phrase | role 白名单过滤、去重、拼 bullet |
| dx3 | 基于分类结果重新综合、选择、改写 Diagnosis | 清理 heading、拼接 Course/Instructions |

因此 dx3 验证的是：

> classifier + agentic diagnosis verbalizer 是否优于 classifier + rule verbalizer。

### 4.11.1 dx3 结果

| Metric | 方法 | Diagnosis | Hospital Course | Discharge Instructions |
|---|---|---:|---:|---:|
| ROUGE-L | Base 1 | 12.30 | 17.92 | 10.11 |
| ROUGE-L | Ours2-v4-final | 9.97 | 18.13 | 12.04 |
| ROUGE-L | Ours2-v4-dx2 | 9.50 | 18.13 | 12.04 |
| ROUGE-L | Ours2-v4-dx3 | 10.93 | 18.13 | 12.04 |
| SapBERT | Base 1 | 59.81 | 74.92 | 70.49 |
| SapBERT | Ours2-v4-final | 59.46 | 73.22 | 72.92 |
| SapBERT | Ours2-v4-dx2 | 57.09 | 73.22 | 72.92 |
| SapBERT | Ours2-v4-dx3 | 58.57 | 73.22 | 72.92 |
| Exact UMLS CUI-F1 | Base 1 | 22.81 | 25.97 | 24.62 |
| Exact UMLS CUI-F1 | Ours2-v4-final | 22.26 | 27.00 | 25.85 |
| Exact UMLS CUI-F1 | Ours2-v4-dx2 | 21.27 | 27.00 | 25.85 |
| Exact UMLS CUI-F1 | Ours2-v4-dx3 | 22.86 | 27.00 | 25.85 |

Light ROUGE-L：

| 方法 | Diagnosis | Hospital Course | Discharge Instructions |
|---|---:|---:|---:|
| Base 1 | 13.83 | 11.05 | 5.22 |
| Ours2-v4-final | 11.55 | 12.08 | 5.63 |
| Ours2-v4-dx2 | 11.03 | 12.08 | 5.63 |
| Ours2-v4-dx3 | 11.92 | 12.08 | 5.63 |

详细对比：

```text
outputs/ds_v2_variants_10/diagnosis_variant_comparison_dx3.md
```

### 4.11.2 dx3 结论

dx3 相比 dx2 明显更好：

| Metric | dx2 Diagnosis | dx3 Diagnosis | Delta |
|---|---:|---:|---:|
| ROUGE-L | 9.50 | 10.93 | +1.43 |
| SapBERT | 57.09 | 58.57 | +1.48 |
| Exact UMLS CUI-F1 | 21.27 | 22.86 | +1.59 |

这说明用户判断是正确的：

> dx2 指标低的一部分原因确实是最后一步没有重新跑 Diagnosis agent，而是用了规则式 verbalizer。

dx3 恢复了 semantic diagnosis coverage：

- Diagnosis CUI-F1: `22.86`，略高于 Base `22.81`；
- Course / Instructions 继续保持 Ours2-v3 的优势。

但 dx3 仍没有完全打平 Diagnosis ROUGE-L / SapBERT：

- Diagnosis ROUGE-L: `10.93` vs Base `12.30`
- Diagnosis SapBERT: `58.57` vs Base `59.81`

说明剩余问题主要是：

```text
surface phrasing / gold-style wording
```

而不是：

```text
semantic diagnosis coverage
```

### 4.11.3 当前最合理结论

当前真实方法排序：

```text
dx2 < v4-final < dx3 < v4-ablation upper-bound
```

解释：

- dx2：规则 verbalizer 太硬；
- v4-final：LLM selector 接近 Base semantics，但 surface phrasing 不足；
- dx3：agentic verbalizer 改善明显，CUI-F1 已超过 Base；
- v4-ablation：复制 Base diagnosis，是 diagnosis upper-bound，不是正式方法。

因此，若写论文，建议把 dx3 作为当前正式 DS 方法的更合理版本：

```text
Ours2-v4-dx3 =
  role-aware diagnosis agent
  + Ours2-v3 course/instruction verifier
```

但结论必须谨慎：

> Ours2-v4-dx3 improves discharge-plan and hospital-course metrics and recovers diagnosis CUI-F1, but diagnosis surface-form alignment remains below the full-context baseline.

## 5. 实际 LLM 实验命令

当前 shell 中尚未检测到 `DEEPSEEK_API_KEY` 环境变量。设置后可运行：

```bash
python scripts/run_ds_minimal_closed_loop.py --limit 10 --case-selection shortest
```

运行完成后评估三组方法：

```bash
python evaluation/evaluate_ds_light.py \
  --gen-dir outputs/ds_minimal_closed_loop_10/method_outputs/base1_full_context_direct \
  --gt-dir data/DS_fixed_composed/full/gold \
  --output-csv outputs/ds_minimal_closed_loop_10/eval_base1.csv

python evaluation/evaluate_ds_light.py \
  --gen-dir outputs/ds_minimal_closed_loop_10/method_outputs/ours1_state_scaffold \
  --gt-dir data/DS_fixed_composed/full/gold \
  --output-csv outputs/ds_minimal_closed_loop_10/eval_ours1.csv

python evaluation/evaluate_ds_light.py \
  --gen-dir outputs/ds_minimal_closed_loop_10/method_outputs/ours2_global_judge_revise \
  --gt-dir data/DS_fixed_composed/full/gold \
  --output-csv outputs/ds_minimal_closed_loop_10/eval_ours2.csv
```

## 6. 当前检查结论

最小闭环的数据和工程路径可以跑通，但暴露了三个需要优化的问题：

1. full DS 数据为 100 条，input/gold 均存在。
2. 10 条短样本可以完整进入 Base 1 / Ours 1 / Ours 2。
3. Ours 1 的 state tracking 和 scaffold 中间文件已经设计好保存路径。
4. Ours 2 的 global DS judge 与 minimal revise 已从 A&P daily verifier 逻辑中分离。
5. 输出格式兼容当前轻量 DS evaluator。
6. 当前 Ours 在 ROUGE 上低于 Base，主要因为它更保守，不主动补全 gold 中的出院计划/随访/药物。
7. 当前 state/judge JSON 仍有少量格式错误，需要 retry/repair 机制。

下一步需要实际调用 LLM 生成 10 条结果，然后检查：

- state JSON 是否稳定、是否过度丢信息；
- scaffold 是否覆盖 diagnosis / hospital course / discharge plan；
- judge 是否能指出真实 unsupported 和 missed；
- minimal reviser 是否会新增未授权细节；
- Ours 2 是否相对 Ours 1 降低 hallucination，而不是过度删除。
