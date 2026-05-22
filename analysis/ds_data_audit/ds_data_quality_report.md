# DS 数据质量审计报告

生成日期：2026-05-20

本报告用于确认当前 DS 任务数据是否适合下一步模型性能验证。重点检查数据量、input/gold 配对、输入长度、gold 长度、空输入、出院后事件泄漏、多 discharge summary note、gold section 可解析性，以及不同 DS 版本之间的差异。

## 1. 当前仓库中的 DS 数据版本

当前仓库里至少存在 5 套 DS 数据：

| 版本 | input 路径 | gold 路径 | 说明 |
|---|---|---|---|
| raw_24h | `data/DS/input` | `data/DS/gold` | 原始 DS 数据，存在锚点和 gold note 选择问题 |
| fixed_24h | `data/DS_fixed/24h/input` | `data/DS_fixed/24h/gold` | 使用 DISCHTIME 锚定，但 gold 未合并多条 discharge summary |
| fixed_full | `data/DS_fixed/full/input` | `data/DS_fixed/full/gold` | full admission input，但 gold 未合并多条 discharge summary |
| composed_24h | `data/DS_fixed_composed/24h/input` | `data/DS_fixed_composed/24h/gold` | 使用 DISCHTIME 锚定，gold 合并多条 discharge summary，但 input 只有最后 24h |
| composed_full | `data/DS_fixed_composed/full/input` | `data/DS_fixed_composed/full/gold` | 使用 DISCHTIME 锚定，gold 合并多条 discharge summary，input 为全住院时间线 |

## 2. 总体结论

**建议主实验使用：**

```text
data/DS_fixed_composed/full/input
data/DS_fixed_composed/full/gold
```

原因：

1. `composed_full` 有 100 个 input 和 100 个 gold，全部一一配对。
2. 出院后事件数为 0，没有明显 future leakage。
3. gold 合并了同一次 admission 的多条 discharge summary notes，避免原始 gold 只取第一条 note 或短 addendum 的问题。
4. DS 是完整 discharge summary generation，本质上需要全住院证据；full input 比 24h input 更符合任务定义。
5. `composed_full` 中没有空输入，没有 input 少于 500 词的样本，输入支持性明显优于 24h 版本。

**不建议主实验使用：**

```text
data/DS/input
data/DS/gold
```

原始版本存在两个主要问题：

1. 部分 HADM_ID 有多条 discharge summary note，但原始 gold 只取了第一条，可能漏掉后续 addendum 或最终 discharge summary。
2. 原始 input 的 24h 窗口以 admission 最后一条事件作为锚点，而不是以 `ADMISSIONS.DISCHTIME` 作为锚点，存在时间对齐不稳的问题。

**24h 版本可以保留为 ablation，但不适合作为完整 DS 主实验。**

## 3. 数据量与配对情况

| 版本 | input 文件数 | gold 文件数 | 成功配对 | 缺 gold | 缺 input |
|---|---:|---:|---:|---:|---:|
| raw_24h | 100 | 100 | 100 | 0 | 0 |
| fixed_24h | 100 | 100 | 100 | 0 | 0 |
| fixed_full | 100 | 100 | 100 | 0 | 0 |
| composed_24h | 100 | 100 | 100 | 0 | 0 |
| composed_full | 100 | 100 | 100 | 0 | 0 |

从文件数量上看，所有版本都是 100 个 admission，input/gold 都能配对。但是否适合作为 DS 主实验，关键不在数量，而在 input 是否足以支持完整 discharge summary，以及 gold 是否是完整最终 summary。

## 4. 输入长度统计

| 版本 | input rows 平均 / 中位数 | input words 平均 / 中位数 | input words 最小 / 最大 | 空输入 | input <100 词 | input <500 词 |
|---|---:|---:|---:|---:|---:|---:|
| raw_24h | 4.88 / 2.0 | 315.1 / 83.5 | 5 / 3442 | 0 | 57 | 85 |
| fixed_24h | 3.97 / 2.0 | 259.5 / 47.5 | 0 / 2526 | 11 | 76 | 87 |
| fixed_full | 334.1 / 158.5 | 24390.5 / 12219.0 | 1377 / 157649 | 0 | 0 | 0 |
| composed_24h | 3.97 / 2.0 | 259.5 / 47.5 | 0 / 2526 | 11 | 76 | 87 |
| composed_full | 334.1 / 158.5 | 24390.5 / 12219.0 | 1377 / 157649 | 0 | 0 | 0 |

解释：

- 24h 版本输入非常短。`composed_24h` 有 11 个样本完全空输入，76 个样本少于 100 词，87 个样本少于 500 词。
- full 版本输入充足，最短也有 1377 词，95/100 的 input words 大于 gold words。
- 对完整 discharge summary generation 来说，24h 输入无法覆盖入院原因、手术、全住院病程、并发症、出院诊断和出院计划，因此低分不能简单归因于模型能力。

## 5. Gold 长度统计

| 版本 | gold words 平均 / 中位数 | gold words 最小 / 最大 |
|---|---:|---:|
| raw_24h | 1879.1 / 1677.0 | 368 / 4277 |
| fixed_24h | 1646.0 / 1509.5 | 68 / 4277 |
| fixed_full | 1646.0 / 1509.5 | 68 / 4277 |
| composed_24h | 1994.5 / 1809.5 | 379 / 4288 |
| composed_full | 1994.5 / 1809.5 | 379 / 4288 |

解释：

- `fixed_24h/full` 的 gold 最小只有 68 词，说明仍可能只取到短 note/addendum，不适合作为主实验 gold。
- `composed_24h/full` 合并多条 discharge summary 后，gold 平均长度升高到 1994.5 词，最短也有 379 词，更合理。
- 因此应优先使用 `DS_fixed_composed`，而不是 `DS_fixed`。

## 6. 多 discharge summary note 问题

在 `DS_fixed_composed` 中：

- 总 admission 数：100
- 多 discharge summary note 样本数：21
- 每个样本 gold note count 平均：1.25
- 最大 gold note count：4

典型多 note 样本：

| HADM_ID | note count | gold words | discharge dates |
|---:|---:|---:|---|
| 104732 | 2 | 969 | 2114-12-17 / 2114-12-28 |
| 145095 | 4 | 3196 | 2105-02-12 / 2105-03-01 / 2105-03-13 / 2105-03-16 |
| 152809 | 3 | 3324 | 2118-11-16 / 2118-11-18 |
| 168849 | 3 | 2122 | 2172-06-06 |
| 174792 | 2 | 1637 | 2191-01-20 / 2191-01-25 |

这说明原始 gold 如果只取第一条 discharge summary，很容易拿到不完整或非最终版本。`composed_full` 合并这些 notes 更稳妥。

## 7. 出院后事件泄漏

基于 regeneration summary：

| 版本 | events_after_discharge |
|---|---:|
| fixed_24h | 0 |
| fixed_full | 0 |
| composed_24h | 0 |
| composed_full | 0 |

说明 fixed/composed 版本都已经按 `ADMISSIONS.DISCHTIME` 锚定，过滤掉了晚于出院时间的事件。

原始 `data/DS` 版本此前报告中已指出存在锚点问题：它以 admission 最后一条事件为 24h 窗口锚点，而不是以 `DISCHTIME` 为锚点，因此不建议继续作为主实验数据。

## 8. Gold section 可解析性

在 `composed_full` 的 100 个 gold discharge summaries 中，基于当前 evaluator 的正则模式：

| Section | 可检测数量 |
|---|---:|
| Diagnosis | 80 / 100 |
| Hospital Course | 91 / 100 |
| Discharge Instructions / Medications / Diet | 98 / 100 |

存在 section heading 不统一的问题，尤其是 Diagnosis。当前 evaluator 的 section extraction 可能低估某些样本，因为 MIMIC discharge summary 的标题变体较多。

缺失任一 section pattern 的样本包括：

```text
107157, 111519, 115916, 118789, 126472, 128928, 129183, 152809,
154191, 155317, 156549, 163712, 165404, 166270, 167323, 168923,
171450, 172606, 174792, 179679, 195556
```

建议：

- 如果继续使用 section-level ROUGE/SapBERT，需要扩展 evaluator 的 section heading regex。
- 如果使用 LLM judge，可以直接评估完整 DS 的 clinical completeness、faithfulness、hospital course correctness、discharge plan correctness，而不完全依赖 heading extraction。

## 9. 24h 版本的主要问题

以 `composed_24h` 为例：

最短输入样本：

| HADM_ID | rows | input words | gold words |
|---:|---:|---:|---:|
| 105091 | 0 | 0 | 1491 |
| 113981 | 0 | 0 | 660 |
| 126472 | 0 | 0 | 2133 |
| 158136 | 0 | 0 | 1461 |
| 148784 | 0 | 0 | 3474 |
| 153920 | 0 | 0 | 1563 |
| 130975 | 0 | 0 | 2386 |
| 132418 | 0 | 0 | 1526 |
| 174792 | 0 | 0 | 1637 |
| 183130 | 0 | 0 | 1028 |
| 175324 | 0 | 0 | 958 |

这类样本无法支持完整 discharge summary generation。若用 24h input 生成完整 DS，模型只能猜测大量住院过程，评价结果会混入严重的信息不足因素。

## 10. Full 版本的主要问题

`composed_full` 虽然最适合作为主实验，但也有一个新挑战：input 很长。

最长输入样本：

| HADM_ID | rows | input words | gold words | input/gold ratio |
|---:|---:|---:|---:|---:|
| 145095 | 2692 | 157649 | 3196 | 49.3 |
| 173189 | 1957 | 147049 | 1042 | 141.1 |
| 121492 | 2118 | 143142 | 2711 | 52.8 |
| 165133 | 1640 | 126951 | 2658 | 47.8 |
| 156549 | 1568 | 103153 | 727 | 141.9 |
| 199716 | 1487 | 99760 | 2842 | 35.1 |
| 198383 | 1431 | 87743 | 1510 | 58.1 |
| 104732 | 981 | 85373 | 969 | 88.1 |

这说明 full DS 不能直接简单塞给 LLM。它更适合验证你的方法是否能做 longitudinal evidence selection / scaffold / compression。

建议 DS pipeline：

```text
full-admission chronology
  -> evidence selection / event retriever
  -> section-aware scaffold
  -> discharge summary generation
  -> verifier / revision
```

如果直接使用 full input，需要先做 token budget 控制，例如：

- 按 section 选择证据：diagnosis / hospital course / discharge plan；
- 优先保留 admission/discharge notes、procedure notes、progress notes、major diagnoses、med changes、abnormal labs；
- 对高频 flowsheet/lab 做聚合，而不是逐条输入；
- 使用 hierarchical summarization。

## 11. 当前已有 DS 生成输出

当前已有生成目录：

```text
data/DS/generated/DG/deepseek_api_full
```

文件数为 100，文件名形如：

```text
48h_all_abs_<HADM_ID>.txt
```

注意：这些生成结果是基于原始 `data/DS/input` 的 24h 风格数据，不应直接作为 `composed_full` 的主实验结果。

## 12. 建议下一步

### 主实验数据

使用：

```text
data/DS_fixed_composed/full/input
data/DS_fixed_composed/full/gold
```

### Ablation 数据

使用：

```text
data/DS_fixed_composed/24h/input
data/DS_fixed_composed/24h/gold
```

并明确命名为：

```text
last-24h discharge-state summarization
```

而不是完整 discharge summary generation。

### 需要修的 evaluator

当前 `evaluation/evaluate_ds.py` 的 section extraction regex 对 Diagnosis 不够鲁棒。建议补充更多 heading variants，或者新增 LLM judge 版本。

### 建议优先跑的小实验

1. 在 `composed_full` 上选 10-20 个样本做 smoke test。
2. 实现 full-input evidence selector 或 hierarchical compressor。
3. 比较：
   - raw 24h direct generation
   - composed 24h direct generation
   - composed full direct generation
   - composed full + evidence selection
   - composed full + scaffold/revise
4. 用 section-level ROUGE/SapBERT + LLM judge 双评估。

## 13. 生成的审计文件

本次审计生成：

```text
analysis/ds_data_audit/ds_dataset_overview.csv
analysis/ds_data_audit/raw_24h_file_stats.csv
analysis/ds_data_audit/fixed_24h_file_stats.csv
analysis/ds_data_audit/fixed_full_file_stats.csv
analysis/ds_data_audit/composed_24h_file_stats.csv
analysis/ds_data_audit/composed_full_file_stats.csv
analysis/ds_data_audit/composed_full_gold_section_flags.csv
analysis/ds_data_audit/ds_data_quality_report.md
```

