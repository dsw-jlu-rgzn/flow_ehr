# AP100 Memory-Gated Scaffold V2 扩展验证报告

## 当前结论

在新生成的 AP100 数据集中，我们先完成了 120 个 patient-day 的扩展验证。结果与 targeted low-score set 不完全一致：

- ROUGE-L：V2 相对 direct baseline 平均略降；
- LLM evaluation judge：明显偏向 V2，尤其 judge/revise 版本；
- judge/revise：进一步提升 judge wins 和 clinical judge 指标，但 ROUGE 下降略大。

这说明 V2 的收益更偏向 clinical structure、trajectory、grounding 和减少 unsupported/missed problems，而不是提高和原始 gold note 的表面文本重合。

## 数据生成

此前 `data/` 目录只有 10 个 AP admission、57 个可评估 method2 patient-days。为了扩展验证，重新从 MIMIC-III raw 数据生成了一个 AP-eligible 100 admission 数据集。

关键修正：

- 原先随机抽 100 ICU admissions 时，只有 10 个 admission 含符合规则的 Physician Progress Note；
- 已修改 `processing/prepare_mimic3_tasks.py`，新增 `--require-ap-progress-notes`，先筛选有 AP progress note 的 admission，再抽样。

新数据集：

```text
data_ap100_ap/
```

统计：

| item | count |
|---|---:|
| AP admissions | 100 |
| gold progress-note days | 753 |
| method2 evaluable patient-days | 653 |

## 当前 120-Case 子集

第一轮没有直接跑 653 全量，而是选取 14 个 admission，共 120 个可评估 patient-days，用来验证更丰富样本上的趋势。

case list：

```text
outputs/ap_memory_gated_scaffold/case_lists/ap100eval_120_cases.txt
```

全量 653 case list 已准备：

```text
outputs/ap_memory_gated_scaffold/case_lists/ap100ap_full_653_cases.txt
```

## 实验配置

当前扩展验证只跑最接近真实部署的 generated-history setting：

| config | base 对照 | history source | scaffold | generation-time judge/revise |
|---|---|---|---|---|
| `ap100eval_generated_method2_gen_v2` | `deepseek_api_full_gen/gen/method2` | cumulative generated A&P | V2 | 否 |
| `ap100eval_generated_method2_gen_v2_judge_revise` | `deepseek_api_full_gen/gen/method2` | cumulative generated A&P | V2 revised | 是 |

输入公平性：

```text
direct baseline:
  current raw EHR + cumulative generated A&P

V2 no-judge:
  current raw EHR + same cumulative generated A&P + V2 scaffold

V2 judge/revise:
  current raw EHR + same cumulative generated A&P + revised V2 scaffold
```

没有使用当前 day gold A&P 作为生成输入。

## 主要结果

| variant | n | base ROUGE | V2 ROUGE | ROUGE delta | ROUGE wins | eval judge V2 wins | eval judge base wins | ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no-judge V2 | 120 | 7.389 | 7.234 | -0.156 | 46 | 71 | 12 | 37 |
| judge/revise V2 | 120 | 7.389 | 7.128 | -0.262 | 41 | 80 | 10 | 30 |

## Judge 指标 Delta

Delta 表示 V2 minus direct baseline。

| variant | coverage | trajectory | plan specificity | grounding | disposition | unsupported count | missed count |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-judge V2 | +0.567 | +0.883 | +0.483 | +0.700 | +0.525 | -0.642 | -0.675 |
| judge/revise V2 | +0.617 | +0.983 | +0.625 | +0.775 | +0.550 | -0.692 | -0.692 |

解释：

- V2 显著改善 LLM judge 认为的 active problem coverage、trajectory capture、plan specificity 和 evidence grounding；
- unsupported problem 和 missed key problem 均下降；
- judge/revise 版本比 no-judge V2 在 judge 指标上进一步增强；
- 但两者 ROUGE 均低于 direct baseline，说明 V2 可能改变了 gold note 的原始 wording/structure。

## 与 Targeted 10-Case 的差异

早期 targeted low-score set 中，V2 对 ROUGE 和 judge 都是正向。但 AP100 120-case 中：

- judge 仍强正向；
- ROUGE 转为轻微负向。

可能原因：

1. targeted set 偏向失败样本，V2 更容易修复明显错误；
2. AP100 子集包含更多普通样本，direct baseline 已经较强；
3. gold 是完整 progress note 或原始 A&P 风格，V2 生成更结构化，导致 ROUGE 不一定受益；
4. judge/revise 会让输出更保守，进一步降低与 gold wording 的重合。

## 当前判断

V2 在更大样本上没有证明 ROUGE 优势，但证明了较强的 clinical judge 优势。后续论文或实验叙述应避免说“全面优于 baseline”，更准确的表述是：

> Memory-gated scaffold V2 improves clinically judged continuity, trajectory capture, evidence grounding, and unsupported/missed problem rates, but may reduce ROUGE on broader AP samples because it changes note structure and wording.

## 路径

数据：

```text
data_ap100_ap/AP/input/
data_ap100_ap/AP/gold/
data_ap100_ap/AP/generated/DG/deepseek_api_full_gen/gen/method2/
```

V2 outputs：

```text
outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v2/
outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v2_judge_revise/
outputs/ap_memory_gated_scaffold_ap100/scaffolds/
outputs/ap_memory_gated_scaffold_ap100/generation_judges/
```

Summaries：

```text
outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v2_summary.csv
outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v2_judge_revise_summary.csv
outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v2_eval_judge_detail.csv
outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v2_judge_revise_eval_judge_detail.csv
outputs/ap_memory_gated_scaffold_ap100/ap100eval_v2_summary.csv
```

Scripts：

```text
processing/prepare_mimic3_tasks.py
processing/get_chronologies_AP.py
modeling/deepseek_api_generation.py
modeling/ap_memory_gated_scaffold_generation.py
evaluation/judge_augmented_ap.py
```

## 下一步

1. 若目标是 clinical judge 指标，建议继续跑 653 full set。
2. 若目标是 ROUGE，也许需要一个 V3：保留 V2 scaffold，但让 generation 更贴近 original note wording/heading。
3. 对 full set 建议先跑 no-judge V2，再决定是否跑 judge/revise，因为 judge/revise 成本更高，且 ROUGE 下降更多。
