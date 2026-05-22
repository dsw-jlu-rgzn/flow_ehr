# Delta-Truth Verifier 2-Case Metric Comparison

生成日期：2026-05-21

## 1. 实验目的

本轮实验用于补充验证：使用 `gold A&P delta -> trajectory/verifier truth` 替代原始 verifier 后，除了 LLM pairwise judge 以外，基础自动指标是否也有同步提升。

对比方法包括：

| 方法 | 含义 |
|---|---|
| `base` | 原始 full generation 输出 |
| `v2` | Scaffold / memory-gated V2 输出，不加 judge-revise |
| `v2_judge` | 原始 V2 judge-revise 输出 |
| `delta_truth_revised` | 使用人工/伪真值 delta verifier 替代 verifier，再由 LLM minimal reviser 进行局部修订 |

当前只使用 2 个 smoke-test case：

```text
105351_day13
105351_day19
```

因此本报告的结论用于判断方向是否正确，不能作为最终统计显著性结论。

## 2. LLM Pairwise Judge

正式评估模型：

```text
Qwen/Qwen3.6-35B-A3B
```

比较对象：

```text
V2 original
vs
delta-truth verifier + LLM minimal reviser
```

结果：

| Method | Wins | Coverage | Trajectory | Specificity | Grounding | Disposition | Unsupported | Missed | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 | 0 | 3.0 | 2.5 | 3.5 | 2.5 | 4.0 | 1.0 | 2.5 | 3.0 |
| Delta-truth revised | 2 | 5.0 | 5.0 | 5.0 | 4.5 | 4.5 | 0.0 | 0.0 | 5.0 |

解释：

- `delta_truth_revised` 在两个 case 中均胜出。
- 提升主要来自 trajectory consistency、active problem coverage、missed problem reduction。
- Qwen judge 没有触发 JSON repair，说明本次评估结果格式稳定。

## 3. ROUGE-L 自动指标

ROUGE-L F1 只衡量生成文本与 gold A&P 的表层序列重叠，不能完整反映医学正确性，但可以作为基础指标观察是否出现明显退化。

| Method | N | ROUGE-L F1 | Avg. Pred Words | Missing Pred | Delta vs Base |
|---|---:|---:|---:|---:|---:|
| `base` | 2 | 0.0740 | 441.5 | 0 | - |
| `v2` | 2 | 0.0757 | 419.5 | 0 | +0.0017 |
| `v2_judge` | 2 | 0.0751 | 437.0 | 0 | +0.0011 |
| `delta_truth_revised` | 2 | 0.0854 | 404.5 | 0 | +0.0113 |

结论：

- `delta_truth_revised` 的 ROUGE-L F1 最高。
- 相比 `base` 提升约 `+0.0113`。
- 相比 `v2_judge` 也有提升，说明伪真值 verifier 并不是只被 LLM judge 偏好，在基础文本重叠指标上也没有退化。

## 4. UMLS CUI-F1 指标

CUI-F1 使用 UMLS 词表抽取医学概念，比较 prediction 与 gold A&P 的概念集合重叠。相比 ROUGE-L，它更关注医学实体/概念覆盖，但仍然不能判断方向性、时序状态和证据支持关系。

| Method | N | Precision | Recall | CUI-F1 | Avg. Pred CUIs | Avg. Gold CUIs |
|---|---:|---:|---:|---:|---:|---:|
| `base` | 2 | 0.4498 | 0.2509 | 0.3218 | 219 | 392 |
| `v2` | 2 | 0.4602 | 0.2530 | 0.3261 | 215 | 392 |
| `v2_judge` | 2 | 0.4789 | 0.2513 | 0.3295 | 206 | 392 |
| `delta_truth_revised` | 2 | 0.5171 | 0.2772 | 0.3604 | 210 | 392 |

Paired CUI-F1 差值：

| Comparison | Mean Delta F1 | Wins | Losses | Ties |
|---|---:|---:|---:|---:|
| `v2 - base` | +0.0043 | 1 | 1 | 0 |
| `v2_judge - base` | +0.0077 | 1 | 1 | 0 |
| `delta_truth_revised - base` | +0.0386 | 2 | 0 | 0 |
| `v2_judge - v2` | +0.0034 | 1 | 1 | 0 |
| `delta_truth_revised - v2` | +0.0343 | 2 | 0 | 0 |
| `delta_truth_revised - v2_judge` | +0.0309 | 2 | 0 | 0 |

结论：

- `delta_truth_revised` 的 CUI-F1 最高，相比 `base` 提升约 `+0.0386`。
- 它在两个 case 上都优于 `base`、`v2` 和 `v2_judge`。
- Precision 和 recall 同时提高，说明它不是单纯通过输出更长文本获得更多概念召回，而是在更短平均输出长度下获得了更好的医学概念匹配。

## 5. 综合结论

目前 2-case smoke test 支持以下判断：

1. `delta_truth_revised` 不仅在 Qwen LLM judge 下优于 V2，也在 ROUGE-L 和 UMLS CUI-F1 上优于 `base`、`v2`、`v2_judge`。
2. 原始 `v2_judge` 相比 `v2` 有轻微提升，但幅度较小；说明原 verifier/judge-revise 的收益有限。
3. 使用 gold-delta 生成的 trajectory/verifier truth 后，主要改善点是：
   - 减少 missed active problems；
   - 修正错误的病程状态；
   - 保留更正确的医学概念；
   - 在不增加输出长度的情况下提升 CUI-F1。
4. 当前最大限制是样本数只有 2，需要扩展到 30-case，再进一步扩展到完整 AP100。

## 6. 结果文件路径

ROUGE-L summary：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_2case_final/auto_metrics/ap_selected_auto_metrics_summary.csv
```

ROUGE-L detail：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_2case_final/auto_metrics/ap_selected_auto_metrics_detail.csv
```

UMLS CUI-F1 summary：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_2case_final/umls_cui_eval/ap100_umls_cui_f1_summary.csv
```

UMLS CUI-F1 paired comparison：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_2case_final/umls_cui_eval/ap100_umls_cui_f1_paired.csv
```

Qwen pairwise judge：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_2case_final/pairwise_v2_vs_delta_truth_revised_qwen36.csv
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_2case_final/pairwise_v2_vs_delta_truth_revised_qwen36_summary.csv
```

## 7. 下一步建议

下一步应把同样评估扩展到 30 cases：

```text
base
v2
v2_judge
delta_truth_revised
```

需要同时报告：

- Qwen LLM judge；
- ROUGE-L；
- UMLS CUI Precision / Recall / F1；
- unsupported / missed / trajectory consistency；
- case-level wins/losses。

如果 30-case 结果仍然保持同向提升，就可以把该实验作为 claim-level verifier upper bound / trajectory truth supervision 的核心证据。
