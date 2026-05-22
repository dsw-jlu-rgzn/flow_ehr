# DeepSeek-v4-pro Delta Truth 29-Case Upper-Bound Validation

生成日期：2026-05-21

## 1. 实验目的

本实验用于验证：使用 `deepseek-v4-pro` 基于相邻两天 gold A&P 生成 trajectory / verifier truth 后，替代原 verifier，再经过 minimal reviser，是否能在小批量 case 上稳定提升指标。

原计划生成 30 例。当前 selected 文件本身包含 30 例，其中 `198275_day28` 多次返回空响应或 repair error，未能生成可用 trajectory truth。因此本轮有效闭环为 29 例，并将失败 case 单独记录。

失败 case：

```text
198275_day28
```

失败记录：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_30case/failed_truth_generation_cases.txt
```

## 2. 方法设置

Truth 生成：

```text
previous-day input + previous gold A&P
+ current-day input + current gold A&P
+ candidate V2 A&P
-> trajectory_delta_truth
-> verifier_truth
```

模型设置：

| 阶段 | 模型 |
|---|---|
| 一阶段 trajectory/verifier truth 生成 | `deepseek-v4-pro` |
| 空 verifier 二阶段补全 | `deepseek-v4-pro` |
| JSON repair | `deepseek-v4-flash` |
| minimal reviser | 主要使用 `deepseek-v4-flash`，复杂/空输出 case fallback 到 `deepseek-v4-pro` |
| LLM judge | `Qwen/Qwen3.6-35B-A3B` |

并发设置：

- truth 一阶段生成支持 `--workers`，本轮主要使用 3-4 worker；
- verifier 二阶段补全支持 `--workers`，本轮最终使用 2 worker 提高稳定性。

## 3. 数据产物

最终 29-case truth：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_30case/ap_delta_trajectory_verifier_truth_29_final.jsonl
```

29-case selected list：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_30case/selected_29_usable.json
```

29-case case list：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_30case/cases_29.txt
```

Truth revised outputs：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_30case/truth_revised_outputs/ap_delta_truth_verifier_revise_29case_pro/
```

Reviser 空输出 fallback 到 V2 的 case：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_30case/reviser_empty_fallback_to_v2.txt
```

## 4. Truth 生成质量概况

29 条有效 trajectory truth 均有 `problem_threads`。

其中 4 条 case 的 verifier truth 仍为空或补全失败：

```text
105351_day13
191230_day18
143054_day9
181596_day11
```

这 4 条在 minimal reviser 中等价于 no-op，保留原 V2 输出。因此当前 29-case 结果是偏保守估计，并不是纯 oracle upper bound。

## 5. ROUGE-L 结果

| Method | N | ROUGE-L F1 | Avg. Words | Missing Output | Delta vs Base |
|---|---:|---:|---:|---:|---:|
| `base` | 29 | 0.0728 | 497.1 | 0 | - |
| `v2` | 29 | 0.0730 | 468.2 | 0 | +0.0002 |
| `v2_judge` | 29 | 0.0709 | 479.3 | 0 | -0.0019 |
| `pro_delta_truth_revised` | 29 | 0.0816 | 388.8 | 0 | +0.0088 |

结论：

- `pro_delta_truth_revised` 的 ROUGE-L F1 最高。
- 相比 base 提升 `+0.0088`。
- 相比原始 V2 judge-revise 提升更明显，说明原 V2-reviser 在这些高风险 case 上没有稳定改善文本重叠。

## 6. UMLS CUI-F1 结果

| Method | N | Precision | Recall | CUI-F1 | Avg. Pred CUIs |
|---|---:|---:|---:|---:|---:|
| `base` | 29 | 0.4626 | 0.2765 | 0.3311 | 235.7 |
| `v2` | 29 | 0.4463 | 0.2588 | 0.3134 | 226.4 |
| `v2_judge` | 29 | 0.4445 | 0.2618 | 0.3145 | 228.1 |
| `pro_delta_truth_revised` | 29 | 0.5339 | 0.2522 | 0.3268 | 192.3 |

Paired CUI-F1：

| Comparison | Mean Delta F1 | Wins | Losses | Ties |
|---|---:|---:|---:|---:|
| `v2 - base` | -0.0176 | 10 | 18 | 1 |
| `v2_judge - base` | -0.0166 | 10 | 19 | 0 |
| `pro_delta_truth_revised - base` | -0.0043 | 15 | 13 | 1 |
| `pro_delta_truth_revised - v2` | +0.0133 | 17 | 5 | 7 |
| `pro_delta_truth_revised - v2_judge` | +0.0123 | 20 | 9 | 0 |

结论：

- `pro_delta_truth_revised` 的 CUI precision 最高，从 base 的 `0.4626` 提升到 `0.5339`。
- CUI-F1 明显高于 `v2` 和 `v2_judge`，但略低于 base。
- 主要原因是 revised output 更短，平均预测 CUI 从 base 的 `235.7` 降到 `192.3`，recall 有下降。
- 这说明当前真值 verifier 更擅长删 hallucination / unsupported concept，但补 missing concept 的能力仍不足。

## 7. Qwen LLM Judge 结果

Pairwise 设置：

```text
v2_judge
vs
pro_delta_truth_revised
```

评估模型：

```text
Qwen/Qwen3.6-35B-A3B
```

| Method | Wins | Coverage | Trajectory | Specificity | Grounding | Disposition | Unsupported | Missed | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v2_judge` | 10 | 3.45 | 2.90 | 3.38 | 2.52 | 3.17 | 1.31 | 1.07 | 2.93 |
| `pro_delta_truth_revised` | 17 | 3.72 | 3.62 | 3.55 | 3.41 | 3.69 | 0.69 | 1.34 | 3.66 |
| `tie` | 2 | - | - | - | - | - | - | - | - |

结论：

- `pro_delta_truth_revised` 相比 `v2_judge` 获得 17 胜、10 负、2 平。
- trajectory capture 从 `2.90` 提升到 `3.62`。
- evidence grounding 从 `2.52` 提升到 `3.41`。
- unsupported problem count 从 `1.31` 降到 `0.69`。
- missed key problem count 从 `1.07` 升到 `1.34`。

这与 CUI-F1 的现象一致：当前 truth verifier 明显减少 unsupported / hallucination，但可能过度删除或补充不足，导致 missed problem 略升。

## 8. 当前结论

当前 29-case 小批量实验支持以下结论：

1. 使用 gold-delta 生成的 verifier truth 能明显改善 trajectory capture、evidence grounding 和 unsupported hallucination。
2. 相比原始 V2 judge-revise，`pro_delta_truth_revised` 在 ROUGE-L、CUI-F1、Qwen LLM judge overall 上均更好。
3. 相比 base，ROUGE-L 提升，CUI precision 大幅提升，但 CUI-F1 略低，说明 recall / missing concept 仍是主要短板。
4. 当前不是完美 oracle upper bound，因为有 4 条 verifier 为空，3 条 revised output 使用 V2 fallback。因此正式全量实验前需要进一步稳定 truth generation 和 reviser。

## 9. 暴露的问题

### 9.1 Pro 一阶段生成仍不稳定

长 prompt 下有较多 case 只生成 trajectory，不生成 verifier。

改进方向：

- 固定两阶段生成：
  - Stage 1: gold A&P delta -> compact trajectory truth；
  - Stage 2: compact trajectory truth + candidate V2 -> verifier truth。

### 9.2 Verifier truth 偏删除，补漏不足

指标证据：

- unsupported 显著降低；
- CUI precision 显著提升；
- CUI recall 下降；
- Qwen missed count 升高。

改进方向：

- 在 verifier prompt 中强制区分：
  - unsupported claims to remove；
  - missing active problems to add；
  - stable carry-forward problems to preserve；
  - resolved problems to remove。
- 对每个 current gold active problem 要求至少给出一个 keep/add/remove decision。

### 9.3 Reviser 对复杂 verifier 指令不稳定

有 3 条 case 最终使用 V2 fallback：

```text
164479_day22
174752_day51
191230_day22
```

改进方向：

- reviser 也拆成 section-level 局部修订；
- 对空输出自动 retry；
- 如果 retry 失败，fallback 到 original V2 并记录。

## 10. 下一步建议

在生成全量 600+ 数据集前，建议先完成下面两个修复：

1. 改成稳定两阶段 truth generation。
2. 加强 missing active problem / carry-forward preservation 的约束。

然后重新跑 30-case，目标是：

- ROUGE-L 继续高于 base；
- CUI precision 保持提升；
- CUI recall 不低于 base 或至少不明显下降；
- Qwen unsupported 下降；
- Qwen missed 不上升。

如果这 30-case 修复版通过，再扩展到 600+ 全量会更稳。
