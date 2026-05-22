# V3.1 Targeted 20-case 实验报告

## 目的

V3.1 在 V3 基础上只做两类泛化优化，避免 case-specific、disease-specific、medication-specific、lab-specific 或固定词汇规则：

1. `must_carry_forward_sections`：保留上一日 A&P major heading 的文档连续性。如果 prior major heading 没有明确 resolved 或 contradicted，不直接删除，而是作为 primary/secondary/one-line update 保留。
2. `disposition_and_goals`：把 disposition、level of care、transfer/discharge trajectory、goals/code/family communication 作为独立文档轨道，不让它们和 active medical problem 竞争 promotion slot。

这两个优化都是 note-structure / evidence-role 层面的泛化规则，不依赖具体病例或固定医学词表。

## 路径

- 代码：`modeling/ap_memory_gated_scaffold_generation.py`
- Case list：`outputs/ap_memory_gated_scaffold/case_lists/ap100eval_v3_targeted_20_cases.txt`
- V3.1 生成输出：`outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v3_1_targeted20/`
- V3.1 scaffold：`outputs/ap_memory_gated_scaffold_ap100/scaffolds/ap100eval_generated_method2_gen_v3_1_targeted20/`
- V3.1 summary：`outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v3_1_targeted20_summary.csv`
- V3.1 v4-pro judge：`outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v3_1_targeted20_eval_judge_deepseek_v4_pro_detail.csv`

## 配置

- 生成模型：`deepseek-chat`
- 评估模型：`deepseek-v4-pro`
- baseline：`deepseek_api_full_gen / gen / method2`
- memory source：`baseline_method`
- prompt version：`v3`
- config name：`ap100eval_generated_method2_gen_v3_1_targeted20`
- generation-time judge/revise：未启用
- 评估集：同 V3 targeted 20

## 主要结果

| 方法 | n | augmented wins | baseline wins | ties | ROUGE delta | coverage delta | trajectory delta | specificity delta | grounding delta | disposition delta | unsupported delta | missed delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 no-judge | 20 | 5 | 9 | 6 | -0.004 | -0.05 | +0.25 | -0.40 | -0.30 | +0.05 | +0.65 | -0.10 |
| V3 gated scaffold | 20 | 8 | 9 | 3 | -0.006 | -0.20 | +0.35 | -0.40 | +0.15 | -0.20 | -0.25 | +0.05 |
| V3.1 carry-forward + disposition | 20 | 7 | 9 | 4 | -0.005 | +0.10 | +0.05 | -0.15 | -0.05 | +0.25 | -0.70 | +0.10 |

## 相比 V3 的直接变化

| 指标 | V3.1 augmented - V3 augmented |
|---|---:|
| active problem coverage | +0.40 |
| trajectory capture | +0.15 |
| plan specificity | +0.30 |
| evidence grounding | -0.05 |
| disposition context | +0.80 |
| unsupported problem count | -0.75 |
| missed key problem count | -0.60 |
| ROUGE-L | +0.00088 |

V3.1 达到了预期目标：在不加入固定词汇/模式匹配的情况下，补回 V3 的 recall 和 disposition，同时继续压低 unsupported。

## Scaffold 诊断

| 字段 | 均值 |
|---|---:|
| active_ap_problem_count | 5.40 |
| must_carry_forward_section_count | 5.55 |
| disposition_and_goals_present | 1.00 |
| watchlist_count | 2.95 |
| supportive_care_count | 3.85 |
| evidence_event_count | 13.30 |
| prior_problem_state_count | 6.40 |
| rejected_candidate_count | 4.10 |
| contradiction_count | 5.15 |
| previous_claim_revised_count | 0.60 |
| previous_claim_dropped_count | 0.70 |
| augmented_words | 364.70 |

这说明 V3.1 的行为和设计一致：它没有靠扩大 active problem 数量来提升覆盖，而是通过 must-carry-forward 与 disposition/goals 轨道保留 gold-relevant 内容。

## 结论

V3.1 是目前三版里更均衡的版本。

相对 V2：

- unsupported 从 `+0.65` 改到 `-0.70`，明显减少 unsupported active problem；
- coverage 从 `-0.05` 改到 `+0.10`；
- disposition 从 `+0.05` 改到 `+0.25`；
- plan specificity 从 `-0.40` 改到 `-0.15`；
- missed 从 `-0.10` 变成 `+0.10`，略有代价但远好于 V3 的 missed 上升。

相对 V3：

- missed 明显下降；
- unsupported 进一步下降；
- disposition 大幅恢复；
- winner 数从 8 降到 7，说明 v4-pro 的 pairwise winner 未明显提升，但细项更稳。

主要剩余问题：

- trajectory delta 从 V3 的 `+0.35` 回落到 `+0.05`，说明 carry-forward 机制让 note 更稳，但可能削弱“今日状态变化”的突出程度。
- evidence grounding 从 V3 的 `+0.15` 回到 `-0.05`，可能因为保留 prior sections 时证据不够显式。
- baseline wins 仍为 9，说明 targeted hard cases 中仍有部分 V3.1 无法修复。

## 是否可扩大实验

我建议可以把 V3.1 扩大到目前已完成的 71-case 子集，而不是直接全量 653。

理由：

- V3.1 已经修复 V3 最大问题：missed 上升；
- targeted 20 上 unsupported 降幅稳定；
- 优化规则是泛化的结构规则，不是 case patch；
- 但 winner 仍未压过 baseline，因此需要 71-case 检查是否在更宽 admission 分布上仍成立。

建议下一步：

1. 在 current v4-pro 71-case subset 上跑 V3.1。
2. 用 `deepseek-v4-pro` 评估同一 71 条。
3. 判断是否满足扩大标准：
   - unsupported delta 仍为负；
   - missed delta 不超过 +0.2；
   - trajectory delta 不低于 0；
   - coverage/disposition 不低于 V2。

如果 71-case 通过，再跑 AP100 120-case。
