# AP100 当前实验结论与定性分析

## 当前状态

截至目前，AP100 full 653 patient-day 的 V2 生成已经完成，但 Qwen/SiliconFlow 评估尚未全部完成。

| 项目 | 状态 |
|---|---:|
| V2 no-judge 生成 | 653 / 653 |
| V2 judge&revise 生成 | 653 / 653 |
| Qwen/SiliconFlow 评估 V2 no-judge | 372 / 653 |
| Qwen/SiliconFlow 评估 V2 judge&revise | 0 / 653 |

重要路径：

- V2 no-judge full summary: `outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2_summary.csv`
- V2 judge&revise full summary: `outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2_judge_revise_summary.csv`
- Qwen no-judge interim judge: `outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2_eval_judge_qwen_siliconflow_detail.csv`
- V3 targeted report: `analysis/ap_memory_gated_scaffold_v3_targeted20_report_zh.md`
- V3.1 targeted report: `analysis/ap_memory_gated_scaffold_v3_1_targeted20_report_zh.md`

## ROUGE-L 解释

当前 CSV 中 ROUGE-L 是小数形式，例如 `0.075`，等价于百分制报告中的 `7.5`。因此当前结果没有从“7 点多”掉到“0.07”，只是单位显示不同。

当前 full 653 的本地 ROUGE-L：

| 配置 | n | base ROUGE-L | augmented ROUGE-L | delta | augmented ROUGE wins |
|---|---:|---:|---:|---:|---:|
| V2 no-judge | 653 | 0.0750 | 0.0724 | -0.0026 | 274 |
| V2 judge&revise | 653 | 0.0750 | 0.0720 | -0.0030 | 241 |

ROUGE-L 偏低的原因主要有三点：

1. 当前自定义 ROUGE 是 whitespace token LCS，标点、模板、缩写、顺序变化都会被严格惩罚。
2. gold note 平均 823 words，生成平均 436-446 words，长度和模板范围不一致会压低 recall。
3. V2 更偏结构化 problem-state 表达，和真实 ICU note 的模板、原句、顺序不完全一致，因此语义上可能更对，但 lexical overlap 不一定更高。

所以 ROUGE-L 当前更适合作为辅助指标，不适合作为 A&P 状态跟踪的主指标。

## Qwen 评估中期结果

目前 Qwen/SiliconFlow 对 V2 no-judge 已完成 372 条评估，覆盖 52 个 admission。

Winner 分布：

| Winner | Count |
|---|---:|
| V2 no-judge | 163 |
| Base | 116 |
| Tie | 93 |

细项指标：

| 指标 | Base | V2 no-judge | Delta |
|---|---:|---:|---:|
| active problem coverage | 3.129 | 3.153 | +0.024 |
| trajectory capture | 2.522 | 2.852 | +0.331 |
| plan specificity | 2.989 | 2.992 | +0.003 |
| evidence grounding | 2.269 | 2.435 | +0.167 |
| disposition context | 2.962 | 2.978 | +0.016 |
| unsupported problem count | 3.922 | 3.758 | -0.164 |
| missed key problem count | 3.366 | 3.312 | -0.054 |

当前 Qwen 评估支持一个较温和但正向的结论：

> V2 no-judge 相比 base 的主要真实收益是 trajectory capture；同时 evidence grounding、unsupported count、missed count 有小幅改善。

这和之前 `deepseek-v4-pro` 71-case 的悲观结论不同。可能原因包括：

- Qwen judge 对结构化 A&P 的接受度更高；
- 当前 372 条覆盖 admission 更广，不再只集中于早期 71 条；
- 不同 judge 模型对“raw-supported but gold-underwritten”的内容惩罚力度不同。

因此目前最稳妥的说法是：**V2 的 trajectory 提升较可信，整体质量提升仍需要 full 653 Qwen 评估和 judge&revise 对照确认。**

## 是否是真提升

我认为“trajectory capture 提升”比较可信，原因是：

- 在多个评估模型和多个子集上，trajectory 都是最稳定的正向项；
- 定性 case `121846_day24` 显示 base 会把 re-intubated 写成 extubated，而 V2 能修正这种 longitudinal state drift；
- Qwen 372 条中 trajectory delta 为 +0.331，是所有细项中最明显的收益。

但“整体 A&P 质量显著提升”还不能下最终结论，原因是：

- ROUGE-L 下降，说明 V2 没有更接近 gold wording；
- Qwen no-judge 评估还没跑完 full 653；
- judge&revise full 653 尚未用 Qwen 评估；
- 不同 judge 模型结论有差异，说明 LLM judge 本身仍有不确定性。

当前可以确认的提升项：

- trajectory capture：较可信；
- evidence grounding：Qwen 下小幅正向，需 full 确认；
- unsupported / missed：Qwen 下小幅改善，方向有利但幅度不大；
- ROUGE：没有提升。

## V2 失败模式

从 V2 的定性分析和 v4-pro/Qwen 评估看，主要失败模式是：

1. V2 有时会把弱证据升级成 active problem。
   典型来源包括单个 medication administration、单个 lab abnormality、case-management 信息、previous generated note 中的错误状态。

2. V2 final generation 有时会加入 plausible but unsupported 的计划。
   例如自动扩展 consult、antibiotics、dialysis coordination、ventilator changes 等，临床上合理但 gold/raw evidence 不一定支持。

3. judge&revise 可能进一步降低 ROUGE。
   因为 revise 倾向于把输出改得更保守、更结构化、更标准化，减少 unsupported，但也更偏离 gold note 的原始措辞。

4. V2 对 disposition/code status 的处理不够稳定。
   这类信息不一定是 active medical problem，但 gold A&P 往往会保留，因此会影响 disposition_context 和 missed count。

## V3 / V3.1 启示

V3 targeted 20 的目的不是最终性能，而是验证 V2 失败模式能不能被泛化修复。

V3 结果：

- unsupported delta 从 V2 targeted 的 +0.65 改到 -0.25；
- evidence grounding 改善；
- 但 missed key problem 上升，说明 gate 太硬。

V3.1 结果：

- unsupported delta 进一步到 -0.70；
- missed delta 控制在 +0.10；
- disposition delta 改善到 +0.25；
- 但 trajectory delta 从 V3 的 +0.35 回落到 +0.05。

这说明 V3.1 的两个泛化结构是有价值的：

- `must_carry_forward_sections` 能减少 missed；
- `disposition_and_goals` 能补回 disposition/context；
- 但过多 carry-forward 会稀释今日状态变化，需要更好地突出 changed state。

## 优化建议

下一步不要直接继续扩大 V3.1 到 full 653，建议先完成当前 V2 的 full evaluation，再决定。

优先级如下：

1. 补完 Qwen/SiliconFlow 的 V2 no-judge full 653 评估。
2. 跑 Qwen/SiliconFlow 的 V2 judge&revise full 653 评估。
3. 对比 no-judge vs judge&revise：
   - 如果 judge&revise 降低 unsupported 但 missed/ROUGE/coverage 变差，要谨慎使用；
   - 如果 judge&revise 在 Qwen 下也稳定提升 trajectory 和 grounding，再考虑保留。
4. 如果 V2 full 结果确认收益主要是 trajectory，下一版 V3.1 应优化为：
   - 保留 `must_carry_forward_sections`；
   - 保留 `disposition_and_goals`；
   - 增加 `changed_state_highlight`，专门突出今天发生变化的状态，避免 carry-forward 稀释 trajectory。

V3.1 的下一版不应加入固定疾病、药物、lab、关键词规则。优化仍应停留在证据角色和文档结构层面。

## 当前结论

当前最稳妥的实验结论是：

> V2 scaffold 对 AP100 的主要收益是真实但有限的，集中体现在 longitudinal trajectory capture。它能够减少部分历史状态漂移，但不保证更高 ROUGE，也不一定全面提升 gold-note mimicry。Qwen/SiliconFlow 372 条 interim 评估显示 V2 no-judge 相比 base 有正向趋势，但 full 653 和 judge&revise 的异模型评估仍需补完后才能形成最终结论。

