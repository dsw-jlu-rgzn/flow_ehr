# V3 Targeted 20-case 实验报告

## 实验目的

本实验验证 `Evidence-Hierarchy Gated Scaffold V3` 的最小闭环是否能修复 V2 在 `deepseek-v4-pro` 异模型 judge 下暴露出的主要失败模式：V2 能改善 longitudinal trajectory，但容易把弱证据、单个 lab/med、previous generated note 的错误状态升级成 unsupported active problem。

V3 不写 case-specific、disease-specific、medication-specific 规则，只使用通用 evidence hierarchy：

- prior active problem + today support 可以进入 active A&P；
- clinician assessment/plan 明确讨论可以进入 active A&P；
- procedure/imaging/consult 产生诊疗决策可以进入 active A&P；
- major state change 可以进入 active A&P；
- isolated medication、isolated lab、routine ICU care、case-management only、previous generated note only 默认不能进入 active A&P。

## 路径

- V3 代码：`modeling/ap_memory_gated_scaffold_generation.py`
- V3 targeted case list：`outputs/ap_memory_gated_scaffold/case_lists/ap100eval_v3_targeted_20_cases.txt`
- V3 生成输出：`outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v3_targeted20/`
- V3 scaffold：`outputs/ap_memory_gated_scaffold_ap100/scaffolds/ap100eval_generated_method2_gen_v3_targeted20/`
- V3 summary：`outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v3_targeted20_summary.csv`
- V3 v4-pro judge detail：`outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v3_targeted20_eval_judge_deepseek_v4_pro_detail.csv`
- V2 v4-pro judge detail：`outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v2_eval_judge_deepseek_v4_pro_detail.csv`

## 实验配置

- 生成模型：`deepseek-chat`
- 评估模型：`deepseek-v4-pro`
- baseline：`deepseek_api_full_gen / gen / method2`
- memory source：`baseline_method`
- prompt version：`v3`
- generation-time judge/revise：未启用
- 评估集：20 条 targeted diagnostic set

Targeted set 构成：

- 10 条 V2 在 v4-pro judge 下最差的 case；
- 5 条 V2 胜出但仍有 unsupported 问题的 case；
- 5 条 V2 tie 且 unsupported delta 为正的 case。

## 主要结果

| 方法 | n | augmented wins | baseline wins | ties | ROUGE delta | coverage delta | trajectory delta | grounding delta | unsupported delta | missed delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 no-judge, same targeted 20 | 20 | 5 | 9 | 6 | -0.00439 | -0.05 | +0.25 | -0.30 | +0.65 | -0.10 |
| V3 gated scaffold | 20 | 8 | 9 | 3 | -0.00625 | -0.20 | +0.35 | +0.15 | -0.25 | +0.05 |

直接比较 V3 augmented 与 V2 augmented：

| 指标 | V3 - V2 |
|---|---:|
| active problem coverage | -0.20 |
| trajectory capture | -0.20 |
| plan specificity | -0.10 |
| evidence grounding | +0.45 |
| disposition context | -0.50 |
| unsupported problem count | -0.75 |
| missed key problem count | +0.85 |
| ROUGE-L | -0.00186 |

注意：V2 与 V3 都是 pairwise judge against base，baseline 分数会随 pair context 略有变化，因此最稳妥的比较是方向性趋势和 augmented 细项，而不是把两次 judge 当成完全同一个 absolute scale。

## 结论

V3 的方向是有效的，但当前版本过于保守且仍有 hallucination 残留。

有效点：

- unsupported problem count 明显下降：V2 targeted 20 为 `+0.65`，V3 为 `-0.25`。
- evidence grounding 从 V2 的 `-0.30` 改到 V3 的 `+0.15`。
- winner 从 V2 的 `5/9/6` 改到 V3 的 `8/9/3`，说明 V3 在 targeted failure set 上确实修复了一部分失败。
- trajectory delta 仍为正：V3 `+0.35`，说明门控没有完全破坏 longitudinal 目标。

主要问题：

- missed key problem count 上升，V3 augmented 比 V2 augmented 平均多 `+0.85`。
- disposition context 下降明显，V3 常漏掉 code status、transfer、hospice、family meeting、floor disposition 等 gold A&P 关注项。
- active problem coverage 下降，说明 gate 把部分该保留的 chronic/secondary active problems 也降级了。
- 个别失败 case 仍出现 hallucination，例如 dialysis/instability、dramatic hypoxemia、unsupported shock、unsupported NSTEMI 等。

## V3 失败模式

1. Gate 降低 unsupported，但 recall 变差。
   V3 更少写无证据 active problem，但也更容易漏掉 gold 中的 chronic active problem、secondary ICU issue、code/disposition。

2. “clinician assessment/plan” 没有被充分优先。
   一些 gold A&P 的问题来自 physician note 的问题列表，而 V3 对 raw input 中的局部 evidence 更敏感，对 prior/gold-like heading continuity 不够稳。

3. Conservative final prompt 太短。
   V3 平均输出约 352 words，明显比 V2 更短，因此 plan specificity、disposition、secondary active problem 容易被压掉。

4. Weak evidence hallucination 仍未完全消除。
   虽然 prompt 禁止从单个 lab/med 推断，但模型仍会在部分 case 中创造 dramatic deterioration 或 unsupported diagnosis。

## V3.1 建议

下一版不应放弃 V3，而应做 recall-preserving gate：

1. 增加 `must_carry_forward_sections`。
   prior A&P 的 major heading 如果今天没有明确 resolved，不允许直接 drop；最多标为 `continued_uncertain`，final 中用一句 monitoring 保留。

2. 把 disposition/code status 作为独立保留轨道。
   `disposition_context` 不应和 active medical problem 抢 promotion slot；floor transfer、ICU stay、CMO/hospice、family meeting、code status 应单独进入 `disposition_and_goals`。

3. 区分 `active_primary_problem` 和 `active_secondary_problem`。
   当前 V3 只有 active/watchlist/supportive，容易把 secondary but gold-relevant 的问题丢掉。建议 schema 加：
   - `primary_ap_sections`
   - `secondary_ap_updates`
   - `routine_supportive_care`

4. 对 high-risk state hallucination 加闭合枚举。
   对 ventilation、pressor、antibiotics、nutrition、dialysis、CMO 等状态，要求输出：
   `present / absent / unchanged / unclear`，不能自由生成。

5. final prompt 放宽长度但不放宽证据。
   允许每个 carried-forward secondary problem 写 1 句，避免 missed 上升；但 plan action 仍必须来自 allowed actions。

## 是否建议扩大实验

不建议直接把当前 V3 扩大到 120 或 653 作为最终版本，因为 missed key problem 上升太明显。

建议先实现 V3.1 的两项最小修正：

- `must_carry_forward_sections`
- `disposition_and_goals`

然后在同一 targeted 20 上复跑。如果 V3.1 能保持 unsupported 下降，同时把 missed delta 拉回到接近 0，再扩大到 71/120。
