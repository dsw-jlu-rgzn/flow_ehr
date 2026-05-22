# ICU A&P Generation 当前实验整理

本文档整理当前项目的论文动机、introduction 草稿、核心 insight、基础实验指标、实验进展、已有消融实验和证明性实验。它的目标是作为后续论文撰写和实验补齐的工作底稿。

## 1. 任务背景与 Motivation

ICU Assessment & Plan (A&P) 生成不是普通的临床摘要任务。普通摘要更关注从当前病历中压缩信息，而 ICU A&P 需要持续维护一个随时间变化的 active problem state。

在连续 ICU admission 中，一个高质量 A&P 至少需要处理以下问题：

- 哪些历史问题今天仍然 active，需要继续 carry forward？
- 哪些历史问题已经 resolved、downgraded 或不应再出现在 active A&P 中？
- 哪些当日新证据应该升级成新的 active problem？
- 每个 active problem 的 trajectory、assessment、plan、disposition 是否被正确更新？
- 模型是否引入了当前证据不支持的药物、操作、诊断、device status、pressor/ventilation/dialysis 状态、code status 或 disposition？

直接让 LLM 根据当日 EHR 输入生成 A&P 容易出现两个结构性失败：

1. **Problem-state drift**：active problem list 不稳定，表现为遗漏关键 active problems、错误继承旧问题、把 PMH 写成 active problem，或者在长住院后期丢失病程主线。
2. **Post-generation hallucination**：即使输入中有正确线索，LLM 在最终 A&P 中仍可能添加 unsupported details，例如错误的抗生素、透析状态、呼吸机状态、输血、出血、感染或处置计划。

因此，当前方法的核心 motivation 是：

> ICU A&P generation requires explicit longitudinal problem-state tracking and evidence-grounded post-generation verification. A scaffolded problem-state representation can guide generation, while verifier/reviser modules can reduce unsupported claims, recover missed updates, and improve longitudinal trajectory consistency.

## 2. Introduction 草稿

ICU patients often have long and complex admissions where the clinically relevant Assessment & Plan evolves over time. Generating a daily A&P requires more than summarizing the current note: the model must preserve active problems, retire resolved problems, incorporate new evidence, and update plans without introducing unsupported clinical claims. Existing LLM-based generation pipelines tend to treat each day as a local generation task, which makes them vulnerable to longitudinal drift. In later hospital days, models may lose track of the active problem list, carry forward stale diagnoses, or hallucinate problem threads that are no longer supported by the evidence.

To address this, we study scaffolded A&P generation for ICU longitudinal problem tracking. Our V2 system first constructs a structured problem-state scaffold from historical A&P and current-day evidence, then generates the final A&P conditioned on this scaffold. We further explore judge/revise and verifier-guided revision modules that inspect the generated output for unsupported claims, missing updates, and forgotten carried-forward problems. Our experiments show that the base generator degrades over time within the same admission, while V2 provides larger improvements in later admission stages. However, claim-level verification alone is insufficient: remaining failures concentrate in wrong active problem selection and stale problem carry-over, motivating a stronger problem-level verifier.

## 3. 核心 Insight

### 3.1 A&P 的关键不是摘要，而是 problem-state maintenance

当前实验反复显示，A&P 失败常常不是某一句话写错，而是整个 active problem thread 错了。例如模型可能继续围绕 pneumonia、GI bleed、pressor、dialysis 或 anticoagulation 写一整段，但当天 gold A&P 的主线已经转向另一个 active problem。

因此，方法设计不能只做 sentence-level factual correction，而需要：

- 识别 active / inactive / uncertain problems；
- 决定哪些 problem sections 必须保留、删除或重写；
- 对每个 problem section 再做 claim-level evidence grounding。

### 3.2 V2 的主要价值是缓解 longitudinal drift

Full 653 Qwen/SiliconFlow 评估显示，base generator 在同一 admission 后期质量下降明显。V2 的绝对质量不一定随 day 单调上升，但它相对 base 的收益在后期变大。

这说明 V2 更像是在抵消 base 的后期退化，而不是让所有后期病例都变得容易。

### 3.3 Claim-level verifier 有用，但不是充分解法

Claim-level verifier 可以降低 unsupported details，但如果它只会 KEEP/FIX/DELETE 单句 claims，就容易出现两个问题：

- 删除后 note 变短，coverage 和 specificity 下降；
- 对 wrong problem thread 无能为力，因为它需要整段 remove/rebuild，而不是只删一句。

因此，当前最重要的下一步是 problem-level verifier + problem-first reviser。

## 4. 当前方法概述

### 4.1 Base

Base 是直接 A&P generation baseline。它主要依赖当前输入和原始上下文进行生成，没有显式 problem-state scaffold，也没有后置 verifier/reviser。

### 4.2 V2 scaffold-only

V2 是当前 scaffolded generation 主方法。工作流为：

```text
historical A&P / memory
+ current-day EHR input
  -> memory-gated scaffold builder
  -> problem-state scaffold
  -> LLM A&P generation
```

V2 scaffold 中包含：

- active A&P problems
- watchlist problems
- supportive care
- carried-forward prior problems
- rejected candidate problems
- contradiction / low-confidence signals

它的作用是 pre-generation control：先把 problem state 显式化，再让 LLM 生成 A&P。

### 4.3 V2 judge-revise

V2 judge-revise 在 scaffold/generation 后增加 generation judge 和 revision：

```text
scaffold
  -> LLM A&P generation
  -> generation judge
  -> revised scaffold / revised generation
```

judge 主要检查：

- unsupported changes
- missing updates
- forgotten carried-forward problems
- scaffold revision suggestions

相对 V2 scaffold-only，judge-revise 能进一步恢复部分 missing updates，并减少部分 unsupported changes。

### 4.4 Claim-level verifier upper-bound

为了验证 verifier/reviser 方向的上限，我们做了 oracle / pseudo-oracle claim-level verifier 实验：

```text
V2 output
  -> claim-level verifier truth
  -> LLM minimal evidence-grounded reviser
  -> judge evaluation
```

实验目的不是部署真实系统，而是回答：

> 如果 verifier 接近真值，后置修订能带来多大提升？

该实验进一步分为：

- pseudo-oracle claim verifier：自动/半自动生成 KEEP/FIX/DELETE/REWRITE；
- curated claim verifier：人工修正明显错误，并加入 missing supported claims / carried-forward problems。

## 5. 方法形式化与数学表达

本节给出当前 V2 judge-revise 方法的形式化表达，用于支撑论文方法段。核心思想是把 A&P generation 表述为一个 longitudinal problem-state tracking 问题，而不是单日文本生成问题。

### 5.1 Longitudinal input

对同一个 admission 的第 \(t\) 天，输入可以表示为：

```math
X_t = \{E_t, A_{<t}, P_{<t}\}
```

其中：

- \(E_t\)：当天 EHR evidence，包括 note、labs、medications、procedures、device status 等。
- \(A_{<t}\)：历史 A&P 文本。
- \(P_{<t}\)：历史 problem states 或 memory。
- \(X_t\)：第 \(t\) 天用于生成 A&P 的完整纵向输入。

目标是生成当天 A&P：

```math
Y_t = \{y_{t,1}, y_{t,2}, \ldots, y_{t,n}\}
```

其中 \(y_{t,i}\) 表示第 \(i\) 个 A&P problem section 或 section-level statement。

### 5.2 Base generation

Base 方法可以抽象为直接生成：

```math
Y_t^{base} = g_{\phi}^{base}(E_t, A_{<t})
```

它没有显式建模当天的 problem state，也没有独立的 post-generation verifier。因此，base 容易在 admission 后期出现 problem-state drift。

### 5.3 V2 scaffold construction

V2 首先构造当天的 problem-state scaffold：

```math
S_t = f_{\theta}(E_t, A_{<t}, P_{<t})
```

其中 scaffold 可以拆解为：

```math
S_t =
\left\{
P_t^{active},
P_t^{carry},
P_t^{watch},
P_t^{reject},
C_t
\right\}
```

各部分含义如下：

- \(P_t^{active}\)：当天应进入 A&P 的 active problems。
- \(P_t^{carry}\)：需要从历史 A&P 中继续 carry forward 的问题。
- \(P_t^{watch}\)：需要关注但不一定写入 active A&P 的 watchlist problems。
- \(P_t^{reject}\)：不应写入 A&P 的 rejected candidate problems。
- \(C_t\)：contradiction 或 low-confidence signals。

该步骤的作用是把隐式的 longitudinal clinical state 显式化，从而降低直接生成时的 active problem drift。

### 5.4 Scaffold-guided generation

V2 的初始 A&P 生成过程为：

```math
Y_t^{0} = g_{\phi}(E_t, A_{<t}, S_t)
```

其中 \(Y_t^{0}\) 是 scaffold-guided generation 的初版输出。相比 base，V2 的生成过程显式依赖 \(S_t\)，因此更容易保留历史 active problems 并更新当前 trajectory。

### 5.5 Generation judge

V2 judge-revise 在初版输出后加入 judge：

```math
J_t = h_{\psi}(Y_t^{0}, E_t, S_t)
```

其中 judge 输出可以表示为：

```math
J_t =
\left\{
U_t,
M_t,
F_t,
R_t
\right\}
```

各部分含义如下：

- \(U_t\)：unsupported claims 或 unsupported problem threads。
- \(M_t\)：missed updates 或 missed active problems。
- \(F_t\)：forgotten carried-forward problems。
- \(R_t\)：revision suggestions。

这个模块用于诊断生成结果与 evidence/scaffold 之间的不一致。

### 5.6 Revision

最终输出由 reviser 生成：

```math
Y_t^{*} = r_{\omega}(Y_t^{0}, E_t, S_t, J_t)
```

也可以写成组合形式：

```math
Y_t^{*}
=
r_{\omega}
\left(
Y_t^{0},
E_t,
S_t,
h_{\psi}(Y_t^{0}, E_t, S_t)
\right)
```

其中 \(Y_t^{*}\) 是 V2 judge-revise 的最终 A&P。

从优化角度，可以把 revision 理解为在保证质量的同时惩罚 unsupported、missed 和 drift：

```math
Y_t^{*}
=
\arg\max_Y
\left[
Q(Y, E_t, S_t)
- \lambda_1 U(Y, E_t)
- \lambda_2 M(Y, S_t)
- \lambda_3 D(Y, S_t)
\right]
```

其中：

- \(Q(Y, E_t, S_t)\)：整体 A&P quality。
- \(U(Y, E_t)\)：unsupported hallucination penalty。
- \(M(Y, S_t)\)：missed active problem penalty。
- \(D(Y, S_t)\)：deviation from scaffold 或 longitudinal problem-state drift penalty。
- \(\lambda_1, \lambda_2, \lambda_3\)：不同错误项的权重。

该式子不是实际训练目标，而是对 LLM judge-revise pipeline 的概念化描述，用于说明方法设计目标。

### 5.7 Drift mitigation evaluation

为了衡量 V2 是否缓解 trajectory drift，定义相对 base 的质量提升：

```math
\Delta_t^q
=
q(Y_t^{method}, G_t)
-
q(Y_t^{base}, G_t)
```

其中：

- \(G_t\)：当天 gold A&P 或 reference evaluation target。
- \(q(\cdot)\)：LLM judge 或 human judge 给出的综合质量分数。
- \(Y_t^{method}\)：V2 或 V2 judge-revise 输出。
- \(\Delta_t^q\)：方法相对 base 的 quality improvement。

如果方法确实缓解 admission 后期 drift，则期望：

```math
\mathbb{E}[\Delta_t^q \mid t \in \text{late}]
>
\mathbb{E}[\Delta_t^q \mid t \in \text{early}]
```

对应当前实验结果：

- V2：early improvement 为 \(-1.67\)，late improvement 为 \(+1.81\)。
- V2 judge-revise：early improvement 为 \(-1.50\)，late improvement 为 \(+2.62\)。

也可以在 admission 内定义趋势斜率。对每个 admission \(a\)，拟合：

```math
\Delta_{a,t}^q = \alpha_a t + \beta_a
```

若 \(\alpha_a > 0\)，说明该 admission 中方法相对 base 的收益随时间增加。当前实验中：

- V2 的 quality improvement 正斜率 admission 比例为 \(78.7\%\)。
- V2 judge-revise 的 quality improvement 正斜率 admission 比例为 \(72.9\%\)。

### 5.8 Trajectory capture evaluation

为了单独验证 longitudinal tracking，定义 trajectory capture score：

```math
\tau(Y_t, G_t)
```

其中 \(\tau\) 衡量生成 A&P 是否正确捕捉患者的纵向病程变化。

方法相对 base 的 trajectory capture improvement 为：

```math
\Delta_t^{\tau}
=
\tau(Y_t^{method}, G_t)
-
\tau(Y_t^{base}, G_t)
```

如果 V2 确实改善 trajectory tracking，则应满足：

```math
\Delta_t^{\tau} > 0
```

并且如果它主要缓解后期 drift，则应满足：

```math
\mathbb{E}[\Delta_t^{\tau} \mid t \in \text{late}]
>
\mathbb{E}[\Delta_t^{\tau} \mid t \in \text{early}]
```

当前 trajectory capture 单独实验支持这一点：

- V2 relative-progress trajectory improvement：early 为 \(-0.04\)，late 为 \(+0.48\)。
- V2 judge-revise relative-progress trajectory improvement：early 为 \(+0.03\)，late 为 \(+0.65\)。
- V2 judge-revise 在 \(>28\) hospital day 的 trajectory improvement 为 \(+0.62\)。

### 5.9 Claim-level verifier upper-bound formulation

Claim-level verifier 实验可以形式化为：

```math
C_t = \text{claim\_extract}(Y_t^{0})
```

其中 \(C_t = \{c_{t,1}, \ldots, c_{t,m}\}\) 是从初版 A&P 中抽取的 claims。

oracle 或 pseudo-oracle verifier 对每个 claim 给出标签：

```math
z_{t,i} = v(c_{t,i}, E_t, G_t)
```

其中：

```math
z_{t,i} \in \{\text{KEEP}, \text{FIX}, \text{DELETE}, \text{REWRITE}\}
```

然后 reviser 根据 verifier truth 生成：

```math
Y_t^{oracle-revise}
=
r_{\omega}(Y_t^0, E_t, \{(c_{t,i}, z_{t,i})\}_{i=1}^{m})
```

这个实验的目标是估计 claim-level verifier 的上限，而不是模拟可部署系统。实验结果表明：

- pseudo-oracle claim-only verifier 不足以超过 V2 judge-revise；
- curated claim verifier 能降低 unsupported/missed，但仍无法稳定修复 wrong problem thread；
- 因此后续需要 problem-level verifier。

### 5.10 Problem-level verifier 的后续形式

后续 problem-level verifier 可以写成：

```math
K_t = \pi_{\eta}(Y_t^0, E_t, S_t)
```

其中：

```math
K_t =
\left\{
P_t^{remove},
P_t^{rewrite},
P_t^{must},
P_t^{forbid},
P_t^{resolved}
\right\}
```

各部分含义：

- \(P_t^{remove}\)：需要删除的错误 problem threads。
- \(P_t^{rewrite}\)：需要整段重写的 problem threads。
- \(P_t^{must}\)：最终 A&P 必须覆盖的 active problems。
- \(P_t^{forbid}\)：最终 A&P 禁止添加的问题。
- \(P_t^{resolved}\)：已经 resolved 或 inactive 的问题。

最终 problem-first revision 可以表示为：

```math
Y_t^{problem-revise}
=
r_{\omega}^{problem}(Y_t^0, E_t, S_t, J_t, K_t)
```

这对应当前 TODO 中的下一步方法升级：从 claim-level sentence revision 转向 problem-level section rebuild。

## 6. 基础实验指标

当前主要使用 LLM-as-judge 评价。核心指标包括：

| Metric | 含义 | 趋势 |
|---|---|---|
| active_problem_coverage | 是否覆盖当天重要 active problems | 越高越好 |
| trajectory_capture | 是否正确捕捉病程轨迹和状态变化 | 越高越好 |
| plan_specificity | plan 是否具体、可执行、贴合问题 | 越高越好 |
| evidence_grounding | 内容是否有证据支撑 | 越高越好 |
| disposition_context | 是否正确表达 disposition/goals/context | 越高越好 |
| unsupported_problem_count | 不被证据支持的问题数 | 越低越好 |
| missed_key_problem_count | 漏掉的关键问题数 | 越低越好 |
| augmented_wins / baseline_wins / ties | pairwise judge 偏好 | wins 越高越好 |

注意：unsupported 和 missed 是负向指标；其他主要维度是正向指标。

## 7. 主实验进展

### 7.1 AP100 evaluation: V2 与 V2 judge-revise

在 AP100 evaluation 中，V2 和 V2 judge-revise 相对 base 均取得明显 LLM-judge 改善。该实验是当前 scaffold/revise 方法的基础有效性证据。

数据路径：

- `outputs/ap_memory_gated_scaffold_ap100/ap100eval_v2_summary.csv`

| method | n | judge wins | baseline wins | ties | coverage delta | trajectory delta | specificity delta | grounding delta | disposition delta | unsupported delta | missed delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 scaffold-only | 120 | 71 | 12 | 37 | +0.567 | +0.883 | +0.483 | +0.700 | +0.525 | -0.642 | -0.675 |
| V2 judge-revise | 120 | 80 | 10 | 30 | +0.617 | +0.983 | +0.625 | +0.775 | +0.550 | -0.692 | -0.692 |

主要结论：

- V2 相对 base 显著提升 coverage、trajectory、specificity、grounding 和 disposition。
- V2 同时降低 unsupported 和 missed counts。
- judge-revise 相比 scaffold-only 进一步提升 trajectory、specificity、grounding，并增加 judge wins。

这说明 scaffold 本身有效，后置 judge-revise 也有增益。

### 7.2 Full 653 trajectory drift analysis

为了验证 V2 是否解决轨迹偏移，我们在 full 653 Qwen/SiliconFlow 评估结果上分析了 admission 内趋势。

数据与报告路径：

- `analysis/trajectory_drift_v2/trajectory_drift_report.md`
- `analysis/trajectory_drift_v2/paper_figures/`
- 复现脚本：`scripts/plot_trajectory_drift_paper_figures.py`

关键发现：

| method / metric | abs day corr | relative progress corr | 解释 |
|---|---:|---:|---|
| V2 trajectory | -0.087 | +0.101 | 绝对 day 不明显变好，admission 内后期略升 |
| base trajectory | -0.214 | -0.128 | base 后期明显变差 |
| V2 trajectory improvement | +0.111 | +0.215 | V2 后期相对 base 提升更明显 |
| V2 quality improvement | +0.190 | +0.282 | V2 对后期 base 退化有抵消作用 |
| V2 unsupported | +0.271 | +0.088 | unsupported 后期仍增加 |
| V2 judge-revise trajectory improvement | +0.175 | +0.232 | judge-revise 后期相对提升更强 |
| V2 judge-revise quality improvement | +0.274 | +0.299 | 后期收益更明显 |

按 admission 相对进度分箱：

| method | stage | method quality | base quality | improvement | win rate |
|---|---|---:|---:|---:|---:|
| V2 | early | 6.83 | 8.50 | -1.67 | 28.1% |
| V2 | late | 6.83 | 5.02 | +1.81 | 49.2% |
| V2 judge-revise | early | 7.82 | 9.32 | -1.50 | 38.9% |
| V2 judge-revise | late | 8.00 | 5.38 | +2.62 | 58.0% |

按绝对 hospital day 分箱：

| method | day bin | method quality | base quality | improvement |
|---|---|---:|---:|---:|
| V2 | <=7 | 7.91 | 8.21 | -0.30 |
| V2 | >28 | 4.65 | 2.65 | +2.00 |
| V2 judge-revise | <=7 | 8.71 | 9.13 | -0.42 |
| V2 judge-revise | >28 | 7.04 | 3.20 | +3.84 |

结论：

> Base 存在明显 longitudinal degradation；V2 能缓解这种后期退化，尤其表现为 admission 后期相对 base 的收益变大。但 V2 尚未完全解决后期累积性 hallucination 和 problem-list drift。

### 7.3 Trajectory capture 单独验证

为了确认收益不是只来自综合 quality，我们额外单独绘制了 trajectory_capture 指标。

图路径：

- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_capture_benefit.pdf`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_capture_benefit.svg`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_capture_benefit.png`

关键数值：

| setting | early improvement | late improvement |
|---|---:|---:|
| V2, relative progress | -0.04 | +0.48 |
| V2 judge-revise, relative progress | +0.03 | +0.65 |

按绝对 hospital day：

| setting | <=7 | 8-14 | 15-28 | >28 |
|---|---:|---:|---:|---:|
| V2 trajectory improvement | +0.15 | +0.40 | +0.45 | +0.38 |
| V2 judge-revise trajectory improvement | +0.19 | +0.44 | +0.77 | +0.62 |

结论：

> V2 不只是提升综合质量，也确实提升 trajectory capture；V2 + judge-revise 在 trajectory capture 上进一步增强，尤其是在 admission 后期和长住院后期。

## 8. 已有消融实验和证明性实验

### 8.1 Scaffold-only vs judge-revise

AP100 evaluation 可以视为 scaffold-only 和 judge-revise 的消融：

- V2 scaffold-only 已显著优于 base；
- V2 judge-revise 在 wins、trajectory、specificity、grounding、unsupported、missed 上进一步改善。

这说明：

> scaffold 是主要结构化控制来源，judge-revise 是额外后置修正来源。

### 8.2 Claim-only pseudo-oracle verifier 上限实验

在 selected 30 cases 上，用 Qwen/Qwen2.5-72B-Instruct 重新评估 base/V2/V2 judge-revise/pseudo-oracle verifier。结果显示 claim-only pseudo-oracle verifier 没有超过 V2 judge-revise。

数据路径：

- `outputs/oracle_claim_verifier_qwen653/qwen25_selected30_upper_bound_comparison/upper_bound_comparison_summary.csv`

| method | coverage | trajectory | specificity | grounding | disposition | unsupported | missed | wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 | 4.53 | 4.17 | 4.53 | 4.50 | 4.20 | 0.50 | 0.47 | 25 |
| V2 judge-revise | 4.60 | 4.23 | 4.60 | 4.47 | 4.30 | 0.37 | 0.33 | 27 |
| V2 + pseudo-oracle verifier + LLM revise | 4.07 | 3.87 | 4.07 | 4.07 | 3.83 | 0.87 | 0.87 | 17 |

解释：

- claim-only verifier 倾向于删除 unsupported claims；
- 删除后 note 变短，coverage / specificity / trajectory 反而下降；
- 它没有充分补回 missed active problems 或 carried-forward problem threads。

这是一个重要 negative finding：

> Claim-level verification alone is not sufficient for robust A&P revision.

### 8.3 Curated claim-level verifier upper-bound

在人工修正后的 curated verifier truth 上，用 DeepSeek judge 评估 selected 30 cases。该实验说明：如果 verifier truth 更准确，并允许补充 missing/carry-forward items，后置 reviser 可以带来更明显收益。

数据路径：

- `outputs/oracle_claim_verifier_qwen653/curated_verifier_deepseek_upper_bound_comparison/upper_bound_comparison_summary_clean.csv`

| method | coverage | trajectory | specificity | grounding | disposition | unsupported | missed | wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 | 3.07 | 2.43 | 3.00 | 2.43 | 2.13 | 2.43 | 2.47 | 14 |
| V2 judge-revise | 3.20 | 2.67 | 3.10 | 2.77 | 2.20 | 2.10 | 2.23 | 18 |
| V2 + curated claim verifier + LLM revise | 3.27 | 2.63 | 3.03 | 2.80 | 2.37 | 1.77 | 2.03 | 22 |

相对 V2：

- unsupported 从 `2.43` 降到 `1.77`；
- missed 从 `2.47` 降到 `2.03`；
- evidence grounding 从 `2.43` 升到 `2.80`；
- wins 从 `14` 升到 `22`。

相对 V2 judge-revise：

- unsupported 从 `2.10` 降到 `1.77`；
- missed 从 `2.23` 降到 `2.03`；
- evidence grounding 从 `2.77` 升到 `2.80`；
- wins 从 `18` 升到 `22`。

结论：

> 更准确的 verifier/reviser 确实有上限收益，尤其能降低 unsupported 和 missed；但 curated claim-level verifier 仍没有稳定解决 problem-list 主线错误。

### 8.4 Admission-level slope 证明实验

为了避免分箱平均造成误导，我们对每个 admission 单独拟合：

```text
metric = a * hospital_day + b
```

然后统计 `a > 0` 的 admission 比例。

关键结果：

| setting | metric | positive slope admissions |
|---|---|---:|
| V2 | base quality | 32.0% |
| V2 | quality improvement | 78.7% |
| V2 | trajectory improvement | 72.0% |
| V2 | unsupported | 69.3% |
| V2 judge-revise | base quality | 28.8% |
| V2 judge-revise | quality improvement | 72.9% |
| V2 judge-revise | trajectory improvement | 71.2% |
| V2 judge-revise | unsupported | 74.6% |

结论：

> 在多数 admission 内，base quality 随时间下降，而 V2 相对 base 的 improvement 随时间增加。这说明 drift mitigation 不是单纯 case-mix 平均效应。但 unsupported 也经常随时间增加，提示 stale/unsupported problem carry-over 是当前主要残留失败。

## 9. 当前实验图与复现路径

论文图目录：

- `analysis/trajectory_drift_v2/paper_figures/`

复现脚本：

- `scripts/plot_trajectory_drift_paper_figures.py`

复现命令：

```bash
python scripts/plot_trajectory_drift_paper_figures.py
```

已生成图：

| Figure | 文件 | 作用 |
|---|---|---|
| Main drift figure | `fig_trajectory_drift_main.pdf/svg/png` | 证明 base 后期退化，V2 后期收益更大 |
| Absolute day figure | `fig_trajectory_drift_absolute_day.pdf/svg/png` | 证明长住院后期 base 退化，V2/JR 在后期收益更大 |
| Trajectory capture figure | `fig_trajectory_capture_benefit.pdf/svg/png` | 单独证明 trajectory_capture 指标也被提升 |
| Within-admission slope figure | `fig_trajectory_drift_within_admission_slopes.pdf/svg/png` | 证明趋势在 admission 内部成立，不只是分箱平均 |

图注和复现说明：

- `analysis/trajectory_drift_v2/paper_figures/figure_captions_and_reproduction.md`

## 10. 当前主要失败模式

### 10.1 Problem-list hallucination

很多失败不是单个 claim 错，而是整个 active problem thread 错。例如：

- 模型保留 pneumonia/fever/mini-BAL 主线，但 gold 重点已经变成 groin hematoma、alcohol withdrawal、AVR anticoagulation、IVC thrombosis。
- 模型添加 sepsis、PRBC、multiple antibiotics，但遗漏 UTI、altered mental status、cough。
- 模型引入 PMH/problem list 幻觉，例如 cirrhosis、colon cancer、IgA nephropathy，同时漏掉 decubitus ulcer、constipation、septic arthritis。

这说明当前 verifier 还没有稳定解决 active problem selection。

### 10.2 Claim-level 修订无法 rebuild section

对 wrong problem thread，reviser 需要做的是：

```text
remove entire wrong section
rebuild section around correct active problem
```

而不是：

```text
delete one unsupported sentence
keep old heading and surrounding plan
```

### 10.3 Reviser 可能引入新 unsupported details

即使 verifier truth 正确，LLM reviser 仍可能为了让段落更连贯而添加未授权细节。因此需要 final self-check 或 stricter revision planner。

### 10.4 Evaluation 仍需补强

当前主要是 LLM-as-judge。还需要：

- Qwen2.5 curated verifier 重新评估；
- DeepSeek/Qwen 双 judge 对比；
- human validation；
- paired bootstrap / sign test / Wilcoxon；
- per-admission clustered analysis。

## 11. 当前论文定位

当前最稳妥的论文主张是：

> ICU A&P generation suffers from longitudinal problem-state drift. A scaffolded problem-state generation pipeline mitigates this drift, especially in later admission stages, and verifier-guided revision can further reduce unsupported and missed problems. However, claim-level verification alone is insufficient; robust A&P generation requires problem-level state verification.

不建议目前直接声称：

> V2 solves trajectory drift.

更稳妥的表述：

> V2 mitigates longitudinal drift relative to the base generator, but remaining errors concentrate in problem-list hallucination and stale problem carry-over.

## 12. 后续 TODO

### TODO 0: DS 任务扩展方法

DS 任务不能直接复用 A&P 的 daily judge-revise 逻辑。A&P verifier 的核心是判断“今天相对昨天的变化是否被当天证据支持”，而 DS 的目标是生成整次住院的最终 discharge summary，需要覆盖已解决和未解决问题、主要诊断、操作、并发症、hospital course、出院状态、药物和随访。

因此，DS 方法应改为：

```text
Full admission chronology
  -> chronological chunks
  -> sequential discharge-state tracking
  -> final discharge state
  -> DS scaffold
  -> initial discharge summary
  -> global admission-level judge
  -> minimal section-level revision
  -> final discharge summary
```

主实验建议只保留一个 direct baseline：

| 方法 | 定义 | 目的 |
|---|---|---|
| Base 1: Full-Context Direct | 完整病程直接输入 LLM 生成 DS | 测试长上下文直接生成能力边界 |
| Ours 1: Sequential State + Scaffold | chunk-wise 维护 final discharge state，再 scaffold-guided 生成 DS | 验证纵向状态压缩和 scaffold 的收益 |
| Ours 2: Sequential State + Scaffold + Global Judge-Revise | 在 Ours 1 后加入 admission-level verifier 和 minimal reviser | 验证 verifier 是否减少 unsupported、missed、resolved/unresolved status errors |

DS judge 的输入应是 `initial DS + final discharge state + DS scaffold + selected admission evidence`，而不是 `previous DS + current DS`。它需要检查：

- unsupported diagnosis/procedure/medication/discharge plan；
- missed major diagnoses, procedures, complications, hospital course events；
- resolved vs unresolved problem status errors；
- wrong temporal order；
- stale or irrelevant problem carry-over；
- discharge medication, follow-up, disposition, diet/activity errors。

完整 DS 方法设计已写入：

- `analysis/ds_admission_level_method_design_zh.md`

### TODO 1: Problem-level verifier

设计 Problem List Verifier 输出：

```json
{
  "wrong_problem_threads_to_remove": [],
  "problem_threads_to_rewrite": [],
  "must_cover_problem_list": [],
  "must_not_add_problem_list": [],
  "inactive_or_resolved_problem_list": []
}
```

目标：

- 降低 problem-list hallucination；
- 减少 stale carry-over；
- 提升 missed active problem recovery；
- 支持 section-level rebuild。

### TODO 2: Problem-first reviser

将 reviser 从 sentence-level minimal reviser 升级为：

```text
1. remove wrong problem threads
2. rebuild required problem sections
3. apply claim-level fixes
4. add missing plan points
5. remove empty sections
6. avoid must-not-add items
```

### TODO 3: Final self-check

检查：

- must-cover problem 是否出现；
- must-not-add 是否违反；
- 是否新增未授权 medication / number / procedure；
- 是否仍有 unsupported PMH/problem list；
- 是否有空 section；
- 是否泄露 gold/oracle/verifier 字样。

### TODO 4: Full evaluation

建议补齐：

- AP100 random / full；
- Qwen653 full；
- failure-enriched selected 30；
- long-stay subset；
- ventilation / renal replacement / infection / disposition 高风险 subset。

### TODO 5: Human validation

建议人工评估 20-50 cases：

- active problem coverage；
- clinically important unsupported hallucinations；
- missed clinically important problems；
- trajectory correctness；
- plan actionability；
- disposition/goals correctness。

## 13. 当前结论摘要

1. ICU A&P generation 的核心难点是 longitudinal problem-state tracking，而不是单日摘要。
2. Base generator 存在明显后期退化，尤其在 long admission 和 admission late stage。
3. V2 scaffold 可以缓解 drift，且后期相对 base 的收益更大。
4. V2 judge-revise 进一步增强 trajectory、grounding 和后期质量稳定性。
5. Trajectory capture 单独分析证明，V2 的收益确实作用在 longitudinal tracking 上。
6. Claim-only verifier 是不充分的，可能降低 coverage 或无法修复 wrong problem thread。
7. Curated verifier upper-bound 说明 verifier/reviser 方向有潜力，但关键瓶颈是 problem-level verification。
8. 下一步最重要的是 problem-level verifier、problem-first reviser、full evaluation 和 human validation。
