# A&P 与 DS 统一论文大纲中文版

生成日期：2026-05-21

本文档将当前 A&P 与 Discharge Summary (DS) 两条实验线统一为一篇论文的中文大纲。目标是形成论文级结构，而不是实验流水账。核心统一视角是：

> 长程临床文本生成不是一次性摘要任务，而是 longitudinal clinical state tracking + section/problem-aware generation + evidence-grounded verification。

## 1. 暂定标题

### 中文标题

面向长程 ICU 病程的状态跟踪式临床文本生成：从每日 A&P 到出院总结

### 英文标题候选

**State-Tracked Longitudinal Clinical Generation for ICU Assessment & Plan and Discharge Summary**

或：

**Scaffolded Longitudinal Clinical Generation with Evidence-Grounded Verification**

## 2. 摘要草稿

ICU 患者通常具有长住院周期、多问题并行演化和复杂治疗轨迹。直接使用大语言模型生成临床文档容易出现纵向状态漂移：模型可能遗漏仍然 active 的问题、错误继承已解决问题、引入无证据支持的诊断或治疗计划，并在住院后期逐渐丢失病程主线。本文将每日 Assessment & Plan (A&P) 生成和 admission-level Discharge Summary (DS) 生成统一建模为 longitudinal clinical state tracking 问题，而非普通摘要问题。

我们提出一种 scaffolded longitudinal generation framework。对于 A&P，模型显式构建 problem-state scaffold，用于追踪 active、carry-forward、watchlist 和 rejected problems，并通过 judge-revise 模块减少 unsupported 和 missed updates。对于 DS，模型将完整住院病程分块读取，维护 admission-level discharge state，再进行 section-specific generation，其中 diagnosis、hospital course 和 discharge instructions 采用不同的验证与生成策略。实验显示，在 A&P 任务中，V2 scaffold 和 judge-revise 明显提升 trajectory capture、evidence grounding，并在 admission 后期相对 base 获得更大收益；在 DS 10-case smoke test 中，Ours2-v4-dx3 提升 hospital course 和 discharge instructions 的 concept coverage，并在 DeepSeek pairwise LLM judge 下 6/10 case 优于 full-context direct baseline。分析同时表明，claim-level verification 不足以解决 wrong problem-thread 和 diagnosis surface mismatch，后续需要 problem-level / section-level verifier。

## 3. Introduction 大纲

### 3.1 临床背景

ICU 文档生成不同于普通临床摘要。ICU 患者在住院期间会经历持续变化的病理状态、治疗响应、器械支持、药物调整和处置计划。高质量临床文档需要持续维护患者状态，而不是只压缩当前输入。

每日 A&P 与 DS 是两个典型长程临床生成任务：

- A&P 需要每天更新 active problem list、assessment、plan 和 trajectory；
- DS 需要最终总结完整住院过程，包括诊断、操作、并发症、已解决/未解决问题和出院计划。

二者表面形式不同，但共享同一个核心难点：

> 模型必须跨时间维护临床状态，并在最终文本中只表达被证据支持且符合文档目的的内容。

### 3.2 现有 LLM 生成方式的问题

直接让 LLM 进行 full-context 或 current-day generation 会出现：

1. **Longitudinal drift**：随着 admission 进展，模型丢失病程主线；
2. **Problem-state error**：错误保留已解决问题，或漏掉仍 active 的问题；
3. **Unsupported hallucination**：生成无证据支持的诊断、药物、操作或 disposition；
4. **Coverage loss**：过度保守修订会删除内容，导致 missed problems；
5. **Section mismatch**：DS 中不同 section 需要不同粒度，Diagnosis 不能写成 problem list，Instructions 不能过度保守。

### 3.3 本文核心观点

本文提出：

> Longitudinal clinical generation should be state-aware, scaffold-guided, and verification-grounded.

也就是说，临床长程生成应拆成：

```text
state tracking
-> scaffold construction
-> section/problem-aware generation
-> evidence-grounded verification
-> minimal or additive revision
```

### 3.4 贡献点

本文贡献可以写成：

1. **统一问题定义**：将每日 A&P 和 DS 统一建模为 longitudinal clinical state tracking generation。
2. **A&P scaffold 方法**：提出 problem-state scaffold，显式维护 active/carry/watch/rejected problems。
3. **DS admission-level 方法**：提出 sequential discharge-state tracking，并设计 diagnosis / hospital course / discharge instructions 的 section-specific generation。
4. **Verifier 分析**：证明 claim-level verifier 有用但不充分，必须升级为 problem-level 或 section-level verifier。
5. **纵向漂移实证分析**：在 A&P full 653 evaluation 中证明 base 后期退化，V2 在后期收益更大。
6. **跨任务验证**：在 DS 10-case smoke test 中验证同一思想可迁移到 admission-level summarization。

## 4. Related Work 大纲

可分为四类：

### 4.1 Clinical note generation

讨论临床文本生成、progress note generation、discharge summary generation。

### 4.2 Longitudinal EHR summarization

强调长程 EHR 输入、病程压缩、状态追踪与时间顺序。

### 4.3 LLM hallucination and verification

讨论 factuality、evidence grounding、LLM-as-judge、claim verification。

### 4.4 Structured clinical reasoning / problem lists

讨论 problem list management、assessment & plan、clinical state representation。

## 5. Task Formulation

### 5.1 A&P 任务定义

对于同一个 admission 的第 \(t\) 天，输入为：

```math
X_t = \{E_t, A_{<t}, P_{<t}\}
```

其中：

- \(E_t\)：当天 EHR evidence；
- \(A_{<t}\)：历史 A&P；
- \(P_{<t}\)：历史 problem state 或 memory。

目标是生成当天 A&P：

```math
Y_t = \{y_{t,1}, ..., y_{t,n}\}
```

关键挑战：

- active problem coverage；
- trajectory update；
- unsupported problem suppression；
- missed problem recovery。

### 5.2 DS 任务定义

对于一个完整 admission，将病程划分为 chronological chunks：

```math
C = \{C_1, C_2, ..., C_N\}
```

模型顺序更新 discharge state：

```math
Z_i = u_\theta(Z_{i-1}, C_i)
```

最终生成：

```math
Y_{DS} = \{D, H, I\}
```

其中：

- \(D\)：Diagnosis；
- \(H\)：Hospital Course；
- \(I\)：Discharge Instructions。

关键挑战：

- full admission compression；
- resolved/unresolved status；
- discharge medications/follow-up；
- diagnosis compactness；
- evidence-grounded section generation。

## 6. Method

本节不把 A&P 和 DS 写成两个彼此独立的方法，而是提出一个统一框架：

> **Longitudinal State-Tracked Generation with Evidence-Grounded Revision**。

核心思想是：无论目标文档是每天的 A&P，还是整次住院的 discharge summary，本质上都不是一次性摘要，而是从 longitudinal EHR 中维护一个临床状态，再根据目标文档的写作规范生成 section-aware 文本，并通过证据约束的 judge/reviser 做局部修订。

## 6.1 统一问题建模

给定一个 admission 的纵向证据序列：

```math
X = \{E_1, E_2, \ldots, E_T\}
```

其中 \(E_t\) 表示第 \(t\) 个时间单元的 EHR evidence。时间单元可以是一天，也可以是 DS 任务中的一个 chronological chunk。模型维护一个随时间更新的临床状态：

```math
Z_t = u_\theta(Z_{t-1}, E_t)
```

其中 \(Z_t\) 是当前已知的 longitudinal clinical state，包含 active problems、resolved problems、major events、treatments、procedures、discharge-relevant plans、uncertain items 和 must-not-add items。然后从状态中构建目标文档 scaffold：

```math
S_t = f_\theta(Z_t, \mathcal{G})
```

这里 \(\mathcal{G}\) 表示目标文档规范。对于 A&P，\(\mathcal{G}\) 强调 active problem、assessment、plan 和 trajectory update；对于 DS，\(\mathcal{G}\) 强调 final diagnosis、hospital course、resolved/unresolved status 和 discharge instructions。

初版文档生成：

```math
Y_t^0 = g_\phi(E_{\le t}, Z_t, S_t, \mathcal{G})
```

证据约束 judge：

```math
J_t = h_\psi(Y_t^0, E_{\le t}, Z_t, S_t, \mathcal{G})
```

最终局部修订：

```math
Y_t^* = r_\omega(Y_t^0, J_t, E_{\le t}, Z_t, S_t, \mathcal{G})
```

因此，完整流程可以写成：

```text
Longitudinal EHR evidence
  -> longitudinal clinical state tracking
  -> target-specific scaffold construction
  -> section/problem-aware generation
  -> evidence-grounded judge
  -> minimal or additive revision
  -> final clinical document
```

## 6.2 统一状态表示

我们将 A&P 和 DS 的中间表示统一为 clinical state \(Z\)，但根据任务粒度使用不同字段子集。

| 状态字段 | A&P 中的作用 | DS 中的作用 |
|---|---|---|
| active problems | 当天必须写入 A&P 的问题 | 出院时仍需处理的问题 |
| carried-forward problems | 从历史 A&P 继承且仍 active 的问题 | 住院过程中持续存在的主要问题 |
| resolved problems | 当天不应继续作为 active problem 的问题 | Hospital Course 中需要总结的已解决问题 |
| watchlist / uncertain items | 有证据但不足以写入 active A&P | 需要避免过度确定化的诊断或事件 |
| rejected / must-not-add items | 防止 stale carry-over 和 hallucination | 防止把无证据 PMH、药物、诊断写入 DS |
| trajectory / timeline | 每个 active problem 的当天变化 | 整个 hospital course 的时间线 |
| treatments / plans | 当天 plan 和 disposition | 出院药物、随访、discharge instructions |

统一状态 schema 可以抽象为：

```json
{
  "active_or_ongoing_problems": [],
  "carried_forward_items": [],
  "resolved_problems": [],
  "major_events": [],
  "procedures_and_interventions": [],
  "treatments_and_plans": [],
  "trajectory_or_timeline": [],
  "discharge_relevant_items": [],
  "uncertain_items": [],
  "must_not_add": []
}
```

A&P 使用该 schema 的日级 problem-state 子集；DS 使用该 schema 的 admission-level discharge-state 子集。

## 6.3 目标特异化实例化

统一框架通过 \(\mathcal{G}\) 适配不同文档目标，而不是改变核心方法。

| 维度 | A&P 实例化 | DS 实例化 |
|---|---|---|
| 时间粒度 | day-level update | chunk-level admission update |
| 输入范围 | current-day evidence + historical A&P/memory | full admission chronology |
| 状态目标 | 当前 active problem state | final discharge state |
| scaffold 目标 | active/carry/watch/reject problems | diagnosis/course/instruction scaffold |
| 生成目标 | 当天 Assessment & Plan | admission-level Discharge Summary |
| judge 重点 | unsupported problem threads, missed updates, stale carry-forward | unsupported diagnoses/events, missed major events, wrong resolved/unresolved status |
| revision 方式 | minimal problem-level revision | section-level additive/minimal revision |

### 6.3.1 A&P 实例化

对于第 \(t\) 天 A&P，状态更新目标是维护当天 problem state：

```math
Z_t^{AP} =
\{P_t^{active}, P_t^{carry}, P_t^{watch}, P_t^{resolved}, P_t^{reject}, C_t\}
```

其中 \(P_t^{active}\) 是当天必须写入 A&P 的问题，\(P_t^{carry}\) 是仍需继承的历史问题，\(P_t^{watch}\) 是证据不足但需观察的问题，\(P_t^{reject}\) 是不应写入的问题，\(C_t\) 是 contradiction / low-confidence signals。

A&P scaffold-guided generation：

```math
Y_t^{AP,0} = g_\phi(E_t, A_{<t}, Z_t^{AP}, S_t^{AP})
```

A&P judge 检查：

- 是否存在 unsupported active problem 或 unsupported plan；
- 是否遗漏当天新增或仍 active 的 key problem；
- 是否错误 carry forward 已 resolved 的问题；
- trajectory、device、pressor、dialysis、infection、disposition 等状态是否更新正确。

最终：

```math
Y_t^{AP,*} =
r_\omega(Y_t^{AP,0}, J_t^{AP}, E_t, Z_t^{AP}, S_t^{AP})
```

### 6.3.2 DS 实例化

对于 DS，模型按 chronological chunks 顺序维护 admission-level discharge state：

```math
Z_i^{DS} = u_\theta(Z_{i-1}^{DS}, C_i)
```

最终状态：

```math
Z_N^{DS} =
\{
D, H, R, U, P, M, F, N
\}
```

其中 \(D\) 表示 diagnoses，\(H\) 表示 hospital course timeline，\(R\) 表示 resolved problems，\(U\) 表示 unresolved problems at discharge，\(P\) 表示 procedures/interventions，\(M\) 表示 discharge medications，\(F\) 表示 follow-up/disposition/instructions，\(N\) 表示 must-not-add 或 uncertain items。

DS scaffold-guided generation：

```math
Y_{DS}^{0} = g_\phi(E_{\le N}, Z_N^{DS}, S_{DS})
```

DS judge 检查：

- Diagnosis 是否是 compact final diagnosis，而不是 broad problem list；
- Hospital Course 是否覆盖主要事件、治疗、并发症和时间顺序；
- resolved/unresolved status 是否符合最终出院状态；
- discharge medications、follow-up、disposition、instructions 是否有证据支持；
- 是否引入无证据诊断、药物、操作或随访计划。

最终：

```math
Y_{DS}^{*} =
r_\omega(Y_{DS}^{0}, J_{DS}, E_{\le N}, Z_N^{DS}, S_{DS})
```

## 6.4 当前实现版本

当前实验中，A&P 和 DS 是同一框架的两个实例化版本：

| 模块 | A&P 当前实现 | DS 当前实现 |
|---|---|---|
| State tracker | memory-gated problem scaffold builder | sequential discharge-state tracker |
| Scaffold | active/carry/watch/reject problem scaffold | diagnosis/course/instruction scaffold |
| Generator | scaffold-conditioned A&P generator | section-wise DS generator |
| Judge | V2 generation judge | global DS evidence judge / diagnosis agent |
| Reviser | LLM minimal reviser | section-level additive/minimal reviser |
| Final output | daily A&P | discharge summary |

因此，论文中的方法名称应统一，不建议分别命名为 “A&P V2” 和 “DS Ours2-v4-dx3”。更好的写法是：

> 我们提出一个统一的 longitudinal state-tracked generation framework，并分别在 daily A&P generation 和 admission-level DS generation 中实例化。A&P 版本侧重 problem-state maintenance，DS 版本侧重 discharge-state compression 与 section-aware revision。

实验中的 V2、Ours2-v4-dx3 可以作为 implementation variants 或 ablation settings 出现，而不应作为论文主方法的两个割裂名称。

## 7. Experiments

## 7.1 A&P Experiments

### 7.1.1 AP100 120-case evaluation

比较：

- Base；
- V2 scaffold-only；
- V2 judge-revise。

指标：

- active problem coverage；
- trajectory capture；
- plan specificity；
- evidence grounding；
- disposition context；
- unsupported problem count；
- missed key problem count；
- pairwise judge wins。

### 7.1.2 Full 653 trajectory drift analysis

分析：

- relative admission progress；
- absolute hospital day；
- trajectory capture；
- within-admission slope。

### 7.1.3 Verifier upper-bound

比较：

- pseudo-oracle claim verifier；
- curated claim verifier；
- V2 judge-revise。

目的：

验证 claim-level verifier 是否足够。

## 7.2 DS Experiments

数据：

```text
data/DS_fixed_composed/full/input
data/DS_fixed_composed/full/gold
```

当前 smoke test：

- 10 shortest full DS cases；
- 后续应扩展到 100 cases。

比较：

- Base 1: Full-context direct；
- Ours1/Ours2 early variants；
- Ours2-v3；
- Ours2-v4-final；
- Ours2-v4-dx2；
- Ours2-v4-dx3。

指标：

- ROUGE-L；
- SapBERT；
- Exact UMLS CUI-F1；
- pairwise LLM judge。

## 8. Results

## 8.1 A&P AP100 结果

| Method | n | Judge wins | Baseline wins | Ties | Coverage Δ | Trajectory Δ | Specificity Δ | Grounding Δ | Disposition Δ | Unsupported Δ | Missed Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 scaffold-only | 120 | 71 | 12 | 37 | +0.567 | +0.883 | +0.483 | +0.700 | +0.525 | -0.642 | -0.675 |
| V2 judge-revise | 120 | 80 | 10 | 30 | +0.617 | +0.983 | +0.625 | +0.775 | +0.550 | -0.692 | -0.692 |

结论：

- scaffold-only 已显著优于 base；
- judge-revise 进一步提升 wins、trajectory、grounding；
- unsupported 和 missed 均下降。

## 8.2 A&P Trajectory Drift 结果

| Result | Value |
|---|---|
| Base quality early -> late | 8.50 -> 5.02 |
| V2 improvement early -> late | -1.67 -> +1.81 |
| V2 judge-revise improvement early -> late | -1.50 -> +2.62 |
| V2 positive improvement slope | 78.7% admissions |
| V2 judge-revise positive improvement slope | 72.9% admissions |

结论：

> V2 主要缓解 admission 后期 drift，而不是均匀提升所有 case。

## 8.3 A&P Verifier 结论

Claim-only pseudo-oracle 没有超过 V2 judge-revise：

| Method | Coverage | Trajectory | Grounding | Unsupported | Missed | Wins |
|---|---:|---:|---:|---:|---:|---:|
| V2 judge-revise | 4.60 | 4.23 | 4.47 | 0.37 | 0.33 | 27 |
| pseudo-oracle claim verifier | 4.07 | 3.87 | 4.07 | 0.87 | 0.87 | 17 |

Curated verifier 有收益，但仍不充分：

| Method | Coverage | Trajectory | Grounding | Unsupported | Missed | Wins |
|---|---:|---:|---:|---:|---:|---:|
| V2 judge-revise | 3.20 | 2.67 | 2.77 | 2.10 | 2.23 | 18 |
| curated verifier | 3.27 | 2.63 | 2.80 | 1.77 | 2.03 | 22 |

结论：

> Claim-level verifier can help but cannot solve wrong problem-thread errors.

## 8.4 DS 自动指标

`Ours2-v4-dx3` vs Base：

| Metric | Section | Base | Ours2-v4-dx3 | Delta |
|---|---|---:|---:|---:|
| ROUGE-L | Diagnosis | 12.30 | 10.93 | -1.37 |
| ROUGE-L | Hospital Course | 17.92 | 18.13 | +0.21 |
| ROUGE-L | Instructions | 10.11 | 12.04 | +1.93 |
| SapBERT | Diagnosis | 59.81 | 58.57 | -1.24 |
| SapBERT | Hospital Course | 74.92 | 73.22 | -1.70 |
| SapBERT | Instructions | 70.49 | 72.92 | +2.43 |
| CUI-F1 | Diagnosis | 22.81 | 22.86 | +0.05 |
| CUI-F1 | Hospital Course | 25.97 | 27.00 | +1.03 |
| CUI-F1 | Instructions | 24.62 | 25.85 | +1.23 |

结论：

- dx3 在 Hospital Course / Instructions 上明显优于 Base；
- Diagnosis CUI-F1 略高于 Base；
- Diagnosis ROUGE-L / SapBERT 仍低于 Base，说明 surface phrasing 仍需优化。

## 8.5 DS LLM Judge 结果

DeepSeek pairwise judge：

| Winner | Count |
|---|---:|
| Ours2-v4-dx3 | 6 |
| Base | 3 |
| Tie | 1 |

平均指标：

| Metric | Base | Ours2-v4-dx3 | Delta |
|---|---:|---:|---:|
| diagnosis_coverage | 3.30 | 3.40 | +0.10 |
| hospital_course_completeness | 3.40 | 3.70 | +0.30 |
| temporal_order_correctness | 3.30 | 3.60 | +0.30 |
| discharge_plan_correctness | 2.30 | 2.50 | +0.20 |
| evidence_grounding | 3.00 | 3.50 | +0.50 |
| unsupported_claim_count | 3.00 | 2.90 | -0.10 |
| missed_major_event_count | 2.50 | 2.10 | -0.40 |
| overall_quality | 2.80 | 3.00 | +0.20 |

结论：

> DeepSeek judge 认为 Ours2-v4-dx3 相比 Base 有临床质量提升，尤其是 grounding、course completeness、temporal order 和 missed major events。

## 9. Discussion

### 9.1 为什么 A&P 和 DS 可以统一

A&P 和 DS 虽然文档形式不同，但都依赖 longitudinal clinical state：

- A&P 是 daily active problem state；
- DS 是 admission-level discharge state。

统一框架是：

```text
clinical state tracking
-> scaffolded generation
-> evidence-grounded verification
```

### 9.2 为什么 verifier 不能只做删除

A&P 的 claim-only verifier 会降低 coverage；DS 的 conservative judge 会降低 instructions recall。这说明 verifier 必须同时做：

```text
remove unsupported
+ add supported missing content
+ preserve required state
```

### 9.3 为什么需要 section/problem-level verifier

A&P 中错误常常是 whole problem thread 错，而不是单句错。

DS 中 Diagnosis、Hospital Course、Instructions 的写作目标不同，不能用同一个 verifier 标准。

因此需要：

- A&P：problem-level verifier；
- DS：section-specific verifier。

## 10. Limitations

### 10.1 A&P

- V2 缓解但没有解决 trajectory drift；
- unsupported/stale carry-over 仍存在；
- claim-level verifier 不足；
- 需要 human validation；
- 需要 cross-judge validation。

### 10.2 DS

- 当前只有 10-case smoke test；
- 还没有 full 100-case DS；
- Qwen judge 因 403 未完成；
- Diagnosis surface phrasing 仍弱于 Base；
- Exact UMLS CUI-F1 是 fallback exact-match，不是 QuickUMLS approximate matching。

## 11. Future Work

### 11.1 A&P

1. Problem-level verifier；
2. Problem-first reviser；
3. final self-check；
4. full 653 / high-risk subset / long-stay subset；
5. human evaluation。

### 11.2 DS

1. 扩展到 100 full DS cases；
2. 修复 Qwen judge；
3. 优化 Diagnosis Agent 的 gold-style phrasing；
4. 加入 section-level LLM judge；
5. 长输入分层实验；
6. human spot-check。

## 12. 当前最终论文主张

最稳妥的论文主张：

> Longitudinal clinical generation requires explicit state tracking and evidence-grounded verification. In daily A&P generation, scaffolded problem-state tracking mitigates longitudinal drift and improves trajectory capture, especially in later admission stages. In discharge summary generation, admission-level state tracking with section-specific verification improves hospital-course and discharge-plan coverage, although diagnosis surface-form alignment remains challenging.

中文版本：

> 长程临床文本生成需要显式状态跟踪和基于证据的验证。对于每日 A&P，problem-state scaffold 能缓解 admission 后期的 longitudinal drift，并提升 trajectory capture；对于 DS，admission-level discharge state 和 section-specific verifier 能提升 hospital course 与 discharge instructions 的覆盖和 grounding。但 Diagnosis 的表面风格对齐仍是当前主要挑战。
