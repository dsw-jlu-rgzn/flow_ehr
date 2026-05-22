# AP100 V2 / V2 Judge-Revised 失败模式与后续优化方案

## 当前评估状态

本文总结 `memory_gated_scaffold_v2` 与 `memory_gated_scaffold_v2_judge_revise` 在 AP100 full-set 上的当前失败模式，并提出下一轮优化方案。

当前有三类结果需要区分：

| 结果类型 | V2 | V2 judge-revise | 说明 |
|---|---:|---:|---|
| generation / inference | 653 / 653 | 653 / 653 | 两个方法的生成均已完成。 |
| ROUGE summary | 653 / 653 | 653 / 653 | 可做 full-set ROUGE 三方比较。 |
| evaluation LLM judge | 约 641 / 653 | 当前 57 / 653 共同可比 | V2 judge-revise 的 full-set evaluation judge 仍在进行中。 |

因此，本文的三方 LLM judge 分析基于当前已完成的 **57 条共同子集**。120-case eval 子集结果也作为参考，但不作为 full-set 最终结论。

## 1. 当前定量结果

### 1.1 57 条共同子集：base / V2 / V2 judge-revise

| metric | base | V2 | V2 judge-revise | 方向 |
|---|---:|---:|---:|---|
| active problem coverage | 3.19 | 3.26 | 3.16 | 越高越好 |
| trajectory capture | 2.68 | 2.93 | 2.93 | 越高越好 |
| plan specificity | 3.12 | 3.05 | 3.07 | 越高越好 |
| evidence grounding | 2.23 | 2.37 | 2.49 | 越高越好 |
| disposition context | 3.19 | 3.35 | 3.21 | 越高越好 |
| unsupported problem count | 4.68 | 4.63 | 4.09 | 越低越好 |
| missed key problem count | 3.40 | 3.25 | 3.16 | 越低越好 |

Winner 统计：

| comparison | augmented wins | base wins | ties |
|---|---:|---:|---:|
| V2 vs base | 28 | 16 | 13 |
| V2 judge-revise vs base | 24 | 18 | 15 |

V2 judge-revise 相对 V2 的变化：

| metric | V2 judge-revise minus V2 | 解释 |
|---|---:|---|
| active problem coverage | -0.11 | judge-revise 会压缩或删掉部分 active problem。 |
| trajectory capture | 0.00 | 没有继续提升 trajectory。 |
| plan specificity | +0.02 | 仅轻微提升。 |
| evidence grounding | +0.12 | grounding 有改善。 |
| disposition context | -0.14 | disposition / context 被削弱。 |
| unsupported problem count | -0.54 | 最大收益：减少 unsupported problem。 |
| missed key problem count | -0.09 | 小幅减少 missed problem。 |

当前结论：

> V2 judge-revise 的主要收益是降低 unsupported problem、提高 evidence grounding；代价是 active problem coverage 和 disposition/context 下降。普通 V2 在当前 57 条 full-set 子集上反而更平衡。

### 1.2 ROUGE full-set 结果

ROUGE full-set 上三方均可比较：

| method | n | ROUGE-L mean | average words |
|---|---:|---:|---:|
| base | 653 | 0.0750 | 473.8 |
| V2 | 653 | 0.0724 | 435.6 |
| V2 judge-revise | 653 | 0.0720 | 445.8 |

ROUGE 说明：

- base 仍然最接近 gold wording；
- V2 和 V2 judge-revise 都更结构化、更短；
- judge-revise 没有提升 ROUGE，反而略低于 V2。

这说明 V2 系列当前的收益不应表述为 lexical overlap 提升，而应表述为 clinical structure / grounding / trajectory 维度的改善。

## 2. 失败模式总览

在当前 57 条共同子集中，V2 judge-revise 没有赢 base 的 case 中，主要失败类别如下：

| failure mode | count |
|---|---:|
| unsupported / fabricated content | 27 |
| trajectory / state error | 13 |
| missed key problems / context | 12 |
| disposition / context missing | 6 |
| generic / weak plan | 5 |
| other | 2 |

最主要问题仍然是 **unsupported / fabricated content**。这点很关键：judge-revise 虽然降低了 unsupported count 的均值，但没有完全解决 hallucination；在部分 case 中，judge-revise 反而把错误 scaffold 写得更具体。

## 3. V2 的失败模式

### 3.1 Coverage 有提升，但容易 broad recall

V2 scaffold 倾向于保留更多候选问题和 carried-forward context。这带来两个效果：

- 优点：trajectory 和 active problem continuity 比 base 好；
- 缺点：candidate problem pool 有时过宽，容易把弱证据、历史问题、routine care 或单个 lab abnormality 推进 final A&P。

典型表现：

- 将 medication administration 推断为 active diagnosis；
- 将单个 lab abnormality 升级为 active problem；
- 将历史状态 carry-forward 到当前 day，但当前 evidence 已不支持；
- 将 watchlist 内容写成主 A&P problem。

具体例子：

| case | 现象 | 失败点 | 优化启发 |
|---|---|---|---|
| `102603_day3` | V2 缺少具体 clinical targets，漏掉 diabetes 和 thrombocytopenia，同时加入 unsupported nutritional plan。 | coverage 和 plan 不够精准；supportive care 被写成更像 active plan。 | nutrition/supportive care 需要单独低权重处理，不能替代 active medical problems。 |
| `102603_day4` | V2 覆盖了 thrombocytopenia 等 active problems，但引入 unsupported EKG findings，且 medication specificity 低于 base。 | broad recall 带来 unsupported detail；plan 具体性下降。 | 对 EKG/arrhythmia 这类高风险具体发现增加证据校验。 |
| `105351_day7` | V2 明确写了 trajectory，但引入 unsupported elements，并漏掉 gold 中的 PEEP weaning goal 和 abdominal pressure limitation。 | trajectory 方向有帮助，但关键 plan nuance 丢失。 | protected plan actions 应从 gold/history/current note 中抽取并保护。 |

### 3.2 对复杂非 respiratory 主线病例鲁棒性不足

当前 taxonomy 和 scaffold 对 COPD / respiratory failure / infection / volume / renal 这类 ICU 主线较友好，但对以下类型较弱：

- arrhythmia / VT / AFib with RVR；
- severe aortic stenosis / low EF / cardiogenic physiology；
- post-operative bleeding / retroperitoneal bleed；
- line removal / procedural complications；
- delirium / family meeting / goals of care；
- discharge planning / code status / stepdown planning。

因此，V2 在 respiratory trajectory 上常常有优势，但在复杂心血管、术后、出血、disposition 场景中容易漏掉关键 context。

具体例子：

| case | 现象 | 失败点 | 优化启发 |
|---|---|---|---|
| `105351_day4` | V2 覆盖 major systems，但缺少具体 ICU management，漏掉 AFib、UGIB concern、DMII，并和 gold 的 transfusion threshold 冲突。 | 心血管/出血/内分泌 taxonomy 与 plan 模板不足。 | 扩展 arrhythmia、bleeding/coagulation、diabetes-specific plan。 |
| `106996_day4` | V2 相比 base 少了一些 hallucinated intervention，但仍漏掉 diabetes 和 pancreatitis，plan 粒度不足。 | surgical/medical mixed case 覆盖不足。 | 加入 post-op / pancreatitis / diabetes / procedure complications 类别。 |
| `107901_day4` | V2 漏掉 primary respiratory 和 structural cardiac problems，且 hallucinate ESRD，错误描述 trajectory。 | cardiac + respiratory mixed case 中错误 scaffold 会主导输出。 | 需要 structural heart disease / arrhythmia / renal replacement 的强证据 gate。 |

### 3.3 Plan specificity 不稳定

V2 的 plan 往往更结构化，但不一定更像 gold A&P 的具体 plan。常见问题：

- plan 变成通用模板，例如 monitor labs、continue supportive care；
- 缺少 gold 中的具体动作，例如 hold diuretics、renal ultrasound、urine lytes、specific antibiotic change；
- 有时加入合理但未支持的 ICU 常规动作，例如 broad antibiotics、SBT、transfusion threshold、consults。

具体例子：

| case | 现象 | 失败点 | 优化启发 |
|---|---|---|---|
| `109444_day3` | V2 plan 较模糊，加入 unsupported severe acidosis 和 cardiac findings，漏掉 airway 与 infectious disease 细节。 | generic plan + unsupported high-risk detail 并存。 | airway/infectious disease 的 plan 需要 evidence-linked action list。 |
| `110458_day5` | V2 覆盖 major problems，但在 vasopressor、feeding status、trajectory 上与 gold 冲突。 | plan action 和状态判断没有被 evidence 约束。 | feeding/pressor/trajectory 这类状态必须逐 claim 校验。 |
| `110458_day23` | V2 比 base 更好地捕捉近期 trajectory 和 hold tube feeds，但仍加入 unsupported interventions，漏掉 UTI、rash、cardiac management。 | 局部 trajectory 好，但 multi-problem coverage 不全。 | protected active problem list 应覆盖 non-respiratory problems。 |

## 4. V2 Judge-Revised 的失败模式

### 4.1 最大优势：减少 unsupported problem

当前 57 条中，V2 judge-revise 相比 V2：

```text
 evidence grounding: +0.12
 unsupported problem count: -0.54
 missed key problem count: -0.09
```

这说明 generation-time judge/revise 的方向是有效的：它确实让输出更保守、更 evidence-grounded。

### 4.2 最大代价：coverage 和 context 下降

同时，V2 judge-revise 相比 V2：

```text
 active problem coverage: -0.11
 disposition context: -0.14
 trajectory capture: 0.00
```

这说明 judge-revise 当前更像一个 conservative filter，而不是真正的 clinical correction module。它会删减或降级一些内容，但没有稳定补全 trajectory 和 disposition。

### 4.3 错误 scaffold 被具体化

部分失败 case 中，judge-revise 不是删掉错误，而是把错误写得更具体。例如：

- hallucinated CRRT；
- unsupported vasopressor transitions；
- incorrect failed extubation；
- unsupported dialysis history；
- fabricated ETT malposition；
- fabricated ABG values；
- incorrect respiratory decline；
- unsupported beta-blocker holding。

这类错误比普通 vague hallucination 更危险，因为它们看起来更具体、更可信。

根因可能是：

1. judge/revise 只看到 scaffold + output + evidence，但缺少强制 citation / claim verification；
2. prompt 鼓励“修订”和“补全”，但没有足够强的“不知道就删掉”约束；
3. scaffold 中已经存在错误 problem/state，reviser 没有独立证伪能力；
4. medication / lab / procedure 证据被错误解释为 diagnosis 或 plan indication。

具体例子：

| case | 现象 | 失败点 | 优化启发 |
|---|---|---|---|
| `110458_day10` | V2 judge-revise 编造 ETT malposition 和具体 ABG values，明显偏离 gold 中 stable clinical picture。 | judge-revise 将错误具体化，数值/器械位置缺少 exact evidence check。 | numeric/device/procedure claim 必须要求 evidence exact match。 |
| `105351_day13` | V2 judge-revise 引入 unsupported CRRT、failed extubation、pressor transitions，和 gold note 矛盾。 | renal replacement、vent trajectory、pressor trajectory 被错误推断。 | CRRT/HD/pressor/extubation 都应作为 high-risk claims 单独验证。 |
| `105351_day19` | V2 judge-revise 能追踪 renal/respiratory trajectory，但引入 unsupported dialysis history，且缺少 explicit disposition planning。 | 一边改善 trajectory，一边 hallucinate history。 | history claims 与 active plan claims 都要区分来源和证据等级。 |
| `110458_day8` | V2 judge-revise 捕捉 hypotension trajectory，但夸大 hemodynamic data，矛盾 feeding status，并加入 unsupported diagnoses/treatments。 | trajectory 有方向，但证据细节和 plan action 不可靠。 | trajectory claim 需要绑定原始 vitals/pressor/feed evidence。 |

### 4.4 Judge-revise 对 trajectory 的收益不足

120-case 中 judge-revise 对 trajectory 有明显提升；但当前 full-set 57 条中：

```text
 V2 trajectory: 2.93
 V2 judge-revise trajectory: 2.93
 delta: 0.00
```

这说明当前 judge-revise 对 broad AP100 样本的 trajectory 修复不稳定。可能原因：

- trajectory parser / scaffold 仍主要依赖文本 pattern；
- 对多系统病例，trajectory 不止一个主问题；
- reviser 没有强制逐 problem 输出 trajectory；
- gold note 中 trajectory 有时隐含在 24h events 或 plan 中，而非显式写作。

具体例子：

| case | 现象 | 失败点 | 优化启发 |
|---|---|---|---|
| `105351_day7` | V2 judge-revise 说 PEEP increased，直接违背 gold 中 documented weaning trajectory。 | trajectory direction 反了。 | 对 ventilator parameters 做 time-aware trend extraction。 |
| `105351_day18` | V2 judge-revise 错误 defer weaning，和 gold 的 extubation/weaning trajectory 矛盾。 | 当前状态和下一步计划均不匹配。 | extubated/intubated/SBT/weaning 状态必须形成 finite-state tracker。 |
| `110458_day17` | V2 judge-revise 匹配了 RSBI，但 contradicts trach plan，并引入 unsupported comorbidities/medications。 | 单个 respiratory metric 对了，但整体 trajectory 和 procedure plan 错。 | respiratory trajectory 不能只看 RSBI/vent参数，要同时保护 trach/procedure plan。 |

### 4.5 Judge-revise 过度过滤或弱化 context

V2 judge-revise 的另一个特有问题是：它会降低 unsupported problem，但也可能删弱上下文和计划细节。

具体例子：

| case | 现象 | 失败点 | 优化启发 |
|---|---|---|---|
| `102603_day4` | V2 judge-revise 临床上基本合理，但漏掉 diuresis goals 和 thrombocytopenia tracking，trajectory 粒度弱于 base。 | 保守修订后丢失 plan granularity。 | protected plan actions 应在 revise 前锁定。 |
| `105351_day15` | V2 原本较好捕捉 clinical trajectory 和 lab trends；judge-revise 反而漏掉 AFib rate control 和 diarrhea workup，并错误写 vasopressor/beta-blocker。 | revision 伤害了原本正确的 problem coverage。 | judge-revise 应基于 sentence-level minimal edit，不能重写整段。 |
| `110458_day17` | V2 更好对齐 ventilator parameters 和 RSBI；judge-revise 虽保留 RSBI，却误写 trach plan 和 comorbidities。 | 单点指标保留，关键 context 变差。 | disposition/procedure context 需要 protected guard。 |

## 5. 根因分析

### 5.1 Problem taxonomy 仍偏窄

当前 V2 主要覆盖 ICU common problems，但 AP100 full-set 中出现更多复杂问题：

- surgical / post-op problems；
- cardiology problems；
- bleeding / anticoagulation；
- goals of care；
- neurologic / delirium；
- procedure-specific complications。

taxonomy 偏窄会导致两类错误：

- missed key problem；
- 用已有 taxonomy 强行解释新问题。

### 5.2 Evidence promotion policy 不够细

目前从 evidence 到 active problem 的 promotion 仍不够严格。不同证据类型应有不同含义：

| evidence 类型 | 应支持什么 | 不应直接支持什么 |
|---|---|---|
| medication administration | treatment exposure | active diagnosis |
| isolated lab abnormality | monitoring / watchlist | active problem |
| prior note heading | historical context | current active problem |
| imaging impression | diagnosis candidate | treatment plan |
| plan in respiratory note | respiratory action | global A&P diagnosis |

当前 V2/V2 judge-revise 有时没有区分这些层级。

### 5.3 Reviser 缺少 claim-level verification

V2 judge-revise 的 judge 主要修 scaffold 和 final output，但没有真正做到：

```text
 every clinical claim -> cite evidence -> support status
```

因此它能降低一部分明显 unsupported problem，但对具体数值、药物、趋势、操作计划仍可能 hallucinate。

### 5.4 Disposition/context 没有被保护

当前 judge-revise 的 revision objective 更偏向 clinical grounding，容易牺牲：

- code status；
- ICU vs floor disposition；
- goals of care；
- family meeting；
- pending consult/procedure；
- discharge/rehab planning。

这些内容在 LLM judge 中会影响 disposition_context，但在 scaffold 中权重不足。

## 6. 后续优化方案

### P0：增加 claim-level evidence verifier

优先级最高。

在 final output 前后增加 claim verifier：

```text
 generated A&P
 -> split into clinical claims
 -> assign claim type: diagnosis / trajectory / medication / procedure / numeric / disposition
 -> retrieve evidence
 -> classify support: supported / partially supported / unsupported / contradicted
 -> revise only unsupported or contradicted claims
```

重点规则：

- 数值必须 exact match 或近似 match evidence；
- 药物计划必须有 medication/evidence/gold support；
- diagnosis 必须有 problem-level evidence，不允许只靠 medication 推断；
- trajectory 必须有 before/after 或 explicit event 支持；
- disposition/code status/family meeting 不得凭空添加或删除。

推荐新增指标：

```text
 claim_supported_rate
 claim_unsupported_rate
 contradicted_claim_rate
 numeric_support_rate
 medication_plan_support_rate
 trajectory_claim_support_rate
```

### P1：把 active problem gate 改成 evidence-type aware

将 candidate problem 分成更细的来源：

```json
{
  "problem_id": "...",
  "evidence_source": "gold_heading | current_note | lab | medication | imaging | procedure | vital | respiratory_note",
  "promotion_level": "active_problem | secondary_problem | watchlist | supportive_care | do_not_output",
  "promotion_reason": "...",
  "required_support": [...]
}
```

关键改动：

- medication-only 默认不得升为 diagnosis；
- isolated lab 默认进入 watchlist；
- prior carry-forward 必须有 current evidence 或 explicit active heading；
- routine ICU care 只进 supportive care；
- high-risk diagnosis 需要多证据支持。

### P2：加入 protected coverage guard

为避免 judge-revise 过度保守导致 coverage/context 下降，需要保护一批必须保留的信息：

```text
 protected_active_problems
 protected_trajectory_facts
 protected_disposition_context
 protected_plan_actions
```

reviser 只能删除不在 protected set 中且 unsupported 的内容。这样可以减少：

- active problem coverage 下降；
- disposition context 丢失；
- key plan 被删成 generic plan。

### P3：针对 broad AP100 扩展 taxonomy

当前 full-set 失败提示需要扩展：

| 新类别 | 例子 |
---|---|
| arrhythmia | VT, AFib with RVR, amiodarone, rate control |
| structural heart disease | severe AS, low EF, CHF |
| bleeding/coagulation | retroperitoneal bleed, anemia, coagulopathy |
| post-operative care | surgical complication, wound/drain, ileus |
| delirium/goals of care | mental status, family meeting, code status |
| renal replacement | HD, CRRT, dialysis access |
| disposition | ICU, floor, rehab, goals, family discussion |

这会直接改善 missed key problem 和错误归类。

### P4：让 judge-revise 变成 minimal edit，而不是 regenerate

当前 judge-revise 仍可能重写过多。建议加入 hard constraints：

```text
 - Preserve supported draft sentences.
 - Do not add new diagnosis, medication, procedure, or value unless it appears in evidence.
 - Prefer downgrading unsupported claims to monitoring/watchlist.
 - If no explicit evidence, remove the specific claim rather than replacing it with another specific claim.
 - Keep disposition/code/family context unless contradicted.
```

并增加 edit-distance / sentence-level change 监控：

```text
 changed_sentence_rate
 added_claim_count
 deleted_supported_claim_count
 unsupported_added_claim_count
```

### P5：分层报告，而不是只报总体均值

后续报告建议按 case type 分层：

```text
 respiratory dominant
 infection / sepsis dominant
 cardiovascular dominant
 renal / metabolic dominant
 surgical / bleeding dominant
 disposition-heavy
 long-stay late days
```

当前 V2 的优势很可能集中在 respiratory / trajectory-heavy cases，而失败集中在 complex cardiovascular / surgical / disposition-heavy cases。总体均值会掩盖这些差异。

## 7. 下一轮推荐实验

### 实验 A：V2.1 evidence-type aware gate

改动：

- medication-only 不得升 active diagnosis；
- isolated lab 进入 watchlist；
- prior carry-forward 需要 current support；
- routine ICU care 降为 supportive care；
- high-risk diagnosis 需要至少两类 evidence。

比较：

```text
 base
 V2
 V2.1 gate
```

关注指标：

```text
 unsupported_problem_count
 missed_key_problem_count
 active_problem_coverage
 evidence_grounding
```

### 实验 B：V2.2 claim verifier + minimal revision

改动：

- final A&P 后做 claim-level verification；
- 只改 unsupported / contradicted claims；
- 不重新生成全文。

比较：

```text
 V2
 current V2 judge-revise
 V2.2 claim-verified minimal revision
```

预期：

- unsupported problem count 下降；
- evidence grounding 上升；
- active coverage 不应明显下降；
- ROUGE 不应继续下降。

### 实验 C：protected disposition/context guard

改动：

- 从 gold-like history/current note 中抽 code status、family meeting、ICU/floor disposition、pending consult/procedure；
- judge-revise 不允许无证据删除这些 context。

关注指标：

```text
 disposition_context
 missed_key_problem_count
 plan_specificity
```

### 实验 D：按 case type 的 stratified evaluation

先用规则给 AP100 case 标记类型：

```text
 respiratory
 cardiac
 renal
 infection
 surgical/bleeding
 disposition-heavy
```

然后分层比较 base / V2 / V2 judge-revise。这样可以明确：

- V2 到底在哪些病例中赢；
- judge-revise 到底在哪些病例中伤害 coverage；
- taxonomy 扩展优先覆盖哪些 case type。

## 8. 当前最稳妥结论

当前结果不支持说 V2 judge-revise 全面优于 V2 或 base。更准确的表述是：

> V2 improves trajectory capture and evidence grounding relative to base, especially in structured ICU problem-tracking scenarios. V2 judge-revise further reduces unsupported problem counts and improves grounding, but may over-filter active problems, weaken disposition context, and occasionally make erroneous scaffold content more specific. The next stage should shift from scaffold-level revision to claim-level evidence verification with protected coverage and disposition guards.

中文总结：

> V2 的核心价值是 trajectory 和 grounding；V2 judge-revise 的核心价值是减少 unsupported problem。但当前 judge-revise 仍不是最终形态，因为它有时会牺牲 coverage/context，并把错误 scaffold 具体化。下一步应从“修 scaffold/重写全文”转向“逐 claim 证据校验 + 最小编辑 + protected coverage guard”。
