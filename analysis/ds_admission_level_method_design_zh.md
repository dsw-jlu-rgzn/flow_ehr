# DS 任务方法设计：Admission-Level State Tracking + Global Judge-Revise

本文档根据 discharge summary (DS) 任务特性，重新定义当前 scaffold + judge-revise 方法。核心修改是：DS 不再使用 A&P 的“相邻天变化验证”逻辑，而是使用 **全住院级别的 discharge-state tracking 与 global evidence judge**。

## 1. 为什么 DS 不能直接复用 A&P judge-revise

A&P 任务的目标是每天生成当天 Assessment & Plan。它的 verifier 主要关注：

- 今天相对昨天新增、删除、延续了哪些 active problems；
- 当天 evidence 是否支持这些变化；
- 当前 plan、trajectory、device/pressor/dialysis status 是否更新正确。

DS 任务不同。DS 的目标是生成整次住院的最终总结，而不是判断两个相邻 summary 的差异。一个合格的 discharge summary 需要覆盖：

- 入院原因和主要诊断；
- 住院期间的主要问题、治疗、操作和并发症；
- 已解决问题和出院仍需处理的问题；
- ICU/hospital course 的时间顺序；
- 出院状态、去向、药物、随访和注意事项。

因此，DS judge 不应判断“前后两个 DS 之间的差异是否被证据支持”，而应判断：

> 最终 DS 是否被整次住院证据和 final discharge state 支持，并且是否完整覆盖了已解决和未解决的关键临床问题。

## 2. DS 主实验设置

建议主实验只保留一个 direct baseline，并和两个 ours 设置对比。

| 方法 | 输入来源 | 状态维护 | Scaffold | Judge/Verifier | Reviser | 作用 |
|---|---|---:|---:|---:|---:|---|
| Base 1: Full-Context Direct | full admission chronology | 否 | 否 | 否 | 否 | 测试直接长上下文生成的能力边界 |
| Ours 1: Sequential State + Scaffold | full admission chronology | 是 | 是 | 否 | 否 | 验证纵向状态压缩和 scaffold 的收益 |
| Ours 2: Sequential State + Scaffold + Global Judge-Revise | full admission chronology | 是 | 是 | 是 | 是 | 验证 admission-level verifier 是否进一步减少 unsupported / missed |

主实验数据建议使用：

```text
data/DS_fixed_composed/full/input
data/DS_fixed_composed/full/gold
```

## 3. Base 1: Full-Context Direct

Base 1 是最直接的 DS baseline：

```text
Full admission chronology
  -> LLM directly generates discharge summary
  -> Final DS
```

实施要求：

1. 输入为同一个 admission 的完整时间线。
2. 不做 chunk-wise state update。
3. 不做 scaffold。
4. 不做 judge-revise。
5. 若输入超过模型 context window，需要记录实际处理方式：
   - 使用长上下文模型完整输入；
   - 或按模型窗口截断；
   - 或记录该样本无法完整输入。

推荐在论文中把该 baseline 定义为：

> Full-context direct generation tests whether a single-pass LLM can summarize the entire admission without explicit longitudinal state tracking.

## 4. Ours 1: Sequential Discharge-State Tracking + DS Scaffold

Ours 1 是 DS 版本的 scaffold 方法。它不再按“天”生成中间文本，而是按时间顺序维护一个 admission-level discharge state。

### 4.1 工作流

```text
Full admission chronology
  -> chronological chunks C_1 ... C_N
  -> sequential discharge-state update
  -> final discharge state Z_N
  -> DS scaffold
  -> section-wise DS generation
  -> Final DS draft
```

### 4.2 Chunk-wise state update

将完整病程按时间顺序切成若干 chunk：

```math
C = \{C_1, C_2, \ldots, C_N\}
```

初始化 admission state：

```math
Z_0 = \emptyset
```

每读入一个 chunk，更新 discharge state：

```math
Z_i = u_{\theta}(Z_{i-1}, C_i)
```

最终得到：

```math
Z_N =
\left\{
D, H, P, T, R, U, M, F
\right\}
```

其中：

- \(D\)：diagnoses，包括 principal diagnoses 和 secondary diagnoses；
- \(H\)：hospital course timeline；
- \(P\)：procedures / operations / major interventions；
- \(T\)：treatments and medication trajectories；
- \(R\)：resolved problems；
- \(U\)：unresolved problems requiring discharge follow-up；
- \(M\)：discharge medications / medication changes；
- \(F\)：follow-up, disposition, diet/activity/instructions。

### 4.3 推荐 discharge state schema

```json
{
  "admission_reason": [],
  "principal_diagnoses": [],
  "secondary_diagnoses": [],
  "major_procedures": [],
  "hospital_course_timeline": [],
  "icu_course": [],
  "complications": [],
  "treatments": [],
  "resolved_problems": [],
  "unresolved_problems_at_discharge": [],
  "discharge_condition": [],
  "discharge_disposition": [],
  "discharge_medications": [],
  "follow_up": [],
  "diet_activity_instructions": [],
  "must_not_add": [],
  "uncertain_items": []
}
```

### 4.4 State update prompt 关键约束

每个 chunk 的 state updater 应该遵守：

```text
You are maintaining a discharge-summary state for one hospital admission.
Update the existing state using only the new chronology chunk.
Do not write the final discharge summary.
Preserve important resolved and unresolved problems.
Mark whether each problem is resolved, improved, worsened, ongoing, or uncertain.
Do not invent diagnoses, procedures, medications, discharge disposition, or follow-up plans.
If a previously tracked item is contradicted or no longer supported, move it to uncertain_items or must_not_add.
Return structured JSON only.
```

### 4.5 DS scaffold

从最终 state 生成 DS scaffold：

```text
Final discharge state Z_N
  -> DS scaffold S_DS
```

推荐 scaffold 包含：

```json
{
  "required_sections": [
    "Discharge Diagnosis",
    "Major Procedures",
    "Hospital Course",
    "Discharge Condition",
    "Discharge Disposition",
    "Discharge Medications",
    "Follow-up and Instructions"
  ],
  "must_cover": [],
  "resolved_problem_summary": [],
  "unresolved_problem_summary": [],
  "temporal_course_outline": [],
  "must_not_add": [],
  "evidence_gaps": []
}
```

### 4.6 Section-wise generation

最终 DS 生成不应让 LLM 自由重写所有临床事实，而应要求它严格根据 `Z_N + S_DS` 生成：

```math
Y^{0}_{DS} = g_{\phi}(Z_N, S_{DS}, E_{select})
```

其中 \(E_{select}\) 是从完整病程中选出的关键证据包，主要用于支持诊断、治疗、操作、并发症和出院计划。

## 5. Ours 2: Global Evidence Judge + Minimal Reviser

Ours 2 是 DS 完整方法。它在 Ours 1 的初版 DS 后加入 admission-level judge-revise。

### 5.1 工作流

```text
Full admission chronology
  -> chronological chunks
  -> sequential discharge-state update
  -> final discharge state Z_N
  -> DS scaffold S_DS
  -> initial DS draft Y_DS^0
  -> global DS judge J_DS
  -> minimal section-level reviser
  -> final DS Y_DS^*
```

### 5.2 Global DS judge 的任务

Global DS judge 不判断“两个 DS 的差异”，而判断最终 DS 是否与全住院 state/evidence 一致：

```math
J_{DS} = h_{\psi}(Y^{0}_{DS}, Z_N, S_{DS}, E_{select})
```

judge 输出：

```json
{
  "unsupported_claims": [],
  "missed_major_events": [],
  "wrong_temporal_order": [],
  "missing_diagnoses": [],
  "missing_procedures": [],
  "missing_complications": [],
  "resolved_status_errors": [],
  "unresolved_status_errors": [],
  "discharge_medication_errors": [],
  "follow_up_or_disposition_errors": [],
  "stale_or_irrelevant_problem_carryover": [],
  "must_remove": [],
  "must_add": [],
  "do_not_change": []
}
```

### 5.3 Judge 判定标准

judge 需要逐项检查：

1. **Faithfulness**：DS 中每个诊断、操作、治疗、药物、出院计划是否有 state/evidence 支持。
2. **Completeness**：是否遗漏 principal diagnosis、major procedure、ICU course、并发症、关键治疗、出院计划。
3. **Resolved vs unresolved status**：已解决问题是否被错误写成 ongoing，出院仍需处理的问题是否被错误写成 resolved。
4. **Temporal order**：hospital course 是否符合病程先后顺序。
5. **Discharge-specific correctness**：出院状态、去向、药物、随访、饮食活动限制是否有证据支持。
6. **Stale carry-over**：是否把无关 PMH、早期已解决问题、未支持 problem list 错误带入最终 DS。

### 5.4 Minimal reviser

reviser 的目标是局部修复，而不是重写全文：

```math
Y^{*}_{DS} = r_{\omega}(Y^{0}_{DS}, J_{DS}, Z_N, S_{DS}, E_{select})
```

修订规则：

```text
1. Remove claims listed in must_remove.
2. Add only events listed in must_add.
3. Correct resolved/unresolved status errors.
4. Correct temporal order errors.
5. Fix discharge medication, follow-up, disposition, diet/activity only when judge provides support.
6. Preserve do_not_change content.
7. Do not introduce new diagnoses, procedures, medications, lab values, dates, or follow-up plans.
8. Do not mention judge, verifier, scaffold, oracle, or gold truth.
```

## 6. 与 A&P 方法的对应关系

| 维度 | A&P 方法 | DS 方法 |
|---|---|---|
| 时间单位 | hospital day | full admission chunks |
| 中间状态 | daily active problem state | admission-level discharge state |
| 生成目标 | 当天 A&P | 最终 discharge summary |
| verifier 重点 | 今天的变化是否支持 | 整段 summary 是否完整且有证据支持 |
| revise 重点 | 修正 active problem/update/plan | 修正 diagnosis/course/resolved-unresolved/discharge plan |
| 主要错误 | problem-state drift, stale active problem | missed major course, unsupported diagnosis/procedure/plan, wrong resolved status |

一句话概括：

> A&P verifier checks how the patient changes today; DS verifier checks whether the final admission summary is complete, faithful, and discharge-specific.

## 7. 推荐实验假设

### H1: Ours 1 vs Base 1

如果 Ours 1 优于 Base 1，说明：

> Sequential discharge-state tracking and scaffolded generation are more effective than single-pass full-context summarization for long admissions.

预期提升：

- hospital course completeness；
- temporal order correctness；
- major diagnosis/procedure coverage；
- reduced missed major events。

### H2: Ours 2 vs Ours 1

如果 Ours 2 优于 Ours 1，说明：

> Admission-level global judge-revise can further improve faithfulness and discharge-specific correctness after scaffolded generation.

预期提升：

- unsupported claims 降低；
- medication/follow-up/disposition errors 降低；
- resolved/unresolved status errors 降低；
- stale or irrelevant problem carry-over 降低。

## 8. DS 评估指标建议

LLM judge 指标建议：

```json
{
  "diagnosis_coverage": "1-5",
  "hospital_course_completeness": "1-5",
  "temporal_order_correctness": "1-5",
  "procedure_and_treatment_coverage": "1-5",
  "resolved_unresolved_status_correctness": "1-5",
  "discharge_plan_correctness": "1-5",
  "evidence_grounding": "1-5",
  "unsupported_claim_count": "integer",
  "missed_major_event_count": "integer",
  "overall_preference": "base / ours1 / ours2 / tie"
}
```

自动指标可以保留 ROUGE-L、BERTScore、SapBERT，但不要只依赖它们，因为 DS 的 gold note 风格和 section heading 变化较大。

## 9. 后续实现 TODO

1. 新增 DS chunker：读取 `data/DS_fixed_composed/full/input/*.csv`，按时间和 token budget 切 chunk。
2. 新增 discharge-state updater：逐 chunk 维护结构化 `Z_i`。
3. 新增 DS scaffold builder：从 `Z_N` 生成 section scaffold。
4. 新增 Base 1 direct generator：完整输入直接生成 DS，并记录 context/truncation。
5. 新增 Ours 1 generator：`Z_N + S_DS + selected evidence` 生成 DS。
6. 新增 Ours 2 global judge：检查完整 DS 的 admission-level coverage/faithfulness。
7. 新增 minimal reviser：按 judge 输出做 section-level 局部修订。
8. 新增 DS LLM evaluator：同时比较 Base 1 / Ours 1 / Ours 2。

## 10. 当前方法最终定义

DS 任务中的最终方法应写成：

> We extend scaffolded longitudinal generation from daily A&P to discharge summary generation by replacing day-level problem-state tracking with admission-level discharge-state tracking. The model reads the full hospitalization chronologically, updates a structured discharge state across chunks, constructs a section-aware discharge-summary scaffold, and generates the final DS from the accumulated state. A global admission-level judge then verifies whether the generated DS faithfully covers major diagnoses, procedures, hospital course, resolved and unresolved problems, and discharge plans, followed by a constrained minimal revision step.

