# DS Ours2-v4-dx3 实验报告

生成日期：2026-05-20

本文档整理 DS 任务中 `Ours2-v4-dx3` 的方法设计、代码入口、LLM/规则分工、实验路径、指标结果与当前结论。

## 1. 背景

在 full DS 10-case smoke test 中，Base 1 是 full-context direct generation。Ours 系列希望通过 admission-level state tracking、scaffold、global judge-revise 和 evidence-verified recall 来提升 discharge summary 质量。

前序实验发现：

1. `Ours2-v3` 已经在 Hospital Course 和 Discharge Instructions 上明显改善。
2. 但 `Ours2-v3` 的 Diagnosis 指标下降，主要因为 diagnosis list 过宽，像 problem list 而不是 compact discharge diagnosis。
3. `Ours2-v4-ablation` 直接使用 Base compact diagnosis，可以同时保持 Base-level Diagnosis 和 Ours2-v3 的 Course/Instructions 增益，但它不是正式方法。
4. `Ours2-v4-final` 使用 LLM verified compact diagnosis selector，不直接复制 Base，但 Diagnosis ROUGE-L 仍低。
5. `Ours2-v4-dx2` 使用 LLM role classification + 规则 verbalizer，结果更低，说明规则式拼接太硬。

因此新增 `Ours2-v4-dx3`：保留 dx2 的 role classification，但最后重新调用 Diagnosis Agent 生成 compact gold-style Diagnosis。

## 2. 方法定义

`Ours2-v4-dx3` 是一个 section-wise controlled discharge summary generation 方法：

```text
Full admission chronology
  -> sequential discharge-state tracking
  -> final discharge state
  -> DS scaffold
  -> Ours2-v3 course/instruction generation
  -> dx2 role classification for diagnosis candidates
  -> Diagnosis Agent synthesis
  -> final DS
```

最终输出由三部分组成：

```text
Diagnosis = dx3 Diagnosis Agent
Hospital Course = Ours2-v3 enhanced hospital course
Discharge Instructions = Ours2-v3 enhanced discharge instructions
```

## 3. dx3 的核心思想

dx2 的问题是：

```text
LLM role classification
  -> rule-based filtering
  -> rule-based bullet verbalizer
```

规则 verbalizer 只根据 role 白名单保留候选，缺少二次综合、合并和 gold-style 改写。因此 dx2 虽然泛化，但 Diagnosis section 表面风格和 gold 不匹配。

dx3 改为：

```text
LLM role classification
  -> Diagnosis Agent
  -> compact gold-style diagnosis section
```

Diagnosis Agent 会根据分类结果、候选诊断、final state 和 evidence 重新判断哪些 diagnosis 应写入最终 discharge diagnosis。

## 4. LLM 与规则分工

### 4.1 Ours2-v3 阶段

| 步骤 | 执行者 |
|---|---|
| Discharge-plan evidence extraction | LLM |
| Ours1-v2 generation | LLM |
| Additive judge-revise | LLM |
| Base-as-recall candidate extraction | LLM |
| Base candidate evidence verification | LLM |
| Ours2-v3 final course/instruction generation | LLM |

### 4.2 dx2 role classification 阶段

| 步骤 | 执行者 |
|---|---|
| 读取 Base / Ours1-v2 / Ours2-v3 Diagnosis candidates | 规则 |
| 对候选 diagnosis 进行 role classification | LLM |
| 判断 include_in_diagnosis | LLM |
| 给出 reason 和 final_phrase | LLM |

dx2 的 role 包括：

```text
principal_discharge_diagnosis
major_secondary_diagnosis
major_complication_affecting_course_or_discharge
procedure_related_diagnosis
chronic_comorbidity_relevant_to_discharge
transient_lab_or_minor_resolved_issue
symptom_or_uncertain_finding
past_history_only
duplicate_or_subsumed
```

### 4.3 dx3 Diagnosis Agent 阶段

| 步骤 | 执行者 |
|---|---|
| 输入 dx2 classification JSON | 规则读取 |
| 输入 Base/Ours diagnosis candidates | 规则读取 |
| 输入 final state 和 admission evidence | 规则读取 |
| 重新综合、合并、选择、改写 Diagnosis | LLM |
| 清理 heading / markdown fence | 规则 |
| 拼接 Course / Instructions | 规则 |

因此 dx3 与 dx2 的根本区别是：

| 版本 | Diagnosis 最后生成方式 |
|---|---|
| dx2 | 规则过滤 + 规则 verbalizer |
| dx3 | Diagnosis Agent synthesis |

## 5. 代码与路径

### 5.1 主要脚本

dx3 脚本：

```text
scripts/run_ds_ours2_v4_dx3_agent.py
```

依赖前序输出：

```text
outputs/ds_minimal_closed_loop_10_format_aligned
outputs/ds_v2_variants_10
```

### 5.2 运行命令

```bash
python scripts/run_ds_ours2_v4_dx3_agent.py --limit 10 --case-selection shortest
```

脚本默认读取：

```text
data/DS_fixed_composed/full/input
outputs/ds_minimal_closed_loop_10_format_aligned
outputs/ds_v2_variants_10
```

### 5.3 输出路径

dx3 最终 DS：

```text
outputs/ds_v2_variants_10/method_outputs/ours2_v4_dx3_agent_diagnosis
```

dx3 诊断 agent prompt：

```text
outputs/ds_v2_variants_10/cases/<HADM_ID>/prompts/ours2_v4_dx3_agent.md
```

dx3 单独 Diagnosis 输出：

```text
outputs/ds_v2_variants_10/cases/<HADM_ID>/ours2_v4_dx3_diagnosis.txt
```

dx3 summary：

```text
outputs/ds_v2_variants_10/ours2_v4_dx3_summary.csv
```

### 5.4 评估输出

Light ROUGE-L：

```text
outputs/ds_v2_variants_10/eval_ours2_v4_dx3_light.csv
```

Full evaluator：

```text
outputs/ds_v2_variants_10/eval_ours2_v4_dx3_full.txt
```

Exact UMLS CUI-F1：

```text
outputs/ds_v2_variants_10/cui_ours2_v4_dx3_exact.csv
```

dx3 对比文档：

```text
outputs/ds_v2_variants_10/diagnosis_variant_comparison_dx3.md
```

## 6. 指标结果

### 6.1 Diagnosis variants 对比

| Metric | Base 1 | Ours2-v4-final | Ours2-v4-dx2 | Ours2-v4-dx3 | Ours2-v4-ablation |
|---|---:|---:|---:|---:|---:|
| Diagnosis ROUGE-L | 12.30 | 9.97 | 9.50 | 10.93 | 12.30 |
| Diagnosis SapBERT | 59.81 | 59.46 | 57.09 | 58.57 | 59.81 |
| Diagnosis Exact UMLS CUI-F1 | 22.81 | 22.26 | 21.27 | 22.86 | 22.81 |
| Diagnosis Light ROUGE-L | 13.83 | 11.55 | 11.03 | 11.92 | 13.83 |

### 6.2 dx3 与 Base 的完整对比

| Metric | Section | Base 1 | Ours2-v4-dx3 | Delta |
|---|---|---:|---:|---:|
| ROUGE-L | Diagnosis | 12.30 | 10.93 | -1.37 |
| ROUGE-L | Hospital Course | 17.92 | 18.13 | +0.21 |
| ROUGE-L | Discharge Instructions | 10.11 | 12.04 | +1.93 |
| SapBERT | Diagnosis | 59.81 | 58.57 | -1.24 |
| SapBERT | Hospital Course | 74.92 | 73.22 | -1.70 |
| SapBERT | Discharge Instructions | 70.49 | 72.92 | +2.43 |
| Exact UMLS CUI-F1 | Diagnosis | 22.81 | 22.86 | +0.05 |
| Exact UMLS CUI-F1 | Hospital Course | 25.97 | 27.00 | +1.03 |
| Exact UMLS CUI-F1 | Discharge Instructions | 24.62 | 25.85 | +1.23 |

### 6.3 dx3 相对 dx2 的提升

| Metric | dx2 Diagnosis | dx3 Diagnosis | Delta |
|---|---:|---:|---:|
| ROUGE-L | 9.50 | 10.93 | +1.43 |
| SapBERT | 57.09 | 58.57 | +1.48 |
| Exact UMLS CUI-F1 | 21.27 | 22.86 | +1.59 |
| Light ROUGE-L | 11.03 | 11.92 | +0.89 |

## 7. 结论

### 7.1 主要发现

1. dx3 明显优于 dx2，说明 Diagnosis 最后一步重新跑 agent 是必要的。
2. dx3 的 Diagnosis Exact UMLS CUI-F1 已经略高于 Base：`22.86` vs `22.81`。
3. dx3 保留了 Ours2-v3 在 Hospital Course 和 Discharge Instructions 上的优势：
   - Hospital Course ROUGE-L: `18.13` vs Base `17.92`
   - Discharge Instructions ROUGE-L: `12.04` vs Base `10.11`
   - Hospital Course CUI-F1: `27.00` vs Base `25.97`
   - Discharge Instructions CUI-F1: `25.85` vs Base `24.62`
   - Discharge Instructions SapBERT: `72.92` vs Base `70.49`
4. dx3 仍未打平 Diagnosis ROUGE-L 和 Diagnosis SapBERT。

### 7.2 当前最准确结论

当前可以说：

> Ours2-v4-dx3 recovers diagnosis semantic coverage while preserving the verified gains in Hospital Course and Discharge Instructions. However, diagnosis surface-form alignment remains weaker than the full-context direct baseline.

中文表述：

> Ours2-v4-dx3 已恢复 Diagnosis 的语义覆盖，并保留了 Hospital Course / Discharge Instructions 的优势；但 Diagnosis 的 gold-style 表面表达仍弱于 Base。

### 7.3 不能过度声称的内容

目前不能说：

```text
Ours2-v4-dx3 全面超过 Base。
```

因为：

- Diagnosis ROUGE-L 仍低于 Base；
- Diagnosis SapBERT 仍低于 Base；
- Hospital Course SapBERT 仍低于 Base。

更稳妥的说法是：

```text
Ours2-v4-dx3 improves discharge-plan and hospital-course factual concept coverage,
and recovers diagnosis CUI-F1, but still requires better diagnosis phrasing control.
```

## 8. 后续优化建议

下一步不是再改 Course / Instructions，而是优化 Diagnosis Agent 的 gold-style phrasing。

建议 dx4：

```text
Diagnosis Agent
  -> generate 2 compact candidate diagnosis lists
  -> choose the one with:
       fewer broad course complications
       more discharge-diagnosis style phrasing
       better match to Base/gold-style candidate terms
```

或者让 agent 输出：

```json
{
  "final_diagnoses": [],
  "excluded_course_problems": [],
  "excluded_reason": []
}
```

并加强约束：

```text
Prefer final discharge diagnosis terminology over ICU problem-list terminology.
Do not include a condition in Diagnosis only because it appears in Hospital Course.
If a condition is primarily a course detail, leave it in Hospital Course.
```

目标：

```text
Diagnosis ROUGE-L closer to Base
Diagnosis SapBERT closer to Base
Diagnosis CUI-F1 stays >= Base
Course/Instructions retain dx3 gains
```

## 9. LLM Judge 评估

为补充 ROUGE / SapBERT / CUI-F1，本节加入 LLM-as-judge 评估。评估对象为：

```text
Base 1 vs Ours2-v4-dx3
```

### 9.1 评估脚本

新增脚本：

```text
evaluation/judge_ds_pairwise_llm.py
```

该脚本会：

1. 读取 Base 和 Ours 的 DS 输出；
2. 读取 gold DS；
3. 读取 full admission evidence；
4. 将两个输出匿名为 Output A / Output B，并交替交换顺序；
5. 让 LLM judge 输出各项 1-5 分、unsupported/missed count 和 winner；
6. 还原 method winner 并生成 CSV。

评估指标：

```text
diagnosis_coverage
hospital_course_completeness
temporal_order_correctness
discharge_plan_correctness
evidence_grounding
unsupported_claim_count
missed_major_event_count
overall_quality
winner
```

### 9.2 Qwen 评估状态

尝试使用：

```text
Qwen/Qwen2.5-72B-Instruct
SiliconFlow endpoint
```

但接口返回：

```text
HTTP Error 403: Forbidden
```

因此本轮未能完成 Qwen judge。脚本已保留，Qwen key / endpoint 修复后可复跑。

### 9.3 DeepSeek LLM Judge 结果

本轮使用 DeepSeek judge 完成 10-case pairwise evaluation。

输出文件：

```text
outputs/ds_v2_variants_10/llm_judge_base_vs_dx3_deepseek.csv
outputs/ds_v2_variants_10/llm_judge_base_vs_dx3_deepseek_summary.csv
```

Winner counts：

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

解释：

- 越高越好：diagnosis/course/temporal/discharge/grounding/overall。
- 越低越好：unsupported_claim_count / missed_major_event_count。

### 9.4 LLM Judge 结论

DeepSeek LLM judge 下，`Ours2-v4-dx3` 有明显正向趋势：

1. Pairwise winner：`6 win / 3 loss / 1 tie`。
2. Evidence grounding 提升最大：`+0.50`。
3. Missed major event count 降低：`-0.40`。
4. Hospital course completeness 和 temporal order 均提升：`+0.30`。
5. Overall quality 小幅提升：`+0.20`。

这说明自动指标中没有完全体现的优势，在 LLM judge 下更清楚：

> Ours2-v4-dx3 更注重证据 grounding、病程完整性、时间顺序和减少 missed major events。

但需要谨慎：

1. 目前只有 10 条 smoke-test case；
2. 只有 DeepSeek judge，Qwen judge 未完成；
3. LLM judge 仍可能偏好更结构化、更详细的输出；
4. 仍需要 100-case 验证和 ideally 双 judge / human validation。

当前可写结论：

> In a 10-case smoke test, DeepSeek pairwise LLM judge preferred Ours2-v4-dx3 over Base in 6/10 cases, with improvements in evidence grounding, hospital-course completeness, temporal order, and missed major event count. This suggests that the proposed DS pipeline provides clinically meaningful gains beyond surface lexical metrics, although larger-scale and cross-judge validation is still required.
