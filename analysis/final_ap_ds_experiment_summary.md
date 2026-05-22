# A&P 与 DS 当前最终实验结论汇总

生成日期：2026-05-21

本文档汇总当前 ICU A&P 与 Discharge Summary (DS) 两条实验线的最终方法版本、关键指标、主要结论、限制与后续建议。它用于作为论文撰写和后续 full-scale validation 的统一工作底稿。

## 1. 总体定位

当前工作的核心主题是：

> Longitudinal clinical generation should not be treated as one-shot summarization. It requires explicit state tracking, scaffolded generation, and evidence-grounded verification/revision.

两条任务线分别验证这个观点：

| 任务 | 目标 | 核心难点 | 当前最终方法 |
|---|---|---|---|
| A&P | 每日 ICU Assessment & Plan 生成 | longitudinal problem-state drift、active problem carry-over、unsupported problem hallucination | V2 scaffold + V2 judge-revise |
| DS | admission-level discharge summary 生成 | full-admission compression、resolved/unresolved coverage、discharge-plan completeness、diagnosis style | Ours2-v4-dx3 |

## 2. A&P 最终结论

### 2.1 方法版本

A&P 当前主方法为：

```text
historical A&P / memory
+ current-day EHR evidence
  -> memory-gated problem-state scaffold
  -> scaffold-guided A&P generation
  -> generation judge
  -> minimal revise
  -> final A&P
```

核心设计：

- 显式维护 active / carry-forward / watch / rejected problems；
- 用 scaffold 控制 problem-state selection；
- 用 judge-revise 修正 unsupported changes、missing updates、forgotten carry-forward problems；
- 当前实验表明 claim-only verifier 不足，下一步需要 problem-level verifier。

### 2.2 AP100 120-case 结果

数据路径：

```text
outputs/ap_memory_gated_scaffold_ap100/ap100eval_v2_summary.csv
```

| Method | n | Judge wins | Baseline wins | Ties | Coverage Δ | Trajectory Δ | Specificity Δ | Grounding Δ | Disposition Δ | Unsupported Δ | Missed Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 scaffold-only | 120 | 71 | 12 | 37 | +0.567 | +0.883 | +0.483 | +0.700 | +0.525 | -0.642 | -0.675 |
| V2 judge-revise | 120 | 80 | 10 | 30 | +0.617 | +0.983 | +0.625 | +0.775 | +0.550 | -0.692 | -0.692 |

解释：

- 正向指标越高越好；
- `unsupported` 和 `missed` 是负向指标，越低越好；
- V2 scaffold-only 已明显优于 base；
- V2 judge-revise 在 wins、trajectory、specificity、grounding、unsupported、missed 上进一步改善。

### 2.3 Trajectory drift 结论

报告与图：

```text
analysis/trajectory_drift_v2/trajectory_drift_report.md
analysis/trajectory_drift_v2/paper_figures/
scripts/plot_trajectory_drift_paper_figures.py
```

核心发现：

| Analysis | Key Result |
|---|---|
| Base quality by relative admission progress | early `8.50` -> late `5.02` |
| V2 quality improvement | early `-1.67` -> late `+1.81` |
| V2 judge-revise quality improvement | early `-1.50` -> late `+2.62` |
| V2 quality improvement positive within-admission slope | `78.7%` admissions |
| V2 judge-revise quality improvement positive within-admission slope | `72.9%` admissions |

Trajectory capture 单独分析：

| Setting | Early improvement | Late improvement |
|---|---:|---:|
| V2, relative progress | -0.04 | +0.48 |
| V2 judge-revise, relative progress | +0.03 | +0.65 |

按绝对 hospital day：

| Setting | <=7 | 8-14 | 15-28 | >28 |
|---|---:|---:|---:|---:|
| V2 trajectory improvement | +0.15 | +0.40 | +0.45 | +0.38 |
| V2 judge-revise trajectory improvement | +0.19 | +0.44 | +0.77 | +0.62 |

稳妥结论：

> V2 mitigates longitudinal drift relative to the base generator, especially in later admission stages and long hospital-day bins. However, it does not fully solve trajectory drift, and unsupported/stale problem carry-over remains a major residual failure.

### 2.4 Verifier upper-bound 结论

#### Claim-only pseudo-oracle

数据路径：

```text
outputs/oracle_claim_verifier_qwen653/qwen25_selected30_upper_bound_comparison/upper_bound_comparison_summary.csv
```

| Method | Coverage | Trajectory | Specificity | Grounding | Disposition | Unsupported | Missed | Wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 | 4.53 | 4.17 | 4.53 | 4.50 | 4.20 | 0.50 | 0.47 | 25 |
| V2 judge-revise | 4.60 | 4.23 | 4.60 | 4.47 | 4.30 | 0.37 | 0.33 | 27 |
| V2 + pseudo-oracle verifier + LLM revise | 4.07 | 3.87 | 4.07 | 4.07 | 3.83 | 0.87 | 0.87 | 17 |

结论：

> Claim-level verification alone is not sufficient. It may delete unsupported content but can reduce coverage, specificity, and trajectory, and cannot reliably repair wrong problem threads.

#### Curated claim-level verifier

数据路径：

```text
outputs/oracle_claim_verifier_qwen653/curated_verifier_deepseek_upper_bound_comparison/upper_bound_comparison_summary_clean.csv
```

| Method | Coverage | Trajectory | Specificity | Grounding | Disposition | Unsupported | Missed | Wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 | 3.07 | 2.43 | 3.00 | 2.43 | 2.13 | 2.43 | 2.47 | 14 |
| V2 judge-revise | 3.20 | 2.67 | 3.10 | 2.77 | 2.20 | 2.10 | 2.23 | 18 |
| V2 + curated claim verifier + LLM revise | 3.27 | 2.63 | 3.03 | 2.80 | 2.37 | 1.77 | 2.03 | 22 |

结论：

> More accurate verifier/reviser signals can reduce unsupported and missed problems, but claim-level curated verification still does not fully solve problem-list hallucination or wrong problem-thread selection.

### 2.5 A&P 当前最稳论文 claim

可以写：

> ICU A&P generation suffers from longitudinal problem-state drift. A scaffolded problem-state generation pipeline mitigates this drift, especially in later admission stages, and verifier-guided revision further improves trajectory capture, grounding, and unsupported/missed problem counts.

不建议写：

> V2 solves trajectory drift.

更稳妥写法：

> V2 mitigates longitudinal drift but does not eliminate it; remaining errors concentrate in problem-list hallucination and stale problem carry-over, motivating problem-level verification.

## 3. DS 最终结论

### 3.1 数据与实验设置

推荐主实验数据：

```text
data/DS_fixed_composed/full/input
data/DS_fixed_composed/full/gold
```

数据量：

| Data | Count |
|---|---:|
| full DS input CSV | 100 |
| full DS gold TXT | 100 |

当前 smoke test 使用 10 条最短 full DS case，目的是验证闭环和方法方向。

### 3.2 DS 方法演进

| Version | Main Change | Status |
|---|---|---|
| Base 1 | Full-context direct DS generation | baseline |
| Ours 1 | Sequential discharge-state tracking + scaffold generation | scaffold baseline |
| Ours 2 | Global judge-revise, conservative filtering | early DS judge |
| Ours1-v2 | Discharge-plan extractor + gold-compatible verbalizer | improves instructions |
| Ours2-v2 | Additive judge-revise | recovers supported missing discharge content |
| Ours2-v3 | Base-as-recall candidates + evidence verification | strongest course/instruction gains |
| Ours2-v4-final | LLM verified compact diagnosis selector + Ours2-v3 course/instructions | formal v4 |
| Ours2-v4-dx2 | Role classifier + rule verbalizer | negative finding |
| Ours2-v4-dx3 | Role-aware Diagnosis Agent + Ours2-v3 course/instructions | current best formal DS method |

### 3.3 当前最终 DS 方法：Ours2-v4-dx3

最终方案：

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

最终输出：

```text
Diagnosis = dx3 Diagnosis Agent
Hospital Course = Ours2-v3 enhanced hospital course
Discharge Instructions = Ours2-v3 enhanced discharge instructions
```

代码与报告：

```text
scripts/run_ds_ours2_v4_dx3_agent.py
analysis/ds_ours2_v4_dx3_report.md
outputs/ds_v2_variants_10/diagnosis_variant_comparison_dx3.md
```

### 3.4 DS 自动指标：Base vs Ours2-v4-dx3

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

dx3 相对 dx2 的 Diagnosis 改善：

| Metric | dx2 Diagnosis | dx3 Diagnosis | Delta |
|---|---:|---:|---:|
| ROUGE-L | 9.50 | 10.93 | +1.43 |
| SapBERT | 57.09 | 58.57 | +1.48 |
| Exact UMLS CUI-F1 | 21.27 | 22.86 | +1.59 |
| Light ROUGE-L | 11.03 | 11.92 | +0.89 |

解释：

- dx3 明显优于 dx2，说明最后一步重新跑 Diagnosis Agent 是必要的；
- dx3 在 Diagnosis CUI-F1 上已略高于 Base；
- dx3 在 Hospital Course / Discharge Instructions 的 ROUGE-L 与 CUI-F1 上超过 Base；
- Diagnosis ROUGE-L 与 SapBERT 仍低于 Base，说明剩余差距主要是 surface phrasing / gold-style wording。

### 3.5 DS LLM Judge 结果

LLM judge 脚本：

```text
evaluation/judge_ds_pairwise_llm.py
```

DeepSeek pairwise judge 输出：

```text
outputs/ds_v2_variants_10/llm_judge_base_vs_dx3_deepseek.csv
outputs/ds_v2_variants_10/llm_judge_base_vs_dx3_deepseek_summary.csv
```

Qwen judge 状态：

```text
Qwen/Qwen2.5-72B-Instruct via SiliconFlow returned HTTP 403 Forbidden.
```

因此当前 LLM judge 结论来自 DeepSeek judge。

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

结论：

> In a 10-case smoke test, DeepSeek pairwise LLM judge preferred Ours2-v4-dx3 over Base in 6/10 cases, with improvements in evidence grounding, hospital-course completeness, temporal order, and missed major event count.

### 3.6 DS 当前最稳论文 claim

可以写：

> For discharge summary generation, an admission-level state tracking pipeline with section-specific verification improves hospital-course and discharge-plan coverage. The current Ours2-v4-dx3 method improves concept coverage and LLM-judge preference over full-context direct generation in a 10-case smoke test, especially for hospital course completeness, evidence grounding, and discharge instructions.

需要谨慎：

> Ours2-v4-dx3 does not yet fully dominate Base on all automatic metrics; diagnosis surface-form alignment remains weaker than the full-context baseline.

不建议写：

> DS 方法已经全面超过 Base。

更稳妥写法：

> DS results provide promising evidence that state/scaffold/verifier design transfers beyond daily A&P to admission-level summarization, but requires full 100-case evaluation and cross-judge validation.

## 4. A&P 与 DS 的统一 insight

### 4.1 共同规律

| 观察 | A&P | DS |
|---|---|---|
| 直接生成容易漂移 | active problem drift | full admission course / discharge plan omission |
| scaffold 有帮助 | 显式 problem-state scaffold | discharge-state scaffold |
| verifier 不能只删错 | claim-only verifier 降低 coverage | conservative DS judge 降低 instructions recall |
| 需要 section/problem-level 控制 | problem-level verifier | diagnosis/course/instruction 分 section 控制 |
| 后置 revision 有潜力 | judge-revise 提升 trajectory/grounding | additive judge + base recall 提升 course/instructions |

### 4.2 核心方法学结论

当前最重要的方法学结论是：

> Verification should not be purely claim-deletion. For longitudinal clinical generation, verification must be state-aware, section-aware, and coverage-aware.

在 A&P 中，这意味着：

```text
problem-level verifier + problem-first reviser
```

在 DS 中，这意味着：

```text
admission-level state tracking
+ section-specific verifier
+ diagnosis-specific agent
+ discharge-plan additive revision
```

## 5. 当前主要限制

### 5.1 A&P 限制

- V2 mitigates drift，但没有完全解决；
- unsupported/stale carry-over 在 admission 后期仍然存在；
- claim-only verifier 不足以修复 wrong problem thread；
- 仍需要 problem-level verifier 和 human validation；
- Qwen / DeepSeek judge 风格不同，需要 cross-judge 结果支撑。

### 5.2 DS 限制

- 当前 DS 只在 10 条 shortest full DS case 上做 smoke test；
- dx3 还未在 100 条 full DS 上验证；
- Qwen judge 当前 403，尚未完成 cross-judge；
- Diagnosis ROUGE-L / SapBERT 仍低于 Base；
- Ours2-v4-ablation 证明上限，但不是正式方法；
- 当前 UMLS CUI-F1 是 exact-match UMLS fallback，不是 QuickUMLS approximate matching。

## 6. 下一步建议

### 6.1 A&P

1. 实现 problem-level verifier：

```json
{
  "wrong_problem_threads_to_remove": [],
  "problem_threads_to_rewrite": [],
  "must_cover_problem_list": [],
  "must_not_add_problem_list": [],
  "inactive_or_resolved_problem_list": []
}
```

2. 实现 problem-first reviser：

```text
remove wrong problem threads
rebuild required problem sections
apply claim-level fixes
add missing plan points
avoid must-not-add items
```

3. 补充 full evaluation：

- AP100 random / full；
- Qwen653 full；
- long-stay subset；
- ventilation / renal / infection / disposition high-risk subset；
- human validation。

### 6.2 DS

1. 扩展到 100 条 `DS_fixed_composed/full`。
2. 修复 Qwen judge 或增加另一个独立 judge。
3. 继续优化 Diagnosis Agent 的 gold-style surface phrasing。
4. 加入 section-level LLM judge 和 human spot-check。
5. 做长输入分层：short / medium / long admissions。

## 7. 当前一句话总结

> A&P experiments show that scaffolded problem-state tracking and judge-revise mitigate longitudinal drift, especially in later admission stages, but remaining failures require problem-level verification. DS experiments show that the same longitudinal-state idea can transfer to admission-level summarization: Ours2-v4-dx3 improves hospital-course and discharge-plan concept coverage and wins a DeepSeek pairwise judge in 6/10 smoke-test cases, while diagnosis surface-form alignment remains the key unresolved issue.

