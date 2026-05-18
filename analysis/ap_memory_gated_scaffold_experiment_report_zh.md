# Memory-Gated Problem-State Scaffold 小批量实验报告

## V2 更新结论

在 V1 之后，我们做了一版通用 prompt/schema 优化，没有使用任何 case-specific 规则。V2 增加：

- `carry_forward_major_headings`：显式保留上一日 A&P 的主要 heading 粒度；
- `candidate_problem_pool`：保留 broad recall backup，避免 gate 过强导致漏问题；
- `section_role`：区分 `primary_section`、`merged_into_existing_section`、`brief_monitoring`；
- 更严格的通用 lab-promotion gate：单个异常 lab 或单次 medication 不能独立成为主问题，除非 prior A&P 或今日 clinician text 明确支持，或改变治疗/处置/监测强度。

V2 targeted 结果优于 V1，尤其 `oracle_method2_gt` 从 V1 的 ROUGE 轻微下降变成明显提升。

| config | direct ROUGE | V1 gated ROUGE | V2 gated ROUGE | V1 delta | V2 delta | V2 judge gated wins |
|---|---:|---:|---:|---:|---:|---:|
| `generated_method2_gen` | 5.806 | 5.896 | 5.974 | +0.090 | +0.168 | 8/10 |
| `oracle_method2_gt` | 6.664 | 6.630 | 7.126 | -0.035 | +0.462 | 9/10 |

V2 judge delta：

| config | coverage | trajectory | plan specificity | grounding | disposition | unsupported count | missed count |
|---|---:|---:|---:|---:|---:|---:|---:|
| `generated_method2_gen_v2` | +0.7 | +1.0 | +0.6 | +0.6 | +0.6 | -0.6 | -0.7 |
| `oracle_method2_gt_v2` | +1.0 | +1.6 | +1.0 | +1.6 | +1.0 | -0.9 | -1.0 |

当前判断：V2 已经比 V1 更值得进入下一轮。它没有完全追上 flat scaffold 在 `generated_method2_gen` 上的 ROUGE delta，但 judge 指标明显更稳，且在 `oracle_method2_gt` 上 ROUGE 和 judge 都强。

下一步建议：

1. 先把 V2 跑到更多 generated-history cases，而不是立刻加 judge/revise。
2. 如果 generated-history 的 ROUGE 仍弱于 flat scaffold，可以尝试 V3：让 `candidate_problem_pool` 在生成阶段更强地提供 recall，但只允许 `active_ap_problems` 决定主 section。
3. Judge/revise 应作为第三步，因为当前 no-judge V2 已经证明 gate 有正向信号。

## 当前结论

本轮 targeted no-judge 小批量实验显示：`memory_gated_scaffold_no_judge` 相比 direct baseline 在 LLM judge 上有明显正向信号，但 ROUGE 收益弱于此前的 flat scaffold augmentation。

最值得注意的结果：

| config | direct ROUGE | gated ROUGE | ROUGE delta | judge gated wins | judge baseline wins | ties |
|---|---:|---:|---:|---:|---:|---:|
| `generated_method2_gen` | 5.806 | 5.896 | +0.090 | 7 | 2 | 1 |
| `oracle_method2_gt` | 6.664 | 6.630 | -0.035 | 9 | 1 | 0 |

与此前 `flat_scaffold_augmented` 对比：

| config | flat ROUGE delta | gated ROUGE delta | flat judge wins | gated judge wins |
|---|---:|---:|---:|---:|
| `generated_method2_gen` | +0.296 | +0.090 | 5/10 | 7/10 |
| `oracle_method2_gt` | +0.309 | -0.035 | 7/10 | 9/10 |

因此当前判断是：memory-gated scaffold 的临床结构和 grounding 更好，但生成文本与 gold 的表面匹配不如 flat scaffold。下一步不应直接全量跑，而应先优化 prompt，使 gated scaffold 保留 precision 的同时，避免过度压缩或过度改写。

## 实验 Motivation

此前实验表明：

1. no-training embedding prefilter 没有稳定收益；
2. problem-state replacement 会丢失 raw EHR 细节；
3. flat problem-state augmentation 有收益，但容易把 routine care、lab abnormality、risk item 平铺到 A&P 主体；
4. AP-Memory V2 的 `active_ap_problems / watchlist / supportive_care` 和 problem-promotion gate 可以减少 over-promotion。

所以本实验尝试把 AP-Memory V2 中最有价值的 gate 移植到当前更公平的 augmented generation 框架里。

## 主要设定

方法名：

```text
memory_gated_scaffold_no_judge
```

输入保持与 direct baseline 公平一致：

```text
current-day raw EHR
+ matched history context
+ memory-gated scaffold
-> today's A&P
```

本轮只跑 no-judge，不做 judge/revise。

小批量 targeted cases：

```text
177997 day 3
177997 day 4
196033 day 4
196033 day 5
196033 day 6
196033 day 7
196033 day 8
115691 day 3
126929 day 3
126929 day 4
```

本轮 configs：

| config | baseline | history/scaffold source |
|---|---|---|
| `generated_method2_gen` | `deepseek_api_full_gen/gen/method2` | cumulative generated A&P |
| `oracle_method2_gt` | `deepseek_api_full/gt/method2` | cumulative gold A&P |

## Scaffold 设计

每个 case 先生成一个 AP-Memory-V2 风格 JSON：

```json
{
  "global_status": {},
  "active_ap_problems": [],
  "watchlist": [],
  "supportive_care": [],
  "resolved_problems": [],
  "uncertainties": [],
  "promotion_gate_notes": []
}
```

Promotion gate：

- 只有未解决的 prior A&P 主问题，或今日证据改变诊断、严重程度、治疗、处置、主要并发症时，才进入 `active_ap_problems`。
- isolated lab abnormality、single medication、prophylaxis、nutrition、access、routine monitoring 默认进入 `watchlist/supportive_care`。
- 每日 `active_ap_problems` 最多 6 个。

本轮 scaffold 平均规模：

| config | active AP problems | watchlist | supportive care |
|---|---:|---:|---:|
| `generated_method2_gen` | 5.0 | 2.5 | 4.7 |
| `oracle_method2_gt` | 4.6 | 2.5 | 4.4 |

## Judge 结果

Judge 指标 delta 表示 gated minus direct baseline。

| config | coverage | trajectory | plan specificity | grounding | disposition | unsupported count | missed count |
|---|---:|---:|---:|---:|---:|---:|---:|
| `generated_method2_gen` | +0.4 | +0.8 | +0.4 | +0.6 | +0.3 | -0.4 | -0.4 |
| `oracle_method2_gt` | +0.8 | +1.4 | +0.8 | +1.2 | +0.6 | -0.9 | -0.7 |

这说明 gated scaffold 确实改善了几个本实验最关心的失败项：

- active problem coverage 更高；
- daily trajectory 更明确；
- evidence grounding 更好；
- unsupported problem 更少；
- missed key problem 更少。

## Case-Level 观察

`generated_method2_gen` 中，ROUGE 下降最明显的是：

| case | ROUGE delta | judge winner | 主要现象 |
|---|---:|---|---|
| `177997 day 3` | -1.654 | baseline | scaffold 过度依赖 sparse lab，生成 AKI/anemia/DM，漏掉更完整 clinical context |
| `115691 day 3` | -1.411 | gated | ROUGE 下降但 judge 认为临床覆盖更好 |
| `126929 day 4` | -0.144 | baseline | gated 把 pulmonary edema/hypoxemia 提升为主问题，但 unsupported count 反而更高 |

`oracle_method2_gt` 中，ROUGE 虽轻微下降，但 judge 9/10 选择 gated，说明当 history 更干净时，gate 的临床结构优势更明显。

## 当前问题

1. Scaffold 可能过度压缩，导致 ROUGE 表面匹配不如 flat scaffold。
2. 部分 case 中 gate 会把 sparse lab 解读为新主问题，尤其当 previous generated context 不完整时。
3. Prompt 中虽然要求 trust raw EHR，但还需要更强地要求保留 direct baseline 的主要 A&P 粒度和 wording。
4. 现在的 judge 与生成同源，可能偏好结构化输出，需要后续加入独立 judge 或 problem-level rule metric。

## 下一步建议

暂时不建议直接全量跑当前 no-judge 版本。建议先做一版 prompt 优化：

1. 在 generation prompt 中加入：不要因为 scaffold 改变真实 A&P section 粒度，优先保留 previous A&P 的 major headings。
2. 在 scaffold prompt 中加入：单纯 creatinine、Hgb、glucose 等 abnormal lab 不能独立成为 active problem，除非 previous A&P 或今日 note 明确支持。
3. 对 `watchlist/supportive_care` 加约束：只能作为一句 monitoring context，不得变成主 plan。
4. 可增加 `carry_forward_major_headings` 字段，让模型显式保留上一日主要 A&P headings。
5. 优化后先 rerun `generated_method2_gen` 10 cases；若 ROUGE delta 接近或超过 flat scaffold，同时 judge wins 保持，则再加 judge/revise。

## 路径

脚本：

```text
modeling/ap_memory_gated_scaffold_generation.py
evaluation/judge_augmented_ap.py
```

输出：

```text
outputs/ap_memory_gated_scaffold/
outputs/ap_memory_gated_scaffold/generated_method2_gen_summary.csv
outputs/ap_memory_gated_scaffold/oracle_method2_gt_summary.csv
outputs/ap_memory_gated_scaffold/memory_gated_fair_detail.csv
outputs/ap_memory_gated_scaffold/memory_gated_summary.csv
outputs/ap_memory_gated_scaffold/memory_gated_judge_detail.csv
outputs/ap_memory_gated_scaffold/scaffolds/
```

对照结果：

```text
outputs/ap_problem_state_augmented/augmented_fair_summary.csv
outputs/ap_problem_state_augmented/augmented_judge_summary.csv
```
