# AP-Memory-Gated Scaffold 新实验设计

## 核心判断

`analysis/ap_memory_experiment_report_zh.md` 中的 AP-Memory V2 方案可以借鉴，而且非常适合和当前 problem-state augmentation 合并。

当前 problem-state augmentation 已经证明：在 direct prompt 中保留 raw EHR + 同等 history，再额外加入 problem-state scaffold，比“用 problem-state 替代 EHR”更合理，targeted low-score set 上也有正向信号。

但当前 scaffold 的问题是：它仍然容易把所有抽取到的问题平铺到 `updated_problems`，缺少一个明确的 A&P problem-promotion gate。AP-Memory V2 的核心贡献正是：

```text
active_ap_problems / watchlist / supportive_care
```

这可以帮助区分：

- 今日应该进入 A&P 主体的问题；
- 需要记住但不该展开成独立 A&P section 的观察项；
- routine ICU care、prophylaxis、nutrition、access 等支持治疗。

因此，新的实验应该不是直接复用 AP-Memory V2 替代当前方法，而是形成：

```text
Memory-Gated Problem-State Scaffold Augmentation
```

简称：`memory_gated_scaffold`。

## 新实验 Motivation

当前实验已经说明：

1. 单纯 prefilter 没有稳定收益。
2. problem-state replacement 会丢 raw EHR 细节。
3. problem-state augmentation 有正向信号，但 scaffold 还不够精细。

低分 A&P 的主要失败模式包括：

- active problems 丢失；
- daily trajectory 缺失；
- unsupported diagnosis 或 unsupported plan；
- routine/supportive care 被过度提升为主要问题；
- generated-history setting 下 history drift 累积。

AP-Memory V2 报告中的观察也类似：

- V1 recall 高，但 precision 差；
- V1 容易把 lab abnormality、risk、supportive care 过度提升为 A&P 主问题；
- V2 通过 `active_ap_problems / watchlist / supportive_care` 和 problem-promotion gate 改善 precision 与 unsupported change。

所以新实验的核心假设是：

> 在保留 raw EHR 和同等 history 的前提下，使用 AP-Memory V2 式 problem-promotion gate 生成 scaffold，可以比当前 flat problem-state scaffold 更稳定地提升 A&P 质量，尤其是减少 unsupported / over-promoted problems，同时保留 trajectory capture 的收益。

## 方法设计

### 1. 输入

每个 patient-day 输入：

```text
previous context:
  - gt setting: previous gold A&P
  - gen setting: previous generated A&P

current evidence:
  - current-day EHR rows before A&P

optional previous memory:
  - previous day memory JSON, for autoregressive memory setting
```

### 2. Memory-Gated Scaffold Schema

新的 scaffold 不再使用单一 `updated_problems`，而使用 V2 分层：

```json
{
  "global_status": {
    "overall_trajectory": "improving|worsening|stable|mixed|unclear",
    "current_severity": "critical|serious|stable|unclear",
    "one_sentence_summary": ""
  },
  "active_ap_problems": [
    {
      "problem": "",
      "today_status": "new|improving|worsening|stable|resolved|unclear",
      "why_active_today": "",
      "supporting_evidence": [],
      "plan_actions": [],
      "confidence": "high|medium|low"
    }
  ],
  "watchlist": [
    {
      "item": "",
      "reason_not_active_ap_problem": "",
      "monitoring_plan": ""
    }
  ],
  "supportive_care": [
    {
      "item": "",
      "reason_not_active_ap_problem": "",
      "routine_plan": ""
    }
  ],
  "resolved_problems": [],
  "uncertainties": [],
  "promotion_gate_notes": []
}
```

### 3. 通用 Problem-Promotion Gate

只允许满足以下条件的问题进入 `active_ap_problems`：

- previous A&P 中仍未解决的主要问题；
- 今日 evidence 显示诊断、严重程度、治疗决策、处置计划或主要并发症发生变化；
- disposition blocker；
- 需要独立 A&P section 的主要 syndrome / organ system problem。

默认不进入 `active_ap_problems`，而进入 `watchlist` 或 `supportive_care`：

- isolated lab abnormality；
- single medication administration；
- prophylaxis；
- nutrition/access/routine monitoring；
- generic risk；
- unclear/low-confidence issue；
- 低优先级 carried-forward history。

每日 `active_ap_problems` 限制为最多 6 个，避免 checklist 化。

### 4. 生成方式

新方法不替代 direct prompt，而是在 direct prompt 中加入 gated scaffold：

```text
current EHR
+ matched history context
+ memory-gated scaffold
-> today's A&P
```

生成规则：

- A&P section 主要来自 `active_ap_problems`。
- `watchlist` 只能作为 monitoring / uncertainty 简短提及。
- `supportive_care` 只能合并到 ICU care / prophylaxis / nutrition，不展开成主问题。
- 若 raw EHR 和 scaffold 冲突，以 raw EHR 和明确证据为准。

## 实验组设计

### Targeted Low-Score Set

先使用当前 10 个低分 targeted day：

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

### Fair Configs

与当前 augmented 实验保持一致：

| config | direct baseline | scaffold history source |
|---|---|---|
| `oracle_method1_gt` | `method1/gt` | previous gold A&P |
| `oracle_method2_gt` | `method2/gt` | cumulative gold A&P |
| `generated_method1_gen` | `method1/gen` | previous generated A&P |
| `generated_method2_gen` | `method2/gen` | cumulative generated A&P |
| `none_method-1` | `method-1` | no history |

### 方法对照

建议比较四组：

| method | 含义 |
|---|---|
| `direct` | 原始 direct baseline。 |
| `flat_scaffold_augmented` | 当前 problem-state augmented。 |
| `memory_gated_scaffold_no_judge` | 新 V2-gated scaffold，不做 judge/revise。 |
| `memory_gated_scaffold_judge` | 新 V2-gated scaffold + judge/revise。 |

如果为了节省 API 成本，第一轮可以只跑：

```text
direct
flat_scaffold_augmented
memory_gated_scaffold_no_judge
```

确认有收益后再加 judge/revise。

## 指标设计

沿用当前指标：

- ROUGE-L
- active_problem_coverage
- trajectory_capture
- plan_specificity
- evidence_grounding
- disposition_context
- unsupported_problem_count
- missed_key_problem_count
- judge winner

增加 AP-Memory 报告中的问题级指标：

- Problem Precision
- Problem Recall
- Problem F1
- Unsupported Change
- Forgotten Carried Problem

这些指标能更直接验证 V2 gate 是否减少：

- over-promotion；
- unsupported new problems；
- carried-forward problem forgetting；
- generated-history drift。

## 预期收益

相比当前 flat scaffold，memory-gated scaffold 预期能：

- 保留 trajectory capture 的收益；
- 提高 active problem precision；
- 降低 unsupported problem count；
- 降低 routine/supportive care 被展开成主问题的概率；
- 在 generated-history setting 下减少 history drift；
- 让 scaffold 更像真实 ICU A&P 的问题粒度。

## 主要风险

1. Gate 过强，导致漏掉 gold A&P 中较细的问题。
2. `watchlist/supportive_care` 被生成器忽略，导致 recall 下降。
3. Judge/revise 增加 API 成本和不稳定性。
4. 问题级 taxonomy 指标可能不完全等价于临床质量。
5. Targeted low-score set 不能代表全量效果。

## 推荐执行顺序

### Step 1: Targeted no-judge

在 10 个低分 day 上跑：

```text
memory_gated_scaffold_no_judge
```

对比：

```text
direct
flat_scaffold_augmented
```

通过条件：

- ROUGE-L 不低于 flat scaffold；
- trajectory_capture 不下降；
- unsupported_problem_count 下降；
- missed_key_problem_count 不上升；
- judge winner 至少不弱于 flat scaffold。

### Step 2: Targeted judge/revise

如果 Step 1 有信号，再跑：

```text
memory_gated_scaffold_judge
```

重点观察：

- unsupported change 是否下降；
- active problem precision 是否提升；
- plan specificity 是否被削弱。

### Step 3: Full AP Set

优先全量跑两个 config：

```text
generated_method2_gen
oracle_method2_gt
```

因为：

- `generated_method2_gen` 最接近真实部署；
- `oracle_method2_gt` 是 upper bound；
- method2 是当前 direct history 中更强、更稳定的历史方式。

### Step 4: Longer-Horizon / Autoregressive

最终应评估 longer-horizon：

```text
day 2 -> day T
```

按 early/middle/late days 分析：

- Problem F1 是否随天数下降；
- Forgotten 是否随天数上升；
- Unsupported Change 是否累积；
- memory-gated scaffold 是否比 direct generated history 更抗漂移。

## 与当前实验的关系

当前实验结论：

```text
flat problem-state scaffold augmentation 有正向信号。
```

新实验是在这个基础上的增强：

```text
flat problem-state scaffold
  -> memory-gated scaffold with active_ap_problems / watchlist / supportive_care
```

不是推翻当前方案，而是把 AP-Memory V2 中最有价值的“问题晋升门控”移植到当前更公平、更有效的 augmented generation 框架中。

