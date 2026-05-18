# Memory-Gated Problem-State Scaffold V2 评估报告

## 一句话结论

`memory_gated_scaffold_v2` 在 targeted low-score set 上相对 direct baseline 有稳定正向信号：`generated_method2_gen` 和 `oracle_method2_gt` 两个配置均提升 ROUGE，并且 LLM judge 大多数 case 选择 V2。相比 V1，V2 的提升更明显；相比 flat scaffold，V2 的 ROUGE 在 generated-history setting 中仍较弱，但 clinical judge 指标更稳。

## 实验 Motivation

当前 A&P 生成的主要失败模式不是单纯缺少 evidence，而是：

1. 丢失上一日未解决的 active problems；
2. 缺少 daily trajectory，只写静态诊断；
3. generated-history setting 中容易发生 history drift；
4. routine ICU care、isolated lab abnormality、single medication 被过度提升为 A&P 主问题；
5. direct baseline 能看到 raw EHR，但缺少结构化问题状态约束。

此前 flat problem-state augmentation 已有正向信号，但它的问题是 scaffold 比较“平铺”，容易提高 recall 却牺牲 precision。AP-Memory V2 的价值在于 problem-promotion gate：把问题分成 `active_ap_problems / watchlist / supportive_care`，减少 over-promotion。

因此 V2 实验的目标是：在不训练模型、不写死疾病规则、不改变 raw EHR 输入面的前提下，加入一个更通用、更有 gate 的 scaffold，验证是否能改善 A&P 的 continuity、trajectory、grounding 和 unsupported problem。

## 方法设计

方法名：

```text
memory_gated_scaffold_v2
```

生成流程：

```text
previous context + current-day raw EHR
-> memory-gated scaffold V2 JSON
-> current-day raw EHR + matched history context + scaffold
-> today's A&P
```

注意：V2 是 augmentation，不是 replacement。生成 A&P 时仍然保留 direct baseline 的完整 raw EHR 和同等 history context。

## V2 Scaffold Schema

V2 相比 V1 的主要变化是增加三个通用字段：

```json
{
  "carry_forward_major_headings": [],
  "candidate_problem_pool": [],
  "active_ap_problems": [
    {
      "section_role": "primary_section|merged_into_existing_section|brief_monitoring"
    }
  ],
  "watchlist": [],
  "supportive_care": []
}
```

### 1. `carry_forward_major_headings`

用于显式保留上一日 A&P 的主要 heading 粒度，避免 scaffold 过度压缩后丢掉 gold note 中常见的主问题表达。

### 2. `candidate_problem_pool`

用于保留 recall backup。并不是所有 candidate 都进入主 A&P section，但它可以防止 gate 过强导致漏掉关键问题。

### 3. `section_role`

控制问题如何进入最终 A&P：

| role | 含义 |
|---|---|
| `primary_section` | 应该成为独立 A&P subsection |
| `merged_into_existing_section` | 应合并到更大的 organ-system/problem section |
| `brief_monitoring` | 只作为 monitoring/update 简短提及 |

## 通用 Promotion Gate

V2 没有写 case-specific 规则，只使用通用 gate：

- prior A&P 中未解决的 major problem 可以 carry forward；
- 今日 evidence 改变 diagnosis、severity、treatment decision、disposition 或 major complication 时，可以进入 `active_ap_problems`；
- isolated abnormal lab 或 single medication 不能独立成为主问题，除非满足：
  - prior A&P 已把它作为 major problem；
  - 今日 clinician text 明确把它作为 diagnosis/treatment target；
  - 它改变 treatment、consult、disposition 或 monitoring intensity；
- prophylaxis、nutrition、access、routine monitoring、generic risk 默认进入 `watchlist/supportive_care`；
- 每日 `active_ap_problems` 最多 6 个。

## 当前实验设定

### Targeted Cases

本轮使用此前识别的 10 个 low-score A&P day：

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

### Configs

| config | direct baseline | history source | 作用 |
|---|---|---|---|
| `generated_method2_gen_v2` | `deepseek_api_full_gen/gen/method2` | cumulative generated A&P | 更接近真实部署，测试 generated-history drift |
| `oracle_method2_gt_v2` | `deepseek_api_full/gt/method2` | cumulative gold A&P | 较干净 history，测试方法上限 |

### 实验配置说明

| 实验配置 | 最终 A&P 输入 | base 对照 | scaffold 来源 | 是否使用 gold history | 是否加入 generation-time judge/revise | 说明 |
|---|---|---|---|---|---|---|
| `direct generated_method2_gen` | current raw EHR + cumulative generated A&P | 本身 | 无 | 否 | 否 | 真实部署近似设置，base 有历史信息，不是 method-1。 |
| `memory_gated generated_method2_gen_v1` | current raw EHR + cumulative generated A&P + V1 scaffold | `direct generated_method2_gen` | cumulative generated A&P + current raw EHR | 否 | 否 | 初版 gate scaffold。 |
| `memory_gated generated_method2_gen_v2` | current raw EHR + cumulative generated A&P + V2 scaffold | `direct generated_method2_gen` | cumulative generated A&P + current raw EHR | 否 | 否 | 当前最接近部署的 V2 主实验。 |
| `direct oracle_method2_gt` | current raw EHR + cumulative gold A&P | 本身 | 无 | 是 | 否 | upper-bound 设置，不代表真实部署。 |
| `memory_gated oracle_method2_gt_v1` | current raw EHR + cumulative gold A&P + V1 scaffold | `direct oracle_method2_gt` | cumulative gold A&P + current raw EHR | 是 | 否 | V1 的 oracle-history 上限。 |
| `memory_gated oracle_method2_gt_v2` | current raw EHR + cumulative gold A&P + V2 scaffold | `direct oracle_method2_gt` | cumulative gold A&P + current raw EHR | 是 | 否 | V2 的 oracle-history 上限。 |
| `memory_gated generated_method2_gen_v2_judge_revise` | current raw EHR + cumulative generated A&P + revised V2 scaffold | `direct generated_method2_gen` | cumulative generated A&P + current raw EHR + generation-time judge feedback | 否 | 是 | 最接近部署的 V2 + judge/revise 闭环。 |
| `memory_gated oracle_method2_gt_v2_judge_revise` | current raw EHR + cumulative gold A&P + revised V2 scaffold | `direct oracle_method2_gt` | cumulative gold A&P + current raw EHR + generation-time judge feedback | 是 | 是 | oracle-history 上限的 V2 + judge/revise 闭环。 |

### 对照组

| method | 含义 |
|---|---|
| direct baseline | 不加 scaffold 的原始生成 |
| flat scaffold augmented | 旧 problem-state augmented |
| memory-gated scaffold V1 | 初版 gate scaffold |
| memory-gated scaffold V2 | 本报告方法 |

## 主要结果

### V2 vs Direct Baseline

| config | base 对照 | history 是否一致 | generation-time judge/revise | n | base ROUGE | V2 ROUGE | ROUGE delta | ROUGE wins | evaluation judge V2 wins | evaluation judge base wins | ties |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `generated_method2_gen_v2` | `deepseek_api_full_gen/gen/method2` | 是，均为 cumulative generated A&P | 未启用 | 10 | 5.806 | 5.974 | +0.168 | 7 | 8 | 1 | 1 |
| `oracle_method2_gt_v2` | `deepseek_api_full/gt/method2` | 是，均为 cumulative gold A&P | 未启用 | 10 | 6.664 | 7.126 | +0.462 | 7 | 9 | 0 | 1 |

这里的 `evaluation judge` 是实验结束后用于比较 direct baseline 与 V2 output 的评估器；它不是生成流程中的 judge/revise。当前 V2 结果均为 no-judge generation。

### Generation-Time Judge/Revise 状态

生成流程中的 judge/revise 已补跑。流程为：

```text
V2 scaffold
-> candidate A&P
-> generation-time transition judge
-> revised V2 scaffold
-> final A&P
-> evaluation judge vs direct baseline
```

这里的 generation-time judge 不使用当前 gold A&P，只使用 latest previous A&P、today raw EHR、candidate scaffold 和 candidate A&P。

| config | candidate scaffold/AP judge | revise scaffold/memory | regenerate final A&P | 当前结果 |
|---|---|---|---|---|
| `generated_method2_gen_v2` | 未运行 | 未运行 | 未运行 | no-judge V2 结果。 |
| `oracle_method2_gt_v2` | 未运行 | 未运行 | 未运行 | no-judge V2 结果。 |
| `generated_method2_gen_v2_judge_revise` | 已运行 | 已运行 | 已运行 | generated-history V2 + judge/revise。 |
| `oracle_method2_gt_v2_judge_revise` | 已运行 | 已运行 | 已运行 | oracle-history V2 + judge/revise。 |

### V2 Judge/Revise vs Direct Baseline

| config | base 对照 | history 是否一致 | generation-time judge/revise | n | base ROUGE | final ROUGE | ROUGE delta | ROUGE wins | evaluation judge final wins | evaluation judge base wins | ties |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `generated_method2_gen_v2_judge_revise` | `deepseek_api_full_gen/gen/method2` | 是，均为 cumulative generated A&P | 已启用 | 10 | 5.806 | 6.016 | +0.210 | 7 | 7 | 3 | 0 |
| `oracle_method2_gt_v2_judge_revise` | `deepseek_api_full/gt/method2` | 是，均为 cumulative gold A&P | 已启用 | 10 | 6.664 | 7.310 | +0.646 | 8 | 9 | 1 | 0 |

### No-Judge V2 vs Judge/Revise V2

| config family | no-judge ROUGE delta | judge/revise ROUGE delta | no-judge eval judge wins | judge/revise eval judge wins | 主要变化 |
|---|---:|---:|---:|---:|---|
| `generated_method2_gen` | +0.168 | +0.210 | 8/10 | 7/10 | ROUGE 小幅上升，但 evaluation judge wins 和 coverage/trajectory 等维度回落。 |
| `oracle_method2_gt` | +0.462 | +0.646 | 9/10 | 9/10 | ROUGE 继续上升，judge wins 持平，整体仍强。 |

Judge/revise 的初步解释：它会删掉不够支持的具体计划，并把部分内容降级到 watchlist/supportive care，因此 final note 更保守。这个保守性带来轻微 ROUGE 增益，但在 generated-history setting 中会牺牲一部分 coverage、trajectory 和 plan specificity。

### V2 Judge Delta

Delta 表示 V2 minus direct baseline。

| config | coverage | trajectory | plan specificity | grounding | disposition | unsupported count | missed count |
|---|---:|---:|---:|---:|---:|---:|---:|
| `generated_method2_gen_v2` | +0.7 | +1.0 | +0.6 | +0.6 | +0.6 | -0.6 | -0.7 |
| `oracle_method2_gt_v2` | +1.0 | +1.6 | +1.0 | +1.6 | +1.0 | -0.9 | -1.0 |

解释：

- `coverage` 上升：更好覆盖 gold active problems；
- `trajectory` 上升：更能表达今日变化；
- `grounding` 上升：unsupported diagnosis/treatment 更少；
- `unsupported count` 下降：少写没有证据支持的问题；
- `missed count` 下降：少漏关键问题。

### V1 vs V2

| config | base ROUGE | V1 ROUGE | V2 ROUGE | V1 delta | V2 delta | V1 judge wins | V2 judge wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| `generated_method2_gen` | 5.806 | 5.896 | 5.974 | +0.090 | +0.168 | 7/10 | 8/10 |
| `oracle_method2_gt` | 6.664 | 6.630 | 7.126 | -0.035 | +0.462 | 9/10 | 9/10 |

V2 相比 V1 的收益主要来自：

- 显式保留上一日 major headings；
- `candidate_problem_pool` 保留 recall；
- `section_role` 降低 checklist 化；
- 更严格的 lab-promotion gate 降低 unsupported/missed problem。

### V2 vs Flat Scaffold

此前 flat scaffold 的结果：

| config | flat ROUGE delta | flat judge wins |
|---|---:|---:|
| `generated_method2_gen` | +0.296 | 5/10 |
| `oracle_method2_gt` | +0.309 | 7/10 |

V2 对照：

| config | V2 ROUGE delta | V2 judge wins |
|---|---:|---:|
| `generated_method2_gen_v2` | +0.168 | 8/10 |
| `oracle_method2_gt_v2` | +0.462 | 9/10 |

解读：

- 在 generated-history setting 中，flat scaffold 的 ROUGE delta 仍更高；
- 但 V2 的 judge wins 和 clinical judge 指标更好；
- 在 oracle-history setting 中，V2 同时超过 flat scaffold 的 ROUGE delta 和 judge wins。

这说明 V2 更偏向提升 clinical structure / grounding / continuity，而 flat scaffold 可能更偏向提高文本 recall。

## 当前实验是否公平

### 目前公平的部分

1. **输入 evidence surface 公平**
   V2 生成最终 A&P 时仍然使用和 direct baseline 相同的 current-day raw EHR。

2. **history context 公平**
   `generated_method2_gen_v2` 使用 cumulative generated A&P，和对应 direct baseline 一致。

3. **没有 gold leakage 到 generated setting**
   `generated_method2_gen_v2` 的 scaffold history source 是 generated A&P，不使用 previous gold A&P。

4. **V2 是 no-training**
   没有训练参数，没有 supervised label tuning，也没有 case-specific prompt。

5. **prompt 规则是通用的**
   promotion gate 不写具体疾病、药物、lab 名称，也不依赖 MIMIC-specific schema。

### 需要警惕的偏差

1. **Targeted low-score set 偏差**
   当前 10 例是低分样本，不代表全量分布。它适合验证“是否修复失败模式”，但不能证明全量平均收益。

2. **同源 LLM judge 偏差**
   生成和 judge 都使用 DeepSeek，judge 可能偏好结构化输出。后续最好加入另一模型 judge 或 problem-level rule metric。

3. **ROUGE 与临床质量冲突**
   V2 的 judge 指标更好，但 generated-history setting 的 ROUGE delta 低于 flat scaffold，说明它可能更改写、更临床抽象，不一定贴近 gold wording。

4. **Oracle config 不能代表部署**
   `oracle_method2_gt_v2` 使用 cumulative gold A&P，是 upper bound。真实部署应优先看 `generated_method2_gen_v2`。

5. **Scaffold 生成增加一次 LLM 调用**
   当前比较的是“多一步 no-training LLM scaffold + generation” vs direct generation，计算成本不相等。若关注成本，需要额外报告 API call 数和 token 成本。

6. **Judge 指标不是 blinded human eval**
   当前 judge 看到 gold、baseline、augmented 三者并直接比较，适合快速筛选，但不是最终临床评估。

## 当前可确认的实验设定

如果继续扩量，建议固定如下设定：

```text
method: memory_gated_scaffold_v2
training: none
generation model: deepseek-chat
temperature: 0
history setting: generated_method2_gen first
evaluation:
  - ROUGE-L vs gold A&P
  - LLM judge vs direct baseline
  - unsupported_problem_count
  - missed_key_problem_count
  - active_problem_coverage
  - trajectory_capture
  - evidence_grounding
```

扩量优先级：

1. 先扩 `generated_method2_gen_v2`，因为最接近真实部署；
2. 再扩 `oracle_method2_gt_v2`，作为上限对照；
3. 暂时不加 judge/revise，避免把 V2 scaffold 本身收益和 revise 收益混在一起。

## 实验路径

### 脚本

```text
modeling/ap_memory_gated_scaffold_generation.py
evaluation/judge_augmented_ap.py
```

### 数据

```text
data/AP/input/
data/AP/gold/
data/AP/generated/DG/deepseek_api_full/
data/AP/generated/DG/deepseek_api_full_gen/
```

### V2 输出

```text
outputs/ap_memory_gated_scaffold/generated_method2_gen_v2/
outputs/ap_memory_gated_scaffold/oracle_method2_gt_v2/
outputs/ap_memory_gated_scaffold/scaffolds/generated_method2_gen_v2/
outputs/ap_memory_gated_scaffold/scaffolds/oracle_method2_gt_v2/
outputs/ap_memory_gated_scaffold/generated_method2_gen_v2_summary.csv
outputs/ap_memory_gated_scaffold/oracle_method2_gt_v2_summary.csv
outputs/ap_memory_gated_scaffold/generated_method2_gen_v2_judge_detail.csv
outputs/ap_memory_gated_scaffold/oracle_method2_gt_v2_judge_detail.csv
outputs/ap_memory_gated_scaffold/generated_method2_gen_v2_judge_revise/
outputs/ap_memory_gated_scaffold/oracle_method2_gt_v2_judge_revise/
outputs/ap_memory_gated_scaffold/scaffolds/generated_method2_gen_v2_judge_revise/
outputs/ap_memory_gated_scaffold/scaffolds/oracle_method2_gt_v2_judge_revise/
outputs/ap_memory_gated_scaffold/generation_judges/generated_method2_gen_v2_judge_revise/
outputs/ap_memory_gated_scaffold/generation_judges/oracle_method2_gt_v2_judge_revise/
outputs/ap_memory_gated_scaffold/generated_method2_gen_v2_judge_revise_summary.csv
outputs/ap_memory_gated_scaffold/oracle_method2_gt_v2_judge_revise_summary.csv
outputs/ap_memory_gated_scaffold/generated_method2_gen_v2_judge_revise_eval_judge_detail.csv
outputs/ap_memory_gated_scaffold/oracle_method2_gt_v2_judge_revise_eval_judge_detail.csv
outputs/ap_memory_gated_scaffold/memory_gated_v2_judge_revise_summary.csv
outputs/ap_memory_gated_scaffold/memory_gated_v2_summary.csv
```

### V1/Flat 对照

```text
outputs/ap_memory_gated_scaffold/memory_gated_summary.csv
outputs/ap_problem_state_augmented/augmented_judge_summary.csv
outputs/ap_problem_state_augmented/augmented_fair_summary.csv
```

## 当前建议

V2 值得进入下一步扩量，但不要马上声称方法最终优于 flat scaffold。更稳妥的表述是：

> V2 在 targeted low-score set 上相对 direct baseline 稳定改善 clinical judge 指标，并在两个 method2 设置中提升 ROUGE；相比 flat scaffold，V2 更强在 clinical grounding 与 continuity，flat scaffold 在 generated-history setting 中 ROUGE recall 仍更高。

下一轮应优先验证：

1. `generated_method2_gen_v2` 在更多样本上是否仍能保持正向 ROUGE delta；
2. judge wins 是否仍显著高于 direct baseline；
3. unsupported/missed problem 是否持续下降；
4. V2 是否在非 targeted cases 上也不伤害普通样本。
