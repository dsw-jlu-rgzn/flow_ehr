# A&P Problem-State No-Training 实验报告

## 当前实验结论与主要设定

当前实验主要设定：

- 任务：MIMIC-III ICU daily A&P 生成。
- Baseline：DeepSeek direct generation。
- 公平比较：
  - `method-1`：无历史 note。
  - `method1`：使用上一天 note。
  - `method2`：使用累计历史 notes。
  - `gt`：历史来自 gold note。
  - `gen`：历史来自模型生成 note。
- 新方法：problem-state augmentation，即保留原始 EHR + 同等历史 note，只额外加入 LLM 抽取的 problem-state scaffold。
- 测试集：10 个低分 targeted day，不是全量随机集。
- 指标：ROUGE-L + LLM judge，包括 active problem coverage、trajectory capture、plan specificity、evidence grounding、disposition context 等。

当前实验结论：

- 单纯 prefilter 没有稳定收益。
- 用 problem-state 替代 EHR 会丢细节，公平比较下效果下降。
- 把 problem-state 作为 scaffold 加到 direct prompt 里，目前最有希望。
- Augmented 在 targeted set 上平均 ROUGE `+0.36`，judge 结果为 `30/50` wins。
- 主要收益来自更好地保留 active problems、捕捉 daily trajectory、减少 key problem 遗漏。
- 但当前只是低分样本 pilot，还不能代表全量效果。
- 下一步应全量验证 `method2/gen + augmented` 和 `method2/gt + augmented`。

## 实验 Motivation

本实验的核心动机是：当前 A&P 生成失败并不主要表现为“EHR 太长，需要简单筛选”，而是表现为 **active problem list 断裂** 和 **daily trajectory 缺失**。

在低分样本中，baseline 经常只根据当天少量 EHR rows 写出局部、泛化的计划。例如某些 day 的当前 EHR 只有 insulin、heparin、PO intake 等记录，模型就倾向于生成“高血糖/预防用药”类 A&P；但 gold A&P 的核心其实是前一天已经建立的 CHF、ESRD/HD、hypoxia、disposition/callout 等问题。这说明单纯 top-k prefilter 很难修复失败，因为问题不是信息过多，而是模型没有稳定维护“病人今天有哪些 active problems，以及这些问题相比昨天如何变化”。

因此，本实验尝试引入一个 no-training 的 problem-state scaffold：

- 从 previous note 中抽取可延续的 active problems 和 prior plans。
- 从 current EHR 中抽取会改变诊断、严重程度、治疗、监测或 disposition 的 daily evidence。
- 合并得到今天的 problem update，包括每个问题的 status、evidence 和 plan action。
- 在最终生成时保留原始 EHR 和相同历史 note，只额外加入 problem-state scaffold，避免丢失原始证据。

这个设计的目标不是替代 LLM 生成器，而是给 direct generation 一个临床上更自然的“问题状态约束”，让模型在写 A&P 时更稳定地覆盖 active problems、表达 daily trajectory，并减少 unsupported 或遗漏的问题。

## 复现路径与实验产物

仓库根目录：

```text
C:\Users\dsw54\Desktop\codex_related\flow_ehr
```

核心数据路径：

```text
data/AP/input/                         # AP 输入时间线，文件名 input_<hadm_id>.csv
data/AP/gold/                          # AP gold progress note，文件名 gt_<hadm_id>.csv
data/AP/generated/DG/deepseek_api_full/ # DeepSeek direct baseline，setting=gt
data/AP/generated/DG/deepseek_api_full_gen/ # DeepSeek direct baseline，setting=gen
```

主要脚本：

```text
modeling/deepseek_api_generation.py                 # direct AP/DS DeepSeek API generation
modeling/deepseek_api_prefilter.py                  # no-training DeepSeek prefilter 实验
modeling/ap_problem_state_experiment.py             # problem memory/evidence/update 抽取与 replacement 生成实验
modeling/ap_problem_state_augmented_generation.py   # problem-state scaffold augmentation 生成实验
evaluation/compare_ap_generation.py                 # AP ROUGE-L paired comparison
evaluation/judge_augmented_ap.py                    # augmented vs baseline 的 LLM judge
processing/validate_mimic3_tasks.py                 # MIMIC-III AP/DS 数据质量检查
```

Problem-state 中间结果：

```text
outputs/ap_problem_state_fair/
  oracle_method1/
  oracle_method2/
  generated_method1/
  generated_method2/
  none_method-1/
```

每个 case 目录中包含：

```text
problem_memory.json      # 从 previous note 抽取的 active problem memory
daily_evidence.json      # 从 current EHR 抽取的 daily evidence
problem_update.json      # 合并 memory 和 evidence 后的 today's problem update
generated_ap.txt         # replacement 版本生成的 A&P
judge_metrics.json       # replacement 版本的 baseline vs problem-state judge
```

Augmented 生成结果：

```text
outputs/ap_problem_state_augmented/
  oracle_method1_gt/
  oracle_method2_gt/
  generated_method1_gen/
  generated_method2_gen/
  none_method-1/
```

每个 augmented case 文件格式：

```text
outputs/ap_problem_state_augmented/<config>/<hadm_id>_day<day>.txt
```

关键汇总文件：

```text
outputs/ap_problem_state_augmented/augmented_fair_summary.csv
outputs/ap_problem_state_augmented/augmented_fair_detail.csv
outputs/ap_problem_state_augmented/augmented_judge_summary.csv
outputs/ap_problem_state_augmented/augmented_judge_detail.csv
outputs/ap_problem_state_fair/fair_summary.csv
outputs/ap_problem_state_fair/fair_detail.csv
outputs/ap_compare_deepseek_prefilters_summary.csv
outputs/ap_compare_deepseek_prefilters.csv
outputs/ap_baseline_day_level_scores.csv
outputs/ap_low_score_case_excerpts.md
```

复现时需要设置 DeepSeek API key：

```powershell
$env:DEEPSEEK_API_KEY='your_api_key'
```

示例命令：

```powershell
# 生成 direct baseline，generated-history setting
python modeling\deepseek_api_generation.py ap `
  --inputdir data\AP\input `
  --outputdir data `
  --run-name deepseek_api_full_gen `
  --setting gen `
  --methods 1 2

# 跑 problem-state extraction / replacement 实验
python modeling\ap_problem_state_experiment.py `
  --output-dir outputs\ap_problem_state_fair\generated_method2 `
  --baseline-run-name deepseek_api_full_gen `
  --baseline-setting gen `
  --baseline-method method2 `
  --memory-source baseline_method

# 跑 augmented generation
python modeling\ap_problem_state_augmented_generation.py `
  --problem-state-dir outputs\ap_problem_state_fair\generated_method2 `
  --config-name generated_method2_gen `
  --baseline-run-name deepseek_api_full_gen `
  --baseline-setting gen `
  --baseline-method method2 `
  --memory-source baseline_method

# 跑 augmented judge
python evaluation\judge_augmented_ap.py `
  --detail-csv outputs\ap_problem_state_augmented\augmented_fair_detail.csv `
  --output-csv outputs\ap_problem_state_augmented\augmented_judge_detail.csv
```

## 1. 背景与目标

本轮实验的目标是验证一个 no-training 的 A&P 生成改进方向：在不训练模型的前提下，用 LLM 抽取和维护“问题状态”（problem state），帮助模型生成更稳定的 ICU daily Assessment & Plan。

前置观察来自低分 A&P 样本分析。baseline 在若干低分 day 上常见失败模式包括：

- 只根据当天稀疏 EHR 写计划，丢失前一天已经确立的 active problem list。
- 能识别一些疾病名，但不能稳定表达“今天变好/变差/稳定”的病程轨迹。
- 生成泛化 ICU boilerplate，或者引入没有证据支持的问题。
- 对 disposition、code status、renal/respiratory trajectory 等 A&P 关键上下文覆盖不足。

因此，实验假设是：

> A&P 失败的关键不是 EHR 信息太多，而是模型缺乏一个稳定、可更新的 problem-state scaffold。若把 previous note 和 current EHR 先转成结构化 problem update，再用于生成或增强生成，可能提升 active problem coverage、trajectory capture 和 grounding。

## 2. 数据与实验样本

数据来自重新用 MIMIC-III 生成的 AP 任务：

- 输入：`data/AP/input/input_<hadm_id>.csv`
- gold：`data/AP/gold/gt_<hadm_id>.csv`
- direct baseline：`data/AP/generated/DG/deepseek_api_full/`
- generated-history direct baseline：`data/AP/generated/DG/deepseek_api_full_gen/`

本轮主要是 targeted low-score pilot，不是全量实验。样本选择为 baseline day-level ROUGE-L 较低、失败模式明显的 10 个 day：

- `177997 day 3`
- `177997 day 4`
- `196033 day 4`
- `196033 day 5`
- `196033 day 6`
- `196033 day 7`
- `196033 day 8`
- `115691 day 3`
- `126929 day 3`
- `126929 day 4`

这些 day 用来判断方法是否确实修复当前暴露出的 A&P 失败模式。它们不是随机样本，因此不能直接代表全量平均效果。

## 3. Direct Baseline 方法定义

AP direct generation 由 `modeling/deepseek_api_generation.py` 产生。它有三个 method：

| method | 含义 |
|---|---|
| `method-1` | 不使用 previous progress note，只用当天 EHR 生成当前 A&P。 |
| `method1` | 使用上一天 progress note 作为历史上下文。 |
| `method2` | 使用累计历史 progress notes 作为历史上下文。 |

它还有两个 setting：

| setting | 含义 |
|---|---|
| `gt` | 历史 note 使用 gold note。相当于 oracle history / teacher-forced setting。 |
| `gen` | 历史 note 使用模型自己前一天生成的 note。更接近真实部署，但会有 history drift。 |

因此，公平比较必须匹配历史来源：

- `method1/gt` 应该对比使用上一天 gold memory 的 problem-state 方法。
- `method2/gt` 应该对比使用历史 gold memory 的 problem-state 方法。
- `method1/gen` 应该对比使用上一天 generated memory 的 problem-state 方法。
- `method2/gen` 应该对比使用历史 generated memory 的 problem-state 方法。
- `method-1` 应该对比 no-memory 的 problem-state 方法。

## 4. 第一版：No-Train Prefilter

第一版实验是 DeepSeek API no-training prefilter：

- `day_context`：用当天 EHR 让 DeepSeek 选择 top-k snippets。
- `previous_note`：用 previous note 作为 query context 选择 top-k snippets。

输出再送入原 AP generator。

### 结果

| method | baseline ROUGE-L | day_context | previous_note |
|---|---:|---:|---:|
| method-1 | 6.60 | 6.53 (-0.07) | 6.67 (+0.07) |
| method1 | 7.96 | 7.83 (-0.12) | 7.87 (-0.09) |
| method2 | 8.21 | 8.16 (-0.05) | 8.07 (-0.14) |

### 分析

这个方案没有稳定收益。主要原因是当前 MIMIC-III AP 数据平均每天 non-note rows 只有约 17 行，而当时 top-k 设为 40，几乎没有真正压缩。加入 trend snippets 后，有些 day 的输入反而更长。

因此，这个实验不能很好验证“信息筛选”是否有帮助；更重要的是，它没有直接针对 low-score A&P 的真实失败模式。

## 5. 第二版：Problem-State Replacement

第二版脚本：

- `modeling/ap_problem_state_experiment.py`

流程为：

1. 从 previous note 抽取 `problem_memory.json`。
2. 从 current EHR 抽取 `daily_evidence.json`。
3. 合并为 `problem_update.json`。
4. 只用 `problem_update.json` 生成 A&P。
5. 用 LLM judge 比较 direct baseline、problem-state generation 和 gold。

中间表示大致为：

```json
{
  "updated_problems": [
    {
      "problem": "short clinical problem name",
      "today_status": "new|improving|worsening|stable|resolved|unclear",
      "assessment": "one concise sentence",
      "supporting_evidence": ["evidence item"],
      "plan_actions": ["specific action or monitoring item"],
      "confidence": "high|medium|low"
    }
  ],
  "global_plan": {
    "disposition": "",
    "code_status": "",
    "monitoring_priorities": []
  }
}
```

### 初始不公平 pilot

最早的 pilot 用 previous gold note 抽 problem memory，却对比 no-history `method-1` baseline，因此结果很乐观：

- problem-state wins：10/10
- active problem coverage：+1.8
- trajectory capture：+2.3
- missed key problem count：-2.7

这个结果说明 oracle memory 有潜力，但不是公平比较。

### 公平比较结果

公平比较把 direct baseline 的 history source 和 problem-state 的 memory source 对齐。

| config | direct baseline | memory source | baseline ROUGE | problem-state ROUGE | delta |
|---|---|---|---:|---:|---:|
| oracle_method1_gt | method1/gt | gold | 6.63 | 5.59 | -1.04 |
| oracle_method2_gt | method2/gt | gold | 6.66 | 6.00 | -0.67 |
| generated_method1_gen | method1/gen | generated | 5.84 | 4.62 | -1.22 |
| generated_method2_gen | method2/gen | generated | 5.81 | 4.76 | -1.05 |
| none_method-1 | method-1 | none | 5.19 | 4.12 | -1.07 |

Judge 指标显示它确实提升 trajectory capture，但 plan specificity、disposition context、coverage 不稳定。

### 分析

Problem-state replacement 的核心问题是：它把原始 EHR 细节压缩得太狠。生成器只看结构化 problem update，不再看完整当天 EHR，因此 A&P 更干净、更有 trajectory，但容易丢具体计划、药物、检查、disposition 细节。

结论：problem-state 不能替代 raw EHR evidence。

## 6. 第三版：Problem-State Scaffold Augmentation

第三版是当前最合理的版本。脚本：

- `modeling/ap_problem_state_augmented_generation.py`
- `evaluation/judge_augmented_ap.py`

核心改动：

> 不再用 problem-state 替代 raw EHR，而是在 direct generation prompt 中保留同样的 current EHR 和同样的 history，只额外加入 problem-state scaffold。

也就是：

```text
direct baseline:
  current EHR
  + previous note context
  -> A&P

augmented:
  current EHR
  + previous note context
  + problem-state scaffold
  -> A&P
```

这样比较更公平，因为 augmented 唯一增加的信息是结构化问题状态约束，而没有剥夺模型访问原始证据的能力。

## 7. 指标定义

### 7.1 ROUGE-L F1

ROUGE-L F1 基于 longest common subsequence，衡量生成文本和 gold note 的词序列重叠。

优点：

- 自动化、可复现。
- 能粗略反映生成内容是否覆盖 gold note 的词汇和短语。

局限：

- MIMIC progress note 包含大量 flowsheet、模板和非 A&P 文本，ROUGE 不一定等价于临床质量。
- 简洁但正确的 A&P 可能 ROUGE 偏低。
- 复制大量模板可能 ROUGE 偏高。

因此本实验把 ROUGE 作为辅助指标。

### 7.2 LLM Judge 指标

使用 DeepSeek judge 对 baseline 和 augmented 分别打 1-5 分，并判断 winner。指标如下：

| 指标 | 含义 |
|---|---|
| `active_problem_coverage` | 是否覆盖 gold A&P 中的主要 active problems。 |
| `trajectory_capture` | 是否捕捉 today status，如 improving/worsening/stable/resolved，而不是只列静态诊断。 |
| `plan_specificity` | plan 是否具体、可执行，是否包含治疗、监测、处置动作。 |
| `evidence_grounding` | 问题和计划是否由 current EHR 或 history 支撑，是否避免 unsupported diagnosis。 |
| `disposition_context` | 是否覆盖 disposition、code status、ICU/floor 转归等全局 care context。 |
| `unsupported_problem_count` | 生成中无证据支持的问题数量。越低越好。 |
| `missed_key_problem_count` | gold 中关键问题被漏掉的数量。越低越好。 |
| `winner` | judge 综合判断 baseline / augmented / tie。 |

局限：

- 生成和 judge 都用 DeepSeek，存在同源偏好。
- Judge 可能偏好结构化、清晰的输出。
- 后续应加入独立 judge 或 gold problem extraction 交叉验证。

## 8. Augmented 实验结果

### 8.1 ROUGE-L

| config | baseline setting | method | memory source | baseline ROUGE | augmented ROUGE | delta | augmented wins |
|---|---|---|---|---:|---:|---:|---:|
| generated_method1_gen | gen | method1 | generated | 5.84 | 5.84 | -0.00 | 7/10 |
| generated_method2_gen | gen | method2 | generated | 5.81 | 6.10 | +0.30 | 8/10 |
| none_method-1 | gt | method-1 | none | 5.19 | 5.74 | +0.55 | 7/10 |
| oracle_method1_gt | gt | method1 | gold | 6.63 | 7.30 | +0.67 | 7/10 |
| oracle_method2_gt | gt | method2 | gold | 6.66 | 6.97 | +0.31 | 6/10 |

总体上，augmented 在 4/5 个 config 中提高 ROUGE。唯一基本持平的是 `generated_method1_gen`，平均 delta 为 -0.001。

### 8.2 Judge 结果

| config | augmented wins | baseline wins | ties | ROUGE delta |
|---|---:|---:|---:|---:|
| generated_method1_gen | 6 | 2 | 2 | -0.00 |
| generated_method2_gen | 5 | 4 | 1 | +0.30 |
| none_method-1 | 5 | 2 | 3 | +0.55 |
| oracle_method1_gt | 7 | 2 | 1 | +0.67 |
| oracle_method2_gt | 7 | 2 | 1 | +0.31 |

总体 50 个 pair：

- augmented wins：30
- baseline wins：12
- ties：8
- mean ROUGE delta：+0.36

Judge 平均 delta：

| 指标 | delta |
|---|---:|
| active_problem_coverage | +0.40 |
| trajectory_capture | +0.76 |
| plan_specificity | +0.26 |
| evidence_grounding | +0.50 |
| disposition_context | +0.22 |
| unsupported_problem_count | -0.08 |
| missed_key_problem_count | -0.30 |

### 8.3 结果解读

最稳定的收益是 `trajectory_capture`，这与方法动机一致。Problem-state scaffold 明确提醒模型：每个 active problem 今天是 improving、worsening、stable、resolved 还是 unclear。

第二个较稳定收益是 `evidence_grounding` 和 `active_problem_coverage`。这说明 scaffold 能帮助模型避免只盯当天稀疏 rows，而是保留重要的 carried-forward problems。

`plan_specificity` 有小幅提升，但不大。说明 scaffold 对“该写哪些问题”帮助更明显，对“每个问题具体怎么治”帮助较有限，仍依赖 raw EHR 和 LLM 本身。

`generated_method2_gen` 的 judge 指标比较 mixed：ROUGE 提升，但 plan specificity 和 disposition context 略降。这提示 generated cumulative history 可能已经带有噪声，scaffold 有时会强化或保留这些噪声。

## 9. 三版方案对比

| 方案 | 设计 | 结果 | 结论 |
|---|---|---|---|
| Prefilter | 选择 top-k EHR snippets | 无稳定收益 | 没直接解决 A&P 失败，且 top-k 太宽。 |
| Problem-state replacement | 只用 problem update 生成 A&P | 公平比较下 ROUGE 下降 | 中间表示太压缩，丢 raw EHR 细节。 |
| Problem-state augmentation | raw EHR + history + scaffold | targeted set 有稳定正信号 | 当前最值得全量验证。 |

## 10. 主要 caveats

1. 当前是 targeted low-score set，不是随机或全量评估。
2. 生成和 judge 都使用 DeepSeek，judge 结果可能偏向结构化输出。
3. Gold progress note 包含大量非 A&P 模板内容，ROUGE 不完全等价于临床质量。
4. Problem-state scaffold 来自 LLM 抽取，可能引入错误；需要 verifier 或 confidence gating。
5. Generated-history setting 中，previous generated note 可能已经 drift，scaffold 可能继承错误。

## 11. 推荐下一步实验

### 11.1 全量验证

优先全量跑两组：

1. `method2/gen + augmented`
   - 最接近真实部署。
   - targeted set 中 ROUGE +0.30，judge 5/10 win。

2. `method2/gt + augmented`
   - oracle upper bound。
   - targeted set 中 ROUGE +0.31，judge 7/10 win。

### 11.2 Ablation

建议做以下 ablation：

| ablation | 目的 |
|---|---|
| raw EHR only | 原 direct baseline。 |
| raw EHR + problem memory only | 看 history problem list 是否足够。 |
| raw EHR + daily evidence only | 看当天 evidence structuring 是否有效。 |
| raw EHR + problem update | 当前 augmented。 |
| raw EHR + problem update + verifier revision | 看二次校验是否进一步减少 unsupported/missed problems。 |

### 11.3 更稳的 judge

后续应加入至少一种非同源评估：

- 用另一个 LLM 做 judge。
- 从 gold note 抽 gold problem list，再计算 generated problem coverage。
- 做人工 spot check，尤其关注 `177997` 和 `196033` 这类典型失败病例。

## 12. 总结

当前最可信的结论是：

> Problem-state 作为替代输入不够好；但作为 scaffold augmentation，有明确 no-training 收益信号。

这个方案的 motivation 比 prefilter 更扎实，因为它直接针对 A&P 低分失败：active problem 丢失、trajectory 缺失、history drift 和 unsupported plan。Targeted set 上 augmented 相比公平 direct baseline 有：

- 平均 ROUGE +0.36
- judge augmented wins 30/50
- trajectory capture +0.76
- evidence grounding +0.50
- missed key problem count -0.30

因此，它值得进入全量实验，但还不应该直接声称已经解决 A&P 生成问题。
