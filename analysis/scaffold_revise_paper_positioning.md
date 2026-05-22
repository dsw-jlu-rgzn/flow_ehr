# Scaffold + Verifier/Reviser 方案论文定位与后续计划

## 1. 当前方案的核心 motivation

ICU Assessment & Plan 生成不是普通的临床摘要任务。它的关键难点不是把当天 EHR 信息压缩成一段文字，而是持续维护一个随时间变化的 active problem list：

- 哪些历史问题应继续 carry forward；
- 哪些历史问题今天已经 resolved / downgraded；
- 哪些当日新证据应升级为 active problem；
- 每个 active problem 的 trajectory、plan、disposition 是否更新；
- 最终 A&P 不能引入 unsupported medication、device status、pressor、dialysis、ventilation、infection、transfusion、code status 等高风险幻觉。

直接让 LLM 根据 EHR 写 A&P 容易出现两个结构性错误：

1. problem list 不稳定：active problem 漏掉、旧问题错误延续、PMH 被误写成 active problem；
2. 生成后无校验：即使 scaffold 正确，LLM 仍可能补写 unsupported 数值、药物、计划和诊断。

因此，当前方案的 motivation 是：

> 用 scaffold 显式表示 problem state，再用 verifier/reviser 对生成结果进行 evidence-grounded 的后置校验和修订，从而提升 ICU A&P 的 active problem tracking、trajectory update 和 hallucination control。

这个 motivation 是成立的，而且比单纯 prompt-based A&P generation 更有论文价值。

## 2. 当前方法概述

### 2.1 V2 scaffold-only

V2 不加后置 revise 时，工作流是：

```text
historical A&P / memory
+ current-day EHR input
  -> memory-gated scaffold builder
  -> problem scaffold
  -> LLM A&P generation
```

V2 的核心是 pre-generation control。它在生成前构造 scaffold，包含：

- active A&P problems；
- watchlist；
- supportive care；
- carried-forward prior problems；
- rejected candidate problems；
- contradiction / low-confidence signals。

它解决的是“无结构直接生成”的问题，但生成后不再检查最终 A&P。

### 2.2 V2 judge-revise

V2 judge-revise 在 scaffold/generation 后加入 generation judge，检查：

- unsupported changes；
- missing updates；
- forgotten carried problems；
- scaffold revision suggestions。

工作流是：

```text
scaffold -> LLM generation -> generation judge -> revised scaffold / revised generation
```

相对 V2，它的价值在于不仅能删掉一部分 unsupported 内容，也能提示一些 missing update 和 carried-forward problem。

### 2.3 当前 claim-level verifier 上限实验

我们进一步做了 claim-level verifier upper-bound 实验：

```text
V2 output
  -> claim-level verifier truth
  -> LLM minimal / evidence-grounded reviser
  -> judge evaluation
```

实验分两轮：

- 初始 pseudo-oracle claim verifier：主要做 KEEP/FIX/DELETE/REWRITE；
- curated claim verifier：人工修正明显错标，并加入 missing_supported_claims_to_add 和 carried_forward_problems_to_restore。

这个实验的目标不是部署系统，而是验证“如果 verifier 接近真值，后置修订能带来多少上限提升”。

## 3. 当前实验结论

### 3.1 Qwen2.5 selected-30 旧实验

使用 `Qwen/Qwen2.5-72B-Instruct` 对 selected 30 重新评估时，claim-only pseudo-oracle verifier 并没有超过 V2 / V2 judge-revise：

| method | coverage | trajectory | specificity | grounding | unsupported | missed | wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 | 4.53 | 4.17 | 4.53 | 4.50 | 0.50 | 0.47 | 25 |
| V2 judge-revise | 4.60 | 4.23 | 4.60 | 4.47 | 0.37 | 0.33 | 27 |
| claim-only pseudo-oracle verifier + LLM revise | 4.07 | 3.87 | 4.07 | 4.07 | 0.87 | 0.87 | 17 |

解释：

- claim-only verifier 主要会删除 unsupported claim；
- 删除后 note 变短，coverage 和 specificity 下降；
- 它没有充分补回 missing active problems 和 carried-forward problems。

这说明 claim-level 删除器不是充分解法。

### 3.2 Curated verifier DeepSeek judge 实验

由于 Qwen/SiliconFlow 当前返回余额不足，curated verifier 的完整重跑先使用 DeepSeek judge。三组都在同一 selected 30 cases 上重新评估：

| method | coverage | trajectory | specificity | grounding | unsupported | missed | wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 | 3.07 | 2.43 | 3.00 | 2.43 | 2.43 | 2.47 | 14 |
| V2 judge-revise | 3.20 | 2.67 | 3.10 | 2.77 | 2.10 | 2.23 | 18 |
| curated claim verifier + LLM revise | 3.27 | 2.63 | 3.03 | 2.80 | 1.77 | 2.03 | 22 |

相对 V2：

- unsupported 从 2.43 降到 1.77；
- missed 从 2.47 降到 2.03；
- evidence grounding 从 2.43 升到 2.80；
- wins 从 14 增加到 22。

相对 V2 judge-revise：

- unsupported 从 2.10 降到 1.77；
- missed 从 2.23 降到 2.03；
- evidence grounding 从 2.77 升到 2.80；
- wins 从 18 增加到 22。

按 per-case quality sum：

- curated vs V2：wins 18，ties 7，losses 5；
- curated vs V2 judge-revise：wins 16，ties 4，losses 10。

结论：

> verifier/reviser 方向确实能降低 unsupported 和 missed，但当前 claim-level + additive truth 仍不是完整解法，尤其在 problem list 主线错误时表现不稳。

## 4. 当前方案已经解决的问题

### 4.1 从无结构生成变成 scaffolded generation

V2 已经把 A&P 生成从自由写作变成 problem-state guided generation。它能显式利用历史 A&P 和当日证据，减少无组织输出。

### 4.2 缓解一部分 claim-level hallucination

judge-revise 和 curated verifier 能发现并修正部分高风险错误：

- CRRT/CVVHD vs HD；
- pressor on/off；
- ventilation / SBT / extubation 状态；
- antibiotic / culture claims；
- transfusion / bleeding claims；
- code status / disposition。

### 4.3 能补回部分 missing updates

原版 judge-revise 和 curated verifier 都能加入 missing updates，例如：

- 当日 lab 更新；
- electrolyte / metabolic abnormalities；
- current-day respiratory trajectory；
- PEG / HD / antibiotic completion；
- disposition / hospice / floor transfer。

### 4.4 已经证明 claim-only verifier 不充分

这是一个重要 negative finding：

> 只做 claim-level KEEP/FIX/DELETE 会降低 unsupported，但可能牺牲 coverage；A&P 需要 problem-level + claim-level verification。

这个发现可以作为论文的 failure analysis 或方法升级动机。

## 5. 当前主要缺陷

### 5.1 Problem list hallucination 仍然是核心瓶颈

很多失败 case 不是某句话错，而是整个 active problem thread 错。

例子：

- `132357_day11`：curated 版本仍保留 pneumonia / fever workup / mini-BAL 主线，但 gold 重点是 groin hematoma、alcohol withdrawal、AVR anticoagulation、IVC thrombosis。
- `181596_day11`：curated 版本引入 sepsis、PRBC transfusion、multiple antibiotics，但漏 UTI、altered mental status、cough。
- `199046_day10`：仍有 unsupported PMH/problem list，如 cirrhosis、colon cancer、IgA nephropathy，同时漏 decubitus ulcer、constipation、septic arthritis。

这说明当前 verifier 还没有稳定解决 active problem selection。

### 5.2 Claim-level verifier 不能触发整段 section rebuild

当前 reviser 多数时候按句子删改。对于 wrong problem thread，它需要的是：

```text
remove entire wrong section
rebuild section around correct active problem
```

而不是：

```text
delete one unsupported sentence
keep old heading and surrounding plan
```

### 5.3 Add instructions 有噪声

部分 missing/add 项来自原版 generation judge，但原版 judge 本身不等于 gold truth。噪声 add 会被 reviser 写进最终 A&P，造成新 hallucination。

例子：

- `198275_day28` 中 antiplatelet therapy / blood-tinged sputum 被写入后被 judge 判 unsupported；
- `181596_day11` 中 HIT/DIC、broad antibiotics、PRBC transfusion 等 add/keep 逻辑导致新错误。

### 5.4 Reviser 会引入新 unsupported details

即使 verifier truth 正确，LLM reviser 仍可能为了让文本连贯而添加未授权细节。这说明需要 final self-check 或 stricter revision planner。

### 5.5 Evaluation 还不够稳定

当前存在两个问题：

- Qwen2.5 curated verifier 评估尚未补跑，因为 SiliconFlow 余额不足；
- DeepSeek judge 和 Qwen judge 的评分风格不同，需要双 judge 或 human validation 支撑。

## 6. 目前如果发论文可能被质疑的地方

### 6.1 “是不是只是 prompt engineering？”

Reviewer 可能认为 scaffold + judge-revise 只是多轮 prompting，没有足够方法创新。

需要强调：

- scaffold 是 problem-state representation；
- verifier 不只是评分，而是结构化诊断 unsupported / missing / forgotten problems；
- 后续 problem-level verifier 是任务特定的 clinical reasoning module。

### 6.2 “LLM judge 是否可信？”

目前主要指标来自 LLM-as-judge。Reviewer 会质疑 judge bias、judge leakage、模型偏好、可重复性。

需要补充：

- 至少两个不同 judge；
- small-scale clinician / medical expert validation；
- judge agreement；
- bootstrap confidence intervals 和 paired significance test。

### 6.3 “gold A&P 是否被用于上限实验，会不会泄漏？”

上限实验允许使用 gold A&P 生成 oracle verifier truth，但论文里必须区分：

- upper-bound oracle experiment；
- deployable evidence-only setting。

不能把 oracle verifier 结果描述成真实部署性能。

### 6.4 “selected 30 cases 是否有代表性？”

当前 30 cases 是从 Qwen653 结果中挑出的 failure-enriched cases，适合做 upper-bound 和 failure analysis，但不能代表全量性能。

需要补：

- AP100 full / 653 full 上的完整自动评估；
- failure-enriched subset 与 random subset 分开报告；
- case stratification：respiratory、renal、infection、hemodynamic、disposition。

### 6.5 “claim-level verifier 为什么不够？”

如果论文只写 claim-level verifier，reviewer 会指出 unsupported/missed 仍高。需要把 claim-only failure 明确写成发现，并升级到 problem+claim verifier。

### 6.6 “reviser 是否按 verifier 执行？”

当前 reviser 有执行偏差，会引入新 unsupported details。需要有 revision adherence metric：

- fix instruction applied rate；
- add instruction applied rate；
- unsupported new claim rate；
- empty section rate；
- must-not-add violation rate。

## 7. 让方法更 solid 需要做的实验

### 7.1 主实验矩阵

建议最终至少比较以下系统：

| System | Purpose |
|---|---|
| Direct generation baseline | 证明 scaffold 有必要 |
| V2 scaffold-only | scaffold 的贡献 |
| V2 judge-revise | 当前 revision 贡献 |
| Claim-only oracle verifier + reviser | 证明 claim-level 上限有限 |
| Problem+claim oracle verifier + planner+reviser | 证明 problem-level verification 带来关键提升 |
| Evidence-only problem+claim verifier | 接近部署设置 |

### 7.2 Problem List Verifier 实验

新增 agent：

```text
Problem List Verifier
  -> wrong_problem_threads_to_remove
  -> problem_threads_to_rewrite
  -> must_cover_problem_list
  -> must_not_add_problem_list
```

需要验证：

- problem hallucination 是否下降；
- missed active problems 是否下降；
- wrong PMH/diagnosis 是否下降；
- section rebuild 是否优于 sentence-level revise。

### 7.3 Revision Planner / Reviser ablation

比较：

- minimal sentence reviser；
- evidence-grounded reviser；
- planner + reviser；
- planner + reviser + final self-check。

需要报告：

- unsupported；
- missed；
- coverage；
- adherence；
- generated note length；
- empty heading / section cleanup。

### 7.4 Human validation

建议做 20-50 cases 的人工评价，至少 double annotation 一部分：

- active problem coverage；
- unsupported clinically important hallucinations；
- missed clinically important problems；
- trajectory correctness；
- plan actionability；
- disposition/goals correctness。

如果没有医生资源，可以先用 medical student / clinician-in-training，但论文中要透明说明。

### 7.5 Robustness / Generalization

需要在不同 subset 上报告：

- random cases；
- failure-enriched cases；
- long ICU stay cases；
- high-risk device cases；
- renal replacement cases；
- ventilation / extubation cases；
- infection / antibiotic cases。

### 7.6 Statistical testing

建议：

- paired bootstrap confidence intervals；
- Wilcoxon signed-rank / sign test；
- per-admission clustered analysis；
- judge agreement between Qwen and DeepSeek；
- human vs LLM judge correlation。

## 8. 后续 TODO

### TODO 1：实现 Problem List Verifier truth

为 selected 30 cases 先人工/LLM 生成：

```json
{
  "wrong_problem_threads_to_remove": [],
  "problem_threads_to_rewrite": [],
  "must_cover_problem_list": [],
  "must_not_add_problem_list": []
}
```

优先修复当前失败 case：

- `132357_day11`
- `181596_day11`
- `198275_day28`
- `174752_day51`
- `199046_day10`
- `126783_day10`

### TODO 2：更新 reviser

从 minimal sentence reviser 改为 problem-first reviser：

```text
1. remove wrong problem threads
2. rebuild required problem sections
3. apply claim-level fixes
4. add missing plan points
5. remove empty sections
6. avoid must-not-add items
```

### TODO 3：增加 final self-check

检查：

- must-cover problem 是否出现；
- must-not-add 是否违反；
- 是否新增未授权 medication / number / procedure；
- 是否仍有 unsupported PMH/problem list；
- 是否有空 section；
- 是否泄露 gold/oracle/verifier 字样。

### TODO 4：补跑 Qwen2.5 curated verifier 评估

当前 Qwen2.5 curated 版本未完成，因为 SiliconFlow 余额不足。余额恢复后，需要用同一批 30 cases 重新跑：

- V2；
- V2 judge-revise；
- curated claim verifier + LLM revise；
- problem+claim verifier + planner/reviser。

### TODO 5：扩大到 AP100 / 653

selected 30 适合 failure analysis，但论文主表需要更大样本。建议：

- AP100 random；
- Qwen653 full；
- failure-enriched subset 单独报告。

### TODO 6：人工评估小样本

至少做 20-50 cases，人工标注 clinically important unsupported / missed problems。用于支撑 LLM judge 的有效性。

## 9. 当前论文定位建议

### 如果现在投稿

当前更适合：

- workshop；
- short paper；
- methods / system paper；
- clinical NLP demo + failure analysis。

可声称：

> Scaffolded problem-state tracking plus verifier-guided revision improves structure and reduces some unsupported/missed problems, but claim-level verification alone is insufficient for robust ICU A&P generation.

### 如果目标是完整 conference paper

建议补齐 problem+claim verifier 和更扎实的 evaluation，再投稿。

更强的主张应是：

> ICU A&P generation requires problem-level state verification in addition to claim-level factual verification. A problem-first verifier/reviser substantially improves active problem coverage, trajectory correctness, and hallucination control over scaffold-only and claim-only revision.

## 10. 当前最重要的结论

1. Motivation 足够：ICU A&P 生成确实需要 problem-state scaffold 和 verifier/reviser。
2. V2 scaffold 已经解决了无结构生成的一部分问题。
3. V2 judge-revise 能进一步改善 missing updates 和 unsupported changes。
4. Claim-only verifier 不足，容易删短 note，不能修主线。
5. 当前最核心瓶颈是 problem list hallucination 和 wrong problem thread。
6. 下一步必须引入 Problem List Verifier 和 problem-first reviser。
7. 论文要 solid，需要 full evaluation、problem-level ablation、human validation 和 statistical testing。

