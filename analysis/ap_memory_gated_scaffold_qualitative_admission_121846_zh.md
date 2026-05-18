# Admission 121846 定性分析：base vs no-judge V2 vs judge&revise V2

## 结论摘要

本次选择 `admission_id=121846` 做定性分析，因为它在 AP100 120-case 评估中具有典型的指标分歧：V2 在 LLM judge 上明显优于 base，但 ROUGE-L 略低。该 admission 覆盖 day13-day36 共 24 个评估日，no-judge V2 的 judge 结果为 16 胜 / 1 负 / 7 平，judge&revise V2 为 18 胜 / 0 负 / 6 平；但两者 admission-level 平均 ROUGE delta 均为负。

最典型的 case 是 `121846_day24`：base 的 ROUGE 更高，但它把当天状态写成“extubated/on nasal cannula”，而 gold 与当天输入均明确显示患者因低氧恶化被重新插管。V2 抓住了 re-intubation、pleural effusion、diuresis、CT PA negative for PE 等关键临床状态，因此 judge 判 V2 赢是有临床依据的。ROUGE 下降主要不是因为 V2 质量更差，而是因为 V2 使用了更结构化、更标准化、更“问题状态跟踪式”的表达，和 gold 的模板化临床原文存在词面/顺序差异；同时 V2 也纳入了一些当天 input 中出现、但 gold A&P 不一定重点写入的事件，导致 lexical overlap 被稀释。

## 复现路径

- Gold note: `data_ap100_ap/AP/gold/gt_121846.csv`
- Base output: `data_ap100_ap/AP/generated/DG/deepseek_api_full_gen/gen/method2/genpns_121846.csv`
- Raw input: `data_ap100_ap/AP/input/input_121846.csv`
- No-judge V2 output: `outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v2/121846_day24.txt`
- Judge&revise V2 output: `outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v2_judge_revise/121846_day24.txt`
- Judge&revise candidate output: `outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v2_judge_revise/121846_day24.candidate.txt`
- No-judge scaffold: `outputs/ap_memory_gated_scaffold_ap100/scaffolds/ap100eval_generated_method2_gen_v2/121846_day24.json`
- Judge&revise initial scaffold: `outputs/ap_memory_gated_scaffold_ap100/scaffolds/ap100eval_generated_method2_gen_v2_judge_revise/121846_day24.json`
- Judge&revise revised scaffold: `outputs/ap_memory_gated_scaffold_ap100/scaffolds/ap100eval_generated_method2_gen_v2_judge_revise/121846_day24.revised.json`
- Generation-time judge: `outputs/ap_memory_gated_scaffold_ap100/generation_judges/ap100eval_generated_method2_gen_v2_judge_revise/121846_day24.json`
- Evaluation judge detail:
  - `outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v2_eval_judge_detail.csv`
  - `outputs/ap_memory_gated_scaffold_ap100/ap100eval_generated_method2_gen_v2_judge_revise_eval_judge_detail.csv`

## Admission-level 结果

| 方法 | n | judge 胜/负/平 | 平均 ROUGE delta | 主要 judge 改善 |
|---|---:|---:|---:|---|
| no-judge V2 | 24 | 16 / 1 / 7 | -0.000484 | coverage +0.708, trajectory +1.083, grounding +0.917, unsupported -0.833, missed -0.750 |
| judge&revise V2 | 24 | 18 / 0 / 6 | -0.001952 | coverage +0.750, trajectory +1.167, grounding +0.958, unsupported -0.917, missed -0.833 |

这说明该 admission 不是单个偶然 case：V2 的优势主要来自跨天状态跟踪，而不是某一天 prompt 撞中 gold wording。

## 代表性 day：121846_day24

### 指标对比

| 方法 | ROUGE-L | delta vs base | words | judge winner | coverage | trajectory | grounding | unsupported | missed |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| base | 0.086667 | - | 637 | - | 2 | 1 | 2 | 3 | 3 |
| no-judge V2 | 0.082818 | -0.003848 | 755 | augmented | 4 | 4 | 4 | 1 | 1 |
| judge&revise V2 | 0.078772 | -0.007895 | 635 | augmented | 4 | 4 | 4 | 1 | 1 |

### Gold / raw evidence 的核心事实

Gold A&P 当天明确写到：

- 患者 “Now intubated”。
- Hypoxic respiratory failure 的原因主要是 pleural effusions 与 atelectatic lung shunt。
- 当天 oxygen requirement 增加，A-a gradient 480，最终 re-intubated。
- CT PA negative for PE，但 pleural effusions increased。
- 计划包括继续 diuresis，目标 net negative >2-4 L，讨论 therapeutic drainage，follow CT final read。
- Afib with RVR 改善，rate control 依赖 sedation/ventilator，继续 metoprolol/diltiazem。
- Acute renal failure 回到 baseline，但需监测 contrast nephropathy 或 Lasix-induced pre-renal azotemia。
- ICU care 中 DVT prophylaxis: heparin SC TID now that platelets improved。

当天 raw input 还包含额外中间证据：

- Critical care note: reintubated for progressive SOB/tachypnea。
- Worsening effusions and edema despite aggressive diuresis。
- Later protected section: fever 102, hypotension, IVF and norepinephrine initiation。
- Oxygenation good but urine output decreased，concern for volume response vs dye-induced ATN。
- Plan BAL in AM to rule out PJP，antibiotics restarted broadly。
- Respiratory/swallow notes显示 patient post-intubation 应 NPO。

因此，V2 纳入“shock / norepi / BAL / PJP workup”等信息是被 raw input 支持的；但这些内容在 gold A&P 中不是同等中心位置，后面会影响 ROUGE 和 judge 置信度。

## 三个输出的关键差异

### Base

Base 的最大问题是状态错位：它沿用了上一日或旧 trajectory，把患者写成 “Now extubated day +3 post-extubation and breathing spontaneously on nasal cannula”。这与 gold 当天 “Now intubated / re-intubated” 直接矛盾。

Base 也把主要 narrative 放在 persistent fevers、hypotension、bronchoscopy/BAL/PJP、AKI worsening 上，但对 gold A&P 的关键呼吸恶化链条覆盖不足：progressive hypoxia -> A-a gradient 480 -> re-intubation -> pleural effusion/atelectasis -> aggressive diuresis/therapeutic drainage。它有较多医学上 plausible 的内容，但 day-level problem state 错了，所以 judge 的 trajectory 分很低。

### No-judge V2

No-judge V2 把主问题改成 “Respiratory Failure / Hypoxia (Reintubation)”，并明确写了 mechanical ventilation、bilateral pleural effusions、post-thoracentesis/no pneumothorax、diuresis、BAL plan、septic shock、AKI monitoring、NPO 等。它明显修正了 base 的最大错误：不再认为患者已经 extubated。

它的问题是内容偏“全量证据收集”：把 raw input 中出现的 shock、thoracentesis、PJP/BAL、right-heart signs 等都纳入，使输出更像 evidence-integrated ICU problem list，而不是完全模仿 gold A&P 的简洁问题列表。这提高了 judge 的 coverage/grounding，但降低了和 gold wording 的 overlap。

### Judge&revise V2

Judge&revise V2 在 candidate 基础上做了更保守的修订：

- 保留 re-intubation、large pleural effusions、CTA negative for PE、right heart failure signs、diuresis、AKI worsening。
- 对 PJP 处理从 “consider adding PJP coverage” 改成 “hold empiric PJP coverage until BAL results; discuss with ID if high suspicion”，避免过度行动。
- 把 daily SBT 改成 “defer until off vasopressors and effusions improved”，更符合 hemodynamic instability。
- 增加 right middle lobe nodule、heparin DVT prophylaxis、Mg/Ca repletion 等 missing updates。

这解释了为什么 judge&revise 的 judge 胜率高于 no-judge：它减少 unsupported/action-too-early 的计划项。但 ROUGE 进一步下降，因为 revised 版本更偏“临床审校后的标准化表达”，和 gold 原始 note 的短句、模板、局部措辞更不一样。

## Judge 指标置信度判断

我认为这例中 judge 的主判断“V2 优于 base”置信度较高，原因是：

- Base 与 gold 在最关键状态上直接冲突：extubated vs re-intubated。
- V2 的 re-intubation、pleural effusion、diuresis、CT PA negative for PE 等均可从 gold 和 raw input 双重验证。
- 同一 admission 内不是孤例，24 天里 V2 多数天胜出，且 trajectory/coverage/grounding 系统性提高。
- Generation-time judge 的 revision 建议能对应到 raw evidence，例如 norepi、BAL、pleural fluid、heparin、right middle lobe nodule，而不是空泛偏好。

但 judge 指标不是完全无偏，置信度应标为“中高”，不是“高到可替代人工评估”：

- Evaluation judge 对 “septic shock” 的 rationale 有轻微不一致：它批评 base invents septic shock，但又认为 augmented captures septic shock。实际上 shock/norepi 在 raw input 中有证据，但 gold A&P 中没有作为核心问题展开。这说明 judge 在“gold-only fidelity”和“raw-input clinical usefulness”之间有时会混用标准。
- Judge 更偏好结构化、覆盖完整、显式证据链的输出，可能天然偏向 V2 这种 scaffold 风格。
- 评估 judge 和 generation judge 都是 LLM，存在同类偏好相关性，后续需要 human spot-check 或规则化 factuality audit 来校准。

## 为什么 ROUGE 变差

ROUGE 下降主要来自四类原因：

1. 状态正确但词面不同。Gold 写 “Hypoxic Respiratory failure / atelectatic lung / A-a gradient / cont diuresis”；V2 写 “progressive hypoxic respiratory failure / lung-protective ventilation / pleural fluid / BAL / vasopressor”。临床语义更完整，但 token overlap 不一定更高。

2. V2 纳入 raw evidence 中但 gold A&P 不强调的事件。比如 norepi、BAL/PJP、thoracentesis、right middle lobe nodule、heparin prophylaxis。这些内容在 input 中有证据，临床上有用，但会稀释与 gold A&P 的 ROUGE。

3. V2 的组织方式更标准化。它按 “Respiratory / Septic Shock / AKI / Afib / Delirium / Heme / FEN / ICU Care” 展开；gold 更像真实 ICU note，保留模板、简写和局部计划句。结构越规范，越可能丢失模板化 lexical overlap。

4. Judge&revise 会删除或降级一些不够稳的行动项。比如把 SBT/PJP coverage 改得更保守，这提高 factual safety，但也可能丢掉与 gold 或 base 偶然重合的词。

## 对后续实验的启示

- 当前 V2 的核心收益是真实的：它能修复 base 的 longitudinal state drift，尤其是 A&P 中最危险的“上一日状态残留”问题。
- 仅看 ROUGE 会低估这种收益；建议主表继续保留 ROUGE，但把 trajectory capture、unsupported/missed count、state contradiction audit 作为主要临床指标。
- 下一版可以增加一个 no-training factual audit：自动检测 extubated vs intubated、pressor on/off、AKI worsening/baseline、antibiotics on/off、diet/NPO 等离散状态，与 gold/raw evidence 对齐。
- 对 judge 评估需要区分两种目标：`gold-note mimicry` 和 `raw-evidence clinical usefulness`。如果目标是生成更可用的 A&P，允许 raw-supported-but-gold-underwritten 的内容；如果目标是 mimic gold note，需要惩罚这些额外内容。
- 为解释 ROUGE 下降，建议额外报告 problem-level recall/precision 和 contradiction count，而不是只报告 lexical overlap。
