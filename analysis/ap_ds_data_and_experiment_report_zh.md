# AP/DS 数据输入、真值支持性与实验结果分析

生成日期：2026-05-14

本报告基于当前仓库中的 MIMIC-III 处理结果与 DeepSeek API 推理结果：

- AP 输入：`data/AP/input`
- AP 真值：`data/AP/gold`
- AP 生成：`data/AP/generated/DG/deepseek_api_full/gt`
- DS 输入：`data/DS/input`
- DS 真值：`data/DS/gold`
- DS 生成：`data/DS/generated/DG/deepseek_api_full`

## 1. 当前数据规模

### AP 任务

AP 是按一次住院 admission 内的多天 progress note 构造的每日预测任务。

- admission 数：10
- AP gold 文件数：10
- AP gold day rows：67
- AP 生成文件数：每个 method 10 个 admission 文件
- AP 生成 day rows：每个 method 57 条

AP gold 有 67 条，而生成只有 57 条，是因为每个 admission 的第一个 progress note 被用作历史上下文，不作为预测目标。因此 10 个 admission 会少 10 条首日 note，最终生成 57 条。

当前 AP 输入统计：

| 指标 | 平均值 | 最小值 | 最大值 |
|---|---:|---:|---:|
| 每个 admission 输入行数 | 221.2 | 61 | 579 |
| 每个 admission 天数 | 10.7 | 4 | 18 |
| 每个 admission note 行数 | 36.2 | 13 | 96 |
| 每个 admission 非 note 输入行数 | 185.0 | 42 | 483 |
| 每个 admission gold day rows | 6.7 | 3 | 15 |
| 每个 admission 非 note 输入词数 | 8900.3 | 3538 | 20435 |
| 每个 admission gold 词数 | 4907.8 | 1551 | 11174 |

### DS 任务

DS 是按 admission 生成 discharge summary 的任务。

- admission 数：100
- DS gold 文件数：100
- DS 生成文件数：100

当前 DS 输入统计：

| 指标 | 平均值 | 中位数 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|
| 每个 admission 输入行数 | 4.88 | 2 | 1 | 41 |
| 每个 admission 输入词数 | 294.85 | 69 | 4 | 3241 |
| 每个 admission gold 词数 | 1804.49 | 1595 | 340 | 4182 |
| 输入词数 / gold 词数 | 0.232 | - | 0.002 | 2.752 |

DS 的输入和真值长度差异非常大。多数样本只有少量 lab/med/event 输入，却要生成完整出院小结。100 条 DS 真值中：

- 89/100 能检测到 discharge diagnosis 相关标题
- 95/100 能检测到 hospital course 相关标题
- 90/100 能检测到 discharge instructions / discharge medications / diet 等出院指导相关内容

这说明真值本身多数有可评估结构，但输入并不总是足以支撑完整真值。

## 2. AP 输入是否支持输出真值

### AP 示例：HADM_ID 109079

AP 示例文件：

- 输入：`data/AP/input/input_109079.csv`
- 真值：`data/AP/gold/gt_109079.csv`
- 生成：`data/AP/generated/DG/deepseek_api_full/gt/method2/genpns_109079.csv`

该 admission 的 day 对齐情况：

```text
gold days: [1, 2, 3, 4, 5, 6]
generated days: [2, 3, 4, 5, 6]
```

Day 1 的 progress note 被当作上下文，所以从 day 2 开始生成。

Day 2 输入里包含较多和真值相关的证据，例如：

- 呼吸机状态：continuous invasive ventilation
- COPD / wheezing / secretion / intubation 信息
- Vancomycin、Levofloxacin、Furosemide、Midazolam、Fentanyl 等用药
- ABG 信息：pCO2、pO2、total CO2
- tube feeding / insulin / free water 等 ICU 治疗事件

Day 2 gold 的核心内容包括：

- hypoxemia / respiratory failure
- severe COPD
- ED intubation，pCO2 very high
- overnight PSV intolerance because of oversedation
- antibiotics、sedation、ventilator、ICU medications
- Assessment and Plan section

DeepSeek method2 生成的 day 2 内容覆盖了不少核心问题：

- hypoxic and hypercarbic respiratory failure
- COPD exacerbation
- intubated and mechanically ventilated
- hypercarbia / ventilator adjustment
- cellulitis / infection treatment
- bronchodilators、steroids、daily SBT/RSBI 等计划

### AP 支持性判断

AP 输入总体是支持真值生成的。原因是：

1. AP 是局部每日任务，输入和输出都围绕同一天或相邻几天的 ICU 状态。
2. 非 note 输入数量较充足，包含 lab、medication、respiratory note、flowsheet 等证据。
3. method1/method2 加入历史 progress note 后，可以补足诊断、病程背景和问题列表连续性。

但 AP 也有几个明显问题：

1. 当前 gold 是完整 progress note，而模型 prompt 要求生成 Assessment and Plan。这会导致输出目标和评估真值不完全一致。
2. 评估脚本按 admission 拼接文本比较，没有严格按 day 对齐。gold 中包含首日上下文 note，而 generated 不包含首日 note。
3. ROUGE 对完整 note 的模板、检查结果、客观数据非常敏感；如果模型只生成 A&P，会在表面文本重合上吃亏。
4. SapBERT 更能反映医学概念覆盖，所以 AP 的 SapBERT 稳定高于 ROUGE。

因此 AP 的主要问题不是输入不足，而是“目标定义和评估对齐”需要进一步规范：建议只抽取 gold 中的 Assessment and Plan section，并按 day 一一对齐评估。

## 3. DS 输入是否支持输出真值

### DS 示例：HADM_ID 104732

DS 示例文件：

- 输入：`data/DS/input/24_both_104732.csv`
- 真值：`data/DS/gold/gtsummary_104732.txt`
- 生成：`data/DS/generated/DG/deepseek_api_full/48h_all_abs_104732.txt`

该样本输入非常短：

```text
input rows: 3
input words: 110
gold words: 635
```

输入主要包含：

- Transferrin 75 mg/dL
- Calcium 7.20 mg/dL
- Creatinine 3.10 -> 2.90 mg/dL
- BUN 64 -> 63 mg/dL
- Hemoglobin 8.60 -> 8.90 g/dL
- Platelet 119 -> 132 K/uL
- Magnesium Oxide administered

但 gold discharge summary 的核心内容是：

- ruptured aortic aneurysm
- emergent open abdominal aortic repair
- delayed closure / G-tube placement
- prolonged ventilation / extubation
- aspiration / swallow evaluation
- pneumonia / sepsis / antibiotics
- pancytopenia due to polypharmacy
- discharge medications and diet

这些关键信息大多不在 DS 输入中。DeepSeek 生成结果因此主要围绕可见 lab 异常展开：

- AKI
- anemia
- hypocalcemia
- thrombocytopenia
- leukopenia
- renal monitoring / electrolyte management

这和输入证据是一致的，但和完整 discharge summary 真值并不一致。

### DS 支持性判断

DS 当前输入对完整真值的支持明显不足。主要难点是：

1. 输入窗口太短，平均只有 294.85 个词，而 gold 平均 1804.49 个词。
2. 很多样本输入只有 1-2 行，最低输入词数只有 4-9 个词。
3. discharge summary 是全住院级别文档，包含入院原因、手术、长期 hospital course、并发症、出院诊断和出院计划。
4. 当前输入更多像 discharge 前后局部事件窗口，无法恢复全住院时间线。
5. Diagnosis 最难，因为最终出院诊断往往来自完整住院病程和医生归纳，而不是最后 24/48 小时的 lab/medication。

因此 DS 的低分不应简单解释为模型能力差，更主要是任务输入和真值之间存在结构性信息缺口。

## 4. 当前 DeepSeek 实验结果

### AP 官方评估结果

当前官方脚本按 admission 拼接文本评估，结果如下：

| Method | ROUGE-L F1 | SapBERT F1 | CUI-F1 |
|---|---:|---:|---:|
| method-1 | 18.65 ± 3.52 | 73.93 ± 3.53 | 0.00 ± 0.00 |
| method1 | 22.61 ± 2.73 | 74.68 ± 3.49 | 0.00 ± 0.00 |
| method2 | 23.13 ± 2.38 | 74.95 ± 3.25 | 0.00 ± 0.00 |

趋势上，`method2 > method1 > method-1`。说明历史 progress note 对 AP 有帮助，累积历史上下文最好。

### AP 按 day 对齐后的补充评估

由于官方 AP 评估会把首日 gold note 拼进去，而生成结果从第二个 note day 开始，本次额外计算了按 generated day 对齐后的指标：

| Method | Aligned rows | ROUGE-L F1 | SapBERT F1 |
|---|---:|---:|---:|
| method-1 | 57 | 11.75 ± 2.24 | 73.66 ± 3.30 |
| method1 | 57 | 14.46 ± 2.59 | 74.00 ± 3.32 |
| method2 | 57 | 14.82 ± 2.51 | 73.91 ± 3.24 |

对齐后，ROUGE-L 绝对值下降，但上下文带来提升的趋势仍然存在。SapBERT 差异很小，说明三种方法对医学概念覆盖相近，历史上下文更多改善了文本组织和局部内容匹配。

### DS 官方评估结果

| Section | ROUGE-L F1 | SapBERT F1 | CUI-F1 |
|---|---:|---:|---:|
| Diagnosis | 5.07 ± 6.30 | 49.50 ± 11.74 | 0.00 ± 0.00 |
| Hospital Course | 12.75 ± 3.83 | 64.65 ± 7.48 | 0.00 ± 0.00 |
| Discharge Instructions | 11.65 ± 4.71 | 62.49 ± 12.88 | 0.00 ± 0.00 |

DS 中 Hospital Course 表现最好，Diagnosis 最差。这符合数据支持性分析：Hospital Course 更容易从近期事件中总结，而最终 Diagnosis 往往依赖完整住院信息。

CUI-F1 为 0 是因为当前 Windows 环境无法安装 QuickUMLS 的 leveldb 依赖，脚本自动跳过 CUI 计算。ROUGE-L 和 SapBERT 是当前可用的主要指标。

## 5. AP 与 DS 的核心差异

AP 和 DS 的任务难度不同：

AP 是局部每日生成任务。输入包含当天/历史病程和 ICU 事件，输出也是进展记录中的每日问题和计划，因此输入和真值较接近。

DS 是全局出院总结任务。输入常常只是很短的局部窗口，但真值是完整住院总结，因此存在天然的信息缺失。

这解释了当前结果：

- AP SapBERT 约 74-75，说明医学概念覆盖较稳定。
- DS Hospital Course SapBERT 约 64.65，仍可从局部事件中总结部分病程。
- DS Diagnosis SapBERT 只有 49.50，说明最终诊断很难从当前输入中恢复。

## 6. 数据处理和后续实验建议

### AP 建议

1. 将 gold progress note 中的 Assessment and Plan section 单独抽取出来，作为 AP 真值。
2. 评估时严格按 HADM_ID + DAY 对齐，不要直接拼接整个 admission 文件。
3. 保留首日 progress note 作为上下文，但不要把首日 gold 纳入预测目标。
4. 分别报告：
   - 无历史 note：method-1
   - 上一日 note：method1
   - 累积历史 note：method2
5. flow matching / 病程压缩小模型优先在 AP 上验证，因为 AP 对每日病程证据和历史上下文更敏感。

### DS 建议

1. 扩大输入窗口，不要只用 discharge 附近很短的局部事件。
2. 为 DS 构造全住院时间线，包括手术、诊断、抗生素、ICU 转入转出、重大检查、consult、procedure、microbiology 等。
3. 将 discharge summary 拆成子任务：
   - Diagnosis generation
   - Hospital Course summarization
   - Discharge Instructions generation
4. 对 Diagnosis 单独加入检索或分类辅助，例如从 ICD、problem list、radiology/procedure、最后 note 中检索诊断证据。
5. 对输入极短样本进行过滤或分层报告，例如输入少于 50 词的 DS 样本单独统计。

## 7. 实验结论

当前实验支持以下结论：

1. AP 输入基本能够支持真值生成，但 gold 和 prompt 的目标范围需要进一步对齐。
2. AP 中历史 progress note 明显有帮助，累积历史上下文 method2 整体最好。
3. AP 的 SapBERT 高而 ROUGE-L 低，说明模型抓到了医学语义，但和原始 note 的表面文本、模板和完整结构仍有差距。
4. DS 当前输入明显不足以支持完整 discharge summary，尤其不足以恢复最终出院诊断。
5. DS 的低分更多来自数据构造的信息缺口，而不是单纯生成模型能力不足。
6. 后续如果要验证 flow matching 或病程压缩/检索排序小模型，AP 是更合适的最小闭环任务；DS 需要先改进输入构造，否则优化模型前端也很难突破真值不可观测的问题。

一句话总结：AP 当前已经能看到合理的上下文增益，适合作为后续 flow matching 小模型的首个验证场景；DS 需要优先重构输入证据链，让模型能看到足够的全住院信息，再谈生成模型或前端压缩模型的优化。
