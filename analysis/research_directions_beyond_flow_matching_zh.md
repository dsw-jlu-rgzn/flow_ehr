# Flow Matching 之外的可行研究方案

生成日期：2026-05-14

本文档整理当前 `flow_ehr` repo 在 AP/DS 临床文档生成任务上的可行改进方向。目标不是简单提高一个 baseline 分数，而是形成能够支撑中科院一区或 A 会投稿的研究主线。

## 1. 当前问题背景

当前实验已经说明：

- AP 任务比 DS 更容易形成有效闭环。
- AP 中历史 progress note 有帮助，`method2 > method1 > method-1`。
- AP 当前主要问题是 gold 和 prompt 目标不完全一致：gold 是完整 progress note，而 prompt 生成 A&P。
- DS 当前主要问题是输入证据不足：输入平均约 295 词，而 gold discharge summary 平均约 1804 词。
- DS 的 Diagnosis 最难，因为最终诊断依赖全住院病程，而当前输入多是局部窗口事件。

因此，后续研究不应只是“换一个更强 LLM”，而应关注：

1. 如何构造 evidence-adequate 的任务输入。
2. 如何检索、压缩和组织纵向 EHR 证据。
3. 如何减少临床文档生成中的 hallucination。
4. 如何评估生成内容是否被输入证据支持。

## 2. 方案一：Evidence Retrieval + LLM

### 核心思想

不把所有 EHR 事件直接塞给 LLM，而是先检索最相关的临床证据，再让 LLM 基于这些 evidence 生成 AP 或 DS。

```text
EHR events / notes
-> evidence retriever
-> top-k evidence
-> LLM generation
-> clinical note
```

### 可选 retriever

- BM25
- SapBERT embedding
- ClinicalBERT embedding
- time-aware retrieval
- problem-aware retrieval

### AP 中可检索的证据

- 当天 abnormal labs
- 当天 medication change
- respiratory / nursing / physician note
- 上一日 A&P 中的问题列表
- ICU intervention，例如 intubation、pressor、antibiotic、dialysis

### 适合回答的问题

- 哪些 EHR event 最支持当天 A&P？
- 检索证据是否比输入全部事件更好？
- 检索模块是否能减少 hallucination？

### 优点

- 实现简单。
- 可解释性强。
- 容易和 DeepSeek/Llama 等 LLM 结合。
- 很适合写成 evidence-grounded clinical note generation。

### 风险

- 如果只做 BM25/embedding，方法创新可能偏弱。
- 需要增加 evidence-level 或 claim-level 评估，否则容易变成普通 RAG baseline。

## 3. 方案二：Temporal Trend Feature + LLM

### 核心思想

把单点 lab/event 转换成趋势特征，让 LLM 看到病程变化。

例如：

```text
Glucose: 180 -> 220 -> 260, increasing
Creatinine: 3.10 -> 2.90, improving
WBC: 18 -> 12, decreasing
FiO2: 60% -> 40%, improving oxygenation
Norepinephrine: stopped
Vancomycin: started
```

### Pipeline

```text
raw EHR events
-> lab/medication/ventilator trend extraction
-> structured trend summary
-> LLM generation
```

### 可构造的趋势

- lab latest value
- lab delta
- lab slope
- abnormality direction
- medication started / stopped / dose changed
- ventilator support improved / worsened
- vasopressor started / stopped
- renal function improved / worsened
- glucose control improved / worsened

### 优点

- 不需要训练。
- 临床可解释性强。
- 能直接解决“血糖升高/降低”等趋势判断问题。
- 对 AP 任务尤其有用。

### 风险

- 规则需要覆盖主要变量。
- 对 notes 中隐含的诊断和计划帮助有限。

## 4. 方案三：Problem-Oriented Generation

### 核心思想

医生写 A&P 通常是 problem-oriented 的。可以先识别 active problems，再围绕每个 problem 检索证据并生成 plan。

```text
EHR events
-> active problem detection
-> problem-specific evidence retrieval
-> problem-wise assessment and plan generation
-> final A&P assembly
```

### Active problems 示例

- acute hypoxic respiratory failure
- COPD exacerbation
- sepsis / infection
- AKI
- hyperglycemia
- anemia
- thrombocytopenia
- nutrition
- DVT prophylaxis

### Active problem detector 可以先不用训练

可由以下信息组合：

- previous note headings
- ICD / diagnosis keywords
- abnormal labs
- medication triggers
- procedure / intervention
- embedding similarity

### 优点

- 非常符合医生写 A&P 的方式。
- 可以逐 problem 评估 recall 和 hallucination。
- 方便加入 evidence attribution。
- 论文故事清楚。

### 风险

- 需要构造 problem taxonomy。
- 如果 active problem 识别错误，后续生成会被带偏。

## 5. 方案四：Hierarchical Summarization

### 核心思想

适合长住院、长时间线和 DS。不要一步从全量 EHR 生成 discharge summary，而是分层总结。

```text
hourly events
-> daily summaries
-> ICU phase summaries
-> hospital course summary
-> discharge summary sections
```

### 对 DS 的意义

DS 真值是全住院级别文档，当前局部输入很难支持。Hierarchical summarization 可以把全住院事件压缩成可控长度的时间线摘要。

### 优点

- 适合长上下文。
- 保留时间顺序。
- 可解释。
- 不一定需要训练。

### 风险

- 每层摘要都会引入误差。
- 需要评估摘要链是否遗漏关键事件。

## 6. 方案五：Contrastive Learning / Dual Encoder

### 核心思想

训练一个小模型，让 EHR day embedding 接近对应 A&P embedding，远离其他 day/admission 的 A&P embedding。

```text
positive pair:
EHR events of day t <-> A&P note of day t

negative pairs:
EHR events of day t <-> other days/admissions
```

训练目标：

```text
InfoNCE / contrastive loss
```

### 推理方式

```text
EHR day
-> dual encoder embedding
-> retrieve similar A&P / evidence / examples
-> LLM generation
```

### 优点

- 比 flow matching 更容易训练。
- 数据需求较低。
- 和 retrieval 天然结合。
- 可以作为“小模型 + LLM”的核心方法。

### 风险

- 需要足够多 AP day-level 样本。
- 如果数据量太少，可能需要先扩大 MIMIC-III 样本。

## 7. 方案六：Supervised Embedding Regression

### 核心思想

这是 flow matching 的简化版。用小模型直接预测目标 note embedding。

```text
input: daily EHR event embedding
target: A&P note embedding
loss: MSE / cosine loss
```

### 推理方式

```text
EHR events
-> predicted A&P embedding
-> retrieve similar examples/evidence
-> LLM generation
```

### 优点

- 实现最简单。
- 可作为 flow matching 前的 sanity check。
- 可以验证“小模型预测病程状态 embedding 是否有用”。

### 风险

- 表达能力有限。
- 可能只学到粗粒度语义，不一定能捕捉复杂病程变化。

## 8. 方案七：Plan-Then-Write

### 核心思想

让 LLM 不直接生成最终 note，而是先生成结构化 clinical plan，再写成自然语言。

Step 1：结构化计划

```json
{
  "problems": [
    {
      "name": "acute hypercapnic respiratory failure",
      "evidence": ["pCO2 74", "intubated", "wheezing"],
      "assessment": "acute on chronic respiratory failure due to COPD exacerbation",
      "plan": ["continue mechanical ventilation", "daily SBT", "bronchodilators"]
    }
  ]
}
```

Step 2：生成 A&P 文本

```text
structured clinical plan
-> final Assessment and Plan
```

### 优点

- 生成更稳定。
- 中间结果可检查。
- 有利于 hallucination 检测。
- 适合做 evidence grounding。

### 风险

- 需要设计结构化 schema。
- LLM 可能输出不合法 JSON，需要容错解析。

## 9. 方案八：Self-Verification / Critic Model

### 核心思想

生成后再让模型检查每个 claim 是否有证据支持。

```text
LLM draft
-> verifier checks unsupported claims / missing evidence / trend errors
-> LLM revision
```

### 检查内容

- 每个 diagnosis 是否有输入证据？
- 每个 medication plan 是否和输入一致？
- 是否遗漏 critical abnormal lab？
- 是否错误描述趋势？
- 是否把不存在的 surgery/procedure 写进 summary？

### 优点

- 能降低 hallucination。
- 很适合临床文档生成。
- 可以和 retrieval、problem-oriented generation 组合。

### 风险

- verifier 自身可能不可靠。
- 需要人工或规则构造一部分 evaluation set。

## 10. 方案九：Section-Specific Models

### 核心思想

AP 和 DS 都可以按 section 拆分，不同 section 使用不同证据来源和生成策略。

### AP section

- Assessment
- Plan
- Problem headings
- Medication plan
- Respiratory plan
- Renal plan

### DS section

- Diagnosis
- Hospital Course
- Discharge Instructions

### DS 中不同 section 的证据来源

```text
Diagnosis:
ICD / problem list / final notes / discharge diagnosis candidates

Hospital Course:
full admission timeline / procedures / major events / progress notes

Discharge Instructions:
discharge medications / diet / follow-up / final assessment
```

### 优点

- 能明显改善 DS。
- 更符合真实 discharge summary 写作。
- 可以逐 section 分析错误。

### 风险

- 需要更复杂的数据处理。
- 每个 section 的 gold 抽取规则要更仔细。

## 11. 方案十：Data-Centric Improvement

### 核心思想

当前很多性能瓶颈来自数据构造，而不是模型。尤其 DS，目前输入不支持完整真值。

### AP 数据改进

- 抽取 A&P-only gold。
- 按 HADM_ID + DAY 对齐评估。
- 首日 note 只作为上下文，不纳入预测目标。
- 过滤极短 note 或无有效 A&P 的样本。

### DS 数据改进

- 扩大输入窗口到全住院。
- 加入 progress notes、procedure、microbiology、radiology、medication、ICD。
- 过滤 low-evidence samples。
- 分 section 构造输入和真值。
- 报告 evidence coverage。

### 优点

- 可能带来最大性能提升。
- 对论文可信度很重要。
- 可以形成 benchmark / dataset contribution。

### 风险

- 工程量较大。
- 需要严谨的数据泄漏控制。

## 12. 推荐路线

### 最快看到结果的路线

```text
AP A&P-only gold
-> day-level alignment
-> trend features
-> evidence retrieval
-> DeepSeek / Llama generation
```

这条路线不需要训练复杂模型，最快能得到可靠增益。

### 最适合作为方法论文的路线

```text
Problem-oriented temporal evidence retrieval
-> structured plan
-> LLM generation
-> self-verification
```

这条路线有清楚的方法贡献，也比较符合临床文档生成逻辑。

### 最适合作为小模型论文的路线

```text
Contrastive EHR-note retriever
-> top-k evidence/examples
-> evidence-grounded LLM generation
```

这比 flow matching 更稳，训练目标更简单，审稿人也更容易理解。

### DS 推荐路线

```text
full-admission timeline
-> hierarchical summarization
-> section-specific generation
-> evidence verification
```

DS 不能继续只用短窗口输入，否则最终诊断和出院总结天然不可观测。

## 13. 建议优先级

| 优先级 | 方案 | 原因 |
|---|---|---|
| P0 | AP A&P-only gold + day-level aligned evaluator | 先修任务定义，否则指标不可靠 |
| P0 | DS full-admission evidence reconstruction | 先解决输入不支持真值 |
| P1 | Trend features + LLM | 快速、可解释、能解决趋势判断 |
| P1 | Evidence retrieval + LLM | 最稳的 RAG baseline |
| P1 | Problem-oriented generation | 临床逻辑强，适合作为论文主线 |
| P2 | Contrastive dual encoder | 适合做可训练小模型 |
| P2 | Plan-then-write + verifier | 提升 factuality 和可解释性 |
| P3 | Flow matching | 可作为高级方法，但不建议一开始押宝 |

## 14. 一句话结论

除了 flow matching，当前最可行、最稳妥的论文路线是：

```text
Problem-oriented temporal evidence retrieval for evidence-grounded AP generation
```

先把 AP 做成严格对齐、证据可验证、能稳定提升的主任务；DS 则先补全 full-admission evidence，再作为扩展任务或第二阶段实验。
