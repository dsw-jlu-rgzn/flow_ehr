# AP100 Full-Set V2 Judge 指标偏低原因分析

## 现象

在 653 条 AP100 patient-day 的 UMLS CUI-F1 下：

| 方法 | Precision | Recall | CUI-F1 | 平均预测 CUI |
|---|---:|---:|---:|---:|
| base | 0.4809 | 0.2801 | 0.3473 | 225.3 |
| V2 | 0.4671 | 0.2554 | 0.3234 | 210.6 |
| V2 judge | 0.4676 | 0.2607 | 0.3279 | 214.6 |

V2 judge 相比 V2 有小幅提升：

- F1：+0.0045
- wins/losses/ties：342 / 309 / 2

但相比 base 仍明显偏低：

- F1：-0.0194
- wins/losses/ties：213 / 438 / 2

核心结论：V2 judge 的低分主要来自 **CUI recall 不足**，不是 precision 崩溃。它比 V2 稍微多覆盖了一些概念，但还没有补回 base 中更完整的临床实体覆盖。

## 与 LLM Judge 的差异

在 Qwen LLM judge 的 common 515 条上，V2 judge 相比 V2 其实略有改善：

| LLM 指标 | V2 | V2 judge | 差异 |
|---|---:|---:|---:|
| active problem coverage | 3.142 | 3.214 | +0.072 |
| trajectory capture | 2.831 | 2.934 | +0.103 |
| plan specificity | 2.942 | 3.052 | +0.111 |
| evidence grounding | 2.429 | 2.536 | +0.107 |
| disposition context | 2.926 | 2.973 | +0.047 |
| unsupported problem count | 3.713 | 3.598 | -0.115 |
| missed key problem count | 3.344 | 3.204 | -0.140 |

Qwen winner counts for V2 judge-revise:

- augmented wins：255
- baseline wins：155
- tie：105

这说明 V2 judge 的临床质量方向并非完全错误。问题是：当前 judge 更偏向“安全、删除不充分支持的内容、使 plan 更合理”，而 UMLS CUI-F1 更强调“具体临床概念覆盖”。两者目标不完全一致。

## 结构性原因

### 1. Judge revision 经常提升 grounding，但降低概念覆盖

在 V2 judge 明显优于 V2 的 63 条中：

| 方法 | CUI-F1 | Recall | Pred CUI |
|---|---:|---:|---:|
| V2 | 0.274 | 0.203 | 182.0 |
| V2 judge | 0.400 | 0.329 | 245.1 |

这类 case 中，judge revision 明显补概念、补 active problems，效果很好。

但在 V2 judge 明显低于 V2 的 44 条中：

| 方法 | CUI-F1 | Recall | Pred CUI |
|---|---:|---:|---:|
| V2 | 0.401 | 0.331 | 239.8 |
| V2 judge | 0.283 | 0.218 | 199.0 |

这类 case 的问题是 revision 后预测 CUI 数明显下降，说明 judge 删除、合并或泛化了原 V2 中有用的概念。

### 2. 当前 judge 不显式保护 candidate AP 的正确 CUI

当前 judge 会标出 unsupported changes，也会建议 scaffold revision，但它没有明确要求：

- 保留 candidate AP 中已经 evidence-supported 的具体概念；
- 如果删除一个 concept，必须说明为何 stale/unsupported；
- revision 后的 CUI 覆盖不能明显低于 candidate；
- 不能把具体概念合并成泛化 heading。

因此它有时会把 `V2` 中本来命中的概念删除或压缩。

### 3. Judge 有时把“缺乏明确适应证”误当作“不要写”

例子：`148910 day 5`

- V2 CUI-F1：0.298
- V2 judge CUI-F1：0.139

gold 当天列出 Voriconazole、Cefepime、Metronidazole、Vancomycin 等抗感染药物。V2 写了这些 broad-spectrum coverage，因此 CUI 命中较多。judge 认为 acyclovir/voriconazole indication 不明确，倾向于弱化或 clarify，从 grounding 角度合理，但 UMLS CUI-F1 会惩罚这些 medication concept 的缺失。

这里的问题不是 judge 一定错，而是评估目标不同：如果 gold 包含 medication exposure，模型应至少保留“administered/clarify indication”，而不是直接删除。

### 4. Judge 合并问题后，CUI recall 下降

例子：`157255 day 14`

- V2 CUI-F1：0.428
- V2 judge CUI-F1：0.250

V2 中覆盖 post-extubation status、volume overload、pleural effusions、sepsis、embolic CVA、AKI、altered mental status、Parkinson's meds、discharge planning 等。judge 后文本更精炼，删除了部分 neuro/AMS/medication/background concepts，临床上更简洁，但 UMLS recall 下降。

### 5. Judge 有时修正状态，但牺牲了 gold 中的模板/历史概念

UMLS gold 来自原始 note text，常包含模板字段、24h events、medication exposure、infusion、allergy、history 等。Judge revision 更像医生写的精炼 A&P，会删去模板化或背景概念。对真实临床可读性可能是好事，但对 CUI-F1 会扣分。

## 典型退化 case

## CUI-F1 低但临床评估更好的反例

### 106996 day 3

这个 case 是论文中可以重点展示的定性例子：base 的 CUI-F1 明显更高，但 V2 judge 在 LLM clinical rubric 上全面更好。

| 方法 | CUI-F1 | LLM judge |
|---|---:|---|
| base | 0.482 | 输 |
| V2 judge | 0.220 | 赢 |

| LLM 指标 | base | V2 judge |
|---|---:|---:|
| active problem coverage | 2 | 4 |
| trajectory capture | 1 | 4 |
| plan specificity | 3 | 4 |
| evidence grounding | 2 | 4 |
| disposition/context | 2 | 3 |
| unsupported problem count | 2 | 0 |
| missed key problem count | 3 | 1 |

gold 轨迹：82 岁 ESRD 患者因 hyperkalemic PEA arrest 入 ICU，已经 extubated；当天有 unplanned line/catheter removal，并记录了 Zosyn/vancomycin/piperacillin、heparin prophylaxis、hydralazine、pantoprazole 等 medication exposure。

base 的 CUI-F1 高，是因为它覆盖了大量 reference concepts：ESRD/hemodialysis、PEA arrest/hyperkalemia、anemia/transfusion、CT abdomen/pelvis、PICC/tunneled dialysis catheter、heparin/pantoprazole、vancomycin/piperacillin-tazobactam、ventilator/sedation 等。但 base 把患者写成 “currently intubated and sedated”，并继续计划 CMV/ASSIST、sedation、daily SBT 和 extubation；这与 gold 中 invasive ventilation 已停止、患者已 extubated 的 trajectory 相矛盾。base 还引入 possible ischemic colitis/pancreatitis 等 unsupported GI diagnosis。

V2 judge 更精炼，漏掉了一部分 template/exposure/background concepts，因此 CUI-F1 低。但它更准确地聚焦 ESRD/hyperkalemia/renal replacement therapy、severe anemia、post-PEA arrest、vascular access、coagulopathy/elevated PTT、hypertension 和 abnormal liver enzymes，且 unsupported count 为 0。这个例子证明：CUI-F1 可以奖励概念堆叠和 reference-style overlap，却不能充分惩罚状态错误；LLM rubric 更能捕捉 trajectory 和 evidence grounding。

### 168571 day 6

| 方法 | CUI-F1 | Pred CUI |
|---|---:|---:|
| V2 | 0.530 | 260 |
| V2 judge | 0.314 | 197 |

Judge 纠正了 vancomycin trough、NPH、decadron 等 grounding 问题，但 revised output 的 CUI 数大幅下降。gold 包含 tube feeds held for OR、active type and screen、vancomycin、fentanyl、midazolam 等 exposure/status concepts。revision 更安全，但减少了 concept coverage。

### 196355 day 8

| 方法 | CUI-F1 | Pred CUI |
|---|---:|---:|
| V2 | 0.539 | 250 |
| V2 judge | 0.326 | 185 |

V2 覆盖 VT storm、ICD、VT ablation、septic shock/E. coli UTI、AKI、quinidine、warfarin/INR、anxiety/PTSD 等。judge 更强调 fever/infection、foley、C diff、CT chest 等修正，但压缩了 cardiology/anticoagulation/psychosocial 等概念。

### 105351 day 7

| 方法 | CUI-F1 | Pred CUI |
|---|---:|---:|
| V2 | 0.463 | 306 |
| V2 judge | 0.295 | 197 |

gold 包含 renal failure、respiratory failure、retroperitoneal bleed、PEEP/FiO2/ABG、Levophed weaning、TPN 等。judge revision 仍保留核心问题，但显著压缩了 respiratory/renal/shock/nutrition 细节，CUI recall 下滑。

## 解决方案

### 方案 1：CUI-aware judge revision

在 judge/revision prompt 中加入显式 coverage 目标：

```json
{
  "supported_concepts_to_preserve": [
    {
      "concept": "",
      "source": "candidate_ap|today_ehr|previous_context",
      "evidence": "",
      "status": "active|resolved|administered|held|uncertain"
    }
  ],
  "unsupported_or_stale_concepts_to_remove": [
    {
      "concept": "",
      "reason": "",
      "evidence_conflict": ""
    }
  ],
  "missing_high_confidence_concepts_to_add": [
    {
      "concept": "",
      "evidence": "",
      "target_section": ""
    }
  ]
}
```

Revision 规则：

- candidate AP 中 evidence-supported 的具体 diagnosis、medication、procedure、device、lab abnormality 不得删除；
- 如果 indication 不明确但 medication administration 明确，写成 “administered; clarify indication/continue per ID”；
- 删除 concept 必须有 today's EHR contradiction 或 clear stale evidence；
- 不允许把 specific concept 改写成 generic heading。

### 方案 2：增加 revision 后 CUI coverage guardrail

在生成 revised AP 后，不用 gold，只比较 `candidate V2` 与 `V2 judge` 的预测 CUI：

- 如果 revised AP 的 CUI 数比 candidate 少超过 15-20%，触发二次 revision；
- 二次 revision 要求补回被删除但仍 supported 的 candidate concepts；
- 如果二次 revision 仍无法恢复，保留原 V2。

一个简单 heuristic 已经显示有效：

- 纯 V2 judge full-set CUI-F1：0.3279
- 如果 `V2 judge pred_cuis >= 0.95 * V2 pred_cuis` 才采用 judge，否则保留 V2，CUI-F1 约为 0.3373
- oracle 级别的 V2/V2 judge per-case 选择上界：0.3422

这说明“选择何时采用 judge revision”本身有较大收益。

### 方案 3：把 judge 从“删除型”改成“状态标注型”

不要直接删除不确定概念，而是保留概念并标注状态：

- `active`
- `resolved/off/discontinued`
- `administered but indication unclear`
- `historical background`
- `uncertain; monitor/clarify`

例如：

不要把 voriconazole 删除；可以写：

> Voriconazole was administered; clarify ongoing indication and de-escalate based on culture/ID guidance.

这样既保持 grounding，又保留 UMLS concept。

### 方案 4：对不同 case 自适应使用 judge

当前 V2 judge 并非对所有 case 都有益：

- V2 judge > V2：342 条
- V2 judge < V2：309 条

建议加一个 gate：

采用 judge revision 的条件：

- candidate 有明显 unsupported/stale concepts；
- judge 没有显著降低 CUI-like concept count；
- revised text 没有明显缩短关键 active sections；
- missing_high_confidence_concepts 被补充而不是只删除。

否则保留原 V2。

### 方案 5：将 UMLS concept checklist 前置到 scaffold

这和 `v2_cui_recall_v1` 的方向一致，但需要加入 judge-revise 流程：

- scaffold 阶段生成 `must_cover_concepts`；
- judge 阶段检查 candidate/revised 是否覆盖；
- revision 阶段只允许删除 `unsupported_or_stale_concepts`；
- final 阶段要求每个 active section heading 含具体 concept。

## 推荐下一版实验

建议命名：

`v2_judge_cui_guard_v1`

核心变化：

1. 先生成 V2 candidate；
2. judge 输出 `supported_concepts_to_preserve`、`unsupported_or_stale_concepts_to_remove`、`missing_high_confidence_concepts_to_add`；
3. revised AP 生成后做无 gold 的 UMLS concept-count guard；
4. 如果 revised concept count 明显低于 candidate，自动保留原 V2 或触发 coverage repair；
5. full-set 和 low-base65 同时评估 UMLS CUI-F1、LLM judge、文本长度和 pred CUI count。

预期：

- UMLS CUI-F1 应该明显高于当前 V2 judge；
- LLM evidence grounding 不应明显下降；
- 相比 base 的 recall 差距应缩小。
