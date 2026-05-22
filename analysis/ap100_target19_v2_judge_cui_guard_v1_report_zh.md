# AP100 Target19 V2 Judge CUI Guard V1 实验报告

## 实验设置

- 实验版本：`ap100target19_generated_method2_gen_v2_judge_cui_guard_v1`
- Prompt version：`v2_judge_cui_guard`
- 样本：19 条 `base CUI-F1 高、V2 judge CUI-F1 低` 的 targeted patient-day
- Case list：`outputs/ap_memory_gated_scaffold_ap100/base_high_judge_low_19_cases.txt`

筛选条件：

- `base_f1 >= 0.38`
- `v2_judge_f1 <= 0.32`
- `base_f1 - v2_judge_f1 >= 0.08`

目标：验证 coverage-preserving judge prompt 是否能缓解 V2 judge 过度删除、泛化具体临床概念的问题。

## Prompt 改动

`v2_judge_cui_guard_v1` 保持 V2 scaffold 基础结构，但修改 judge/revision 目标：

1. Judge 输出三类概念：
   - `supported_concepts_to_preserve`
   - `missing_high_confidence_concepts_to_add`
   - `unsupported_or_stale_concepts_to_remove_or_downgrade`

2. Revision 规则从“删除 unsupported change”改成“修错不删对”：
   - 保留有 evidence 支持的 diagnosis、medication、procedure/device、lab abnormality、microbiology、care-state concepts。
   - 如果 medication/procedure 有记录但 indication 不清楚，写成 “clarify indication”，而不是删除。
   - stale/contradicted concept 用 resolved/off/held/discontinued/historical background 标注。
   - 不允许把具体概念泛化成 `arrhythmia`、`infection`、`renal dysfunction` 之类标题。

3. 为避免 JSON 过长，guard judge 输出加了上限：
   - preserve 最多 12 条
   - add/remove 最多 10 条
   - generic rewrite 最多 8 条
   - 每条 evidence 最多 2 个短 quote

## UMLS CUI-F1 结果

| 方法 | Precision | Recall | CUI-F1 | Pred CUI |
|---|---:|---:|---:|---:|
| base | 0.5863 | 0.3850 | 0.4597 | 227.1 |
| V2 | 0.4085 | 0.1893 | 0.2567 | 162.7 |
| V2 judge | 0.3976 | 0.1874 | 0.2535 | 165.2 |
| V2 judge CUI guard V1 | 0.4693 | 0.2785 | 0.3459 | 202.6 |

配对差异：

| 对比 | 平均 F1 差异 | wins | losses | ties |
|---|---:|---:|---:|---:|
| CUI guard - base | -0.1138 | 4 | 15 | 0 |
| CUI guard - V2 | +0.0892 | 14 | 5 | 0 |
| CUI guard - V2 judge | +0.0923 | 15 | 4 | 0 |

## 结论

1. `v2_judge_cui_guard_v1` 在 targeted failure subset 上显著优于原 V2 judge：
   - CUI-F1 从 0.2535 提升到 0.3459，绝对提升 +0.0923。
   - Recall 从 0.1874 提升到 0.2785。
   - Pred CUI 从 165.2 提升到 202.6。

2. 提升主要来自 recall 恢复，而不是单纯堆无关概念：
   - Precision 也从 0.3976 提升到 0.4693。
   - 说明 coverage-preserving prompt 不只是“写长”，而是更好地保留了 gold 中常见的具体医学概念。

3. 仍未追上 base：
   - base CUI-F1 为 0.4597，CUI guard 为 0.3459。
   - targeted subset 本身是 base 明显强的区域，说明 V2 系列的上游 scaffold/candidate 仍漏掉不少 base 覆盖的概念。

## Case 级表现

最大提升 case：

| Case | base | V2 | V2 judge | CUI guard | guard - judge |
|---|---:|---:|---:|---:|---:|
| 129694 day 3 | 0.557 | 0.278 | 0.234 | 0.462 | +0.228 |
| 141196 day 5 | 0.383 | 0.193 | 0.164 | 0.388 | +0.224 |
| 119898 day 8 | 0.491 | 0.310 | 0.310 | 0.511 | +0.200 |
| 174622 day 7 | 0.539 | 0.186 | 0.239 | 0.419 | +0.179 |
| 162680 day 3 | 0.412 | 0.245 | 0.255 | 0.431 | +0.175 |

退化 case：

| Case | base | V2 | V2 judge | CUI guard | guard - judge |
|---|---:|---:|---:|---:|---:|
| 167242 day 3 | 0.513 | 0.232 | 0.226 | 0.188 | -0.037 |
| 180024 day 2 | 0.392 | 0.273 | 0.278 | 0.260 | -0.018 |
| 186431 day 4 | 0.425 | 0.216 | 0.217 | 0.207 | -0.010 |
| 155513 day 3 | 0.401 | 0.302 | 0.308 | 0.300 | -0.008 |

## 观察

### 129694 day 3

原 V2 judge 主要保留 hypotension、anemia、leukocytosis、hypocalcemia 等，但较短。CUI guard 版本补充了 IV fluid bolus、sepsis/infection possibility、digoxin indication uncertainty、coagulopathy、electrolyte repletion 等概念，CUI-F1 大幅提升。

### 174622 day 7

原 V2/V2 judge 对 VT/amiodarone/diuresis/hyponatremia/renal function 等概念覆盖不足。CUI guard 版本补回 AKI/CKD、hyperglycemia、heparin anticoagulation、antiarrhythmics、furosemide/diuresis、potassium repletion 等，明显提升 recall。

### 141196 day 5

原 V2 judge 把 tachyarrhythmia 写得更抽象。CUI guard 保留 adenosine、esmolol、metoprolol、diltiazem、cefazolin、hydromorphone、lorazepam、diphenhydramine、anemia/hypocalcemia 等 documented concepts，因此指标提升。

## 后续建议

1. 将 `v2_judge_cui_guard` 扩展到 full-set 前，先在更大 targeted set 上测试 80-120 条，确认不会显著增加 unsupported concepts。
2. 加一个无 gold 的 concept-count guard：
   - 如果 revised CUI-like count 低于 candidate 的 85%-90%，触发 coverage repair 或保留原 V2。
3. 在 judge prompt 中进一步区分：
   - clinically active concept
   - documented exposure concept
   - historical/background concept
   - uncertain indication concept
   这样可以提升 CUI recall，同时减少临床误导。
4. 对退化 case 做单独分析，尤其是 `167242 day 3`，确认是否因为 prompt 增加 documented concepts 后稀释了主要问题，或仍然漏掉 gold 的关键诊断。

## 为什么仍然明显低于 Base

虽然 CUI guard 显著优于原 V2 judge，但 targeted19 上仍低于 base：

- base：0.4597
- CUI guard：0.3459
- gap：-0.1138

进一步查看差距最大的 case 后，原因主要不是单纯“guard 写得不够长”，而是下面几类结构性差异。

### 1. 上游 scaffold 仍没有恢复 base/gold 中的主诊断和长期背景

例子：`129694 day 3`

base 明确写出：

- ventricular tachycardia
- heart failure with reduced ejection fraction
- peripheral vascular disease / revascularization
- sacral decubitus ulcer
- wound care

CUI guard 的 scaffold active problems 只有：

- Hypotension
- Anemia
- Leukocytosis
- Hypocalcemia

candidate pool 虽然包含 digoxin、vancomycin、heparin prophylaxis、IV fluids 等，但没有恢复 `ventricular tachycardia` 这个 chief complaint 级主问题，因此 guard 对 gold 的核心 diagnosis recall 仍低。

### 2. Guard 偏向“当前可证据化管理”，base 更像 gold 的模板/历史字段

很多 gold 是原始 progress note，包含：

- Chief Complaint
- Code status
- Allergies
- Review of systems
- Physical examination
- Heart rhythm
- Stress ulcer / DVT prophylaxis
- Glycemic control
- Other medications
- Infusions
- Family history / social context

这些内容不是传统 A&P 的主要 active problems，但会被 UMLS CUI-F1 计入。base 往往会保留更多类似模板化/背景化概念，因此 CUI recall 更高。

例子：`174622 day 7`

base 命中 chronic systolic CHF、BiV ICD、chronic back pain、fluid overload、elevated JVD、stress ulcer、ankle edema 等；guard 主要围绕 AKI/CKD、hyperglycemia、heparin、hyponatremia、arrhythmia、HF/volume overload 等 active management，漏掉不少背景/模板概念。

### 3. Guard 仍会把具体历史诊断泛化

例子：`141196 day 5`

base 覆盖：

- metastatic NSCLC
- RML lobectomy
- bilateral PE
- left calf compartment syndrome
- fasciotomy and closure
- AVNRT / SVT
- anticoagulation

guard 覆盖 SVT、adenosine/esmolol/metoprolol/diltiazem、anemia、cefazolin 等当前用药和管理，但没有完整恢复 NSCLC、lobectomy、PE、fasciotomy/compartment syndrome 等历史背景概念。UMLS 指标会把这些 gold 中出现的历史概念算作 recall 缺失。

### 4. 有些 case 的 guard 引入状态/严重程度偏差，降低 precision

例子：`167242 day 3`

base 和 guard 都围绕 GI bleeding / metastatic RCC / embolization。但 guard 写到 “hemodynamic instability and ongoing transfusion requirements”，而 gold/base 更接近 hemodynamically stable、post-procedure monitoring。这里 guard 既漏掉部分模板概念，又引入更强的不稳定状态，precision 和 recall 都低。

### 5. Base 可能“更像 gold”，但不一定临床更优

base 的高 CUI-F1 有一部分来自：

- 更接近 gold note 的长背景和模板字段；
- 复述 prior/history；
- 覆盖更多 medication/prophylaxis/allergy/ROS/physical exam concepts；
- 更少主动删除 uncertain concepts。

这会提高 UMLS recall，但不一定代表临床推理、状态更新、证据支持都更好。因此后续需要把 UMLS CUI-F1 拆成：

- active problem CUI-F1
- treatment/procedure/device CUI-F1
- documented exposure CUI-F1
- historical/background CUI-F1
- status-aware contradiction score

否则会把“背景模板覆盖”与“真正临床 A&P 质量”混在一起。

## 针对这个差距的下一步方案

### A. 增加 `background_context_concepts`

在 scaffold 中新增一个不进入 active section、但允许出现在 assessment 背景句的列表：

```json
"background_context_concepts": [
  {
    "concept": "metastatic NSCLC s/p RML lobectomy",
    "source": "chief complaint/HPI/previous context",
    "output_rule": "assessment_background_only"
  }
]
```

这样可以保留历史肿瘤、PE、ICD、PVD、procedure history 等 gold 常见概念，而不把它们错误升级为 active problems。

### B. 增加 `documented_exposure_concepts`

对 medication、infusion、line/device、procedure、culture、nutrition route 等，设为 documented exposure：

```json
"documented_exposure_concepts": [
  {
    "concept": "vancomycin",
    "status": "administered",
    "indication": "unclear",
    "output_rule": "mention with clarify/de-escalation plan"
  }
]
```

这能补回 UMLS recall，同时避免把 indication 不明的药物写成强诊断。

### C. 从 EHR 中显式抽取 Chief Complaint / HPI / 24h Events

当前 scaffold 偏向当前结构化 evidence。应增加一个轻量抽取：

- chief complaint concepts
- HPI/background diagnoses
- 24h event concepts
- medications/infusions/antibiotics
- procedures/devices/lines

这些不一定都进入 active problems，但要进入 assessment 或 related section。

### D. Final prompt 增加“背景不升级”规则

目标不是把所有概念都变成 active problem，而是：

- active problems 写在 plan；
- history/background concepts 写在 assessment 第一段；
- documented exposure concepts 写在相关 plan 或 supportive care；
- resolved/stale concepts 用 off/resolved/discontinued 标注。

### E. 报告时不要只用总 CUI-F1

为了公平衡量 V2 系列，建议后续新增分层指标。总 CUI-F1 会偏向“更像原始 note 模板”的输出，而不一定偏向“更临床可用的 A&P”。

## 输出文件

- Generated notes：`outputs/ap_memory_gated_scaffold_ap100/ap100target19_generated_method2_gen_v2_judge_cui_guard_v1/`
- UMLS eval：`outputs/ap_memory_gated_scaffold_ap100/umls_eval_target19_v2_judge_cui_guard_v1/`
- Case list：`outputs/ap_memory_gated_scaffold_ap100/base_high_judge_low_19_cases.txt`
