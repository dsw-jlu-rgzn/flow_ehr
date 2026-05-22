# AP100 A&P UMLS CUI-F1 评估结果

## 评估是否可行

可以评估。当前下载的 `C:\Users\dsw54\Downloads\umls-2026AA-full\2026AA-full` 不是已经展开好的 `META` 目录，而是 UMLS 2026AA 的 `.nlm` ZIP 分卷；其中包含 `MRSTY.RRF.gz` 和 `MRCONSO.RRF.*.gz`。现有 `.venv-eval` 的 QuickUMLS 包位于 python3.10 site-packages，但当前 WSL `python3` 为 3.14，导致 QuickUMLS 不能直接 import。因此本次先采用一个可复现的 UMLS 字典匹配版 CUI-F1：

- 直接从 `.nlm` 中流式读取 UMLS 表，不要求完整解压。
- 只保留原项目 `evaluate_ap.py` 使用的 13 个临床相关 semantic types。
- 在同一批 AP100 patient-day 上比较 `base`、`V2`、`V2 judge` 三个输出与同一天 gold A&P 的 CUI 集合重叠。
- 匹配方式是标准化后的 exact n-gram term match，不是 QuickUMLS 的近似匹配，因此结果更适合作为当前稳定、可运行的 UMLS-CUI proxy；如果后续修复 QuickUMLS 环境，数值可能会有小幅变化。

## 数据覆盖

- summary 中 case 数：653
- 实际评估 patient-day：653
- 缺失：0
- UMLS semantic types 过滤后允许 CUI：1,235,462
- 本批文本候选 n-gram：4,192,272
- 命中的 UMLS term：6,726
- 扫描 MRCONSO 行数：18,064,972

## 总体指标

| 方法 | CUI Precision | CUI Recall | CUI-F1 | 平均预测 CUI 数 | 平均 gold CUI 数 |
|---|---:|---:|---:|---:|---:|
| base | 0.4809 | 0.2801 | 0.3473 | 225.3 | 392.6 |
| V2 | 0.4671 | 0.2554 | 0.3234 | 210.6 | 392.6 |
| V2 judge | 0.4676 | 0.2607 | 0.3279 | 214.6 | 392.6 |

## 配对差异

| 对比 | 平均 F1 差异 | wins | losses | ties |
|---|---:|---:|---:|---:|
| V2 - base | -0.0239 | 192 | 454 | 7 |
| V2 judge - base | -0.0194 | 213 | 438 | 2 |
| V2 judge - V2 | +0.0045 | 342 | 309 | 2 |

## 结论

1. 在 UMLS CUI-F1 口径下，`base` 仍然最高。主要原因不是 precision 更高很多，而是 recall 明显更高：base 平均预测 CUI 数 225.3，V2 为 210.6，V2 judge 为 214.6；V2 系列更短、更筛选，漏掉了一部分 gold 中存在的临床概念。
2. `V2 judge` 相比 `V2` 有小幅改善：F1 +0.0045，653 条中 342 条优于 V2。这说明 judge revision 能补回一部分临床概念，但增益较弱。
3. `V2 judge` 仍低于 base：F1 -0.0194，且 438 条低于 base。说明当前 judge 更偏向“清理/重写/结构化”，还没有稳定地把关键 diagnosis、status、medication、procedure、lab abnormality 补全。
4. 这个 UMLS 指标与之前 ROUGE/LLM judge 的方向基本互补：ROUGE/LLM judge 可以看结构、证据和状态合理性，UMLS CUI-F1 更像“临床概念覆盖率”。V2 可能在可读性/组织性上更好，但概念覆盖暂时不足。

## 典型例子

### CUI-F1 较低但临床轨迹更好：106996 day 3

这个 case 可以说明为什么 UMLS CUI-F1 不能单独代表临床质量。

| 方法 | CUI-F1 | LLM judge 结论 |
|---|---:|---|
| base | 0.482 | 输 |
| V2 judge | 0.220 | 赢 |

LLM judge 维度：

| 指标 | base | V2 judge |
|---|---:|---:|
| active problem coverage | 2 | 4 |
| trajectory capture | 1 | 4 |
| plan specificity | 3 | 4 |
| evidence grounding | 2 | 4 |
| disposition/context | 2 | 3 |
| unsupported problem count | 2 | 0 |
| missed key problem count | 3 | 1 |

gold 的关键轨迹是：82 岁 ESRD 患者因 hyperkalemic PEA arrest 入 ICU，已经 extubated，当天有 unplanned line/catheter removal，并有 Zosyn/vancomycin/piperacillin、heparin prophylaxis、hydralazine、pantoprazole 等 documented exposure。

base 的 CUI-F1 更高，因为它写了更多 gold 中能命中的概念，例如 ESRD/hemodialysis、PEA arrest/hyperkalemia、anemia/transfusion、PICC/tunneled dialysis catheter、heparin/pantoprazole、vancomycin/piperacillin-tazobactam、ventilator/sedation 等。但 base 明显存在状态错误：它写患者 “currently intubated and sedated”，并计划继续 CMV/ASSIST、sedation、daily SBT 和 extubation；而 gold 明确显示 invasive ventilation 已经停止，患者已经 extubated。base 还引入 possible ischemic colitis/pancreatitis 等 unsupported GI diagnosis。

V2 judge 的文本更短，因此 CUI-F1 低；它没有保留那么多 template/exposure/background concepts。但它更准确地抓住了当天临床轨迹：ESRD/hyperkalemia/renal replacement therapy、severe anemia、post-PEA arrest、vascular access、coagulopathy/elevated PTT、hypertension 和 abnormal liver enzymes。LLM judge 因此认为 V2 judge 的 trajectory capture、evidence grounding 和 unsupported claim 控制都更好。

这个例子说明：CUI-F1 会奖励“概念出现”，但不能充分判断概念状态是否正确。base 命中 ventilator/sedation 相关 CUI，反而掩盖了其把已 extubated 患者写成仍 intubated/sedated 的 trajectory 错误。论文中应将 CUI-F1 解释为 reference concept overlap，而不是完整的 clinical reasoning quality。

### base 明显优于 V2：174622 day 7

- gold 重点包含：amiodarone PO BID、amiodarone gtt overlap、Lasix IV BID、hyponatremia、VT runs、K repletion、allergies、infusions 等。
- base CUI-F1 0.539，V2 0.186，V2 judge 0.239。
- base 明确保留 chronic systolic CHF、EF 20-25%、BiV ICD、refractory VT、VT storm、amiodarone/lidocaine、AKI、hyponatremia、hyperglycemia、volume overload、electrolyte repletion 等。
- V2 写成较泛化的 “complex ICU course / cardiac arrhythmia / therapeutic anticoagulation / likely underlying cardiac disease”，具体 VT storm、ICD、antiarrhythmic transition、diuresis、electrolyte targets 等概念减少，导致 CUI recall 下降。
- V2 judge 补充了 AKI/CKD、DM、volume overload、furosemide、free water restriction 等，但仍把核心节律问题写得比较泛，且引入 “presumed thrombotic event” 这类不够稳的表达。

### V2 judge 明显优于 V2：139529 day 3

- base CUI-F1 0.304，V2 0.159，V2 judge 0.434。
- gold 包含 pneumonia、hypoxemia、desaturation、right-sided positional worsening、baseline arthritis、iodine allergy 等。
- V2 覆盖 pneumonia、right pleural effusion、retrocardiac opacity、AKI、hyperkalemia、vancomycin trough，但内容较短。
- V2 judge 把 respiratory status、right pleural effusion、atelectasis/effusion、AKI、hyperkalemia、vancomycin trough、ICU-level care 拆成更明确的问题，概念数从 105 增至 187，召回显著改善。
- 这个例子说明 judge revision 在“把隐含问题拆成独立 active problems”时有效。

### V2 优于 base，但 judge 回退：176840 day 10

- base CUI-F1 0.235，V2 0.433，V2 judge 0.272。
- gold 包含 vancomycin for GPC bacteremia concern、diet advanced、insulin gtt restarted、off dopamine、NSVT/PVC ectopy improvement 等。
- V2 覆盖 CAD、HTN、DM2、PVD、NSTEMI/LCX stent、cardiogenic shock、hypoglycemia、anemia、metabolic alkalosis、IABP removal、dopamine 等，概念覆盖高于 base。
- V2 judge 加入 leukocytosis、possible infection、confirm IABP status 等，但同时保留 “ongoing pressor requirement / cardiogenic shock requiring vasopressor support”，与 gold 的 “Off dopa” 存在状态不一致，且可能稀释了当天真实进展。
- 这个例子提示 judge revision 需要更强的 temporal/status 校验，否则会补概念但错状态。

## 后续优化建议

1. 把 UMLS CUI-F1 拆成两个子指标：`active problem CUI-F1` 和 `management/evidence CUI-F1`。当前一个总 CUI-F1 会把 diagnosis、lab、procedure、medication 混在一起，不利于定位问题。
2. 增加 `status-aware CUI score`：同一个 CUI 如果状态词不匹配，例如 resolved/off/weaned/improving/worsening/active，应该降权或单独记为 contradiction。
3. 对 V2 judge 加入“不得泛化”的约束：revision 必须优先补具体疾病、药物、检查、数值和状态，而不是把文本改成更抽象的 ICU 计划。
4. 在 scaffold 阶段加入 UMLS concept coverage check：生成后抽取 CUI，与 evidence/rubric 中的 must-cover concepts 比较，低 recall 的 patient-day 触发二次补全。
5. 修复 QuickUMLS 环境后复跑 official-style CUI-F1：当前 exact n-gram 版本稳定可用，但 QuickUMLS 可覆盖拼写变体和近似表达，适合作为最终报告指标。

## 输出文件

- 明细：`outputs/ap_memory_gated_scaffold_ap100/umls_eval/ap100_umls_cui_f1_detail.csv`
- 汇总：`outputs/ap_memory_gated_scaffold_ap100/umls_eval/ap100_umls_cui_f1_summary.csv`
- 配对差异：`outputs/ap_memory_gated_scaffold_ap100/umls_eval/ap100_umls_cui_f1_paired.csv`
- 元数据：`outputs/ap_memory_gated_scaffold_ap100/umls_eval/ap100_umls_cui_f1_metadata.json`
- 评估脚本：`evaluation/evaluate_ap100_umls_cui_f1.py`
