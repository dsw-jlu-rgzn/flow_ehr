# AP100 低 Base CUI-F1 Case 分析

## 分析对象

基于 `outputs/ap_memory_gated_scaffold_ap100/umls_eval/ap100_umls_cui_f1_detail.csv`，选取 base CUI-F1 最低的前 10% patient-day：

- case 数：65 / 653
- base 平均 Precision：0.3679
- base 平均 Recall：0.1959
- base 平均 CUI-F1：0.2448
- base 平均预测 CUI 数：197.4
- 平均 gold CUI 数：375.2

在这个低分子集上：

| 方法 | 平均 CUI-F1 | 相对 base 胜出数 |
|---|---:|---:|
| base | 0.2448 | - |
| V2 | 0.2476 | 36 / 65 |
| V2 judge | 0.2542 | 35 / 65 |

结论：虽然 full set 上 base 最高，但在 base 的低分区间，V2/V2 judge 已经有一定优势，说明 V2 的 scaffold/judge 对某些失败模式确实能补救。

## 低分 Base 的主要失败模式

### 1. 欠生成导致低 recall

定义：base precision >= 0.35 且 recall < 0.20。

- 数量：28 / 65
- 平均 F1：base 0.2487，V2 0.2411，V2 judge 0.2500
- 典型特征：gold CUI 很多，base 输出较短，虽然写的内容多数正确，但遗漏大量当天事件、药物、设备、感染、凝血、营养、管路、监测等概念。

例子：`176182 day 49`

| 方法 | CUI-F1 | Precision | Recall | pred CUI | gold CUI |
|---|---:|---:|---:|---:|---:|
| base | 0.188 | 0.560 | 0.113 | 84 | 416 |
| V2 | 0.097 | - | - | 59 | 416 |
| V2 judge | 0.144 | - | - | 97 | 416 |

gold 提到 RA、recent L AKA、necrotizing fasciitis、neutropenic fever、mental status change/agitation、LL PE、bilateral DVT、failed PICC attempts、allergies、infusions 等。base 基本聚焦到 CMO/death、neutropenic fever/sepsis、comfort care，具体概念覆盖不足。

这个类型对 CUI 指标伤害最大，因为它不是“错得离谱”，而是“太压缩”。优化空间是增加 high-confidence concept carry-forward。

### 2. 过生成/日期错位导致低 precision

定义：base precision < 0.30 且 recall >= 0.20。

- 数量：11 / 65
- 平均 F1：base 0.2379，V2 0.2272，V2 judge 0.2347
- 典型特征：gold 当天很短或只强调少数问题，但 base 复述了很长的 ICU course，带入大量当天 gold 不出现的历史问题/旧状态。

例子：`191230 day 26`

| 方法 | CUI-F1 | Precision | Recall | pred CUI | gold CUI |
|---|---:|---:|---:|---:|---:|
| base | 0.190 | 0.117 | 0.511 | 402 | 92 |
| V2 | 0.192 | - | - | 387 | 92 |
| V2 judge | 0.194 | - | - | 372 | 92 |

gold 当天主要强调 PEG tolerated、metoprolol around dialysis days、stable exam、ARDS/CMV/VRE/VAP 等；base/V2/V2 judge 都输出了很长的 ARDS/septic shock/CVVH/multi-organ dysfunction course，出现 acute-on-chronic renal failure、abdominal compartment syndrome、thoracentesis、adrenal insufficiency、SVC 等大量额外概念。

这个类型说明需要 temporal filtering：不能只因为历史 scaffold 存在，就把全部 chronic/old ICU problems 带入当天 A&P。

### 3. precision 和 recall 都低

定义：base precision < 0.35 且 recall < 0.20。

- 数量：14 / 65
- 平均 F1：base 0.2266，V2 0.2441，V2 judge 0.2536
- 这是 V2/V2 judge 最有优化价值的一类：base 既漏了 gold 概念，又引入不少不匹配概念；V2 judge 往往能通过更明确的问题拆分补回来。

例子：`107901 day 4`

| 方法 | CUI-F1 | pred CUI | gold CUI |
|---|---:|---:|---:|
| base | 0.201 | 172 | 385 |
| V2 | 0.100 | 134 | 385 |
| V2 judge | 0.302 | 224 | 385 |

gold 重点包括 hypoxemia、sustained VT episodes、amiodarone bolus/gtt、Tamiflu、discontinued antibiotics、off pressors、多种 allergy/antibiotic 信息。base 把 VT 写成 atrial fibrillation/RVR，V2 又进一步泛化为 arrhythmia/dialysis/disposition，V2 judge 虽然 CUI 数增加，但也引入 ESRD/dialysis 等可疑概念。这里需要同时优化概念覆盖和事实过滤。

## V2 在低 Base 区间的表现

### V2 有优势的场景

1. 当 base 输出太短、遗漏当天活跃干预时，V2 有时会补充更多 active problems。
2. 当 base 只写泛泛 sepsis/respiratory failure，V2 能补充具体药物、感染、代谢异常或 consult plan。
3. 当 base 对复杂 ICU course 的当天变化没更新，V2 judge 有时能通过 revision 补回更当天化的内容。

例子：`176182 day 50`

| 方法 | CUI-F1 | pred CUI | TP |
|---|---:|---:|---:|
| base | 0.192 | 84 | 45 |
| V2 | 0.240 | 166 | 66 |
| V2 judge | 0.275 | 167 | 76 |

base 仍沿用 death/CMO 文本；V2/V2 judge 发现 EHR 当天还有 antibiotics、TPN、heparin、insulin 等 active interventions，并提出 critical discrepancy，因此 CUI recall 明显提高。

### V2 仍失败的场景

1. V2 过度抽象：把 VT storm、ICD、amiodarone/lidocaine transition 写成 `cardiac arrhythmia`。
2. V2 继承 scaffold 错误：如把 hypoxemia/VT/pressors case 写成 ESRD/dialysis/disposition。
3. V2 judge 有时会补更多 CUI，但补的是不确定或状态错误的概念。

## 针对 Base 低分区间的优化空间

### A. 对欠生成 case：补概念 recall

新增一个 `must-cover concept checklist`，在生成前从当天 evidence 中抽取：

- active diagnosis/problem
- medication/treatment
- procedure/device
- lab/physiology abnormality
- infection/microbiology
- goals of care/disposition/status

生成后检查：如果 gold-like evidence 中的高置信概念没有进入 final A&P，触发 revision。

这类优化主要提升 `176182 day 49/50`、`193894 day 5/6`、`188623 day 10` 等高 gold CUI、低 pred CUI 的 case。

### B. 对过生成 case：加 temporal active filter

对每个候选 problem 标注：

- `active_today`
- `resolved_today`
- `historical_context_only`
- `not_supported_today`

只有 `active_today` 和少量必要的 `resolved_today` 可以进入 A&P 主体。历史问题只允许作为 comorbidity 背景短句出现，不能展开成完整 problem section。

这类优化主要针对 `191230 day 26`：当天 gold 很短，但模型把复杂 ICU 历史全部展开，precision 被大幅拉低。

### C. 对两者都低的 case：先事实校验，再补 CUI

这类 case 不能单纯增加长度，因为会进一步引入错误概念。建议使用两阶段 judge：

1. `fact filter`：删除不被当天 evidence 支持或状态冲突的概念。
2. `coverage repair`：补充遗漏的高置信 concept。

特别是 arrhythmia、dialysis、pressor、antibiotic、CMO/death 这些状态敏感问题，需要强制状态检查。

## 对 V2 的具体改法

1. 在 scaffold 中加入 `must_cover_concepts` 字段，每个 concept 附 evidence、status、source time。
2. prompt 要求 section heading 使用具体诊断，不使用泛化 heading，例如用 `Sustained VT on amiodarone infusion`，不要只写 `Arrhythmia`。
3. judge revision 增加 `missing_high_confidence_concepts` 和 `unsupported_or_stale_concepts` 两个列表。
4. 对短 gold/短当天 note case 加入输出长度上限和 stale problem penalty，避免复述完整 ICU course。
5. 对长 gold/复杂当天事件 case 加入输出概念下限，例如 final A&P 至少覆盖 top-K active clinical concepts。

## 总体判断

base 低分区间不是单一问题。大约一半以上是 recall 型问题，V2 可以通过 checklist 和 coverage repair 获益；少数但很典型的是 precision 型过生成，需要 temporal active filter。V2 judge 当前在低 base 子集上平均最高，说明 judge 路径值得继续做，但 judge 的目标应从“润色文本”改成“删除 stale concepts + 补齐 high-confidence CUI + 校验 status”。
