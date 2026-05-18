# AP-Memory A&P 纵向生成实验阶段报告

日期：2026-05-16

## Motivation

本实验希望验证：**LLM 是否可以通过维护一个患者级结构化病程记忆，获得比简单读取上一日 A&P 更稳定的纵向临床状态追踪能力。**

真实临床中的 A&P 本身就是医生对患者当前问题、证据、治疗反应和下一步计划的压缩表达。因此，上一日 A&P 是一个非常强的短程历史 baseline。AP-Memory 的研究价值不应仅被理解为“把历史写得更长”，而是希望把纵向 EHR 推理从：

```text
previous text + today input -> today A&P
```

转化为：

```text
previous patient state + today evidence -> updated patient state -> today A&P
```

也就是让 LLM 学会维护一个随时间演化的患者状态表示，并用 judge LLM 检查相邻两日 A&P 的变化是否能由当日证据解释。理论上，这种显式状态记忆更可能在长程、自回归、历史 note 不完整、或需要下游 outcome/problem tracking 的场景中体现优势。

## Current Conclusion

当前 5 例 DeepSeek smoke test 的结论是：

> **AP-Memory V2 明显改善了 V1 的过度问题升级和 checklist 化问题，但在 teacher-forcing、短程 A&P 生成设置下，还没有相对 `history_ap` baseline 形成明确优势。**

关键结果：

| Method | ROUGE-L | Problem F1 | Precision | Recall | Unsupported Change | Forgotten |
|---|---:|---:|---:|---:|---:|---:|
| `history_ap` | 6.06 | 86.49 | 82.43 | 93.00 | 7.00 | 0.00 |
| `ap_memory` | 5.82 | 83.27 | 73.82 | 100.00 | 30.30 | 0.00 |
| `ap_memory_v2` | 6.20 | 86.28 | 81.94 | 95.00 | 14.00 | 0.00 |

解释：

- `history_ap` 很强，因为上一日 gold A&P 已经是高质量医生状态快照。
- V1 AP-Memory 的 recall 很高，但会把风险、lab abnormality、supportive care 过度提升为 A&P 主问题，导致 precision 和 unsupported-change 变差。
- V2 通过 `active_ap_problems / watchlist / supportive_care` 的通用 problem-promotion gate，显著改善 V1。
- 但 V2 仍未明显超过 `history_ap`，说明后续应重点评估 autoregressive 和 longer-horizon 设置，而不是继续只在短程 teacher-forcing 下微调 prompt。

下一步优先方向：

```text
history_ap_teacher_forcing
ap_memory_v2_teacher_forcing
history_ap_autoregressive
ap_memory_v2_autoregressive
```

重点观察长程自回归下的 `Problem F1`、`Forgotten` 和 `Unsupported Change` 是否出现分化。

## Reproduction Paths

项目根目录：

```text
C:\Users\dsw54\Desktop\codex_related\flow_ehr
```

核心实验脚本：

```text
experiments/problemflow_ap/ap_memory_experiment.py
```

便捷运行入口：

```text
run_ap_memory_experiment.ps1
run_ap_memory_experiment.sh
```

实验说明文档：

```text
experiments/problemflow_ap/AP_MEMORY_EXPERIMENT_zh.md
```

本阶段报告：

```text
analysis/ap_memory_experiment_report_zh.md
```

输入数据路径：

```text
data/AP/input/input_{hadm_id}.csv
data/AP/gold/gt_{hadm_id}.csv
```

本次 DeepSeek 5 例 smoke 使用的构造样本：

```text
experiments/problemflow_ap/outputs_ap_memory_deepseek_smoke/data/ap_samples.jsonl
```

本次 DeepSeek 5 例 smoke 的生成结果：

```text
experiments/problemflow_ap/outputs_ap_memory_deepseek_smoke/generations/direct.jsonl
experiments/problemflow_ap/outputs_ap_memory_deepseek_smoke/generations/history_ap.jsonl
experiments/problemflow_ap/outputs_ap_memory_deepseek_smoke/generations/ap_memory_no_judge.jsonl
experiments/problemflow_ap/outputs_ap_memory_deepseek_smoke/generations/ap_memory.jsonl
experiments/problemflow_ap/outputs_ap_memory_deepseek_smoke/generations/ap_memory_v2_no_judge.jsonl
experiments/problemflow_ap/outputs_ap_memory_deepseek_smoke/generations/ap_memory_v2.jsonl
```

本次 DeepSeek 5 例 smoke 的 memory 与 judge 产物：

```text
experiments/problemflow_ap/outputs_ap_memory_deepseek_smoke/memories/
experiments/problemflow_ap/outputs_ap_memory_deepseek_smoke/judges/
```

本次 DeepSeek 5 例 smoke 的指标文件：

```text
experiments/problemflow_ap/outputs_ap_memory_deepseek_smoke/metrics/ap_memory_metrics.csv
experiments/problemflow_ap/outputs_ap_memory_deepseek_smoke/metrics/ap_memory_metrics_summary.csv
experiments/problemflow_ap/outputs_ap_memory_deepseek_smoke/metrics/ap_memory_metrics_with_v2.csv
experiments/problemflow_ap/outputs_ap_memory_deepseek_smoke/metrics/ap_memory_metrics_with_v2_summary.csv
```

V2 mock smoke 输出，仅用于确认 pipeline 可运行，不用于论文结论：

```text
experiments/problemflow_ap/outputs_ap_memory_v2_mock_smoke/
```

复现本次 DeepSeek smoke 的命令：

```powershell
$env:DEEPSEEK_API_KEY="your-key"
.\run_ap_memory_experiment.ps1 `
  -Llm deepseek `
  -Limit 5 `
  -OutDir experiments\problemflow_ap\outputs_ap_memory_deepseek_smoke
```

如果只想补跑 V2 方法，并复用已有 V1/direct/history 结果：

```powershell
$env:DEEPSEEK_API_KEY="your-key"
& "C:\Users\dsw54\AppData\Local\Programs\Python\Python312\python.exe" `
  experiments\problemflow_ap\ap_memory_experiment.py generate `
  --samples experiments\problemflow_ap\outputs_ap_memory_deepseek_smoke\data\ap_samples.jsonl `
  --outdir experiments\problemflow_ap\outputs_ap_memory_deepseek_smoke `
  --method ap_memory_v2_no_judge `
  --llm deepseek `
  --limit 5 `
  --judge-max-tokens 4000 `
  --memory-max-tokens 3200

& "C:\Users\dsw54\AppData\Local\Programs\Python\Python312\python.exe" `
  experiments\problemflow_ap\ap_memory_experiment.py generate `
  --samples experiments\problemflow_ap\outputs_ap_memory_deepseek_smoke\data\ap_samples.jsonl `
  --outdir experiments\problemflow_ap\outputs_ap_memory_deepseek_smoke `
  --method ap_memory_v2 `
  --llm deepseek `
  --limit 5 `
  --judge-max-tokens 4000 `
  --memory-max-tokens 3200
```

补跑后统一评估：

```powershell
& "C:\Users\dsw54\AppData\Local\Programs\Python\Python312\python.exe" `
  experiments\problemflow_ap\ap_memory_experiment.py evaluate `
  --generation-dir experiments\problemflow_ap\outputs_ap_memory_deepseek_smoke\generations `
  --output experiments\problemflow_ap\outputs_ap_memory_deepseek_smoke\metrics\ap_memory_metrics_with_v2.csv
```

## 1. 实验动机

本实验验证一个 no-training 病程记忆框架是否能提升 ICU 每日 A&P 生成。

核心假设是：相比直接生成或简单加入上一日 A&P，显式维护患者级结构化病程记忆 `M_t`，并用 judge LLM 检查相邻两日 A&P 的变化是否由当日证据支持，可以提升：

- 活跃问题追踪能力
- 时间一致性
- 证据归因
- 历史问题保留能力

形式化表示：

```text
M_t = Update(M_{t-1}, AP_{t-1}, X_t)
AP_t = Generate(M_t, X_t)
```

其中：

- `X_t`: 当日 EHR input
- `AP_{t-1}`: 上一日 A&P
- `M_t`: 第 t 日患者级病程 memory JSON
- `AP_t`: 当日生成 A&P

完整 judge 版本为：

```text
M_{t-1} + AP_{t-1} + X_t -> candidate M_t
candidate M_t + X_t -> candidate AP_t
Judge(AP_{t-1}, candidate AP_t, X_t, M_{t-1}) -> feedback
candidate M_t + feedback -> revised M_t
revised M_t + X_t -> final AP_t
```

## 2. 数据与设置

实验使用项目内 AP 任务数据：

```text
data/AP/input/input_{hadm_id}.csv
data/AP/gold/gt_{hadm_id}.csv
```

样本单位为 `patient-day`。

本阶段 smoke test 使用 5 个 DeepSeek 样例，均来自同一患者 `109079` 的连续天数：

```text
109079_day2
109079_day3
109079_day4
109079_day5
109079_day6
```

当前默认设置为 teacher-forcing history：

```text
history_ap / ap_memory 使用上一日 gold A&P 作为 AP_{t-1}
```

这意味着 `history_ap` baseline 得到了非常强的上一日医生状态快照。该设置适合验证方法上限，但不完全等同真实部署。

## 3. 方法组

### 3.1 Direct

只使用当日 EHR evidence：

```text
X_t -> AP_t
```

### 3.2 History AP

使用上一日 A&P 和当日 EHR evidence：

```text
AP_{t-1} + X_t -> AP_t
```

这是目前最强的短程 baseline，因为 A&P 本身就是医生浓缩后的临床状态。

### 3.3 AP-Memory V1 No Judge

维护患者级 JSON memory，但不做 judge 修订：

```text
M_{t-1} + AP_{t-1} + X_t -> M_t
M_t + X_t -> AP_t
```

V1 memory schema 主要包括：

```json
{
  "global_status": {},
  "active_problems": [],
  "resolved_problems": [],
  "key_events_today": [],
  "interventions_today": [],
  "treatment_response": [],
  "risks": [],
  "uncertainties": [],
  "judge_feedback": []
}
```

### 3.4 AP-Memory V1

在 V1 memory 基础上加入一轮 judge/revise：

```text
generate -> judge -> revise memory -> regenerate
```

Judge 检查：

- 今日 A&P 相比昨日新增了什么
- 哪些问题被删除、弱化或 resolved
- 哪些变化能由今日 evidence 支持
- 哪些变化没有证据支持
- 哪些昨日 active problem 被无证据遗忘
- 哪些今日 evidence 没有进入 A&P/memory

### 3.5 AP-Memory V2 No Judge

V2 的动机来自 V1 问题：V1 容易过度 carry-forward，把所有风险、支持治疗和 lab abnormality 都升级成 A&P 主问题。

V2 将 memory 拆为：

```json
{
  "global_status": {},
  "active_ap_problems": [],
  "watchlist": [],
  "supportive_care": [],
  "resolved_problems": [],
  "key_events_today": [],
  "treatment_response": [],
  "uncertainties": [],
  "judge_feedback": []
}
```

V2 加入通用 problem-promotion gate：

- `active_ap_problems`: 今日应该有独立 A&P section 的主问题
- `watchlist`: lab 异常、风险、趋势、不确定问题
- `supportive_care`: prophylaxis、nutrition、access、routine monitoring 等支持治疗

最终 A&P 只从 `active_ap_problems` 生成。

### 3.6 AP-Memory V2

V2 memory + judge/revise。

修订原则：

- 不针对单个 case 写规则
- 仅将满足通用 problem-promotion gate 的内容提升到 `active_ap_problems`
- 其余信息进入 `watchlist` 或 `supportive_care`
- 避免 checklist 化
- 每日 `active_ap_problems` 最多 6 个

## 4. 指标说明

| 指标 | 含义 | 方向 |
|---|---|---|
| `ROUGE-L` | 生成 A&P 与 gold A&P 的最长公共子序列重合度，偏字面文本相似度 | 越高越好 |
| `Problem Precision` | 生成 A&P 中抽取到的问题有多少也出现在 gold A&P 中 | 越高越好 |
| `Problem Recall` | gold A&P 中的问题有多少被生成 A&P 覆盖 | 越高越好 |
| `Problem F1` | problem precision 与 recall 的调和平均 | 越高越好 |
| `Unsupported Change` | 相比上一日新增/变化的问题中，没有被当日 evidence taxonomy 支持的比例 | 越低越好 |
| `Forgotten` | gold 中上一日和今日都存在的问题，被模型今日漏掉的比例 | 越低越好 |

注意：当前 `Problem F1`、`Unsupported Change` 等是规则 taxonomy 指标，不是医生人工评价。它们适合快速比较趋势，但可能惩罚临床合理但 taxonomy 未覆盖的表达。

## 5. DeepSeek 5 例结果

输出文件：

```text
experiments/problemflow_ap/outputs_ap_memory_deepseek_smoke/metrics/ap_memory_metrics_with_v2.csv
experiments/problemflow_ap/outputs_ap_memory_deepseek_smoke/metrics/ap_memory_metrics_with_v2_summary.csv
```

汇总结果：

| Method | ROUGE-L | Problem F1 | Precision | Recall | Unsupported Change | Forgotten |
|---|---:|---:|---:|---:|---:|---:|
| `direct` | 6.14 | 76.47 | 72.33 | 82.78 | 8.00 | 12.22 |
| `history_ap` | 6.06 | 86.49 | 82.43 | 93.00 | 7.00 | 0.00 |
| `ap_memory_no_judge` | 6.05 | 81.99 | 73.45 | 97.50 | 10.30 | 0.00 |
| `ap_memory` | 5.82 | 83.27 | 73.82 | 100.00 | 30.30 | 0.00 |
| `ap_memory_v2_no_judge` | 6.12 | 83.29 | 79.06 | 93.00 | 27.22 | 2.86 |
| `ap_memory_v2` | 6.20 | 86.28 | 81.94 | 95.00 | 14.00 | 0.00 |

## 6. 主要观察

### 6.1 History AP 是很强 baseline

在 teacher-forcing 设置下，`history_ap` 表现非常强：

```text
Problem F1 = 86.49
Unsupported Change = 7.00
Forgotten = 0.00
```

这说明在短程场景中，上一日 gold A&P 本身已经是一个高质量临床状态快照。简单加入上一日 A&P 就能显著提升 direct baseline。

### 6.2 V1 AP-Memory 会过度保留和过度展开问题

V1 `ap_memory` 的 recall 达到 100.00，但 precision 只有 73.82。

这说明它几乎覆盖了所有 gold 问题，但额外输出了许多 gold 中没有的条目，例如：

```text
heme
renal_aki
glucose_diabetes
pneumonia
neuro_pain_sedation
volume
```

这些可能是临床上应关注的风险或支持治疗，但不一定应该成为当日 A&P 主问题。

因此 V1 的 `Unsupported Change` 升高到 30.30。

### 6.3 V2 改善了 V1 的 checklist 化问题

相比 V1 `ap_memory`，V2 `ap_memory_v2`：

```text
ROUGE-L: 5.82 -> 6.20
Problem F1: 83.27 -> 86.28
Precision: 73.82 -> 81.94
Unsupported Change: 30.30 -> 14.00
Forgotten: 0.00 -> 0.00
```

这说明 V2 的 problem-promotion gate 有效减少了额外问题，提升了 precision 和文本贴近度。

### 6.4 但 V2 仍未明显超过 History AP

V2 与 `history_ap` 对比：

```text
history_ap:
Problem F1 = 86.49
ROUGE-L = 6.06
Unsupported Change = 7.00
Forgotten = 0.00

ap_memory_v2:
Problem F1 = 86.28
ROUGE-L = 6.20
Unsupported Change = 14.00
Forgotten = 0.00
```

结论：V2 相比 V1 有明显改进，但在当前 5 例 teacher-forcing short-horizon 设置下，没有相对 `history_ap` 形成稳定优势。

## 7. 当前阶段结论

当前最稳妥的结论是：

> Previous-day A&P is a strong short-horizon baseline. AP-Memory V2 improves over V1 by reducing over-promotion of problems, but it does not yet clearly outperform the history-A&P baseline in the teacher-forcing setting.

中文表述：

> 在短程、teacher-forcing 设置下，上一日 A&P 本身已经提供了强临床状态信息。AP-Memory V2 相比 V1 明显减少了过度问题升级，但尚未相对简单加入上一日 A&P 的 baseline 形成明确优势。

这不是负面结果，而是说明后续实验应转向更能体现 memory 价值的设置。

## 8. 后续实验建议

### 8.1 Autoregressive setting

当前使用上一日 gold A&P：

```text
gold AP_{t-1} + X_t -> AP_t
```

这对 `history_ap` 非常有利。

下一步应测试：

```text
generated AP_{t-1} + X_t -> generated AP_t
```

也就是使用模型自己昨日输出作为今日历史。此时 `history_ap` 可能累积漂移，而 AP-Memory 理论上可以通过结构化状态减少漂移。

建议比较：

```text
history_ap_teacher_forcing
ap_memory_v2_teacher_forcing
history_ap_autoregressive
ap_memory_v2_autoregressive
```

重点观察：

- `Problem F1` 随天数是否下降
- `Forgotten` 是否累积上升
- `Unsupported Change` 是否累积上升

### 8.2 Longer horizon

当前 5 例来自短程连续天数。AP-Memory 的价值更可能出现在长住院过程：

```text
day 2 -> day T
```

建议按 patient 聚合，分析晚期天数：

```text
early days: day 2-3
middle days: day 4-6
late days: day >= 7
```

如果 memory 有价值，应在 late days 更能减少遗忘和漂移。

### 8.3 Memory state 作为下游状态表示

AP-Memory 不一定只用于提高 A&P 文本 ROUGE。后续可以评估 `M_t` 是否有助于：

- next-day deterioration prediction
- discharge outcome prediction
- active problem tracking
- treatment response tracking
- temporal contradiction detection

这可能比 A&P 文本生成更能体现结构化 memory 的价值。

### 8.4 改进 V2 gate，但避免单例过拟合

下一步 prompt 可以继续加强通用规则：

- judge feedback 默认进入 `watchlist`
- 只有影响今日诊疗决策的内容才进入 final A&P
- routine ICU care 合并到 `ICU Care`
- lab abnormality 只有在改变诊疗计划时才升级
- prior A&P 中低优先级问题可保留在 memory，但不一定生成到今日 A&P

原则是：只改通用 problem-promotion policy，不针对单个患者或单个病种写特殊规则。

## 9. 当前实验代码位置

主脚本：

```text
experiments/problemflow_ap/ap_memory_experiment.py
```

运行入口：

```text
run_ap_memory_experiment.ps1
run_ap_memory_experiment.sh
```

实验说明：

```text
experiments/problemflow_ap/AP_MEMORY_EXPERIMENT_zh.md
```

DeepSeek smoke 输出：

```text
experiments/problemflow_ap/outputs_ap_memory_deepseek_smoke/
```

## 10. 下一步推荐命令

跑 autoregressive 版本：

```powershell
$env:DEEPSEEK_API_KEY="..."
.\run_ap_memory_experiment.ps1 `
  -Llm deepseek `
  -Limit 5 `
  -AutoregressiveHistory `
  -OutDir experiments\problemflow_ap\outputs_ap_memory_deepseek_autoreg_smoke
```

跑更大规模 teacher-forcing：

```powershell
$env:DEEPSEEK_API_KEY="..."
.\run_ap_memory_experiment.ps1 `
  -Llm deepseek `
  -OutDir experiments\problemflow_ap\outputs_ap_memory_deepseek_full
```
