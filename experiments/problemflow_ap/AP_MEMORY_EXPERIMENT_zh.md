# AP-Memory no-training 实验

这个实验用于验证：

> 上一日 A&P + 当日 input + 患者结构化病程 memory JSON + judge/revise 闭环，是否比 direct 生成和简单加入上一日 A&P 更好。

## 方法组

- `direct`: 只使用当天 EHR evidence 生成当日 A&P。
- `history_ap`: 使用上一日 A&P + 当天 EHR evidence 生成当日 A&P。
- `ap_memory_no_judge`: 维护患者级 JSON memory，但不做 judge 修订。
- `ap_memory`: 维护患者级 JSON memory，并用 judge LLM 判断相邻两日 A&P 差异是否由当天 evidence 支持，再修订 memory 后生成最终 A&P。
- `ap_memory_v2_no_judge`: V2 memory，不做 judge 修订。V2 将记忆拆成 `active_ap_problems`、`watchlist`、`supportive_care`，避免把所有风险和支持治疗都升级为 A&P 主问题。
- `ap_memory_v2`: V2 memory + judge 修订。judge feedback 只把满足通用 problem-promotion gate 的内容提升到 `active_ap_problems`，其余进入 `watchlist` 或 `supportive_care`。

## V2 设计原则

V2 不针对单个病例写规则，而是加入通用 A&P problem-promotion gate：

- 只有“上一日 A&P 未解决问题”或“今日证据导致诊断、严重程度、治疗决策、处置障碍、主要并发症发生变化”的内容，才进入 `active_ap_problems`。
- 单个 lab abnormality、单次 medication administration、prophylaxis、nutrition、access、routine monitoring、generic risk 默认进入 `watchlist` 或 `supportive_care`。
- 最终 A&P 只从 `active_ap_problems` 生成；`watchlist` 和 `supportive_care` 仅在今日变成主要临床问题时才展开。
- 每日 `active_ap_problems` 最多 6 个，鼓励接近真实 ICU A&P 的问题粒度，而不是完整 checklist。

## 运行

PowerShell:

```powershell
.\run_ap_memory_experiment.ps1 -Llm mock -Limit 10
```

DeepSeek no-training 实验:

```powershell
$env:DEEPSEEK_API_KEY="your-key"
.\run_ap_memory_experiment.ps1 -Llm deepseek -Limit 10 -OutDir experiments\problemflow_ap\outputs_ap_memory_deepseek_smoke
```

Bash:

```bash
LLM=mock LIMIT=10 bash run_ap_memory_experiment.sh
```

## 输出

默认输出目录：

```text
experiments/problemflow_ap/outputs_ap_memory
```

主要文件：

```text
data/ap_samples.jsonl
generations/direct.jsonl
generations/history_ap.jsonl
generations/ap_memory_no_judge.jsonl
generations/ap_memory.jsonl
generations/ap_memory_v2_no_judge.jsonl
generations/ap_memory_v2.jsonl
memories/*_latest_memory.jsonl
judges/*_judges.jsonl
metrics/ap_memory_metrics.csv
metrics/ap_memory_metrics_summary.csv
```

## 指标

脚本会输出 paired summary，重点指标包括：

- `rouge_l_f1`: 与 gold A&P 的文本相似度。
- `problem_f1`: 基于项目内 problem taxonomy 的问题覆盖 F1。
- `grounded_claim_rate`: 规则 verifier 估计的 claim evidence grounding。
- `unsupported_claim_rate`: 规则 verifier 估计的 unsupported claim 比例。
- `unsupported_problem_rate`: 生成问题不在 gold A&P problem set 中的比例。
- `unsupported_change_rate`: 相对上一日新增、但当天 evidence 未支持的问题变化比例。
- `forgotten_carried_problem_rate`: gold 中连续出现的问题被模型遗忘的比例。

## 推荐论文对比

主表建议比较：

```text
direct
history_ap
ap_memory_no_judge
ap_memory
ap_memory_v2_no_judge
ap_memory_v2
```

如果 `ap_memory` 相比 `history_ap` 在 `problem_f1`、`unsupported_change_rate`、`forgotten_carried_problem_rate` 上更好，就能支持“结构化病程记忆 + judge evidence attribution”对 A&P 纵向生成有增益。
如果 V1 出现过度 carry-forward 或过度 checklist 化，优先比较 `ap_memory_v2` 是否提高 `problem_precision` 并降低 `unsupported_change_rate`。
