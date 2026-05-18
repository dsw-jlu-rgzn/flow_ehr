# ProblemFlow-AP 多智能体实验脚本

这是一个面向 AP 任务的最小多智能体实验闭环。第一版默认不调用真实 LLM，而是用规则/Mock agent 跑通数据流和评估，方便先验证实验设计是否合理。

## 实验目标

比较三类方法：

- `direct`：当天 EHR evidence 直接生成 A&P。
- `trend`：加入趋势抽取后的 A&P 生成。
- `problemflow`：Evidence Agent + Problem Detector + Problem State Agent + Writer + Verifier 的结构化 problem-memory pipeline。
- `problemflow_v6`：在 V4 certainty gate 和 V5 verifier-reviser 基础上，增加覆盖率导向的写作/修订约束，尽量保留有证据支持的主要 A&P 问题。

## 运行方式

Windows PowerShell：

```powershell
.\run_problemflow_ap_mvp.ps1
```

只跑 V6 smoke/full 时可以指定 method、LLM 和输出目录：

```powershell
.\run_problemflow_ap_mvp.ps1 -Method problemflow_v6 -Llm deepseek -Limit 5 -OutDir experiments\problemflow_ap\outputs_deepseek_v6_smoke
.\run_problemflow_ap_mvp.ps1 -Method problemflow_v6 -Llm deepseek -OutDir experiments\problemflow_ap\outputs_deepseek_v6_full
```

或直接运行：

```powershell
& "C:\Users\dsw54\AppData\Local\Programs\Python\Python312\python.exe" experiments\problemflow_ap\problemflow_ap.py run-all
```

Linux/macOS：

```bash
bash run_problemflow_ap_mvp.sh
```

或：

```bash
METHOD=problemflow_v6 LLM=deepseek LIMIT=5 OUT_DIR=experiments/problemflow_ap/outputs_deepseek_v6_smoke bash run_problemflow_ap_mvp.sh
METHOD=problemflow_v6 LLM=deepseek OUT_DIR=experiments/problemflow_ap/outputs_deepseek_v6_full bash run_problemflow_ap_mvp.sh
```

## 输出

默认输出目录：

```text
experiments/problemflow_ap/outputs
```

主要文件：

```text
data/ap_samples.jsonl
logs/failed_ap_extraction.jsonl
generations/direct.jsonl
generations/trend.jsonl
generations/problemflow.jsonl
generations/problemflow_v6.jsonl
memories/problemflow_memory.jsonl
verification/*_verification.jsonl
metrics/metrics.csv
```

## 当前限制

- 当前 writer 是规则/Mock 版本，用于验证实验流程，不代表最终模型性能。
- A&P 抽取使用规则匹配 `Assessment and Plan`、`A/P`、`Assessment:`、`Plan:` 等标题。
- Verifier 是关键词支持率近似，用于快速比较不同 pipeline 的 evidence grounding 趋势。

后续可以把 Writer、Problem State Updater、Verifier 替换为 DeepSeek API agent。
