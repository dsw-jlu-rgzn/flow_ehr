# Truth Prompt V2 Repair Plan

生成日期：2026-05-22

## 1. 当前问题

29-case 小批量实验暴露出两个核心问题：

1. 一阶段 prompt 太长，`deepseek-v4-pro` 经常只生成 `trajectory_delta_truth`，不稳定生成 `verifier_truth`。
2. verifier truth 偏向删除 unsupported claims，但补充 missing active problems / stable carry-forward problems 不足。

指标上的表现是：

- unsupported problem count 明显下降；
- CUI precision 明显提升；
- CUI recall 下降；
- Qwen missed key problem count 上升。

因此修复目标不是继续强化“删除幻觉”，而是把 verifier 从 claim-level 删除器改成 problem-level coverage auditor。

## 2. 修复原则

新的 truth prompt 采用两个原则：

### 2.1 Trajectory First

Stage 1 先保证完整 problem-level trajectory truth：

```text
previous gold A&P
+ current gold A&P
+ previous/current input evidence
-> complete problem_threads
```

重点：

- 覆盖 current gold A&P 中每个有临床意义的问题；
- 包括 stable active problems，不只关注变化大的问题；
- resolved/removed problem 也要记录；
- 当前 gold A&P 是 note-time target，不能使用 gold 之后的 input event 作为真值目标；
- current input support 弱时，允许证据来自 current_gold_ap，但必须标注 weak input support。

### 2.2 Balanced Verification

Stage 2 再把 trajectory truth 转成 verifier truth：

```text
trajectory truth + candidate V2 A&P
-> keep / add / restore / rewrite / delete decisions
```

新的 verifier 不应该只输出 remove/rewrite，而要对每个 active trajectory thread 做决策：

| Candidate 状态 | Verifier 动作 |
|---|---|
| 已正确覆盖 | `supported_claims_to_keep` |
| 完全遗漏 | `missing_supported_claims_to_add` |
| 稳定 carry-forward 问题被遗漏 | `carried_forward_problems_to_restore` |
| 有问题但状态/计划错误 | `unsupported_claims_to_remove_or_rewrite` with `REWRITE` |
| resolved 问题被错误 carry forward | `unsupported_claims_to_remove_or_rewrite` with `DELETE` |

## 3. 已修改脚本

### 3.1 Stage 1 prompt

文件：

```text
scripts/build_ap_delta_trajectory_truth_deepseek.py
```

主要修改：

- 加入 note-time target 约束，避免使用 current gold A&P 之后的事件作为真值。
- 强制覆盖 current gold A&P 中所有 clinically meaningful problems。
- 要求 stable active problems 也生成 trajectory thread。
- 要求每个 thread 给出 `KEEP / ADD / REWRITE / REMOVE` 类型的 revision instruction。
- verifier 部分改为 balanced verifier truth，不能只删除。
- 如果输出长度不足，允许优先保证完整 trajectory truth，再在二阶段补 verifier。

### 3.2 Stage 2 verifier prompt

文件：

```text
scripts/augment_ap_delta_truth_verifier.py
```

主要修改：

- 明确 verifier 的目标是让最终 A&P 匹配 current active trajectory，而不是只删除 hallucination。
- 对每个 active trajectory thread 要求输出 KEEP / ADD / RESTORE / REWRITE 决策。
- 强化 missing active problems 和 stable carry-forward problems。
- 要求优先 REWRITE 而不是 DELETE，避免把整个 active problem 删掉。
- 如果 active trajectory thread 在 candidate 中缺失，则 `missing_supported_claims_to_add` 不能为空。

### 3.3 Truth 质量校验

新增文件：

```text
scripts/validate_ap_delta_truth_quality.py
```

用于检查：

- 是否有 trajectory threads；
- verifier actions 是否为空；
- active trajectory thread 是否没有 add/restore；
- 是否 delete-heavy 但没有 add/restore；
- 是否存在 augment error。

示例命令：

```bash
python scripts/validate_ap_delta_truth_quality.py \
  --truth-jsonl outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_30case/ap_delta_trajectory_verifier_truth_29_final.jsonl \
  --out-csv outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_30case/truth_quality_check.csv \
  --retry-cases outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_30case/truth_quality_retry_cases.txt
```

## 4. 推荐新运行流程

### Step 1: Stage 1 生成 trajectory-first truth

```bash
python scripts/build_ap_delta_trajectory_truth_deepseek.py \
  --model deepseek-v4-pro \
  --repair-model deepseek-v4-flash \
  --limit 30 \
  --force \
  --outdir outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_30case_v2prompt \
  --max-tokens 12000 \
  --repair-max-tokens 16000 \
  --parse-retries 2 \
  --retries 3 \
  --sleep-seconds 2 \
  --workers 3
```

### Step 2: Stage 2 补 verifier truth

```bash
python scripts/augment_ap_delta_truth_verifier.py \
  --truth-jsonl outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_30case_v2prompt/ap_delta_trajectory_verifier_truth.jsonl \
  --out-jsonl outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_30case_v2prompt/ap_delta_trajectory_verifier_truth_final.jsonl \
  --model deepseek-v4-pro \
  --repair-model deepseek-v4-flash \
  --max-tokens 10000 \
  --retries 3 \
  --sleep-seconds 2 \
  --workers 2
```

### Step 3: 质量校验

```bash
python scripts/validate_ap_delta_truth_quality.py \
  --truth-jsonl outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_30case_v2prompt/ap_delta_trajectory_verifier_truth_final.jsonl \
  --out-csv outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_30case_v2prompt/truth_quality_check.csv \
  --retry-cases outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_30case_v2prompt/truth_quality_retry_cases.txt
```

## 5. 预期改善

这版 prompt 主要希望改善：

- verifier 空输出；
- delete-heavy；
- missing active problem 上升；
- CUI recall 下降；
- stable carry-forward problem 被删掉。

预期指标变化：

| 指标 | 预期 |
|---|---|
| Qwen unsupported | 继续下降 |
| Qwen missed | 不再上升，最好低于 V2 judge |
| CUI precision | 保持高于 base/V2 |
| CUI recall | 接近或高于 base |
| ROUGE-L | 保持高于 base |
| Overall LLM judge | 继续高于 V2 judge |

## 6. 判断是否可以上全量

重跑 30-case 后，如果满足以下条件，就可以扩展到 600+：

- trajectory truth 失败率低于 5%；
- verifier truth 空输出低于 5%；
- revised output 空输出为 0；
- Qwen unsupported 下降；
- Qwen missed 不上升；
- CUI-F1 至少高于 V2/V2-judge，并尽量不低于 base。
