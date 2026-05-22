# A&P Delta Trajectory Truth 2-Case Smoke Report

生成日期：2026-05-21

## 1. 目的

本实验验证一个新的 verifier / trajectory truth 构造路线：

```text
previous-day input + previous gold A&P
+ current-day input + current gold A&P
+ candidate V2 A&P
-> problem-level gold delta
-> evidence chain
-> verifier truth
-> minimal revised A&P
```

该 truth 同时作为：

- verify agent 的伪真值；
- 病程轨迹建模真值；
- 后续训练 state / verifier agent 的监督信号。

## 2. 使用模型

DeepSeek endpoint 当前不支持 `deepseek-pro` 这个模型名，接口返回的支持模型名为：

```text
deepseek-v4-pro
deepseek-v4-flash
```

因此本实验采用：

- truth 主生成：`deepseek-v4-pro`
- JSON repair / day19 verifier 补全：`deepseek-v4-flash`
- minimal reviser：`deepseek-v4-flash`
- 初版 pairwise judge：`deepseek-v4-pro`
- 正式复评 pairwise judge：`Qwen/Qwen3.6-35B-A3B`

API key 只通过环境变量传入，没有写入代码或输出文件。

## 3. 新增脚本

```text
scripts/build_ap_delta_trajectory_truth_deepseek.py
scripts/augment_ap_delta_truth_verifier.py
evaluation/judge_ap_pairwise_llm.py
```

## 4. 输出路径

Truth JSONL：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_2case_final/ap_delta_trajectory_verifier_truth.jsonl
```

单 case truth：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_2case_compact_repair/case_truth/105351_day13.json
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_day19_retry/case_truth/105351_day19.json
```

Truth-revised A&P：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_2case_final/truth_revised_outputs/ap_delta_truth_verifier_revise_2case/
```

DeepSeek pairwise judge：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_2case_final/pairwise_v2_vs_delta_truth_revised_projudge.csv
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_2case_final/pairwise_v2_vs_delta_truth_revised_projudge_summary.csv
```

Qwen3.6 pairwise judge：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_2case_final/pairwise_v2_vs_delta_truth_revised_qwen36.csv
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_2case_final/pairwise_v2_vs_delta_truth_revised_qwen36_summary.csv
```

## 5. Truth 生成质量检查

| Case | Previous day | Problem threads | Fix items | Add items | Restore items |
|---|---:|---:|---:|---:|---:|
| 105351_day13 | 12 | 14 | 9 | 7 | 2 |
| 105351_day19 | 18 | 6 | 3 | 3 | 0 |

day13 truth 能较好识别：

- 错误 extubation / ARDS / VAP / re-bleeding / shock / CVVHD；
- 新增 delirium/confusion；
- renal plan 从 CVVH 转向 HD；
- fever plan 需要 reculture + gram-negative coverage；
- pressors off，不能继续写 shock。

day19 truth 能识别：

- respiratory failure 已 post-extubation improving；
- no urgent HD，renal following，diuresis plan；
- CHF plan 包含 beta-blocker 和 Imdur；
- fevers resolved，不应继续主动 infectious workup；
- delirium/confusion 仍需保留。

## 6. 替代 verifier 后的 2-case 结果

Pairwise judge 直接比较：

```text
V2 original
vs
delta-truth verifier + LLM minimal reviser
```

### 6.1 正式 Qwen3.6 Judge 结果

正式评估使用：

```text
Qwen/Qwen3.6-35B-A3B
```

两条 case 均未触发 JSON repair：

```text
judge_repair_used = False
```

Qwen3.6 judge summary：

| Method | Wins | Coverage | Trajectory | Specificity | Grounding | Disposition | Unsupported | Missed | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 | 0 | 3.0 | 2.5 | 3.5 | 2.5 | 4.0 | 1.0 | 2.5 | 3.0 |
| Delta-truth revised | 2 | 5.0 | 5.0 | 5.0 | 4.5 | 4.5 | 0.0 | 0.0 | 5.0 |
| Tie | 0 | - | - | - | - | - | - | - | - |

逐例：

- `105351_day13`：delta-truth revised 胜。Qwen rationale 指出 revised 输出更准确捕捉 trajectory、覆盖关键 active problems，并避免 V2 在 pressor status、dialysis modality 和 hemorrhage trajectory 上的矛盾。
- `105351_day19`：delta-truth revised 胜。Qwen rationale 指出 revised 输出覆盖 CHF、delirium 等 active problems，并符合 medication / trajectory plans；V2 遗漏关键问题且存在 respiratory support 相关矛盾。

### 6.2 初版 DeepSeek Judge 结果

初版 DeepSeek-v4-pro judge summary：

| Method | Wins | Coverage | Trajectory | Specificity | Grounding | Disposition | Unsupported | Missed | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V2 | 0 | 2.5 | 2.0 | 2.5 | 2.5 | 3.0 | 1.5 | 1.5 | 2.0 |
| Delta-truth revised | 1 | 4.0 | 4.0 | 4.0 | 4.0 | 3.5 | 0.0 | 0.0 | 4.0 |
| Tie | 1 | - | - | - | - | - | - | - | - |

逐例：

- `105351_day13`：delta-truth revised 胜。judge 认为 revised 版本更接近 gold A&P，正确处理 HD、fever workup、delirium、pressors off，并去掉 CRRT / vasopressor 等 unsupported 内容。
- `105351_day19`：DeepSeek judge 触发 fallback repair 后为 tie，因此该例不能作为强结论。Qwen3.6 复评后该问题已经解决，且结果为 delta-truth revised 胜。

## 7. 初步结论

该 2-case smoke test 支持以下判断：

1. 使用 consecutive gold A&P delta 生成 trajectory / verifier truth 是可行的。
2. 生成的 truth 能捕捉 problem-level 轨迹变化，而不只是 claim-level 删除。
3. 在 Qwen3.6 judge 下，2 个 case 均显示 delta-truth verifier + minimal reviser 优于 V2。
4. 当前仍需要提升 truth 生成 JSON 稳定性，尤其是 DeepSeek-v4-pro 长 prompt 容易输出格式错误或空 content；评估端使用 Qwen3.6 后本次没有触发 fallback。

## 8. 后续 TODO

1. 把 truth prompt 再压缩，减少 schema 和原始 input 长度。
2. 对每个 case 固定两阶段：
   - `gold delta -> trajectory truth`
   - `trajectory truth + candidate -> verifier truth`
3. 增加 automatic validation：
   - `case_id` 必须存在；
   - `problem_threads` 非空；
   - verifier fix/add 至少有一个，除非 candidate 已完全正确；
   - 禁止 claim_text 中泄露 gold/truth/verifier/oracle。
4. 扩展到 30 cases，再用稳定 judge 评估：
   - V2 original
   - V2 judge-revise
   - delta-truth verifier + minimal reviser
5. 将生成的 truth 转成训练样本：
   - evidence/history -> trajectory delta JSON；
   - candidate section -> verifier action；
   - verifier feedback -> revised A&P。
