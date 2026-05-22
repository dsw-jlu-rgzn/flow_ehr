# DeepSeek-v4-flash vs DeepSeek-v4-pro Truth Generation Comparison

生成日期：2026-05-21

## 1. 对比目的

本实验用于判断 `deepseek-v4-flash` 是否可以替代 `deepseek-v4-pro`，用于后续批量生成 A&P trajectory / verifier pseudo-gold truth。

对比设置保持一致：

```text
previous-day input + previous gold A&P
+ current-day input + current gold A&P
+ candidate V2 A&P
-> trajectory_delta_truth
-> verifier_truth
```

测试 case：

```text
105351_day13
105351_day19
```

## 2. 生成结果路径

Pro 版本：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_pro_2case_final/ap_delta_trajectory_verifier_truth.jsonl
```

Flash 初始版本，`max_tokens=6000`：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_flash_2case/ap_delta_trajectory_verifier_truth.jsonl
```

Flash 高 token 版本，`max_tokens=12000`：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_flash_2case_maxtok12000/ap_delta_trajectory_verifier_truth.jsonl
```

Flash 高 token 版本的 revised outputs：

```text
outputs/ap_delta_trajectory_truth_deepseek/deepseek_v4_flash_2case_maxtok12000/truth_revised_outputs/ap_delta_truth_verifier_revise_2case_flash12000/
```

## 3. 结构完整性对比

### 3.1 Flash `max_tokens=6000`

Flash 在 `max_tokens=6000` 时能够生成合法 JSON，但只生成了 `trajectory_delta_truth`，没有生成可用的 `verifier_truth`。

| Case | Pro problem threads | Flash problem threads | Pro verifier actions | Flash verifier actions |
|---|---:|---:|---:|---:|
| `105351_day13` | 14 | 7 | 21 | 0 |
| `105351_day19` | 6 | 6 | 11 | 0 |

结论：`max_tokens=6000` 下的 flash 不可用，因为 verifier truth 为空。

### 3.2 Flash `max_tokens=12000`

提高 token 后，flash 可以补全 verifier truth，但输出模式和 pro 明显不同。

| Case | Method | Problem Threads | Remove/Rewrite | Keep | Add | Restore | Has Summary |
|---|---|---:|---:|---:|---:|---:|---|
| `105351_day13` | pro | 14 | 9 | 3 | 7 | 2 | yes |
| `105351_day13` | flash12000 | 7 | 8 | 5 | 5 | 2 | yes |
| `105351_day19` | pro | 6 | 3 | 5 | 3 | 0 | no |
| `105351_day19` | flash12000 | 6 | 16 | 6 | 0 | 0 | no |

结论：

- day13：flash 少覆盖了约一半 problem threads。
- day19：flash verifier 偏向大量删除/重写，几乎没有补充 missing claim。
- pro 的 verifier 更像“纠错 + 补漏”，flash 更像“过度审稿式删除”。

## 4. 内容质量差异

### 4.1 day13 的主要差异

Pro 版本较好地识别：

- 新增 delirium/confusion；
- renal plan 从 CVVH 转为 HD；
- fever 需要 reculture + gram-negative coverage；
- hypotension/pressor off；
- RP bleed / Hct stable；
- 删除 V2 中的 ARDS、re-bleeding、shock、CVVHD、unsupported medication 等错误。

Flash 版本的问题：

1. 覆盖问题不足。
   - pro 生成 14 个 problem threads；
   - flash 只生成 7 个。

2. 更容易利用 gold A&P 之后的晚间 evidence。
   - flash 使用了 `2190-11-11 19:16` 的 CPAP/extubation failure 记录；
   - flash 使用了 `2190-11-11 20:19` 的 Hgb 7.2；
   - 但当前 gold A&P 是 `2190-11-11 10:36` 左右的 A&P。

这会导致一个严重问题：flash 可能把 gold A&P 生成时刻之后才发生的事件写入真值，形成时间泄露或 temporal mismatch。

3. Flash 对 candidate 错误的判断不如 pro 稳定。
   - pro 将 V2 的 failed extubation / Hgb 7.2 / re-bleeding 识别为 unsupported 或 contradicted；
   - flash 反而把这些晚间 evidence 纳入 trajectory truth，使其更接近“未来事件 truth”，不适合作为当前 A&P verifier truth。

### 4.2 day19 的主要差异

Pro 版本较好地识别：

- respiratory failure post-extubation improving；
- renal plan：no urgent HD，renal following，will diurese；
- CHF plan：Amio + beta-blocker + Imdur；
- fever resolved / should not carry forward；
- delirium/confusion 仍需保留；
- RP bleed stable。

Flash 版本的问题：

1. verifier action 过多。
   - pro：3 个 remove/rewrite，3 个 add；
   - flash：16 个 remove/rewrite，0 个 add。

2. flash 更偏向删除“gold 中没明确写”的内容。
   - 这会让 revised A&P 变短，可能损失合理临床上下文。

3. flash 没有补充 pro 认为重要的 missing items。
   - 尤其是 CHF、delirium、renal following 等结构化补充不稳定。

## 5. 下游 revised output 快速验证

使用 flash12000 truth 继续跑 LLM minimal reviser，结果如下：

| Case | Fix Items | Add Items | Revised Words |
|---|---:|---:|---:|
| `105351_day13` | 8 | 7 | 0 |
| `105351_day19` | 16 | 0 | 317 |

day13 的 revised output 为空，说明 flash truth + flash reviser 的链路不稳定。这个问题在 pro truth-revised 版本中没有出现。

## 6. 自动指标对比

### 6.1 ROUGE-L

| Method | N | ROUGE-L F1 | Avg. Pred Words | Missing Pred |
|---|---:|---:|---:|---:|
| `base` | 2 | 0.0740 | 441.5 | 0 |
| `v2` | 2 | 0.0757 | 419.5 | 0 |
| `v2_judge` | 2 | 0.0751 | 437.0 | 0 |
| `pro_delta_truth_revised` | 2 | 0.0854 | 404.5 | 0 |
| `flash_delta_truth_revised` | 2 | 0.0340 | 158.5 | 1 |

Flash revised 明显下降，主要原因是 `105351_day13` 输出为空。

### 6.2 UMLS CUI-F1

由于 flash day13 revised output 为空，CUI-F1 只评估到 1 条 case，因此这个结果只能作为错误诊断，不能作为完整对比。

| Method | Evaluated N | CUI-F1 |
|---|---:|---:|
| `base` | 1 | 0.3087 |
| `v2` | 1 | 0.3448 |
| `v2_judge` | 1 | 0.3322 |
| `pro_delta_truth_revised` | 1 | 0.3736 |
| `flash_delta_truth_revised` | 1 | 0.3298 |

在可评估的 day19 上，flash revised 低于 pro revised，也低于 v2。

## 7. 当前结论

当前不建议直接用 `deepseek-v4-flash` 替代 `deepseek-v4-pro` 进行一阶段真值生成。

主要原因：

1. `max_tokens=6000` 时 flash 会生成不完整 truth，缺少 verifier truth。
2. `max_tokens=12000` 虽然结构补全，但内容质量和 pro 差异明显。
3. flash 更容易使用 gold A&P 时间之后的 evidence，造成 temporal leakage。
4. flash verifier 容易过度删除，补充 missing claims 不稳定。
5. flash truth + flash reviser 已经在 day13 出现空输出，链路稳定性不足。

## 8. 后续可行方案

如果希望使用 flash 降低成本，建议不要使用当前“一次性生成 trajectory + verifier”的方式。更可行的是拆成两阶段：

### 方案 A：Pro 生成 trajectory，Flash 生成 verifier

```text
deepseek-v4-pro:
  previous/current gold A&P -> compact problem-level trajectory delta

deepseek-v4-flash:
  compact trajectory delta + candidate V2 -> verifier truth
```

优点：trajectory truth 的关键判断仍由 pro 保证；flash 只做较短上下文的 claim audit。

### 方案 B：Flash 两阶段生成，但增加强校验

```text
Step 1: flash 只生成 trajectory_delta_truth
Step 2: flash 只基于 trajectory_delta_truth + candidate 生成 verifier_truth
Step 3: 自动校验
```

必须加入的校验：

- `verifier_truth` 不能为空；
- `missing_supported_claims_to_add` 不能总是为空；
- 当前 A&P 时刻之后的 input evidence 不允许作为主要 truth evidence；
- `problem_threads` 数量不能显著少于 current gold A&P active problems；
- revised output 不能为空；
- unsupported/remove 数量过高时触发人工复核或 pro fallback。

### 方案 C：Flash 生成，Pro/Qwen 做 validation

```text
flash truth generation
-> validator LLM checks temporal leakage / missing active problems / empty verifier
-> failed cases fallback to pro
```

这个方案更适合大规模批量生成，可以用 flash 覆盖简单 case，用 pro 处理复杂 case。

## 9. 建议

短期 30-case 实验建议继续使用 `deepseek-v4-pro` 生成 truth。

等 30-case 跑通后，再做一个专门的 cost-quality ablation：

```text
pro one-stage truth
flash one-stage truth
flash two-stage truth
flash + validator + pro fallback
```

如果 flash two-stage 或 flash+fallback 的质量接近 pro，再考虑用 flash 作为大规模真值生成主力。
