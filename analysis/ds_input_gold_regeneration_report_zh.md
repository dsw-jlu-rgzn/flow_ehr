# DS input/gold 合理性审计与重生成记录

生成日期：2026-05-16

## 结论

原始 DS 数据不完全合理，主要问题有两个：

1. `data/DS/gold` 中部分 HADM_ID 只取了第一条 discharge summary note。若同一次住院有后续 addendum 或多条 discharge summary，真值会停在较早出院日期。
2. `data/DS/input` 以 admission 的最后一条事件作为 24h 窗口锚点，而不是以 `ADMISSIONS.DISCHTIME` 作为锚点。这会让部分输入事件晚于当前 gold 的 discharge date。

因此已新增 MIMIC-III 专用重生成脚本：

- `processing/regenerate_ds_mimic3.py`

脚本策略：

- gold：按 HADM_ID 拼接所有 `CATEGORY == Discharge summary` 的 notes，按 `CHARTDATE, ROW_ID` 排序，避免短 addendum 覆盖主 discharge report。
- input：以 `filtered_ADMISSIONS.DISCHTIME` 为锚点，只保留 `TIME <= DISCHTIME` 的事件。
- 24h input：保留 `DISCHTIME` 前 24 小时内事件。
- full input：保留整个住院期间、且不晚于 `DISCHTIME` 的事件。

## 原始数据问题

基于 `data/target_population/filtered/filtered_NOTEEVENTS.csv`：

- admissions：100
- discharge summary notes：125
- 有 discharge summary 的 HADM_ID：100
- 有多条 discharge summary notes 的 HADM_ID：21

原始 `data/DS/gold` 中有 12 个 HADM_ID 的 gold discharge date 早于 `ADMISSIONS.DISCHTIME`，典型样本：

- `104732`：原 gold discharge date 为 `2114-12-17`，但 `ADMISSIONS.DISCHTIME` 为 `2114-12-28`。
- `145095`：原 gold discharge date 为 `2105-02-12`，但 `ADMISSIONS.DISCHTIME` 为 `2105-03-16`。
- `174792`：原 gold discharge date 为 `2191-01-20`，但 `ADMISSIONS.DISCHTIME` 为 `2191-01-25`。

## 重生成结果

输出目录：

- `data/DS_fixed_composed/24h/input`
- `data/DS_fixed_composed/24h/gold`
- `data/DS_fixed_composed/24h/ds_regeneration_summary.csv`
- `data/DS_fixed_composed/full/input`
- `data/DS_fixed_composed/full/gold`
- `data/DS_fixed_composed/full/ds_regeneration_summary.csv`

### 24h 版

| 指标 | 数值 |
|---|---:|
| input 文件数 | 100 |
| gold 文件数 | 100 |
| 出院后事件数 | 0 |
| 多 note gold 样本数 | 21 |
| input rows 平均 / 中位数 | 4.0 / 2.0 |
| input words 平均 / 中位数 | 259.5 / 47.5 |
| gold words 平均 / 中位数 | 1994.5 / 1809.5 |
| 空输入样本 | 11 |
| input 少于 100 词 | 76 |

24h 版时间上是干净的，但作为 DS 输入仍然明显不足。11 个样本在出院前 24 小时没有任何可用输入，76 个样本少于 100 词。

### full 版

| 指标 | 数值 |
|---|---:|
| input 文件数 | 100 |
| gold 文件数 | 100 |
| 出院后事件数 | 0 |
| 多 note gold 样本数 | 21 |
| input rows 平均 / 中位数 | 334.1 / 158.5 |
| input words 平均 / 中位数 | 24390.5 / 12219.0 |
| gold words 平均 / 中位数 | 1994.5 / 1809.5 |
| 空输入样本 | 0 |
| input 少于 500 词 | 0 |
| input words >= gold words | 95 |

full 版更符合 discharge summary 任务，因为 DS 是全住院总结，而不是最后 24 小时局部事件总结。

## 建议

后续 DS 实验建议优先使用：

- `data/DS_fixed_composed/full/input`
- `data/DS_fixed_composed/full/gold`

如果必须保持 24h 设定，应将结果单独报告为 “last-24h discharge-state summarization”，不应解释为完整 discharge summary generation。

更推荐的 DS pipeline：

1. 使用 full-admission 输入。
2. 先做 evidence selection 或 hierarchical summarization 压缩。
3. 再按 section 生成 Diagnosis、Hospital Course、Discharge Instructions。
4. 对 24h 版保留为 ablation，而不是主实验设置。
