# Flow EHR 中文实验说明

这个仓库用于 MIMIC-IV 纵向 EHR 生成实验，核心任务包括：

- AP：根据多日 EHR 和历史 progress notes 生成每日 Assessment & Plan。
- DS：根据住院时间线生成 discharge summary。
- EHRShot：诊断预测任务。

当前新增的实验重点是：在原始 LLM 生成前，加入一个轻量的病程信息筛选模块，先压缩/排序每日 EHR，再交给原来的 LLM 生成 AP。

## 数据位置

主要数据目录：

```text
data/AP/input/              原始 AP 输入
data/AP/gold/               AP 真实 progress note
data/DS/full_input/         DS full_input 输入
data/DS/gold/               DS 真实 discharge summary
```

注意：`data/` 目录包含真实输入和生成结果，默认不提交到 GitHub。

## 原始 V2 运行流程

DS V2 使用 `data/DS/full_input`：

```bash
bash run_event_ds_fix_v2.sh mistral
```

AP V2 使用 `data/AP/input`：

```bash
bash run_event_ap_fix_v2.sh mistral gt
```

如果需要监控 V1 跑完后自动启动 V2：

```bash
nohup bash monitor_and_run_v2.sh mistral gt > monitor_v2.log 2>&1 &
```

## 实验 1：无训练 embedding prefilter

这是最小闭环。它不训练任何小模型，直接复用已有 LLM 作为 frozen embedding encoder。

流程：

```text
每日 AP EHR rows
  + 规则生成的 trend snippets
        ↓
旧 LLM 抽 embedding
        ↓
用 query embedding 给 snippets 排序
        ↓
导出 top-k 压缩输入
        ↓
event_ap_fix_v2.py 生成 AP
        ↓
和 baseline 公平比较
```

支持三种 query mode：

- `previous_note`：真实推理可用，用上一天 note 作为检索目标。
- `day_context`：完全不用 note，用当天 EHR 全文作为检索目标。
- `oracle_gt`：上限测试，用当天真实 note 作为检索目标。

运行：

```bash
bash run_embedding_prefilter_no_train.sh 40 previous_note mistral
```

会导出：

```text
data/AP/input_embedding_topk_previous_note/
```

然后用同一个 LLM 生成，注意用 `--run_name` 避免覆盖 baseline：

```bash
python -u modeling/event_ap_fix_v2.py \
  --inputdir data/AP/input_embedding_topk_previous_note \
  --outputdir data/AP/generated \
  --setting gt \
  --model mistral \
  --run_name embedding_previous_note
```

输出目录：

```text
data/AP/generated/EE/mistral/embedding_previous_note/
```

## 实验 2：训练版 flow prefilter MVP

这个版本训练一个很小的模型。它使用每日真实 AP note 作为监督信号，学习从当天 EHR embedding 预测当天 note 的语义方向。

流程：

```text
每日 EHR snippets + trend snippets
        ↓
hashing text encoder 得到 day embedding
        ↓
小 flow/MLP 模型预测 target note embedding
        ↓
用预测 embedding 给 snippets 排序
        ↓
导出 top-k 压缩输入
        ↓
原 AP LLM 生成
```

训练 flow 版本并导出：

```bash
bash run_flow_prefilter_mvp.sh 40 flow
```

也可以跑 MLP baseline：

```bash
bash run_flow_prefilter_mvp.sh 40 mlp
```

导出目录：

```text
data/AP/input_flow_topk/
```

生成 AP：

```bash
python -u modeling/event_ap_fix_v2.py \
  --inputdir data/AP/input_flow_topk \
  --outputdir data/AP/generated \
  --setting gt \
  --model mistral \
  --run_name flow_topk
```

## 趋势片段

为了避免“单条记录打分无法判断趋势”的问题，prefilter 会额外生成 `[Trend]` snippets，例如：

```text
[Trend] glucose rising: 120 -> 260 (delta +140) from 08:00 to 16:00.
[Trend] creatinine falling: 2.1 -> 1.5 (delta -0.6) from day start to day end.
```

当前覆盖的指标包括：

- glucose
- creatinine
- WBC
- potassium
- sodium
- lactate
- hemoglobin
- heart rate
- temperature
- oxygen saturation

这些趋势片段会和原始 EHR rows 一起参与 embedding 排序。

## 公平验证

为了公平比较，验证脚本会对每个 method 只使用 baseline 和实验组共同拥有的 admission IDs。

比较 baseline 和 no-training prefilter：

```bash
bash run_compare_ap_generation.sh \
  data/AP/generated/EE/mistral/gt_v2 \
  data/AP/generated/EE/mistral/embedding_previous_note \
  embedding_previous_note
```

比较 baseline 和 flow prefilter：

```bash
bash run_compare_ap_generation.sh \
  data/AP/generated/EE/mistral/gt_v2 \
  data/AP/generated/EE/mistral/flow_topk \
  flow_topk
```

输出：

```text
outputs/ap_compare_<实验名>.csv
outputs/ap_compare_<实验名>_summary.csv
```

默认指标：

- ROUGE-L F1
- 生成文本长度
- paired delta vs baseline

如果环境里可以加载 SapBERT，也可以直接运行：

```bash
python -u evaluation/compare_ap_generation.py \
  --gt-dir data/AP/gold \
  --run baseline=data/AP/generated/EE/mistral/gt_v2 \
  --run embedding_previous_note=data/AP/generated/EE/mistral/embedding_previous_note \
  --baseline baseline \
  --sapbert
```

## 建议的最小实验顺序

1. 跑原始 AP V2 baseline：

```bash
bash run_event_ap_fix_v2.sh mistral gt
```

2. 跑无训练 prefilter：

```bash
bash run_embedding_prefilter_no_train.sh 40 previous_note mistral
```

3. 用 prefilter 输入重新生成 AP：

```bash
python -u modeling/event_ap_fix_v2.py \
  --inputdir data/AP/input_embedding_topk_previous_note \
  --outputdir data/AP/generated \
  --setting gt \
  --model mistral \
  --run_name embedding_previous_note
```

4. 做公平比较：

```bash
bash run_compare_ap_generation.sh \
  data/AP/generated/EE/mistral/gt_v2 \
  data/AP/generated/EE/mistral/embedding_previous_note \
  embedding_previous_note
```

5. 如果 no-training prefilter 有信号，再跑训练版 flow prefilter：

```bash
bash run_flow_prefilter_mvp.sh 40 flow
```

## 当前新增文件

```text
modeling/embedding_prefilter_no_train.py   无训练 embedding prefilter
modeling/flow_prefilter_mvp.py             训练版 flow/MLP prefilter MVP
evaluation/compare_ap_generation.py        公平对比 AP 生成结果
run_embedding_prefilter_no_train.sh        无训练 prefilter runner
run_flow_prefilter_mvp.sh                  训练版 prefilter runner
run_compare_ap_generation.sh               公平验证 runner
README_zh.md                               中文实验说明
```
