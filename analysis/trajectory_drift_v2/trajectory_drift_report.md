# V2 对轨迹偏移的影响分析

## 结论

基于 full 653 Qwen/SiliconFlow 评估结果，V2 对“轨迹偏移”有缓解作用，但不能说已经解决。

最核心的现象是：同一个 admission 内，越到后期，V2 相对 base 的提升会变大；但 V2 自身的绝对质量并不会稳定变好，且 unsupported problem count 在后期仍有上升趋势。这说明 V2 更像是在抵消 baseline 后期退化，而不是完全建立了可控的纵向 trajectory tracking。

## 论文图

论文风格图文件已保存到 `analysis/trajectory_drift_v2/paper_figures/`，每张图同时提供 `.pdf`、`.svg` 和 `.png`。建议论文投稿优先使用 `.pdf` 或 `.svg`，日常预览使用 `.png`。

复现命令：

```bash
python scripts/plot_trajectory_drift_paper_figures.py
```

图注、指标定义与复现路径已整理在：

`analysis/trajectory_drift_v2/paper_figures/figure_captions_and_reproduction.md`

### 主图：Longitudinal trajectory drift and V2 mitigation

![Paper main figure](C:/Users/dsw54/Desktop/codex_related/flow_ehr/analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_main.png)

路径：

- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_main.pdf`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_main.svg`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_main.png`

**设计意图：**

这张图作为主实验图，用来同时回答三个问题：

1. base generator 是否存在 longitudinal trajectory drift？
2. V2 的收益是否主要体现在 admission 后期，而不是所有时间点均匀提升？
3. V2 和 V2 + judge-revise 是否能稳定后期 A&P 质量？

因此主图被设计成三个 panel：Panel A 先证明问题存在，Panel B 证明方法对 drift 有针对性缓解，Panel C 证明缓解后的绝对质量变化。

**实验目的：**

该图的目标不是只比较平均分，而是验证“随 admission 进展，A&P 生成质量是否退化，以及 V2 是否能缓解这种退化”。这比普通 overall score 更贴近本文的 longitudinal A&P generation motivation。

**读图方式：**

- x 轴是同一个 admission 内的相对进度分箱：`Early`、`Mid-E`、`Mid-L`、`Late`。
- `Mid-E` 表示 mid-early，`Mid-L` 表示 mid-late。
- Panel A 的 y 轴是 evaluation score，包括 base quality 和 base trajectory capture。
- Panel B 的 y 轴是 `Quality improvement over base = method quality - base quality`。
- Panel C 的 y 轴是不同方法的 absolute quality score。

**具体观察：**

- Panel A 中，base quality 从 early 的 `8.50` 降到 late 的 `5.02`，说明 base 在同一 admission 后期整体 A&P 质量明显下降。
- Panel A 中，base trajectory capture 从 early 的 `2.73` 降到 mid/late 阶段约 `2.25-2.41`，说明 base 不只是 general quality 下降，也确实更难维持病程轨迹。
- Panel B 中，V2 在 early 阶段相对 base 为负收益，但从 mid-early 开始转为正收益，并在 late 阶段达到 `+1.81`。
- Panel B 中，V2 + judge-revise 的 late improvement 达到 `+2.62`，高于 V2，说明 revision 进一步增强了后期修复能力。
- Panel C 中，base 从 early 到 late 明显下滑，而 V2/V2 + judge-revise 在后期维持更高质量；尤其 judge-revise 在 late 阶段保持约 `8.00` 的质量分数。

**具体结论：**

这张图支持两个核心结论：

1. base generator 存在明显的 longitudinal degradation：同一 admission 越到后期，A&P 质量和 trajectory capture 越容易下降。
2. V2 不是简单提升整体平均分，而是对 admission 后期更有效；它的 improvement 随 admission 进展变大，因此可以解释为对 trajectory drift 的缓解。

**论文中可写的 claim：**

> The base generator exhibits clear longitudinal degradation: both aggregate A&P quality and trajectory capture decrease as admissions progress. V2 mitigates this drift, with larger gains in later admission stages, and judge-revise further stabilizes late-stage generation quality.

### 补充图：Absolute hospital-day analysis

![Absolute day figure](C:/Users/dsw54/Desktop/codex_related/flow_ehr/analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_absolute_day.png)

路径：

- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_absolute_day.pdf`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_absolute_day.svg`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_absolute_day.png`

**设计意图：**

主图按 admission 内相对进度分箱，可以控制不同 admission 长度的影响；但审稿人可能会问：如果按真实 hospital day 来看，长住院后期是否也存在同样趋势？这张补充图就是为回答这个问题而设计。

**实验目的：**

该图验证 trajectory drift 是否也体现在绝对时间尺度上，即 day `<=7`、`8-14`、`15-28`、`>28`。如果 base 在绝对 day 后期也下降，而 V2 在后期收益更大，就说明 drift 不只是相对分箱造成的统计现象，而是在真实长住院场景中也存在。

**读图方式：**

- x 轴是绝对 hospital day 分箱：`<=7`、`8-14`、`15-28`、`>28`。
- Panel A 的 y 轴是 absolute quality score。
- Panel B 的 y 轴是 `Quality improvement over base`。
- 折线越高表示该方法在对应 day bin 的 A&P 质量越好。
- 柱子越高表示该方法相对 base 的收益越大。

**具体观察：**

- Panel A 中，base quality 从 `day<=7` 的 `8.21` 降到 `>28` 的 `2.65`，说明长住院后期 base 退化非常明显。
- V2 在 `>28` day 的 quality 为 `4.65`，仍高于 base，但绝对质量并不理想，说明单纯 V2 只能部分缓解长住院后期退化。
- V2 + judge-revise 在 `>28` day 的 quality 达到 `7.04`，明显高于 V2 和 base。
- Panel B 中，V2 的 improvement 从 early 的 `-0.30` 增至 `>28` day 的 `+2.00`。
- V2 + judge-revise 的 improvement 从 early 的 `-0.42` 增至 `>28` day 的 `+3.84`，说明 revision 对长住院后期尤其重要。

**具体结论：**

这张图说明 trajectory drift 也存在于绝对 hospital day 维度：base 在长住院后期显著退化。V2 对后期有正向缓解作用，而 V2 + judge-revise 在 long-admission setting 中收益最大。

同时，这张图也提示一个限制：V2 自身在 `>28` day 虽然优于 base，但绝对质量仍低于早期病例，因此不能宣称 V2 完全解决了长住院后期生成问题。

**论文中可写的 claim：**

> When stratified by absolute hospital day, base quality drops sharply in long admissions. V2 provides increasing gains for later hospital-day bins, and judge-revise is particularly beneficial for notes after day 28, suggesting that revision is important for long-context clinical trajectory maintenance.

### 补充图：Trajectory capture benefit

![Trajectory capture benefit](C:/Users/dsw54/Desktop/codex_related/flow_ehr/analysis/trajectory_drift_v2/paper_figures/fig_trajectory_capture_benefit.png)

路径：

- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_capture_benefit.pdf`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_capture_benefit.svg`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_capture_benefit.png`

**设计意图：**

主图已经证明了 base 的 trajectory capture 会随 admission 进展下降，但还需要额外验证：V2 的收益是否真的体现在 `trajectory_capture` 这个核心指标上，而不是只提升了综合 quality。因此这张图专门把 trajectory capture 单独拿出来，对比 base、V2 和 V2 + judge-revise。

**实验目的：**

该图用于回答一个更直接的问题：优化后是否提升了模型对患者纵向病程轨迹的捕捉能力？如果 V2/V2 + judge-revise 在 trajectory capture 上也优于 base，且后期收益更大，就可以更有力地支持“V2 缓解 trajectory drift”的方法 claim。

**读图方式：**

- Panel A：按 admission 内相对进度比较 base、V2、V2 + judge-revise 的 trajectory capture score。
- Panel B：按 admission 内相对进度比较 `trajectory improvement over base = method trajectory capture - base trajectory capture`。
- Panel C：按绝对 hospital day 比较 trajectory capture score。
- Panel D：按绝对 hospital day 比较 trajectory improvement over base。
- y 轴中的 `Trajectory score` 越高越好；`Improvement vs. base` 高于 0 表示方法优于 base。

**具体观察：**

- 按 admission 相对进度看，V2 在 early 阶段 trajectory improvement 为 `-0.04`，基本不优于 base；但 mid-early、mid-late、late 分别提升到 `+0.38`、`+0.46`、`+0.48`。
- V2 + judge-revise 在所有阶段均不低于 base，early 为 `+0.03`，late 达到 `+0.65`。
- 按绝对 hospital day 看，V2 在 `<=7`、`8-14`、`15-28`、`>28` 的 trajectory improvement 分别为 `+0.15`、`+0.40`、`+0.45`、`+0.38`。
- V2 + judge-revise 的提升更明显，分别为 `+0.19`、`+0.44`、`+0.77`、`+0.62`，其中 `15-28` 和 `>28` day 的收益最大。
- Panel C 中 base trajectory capture 从 `<=7` 的 `2.71` 降到 `>28` 的 `2.17`，而 V2 + judge-revise 在 `>28` day 仍保持 `2.76`。

**具体结论：**

这张图直接证明：V2 的优化不仅提升综合 quality，也确实提升了 trajectory capture。更重要的是，trajectory capture 的收益主要出现在 admission 中后期和长住院后期，这与本文的 longitudinal drift motivation 一致。

同时，V2 + judge-revise 在 trajectory capture 上明显强于单独 V2，说明 revision 模块不仅修正局部文本，也能增强病程轨迹的一致性。

**论文中可写的 claim：**

> Direct trajectory-capture analysis shows that V2 improves longitudinal tracking over base, with larger gains in later admission stages and later hospital-day bins. The judge-revise module further improves trajectory capture, indicating that revision contributes to maintaining clinical trajectory consistency rather than only improving surface-level A&P quality.

### 补充图：Within-admission slope analysis

![Slope figure](C:/Users/dsw54/Desktop/codex_related/flow_ehr/analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_within_admission_slopes.png)

路径：

- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_within_admission_slopes.pdf`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_within_admission_slopes.svg`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_within_admission_slopes.png`

**设计意图：**

前两张图是分箱平均结果，可能受到 admission 组成差异影响。为了进一步确认趋势不是由不同 admission 的 case mix 造成，这张图改为 admission-level 分析：每个 admission 单独拟合随 day 变化的趋势，再统计趋势方向。

**实验目的：**

该图用于验证“同一个 admission 内，指标是否随时间系统性变化”。这比全局分箱更严格，因为它避免了把不同 admission 混在一起后产生的混杂。

**计算方式：**

对每个 admission 和每个指标，拟合一条线性趋势：

```text
metric = a * hospital_day + b
```

其中：

- `a > 0`：该 admission 内该指标随 day 上升。
- `a < 0`：该 admission 内该指标随 day 下降。

图中的 y 轴是 `a > 0` 的 admission 比例。

**读图方式：**

- x 轴是被分析的趋势指标：
  - `Base quality`：base 综合质量是否随 day 上升。
  - `Output quality`：V2 或 V2 + judge-revise 的综合质量是否随 day 上升。
  - `Improvement over base`：方法相对 base 的优势是否随 day 上升。
  - `Unsupported problems`：不被证据支持的问题数是否随 day 上升。
- y 轴是正斜率 admission 的比例。
- 虚线 50% 表示多数/少数分界，高于 50% 说明多数 admission 中该指标随 day 上升。

**具体观察：**

- V2 设置下，只有 `32.0%` 的 admission 中 base quality 是正斜率；换句话说，约 `68.0%` 的 admission 中 base quality 随 day 下降。
- V2 设置下，`78.7%` 的 admission 中 quality improvement 是正斜率，说明多数 admission 里 V2 相对 base 的优势越到后期越大。
- V2 + judge-revise 设置下，base quality 正斜率比例为 `28.8%`，仍说明 base 在多数 admission 中后期退化。
- V2 + judge-revise 的 improvement 正斜率比例为 `72.9%`，说明 judge-revise 后的方法优势仍主要体现在 admission 后期。
- 但 unsupported problems 的正斜率比例较高：V2 为 `69.3%`，V2 + judge-revise 为 `74.6%`。这说明虽然质量相对 base 改善，但后期 unsupported/stale problem carry-over 仍然是主要残留错误。

**具体结论：**

这张图从 admission-level trend 的角度支持主结论：base 的质量在多数 admission 中随时间下降，而 V2 的相对优势在多数 admission 中随时间增加。因此，V2 的效果确实与 longitudinal drift mitigation 相关，而不是单纯由某些后期病例带来的平均值偏差。

同时，它也揭示当前方法的主要缺陷：unsupported problem count 也倾向于随 admission 进展上升，说明后期 problem-list hallucination/stale carry-over 没有被完全解决。

**论文中可写的 claim：**

> Admission-level slope analysis confirms that the improvement is not merely a population-level averaging artifact. In most admissions, base quality decreases over time, while V2's improvement over base increases over time. However, unsupported problem counts also tend to increase, highlighting stale or unsupported problem carry-over as the main remaining failure mode.

## 旧版快速检查图

旧版快速检查图仍保留在 `analysis/trajectory_drift_v2/figures/`，但后续论文或汇报建议使用 `paper_figures/` 下的新图。

### Base 后期退化

![Base drift](C:/Users/dsw54/Desktop/codex_related/flow_ehr/analysis/trajectory_drift_v2/figures/base_drift_quality_trajectory.png)

**指标含义：**

- `Base quality`：base 输出的综合质量分数，计算为正向指标之和减去负向错误项，越高越好。
- `Base trajectory`：base 输出对患者纵向病程变化的捕捉能力，越高表示越能正确延续/更新 admission 内的临床轨迹。

**坐标轴：**

- 左图 x 轴：同一个 admission 内的相对进度分箱，`Early -> Mid-early -> Mid-late -> Late` 表示从住院早期到后期。
- 右图 x 轴：绝对住院日分箱，`<=7, 8-14, 15-28, >28`。
- y 轴：对应评估分数，越高越好。

**图中含义：**

- 灰线表示 base 的综合 A&P 质量。
- 紫线表示 base 的 trajectory capture。

**结论：**

这张图最直接验证 base 的 trajectory drift：无论按 admission 内相对进度，还是按绝对 hospital day，base 的 quality 和 trajectory capture 都有后期下滑趋势。

### V2 相对进度趋势

![Quality by relative progress](C:/Users/dsw54/Desktop/codex_related/flow_ehr/analysis/trajectory_drift_v2/figures/quality_by_relative_progress.png)

**指标含义：**

- `Base`：原始 base A&P 生成结果的综合质量分数。
- `V2`：使用 scaffold/memory 后生成的 A&P 综合质量分数。
- `Judge-revise`：在 V2 基础上经过 judge-revise 后的综合质量分数。
- `Improvement`：当前方法相对 base 的提升，计算为 `method quality - base quality`。柱子高于 0 表示方法优于 base，低于 0 表示弱于 base。

**坐标轴：**

- x 轴：同一个 admission 内的相对进度分箱。
  - `Early`：该 admission 内较早的 note。
  - `Mid-early`：中前期。
  - `Mid-late`：中后期。
  - `Late`：该 admission 内较晚的 note。
- y 轴：综合质量分数和相对提升值。折线越高表示 A&P 质量越好；绿色柱子越高表示相对 base 的收益越大。

**图中含义：**

- 灰线：base 的质量。
- 蓝线：V2 的质量。
- 橙线：V2 judge-revise 的质量。
- 绿色柱：相对 base 的 improvement。

**结论：**

这张图展示 V2/V2 judge-revise 与 base 在 admission 早期、中期、后期的质量对比。V2 在 early 阶段不占优，但越到后期，相对 base 的 improvement 越明显，说明 V2 对 admission 后期的 trajectory drift 有缓解作用。

### 绝对 hospital day 趋势

![Quality by absolute day](C:/Users/dsw54/Desktop/codex_related/flow_ehr/analysis/trajectory_drift_v2/figures/quality_by_absolute_day.png)

**指标含义：**

- `Base`：原始 base A&P 生成结果的综合质量分数。
- `V2`：scaffold/memory 方法的综合质量分数。
- `Judge-revise`：V2 加 judge-revise 后的综合质量分数。
- `Improvement`：当前方法相对 base 的质量提升，计算为 `method quality - base quality`。

**坐标轴：**

- x 轴：绝对住院日分箱。
  - `<=7`：住院 7 天以内。
  - `8-14`：住院第 8-14 天。
  - `15-28`：住院第 15-28 天。
  - `>28`：住院超过 28 天。
- y 轴：综合质量分数和相对提升值。折线越高表示 A&P 质量越好；绿色柱子越高表示相对 base 的收益越大。

**图中含义：**

- 灰线：base 质量。
- 蓝线：V2 质量。
- 橙线：V2 judge-revise 质量。
- 绿色柱：相对 base 的 improvement。

**结论：**

这张图说明长住院后期仍然更难：base 随 hospital day 明显退化；V2 相对 base 的提升随 day 增大，但 V2 自身的绝对质量没有单调上升。judge-revise 对 `>28` day 的后期病例改善最明显。

### Admission 内斜率

![Within-admission slope](C:/Users/dsw54/Desktop/codex_related/flow_ehr/analysis/trajectory_drift_v2/figures/within_admission_positive_slope.png)

**指标含义：**

- `Base quality`：每个 admission 内，base 综合质量随 day 变化的斜率。
- `Output quality`：每个 admission 内，V2 或 V2 judge-revise 综合质量随 day 变化的斜率。
- `Improvement`：每个 admission 内，方法相对 base 的提升随 day 变化的斜率。
- `Unsupported`：每个 admission 内，unsupported problem count 随 day 变化的斜率。这个指标是负向指标，越高表示越多不被证据支持的问题。

**坐标轴：**

- x 轴：不同的 admission 内趋势指标。
- y 轴：正斜率 admission 的比例。比如 `70%` 表示 70% 的 admission 中，该指标随着住院进展是上升的。

**图中含义：**

- 灰柱：base quality 正斜率比例。
- 蓝柱/橙柱：V2 或 V2 judge-revise 输出质量正斜率比例。
- 绿柱：相对 base 的 improvement 正斜率比例。
- 红柱：unsupported problem count 正斜率比例。
- 虚线 `50%`：作为多数/少数分界线，高于该线表示多数 admission 呈上升趋势。

**结论：**

这张图汇总每个 admission 内的斜率方向。base quality 多数 admission 是负斜率，而 V2/V2 judge-revise 的 improvement 多数为正斜率，支持“V2 缓解 longitudinal drift”的结论；但 unsupported 的正斜率仍然较高，说明后期幻觉/旧问题继承还没有彻底解决。

## 数据范围

- V2 full 评估：653 条，91 个 admission，day 2-58
- V2 judge-revise 评估：515 条，72 个 admission，day 2-58
- 分析维度：
  - 绝对住院日 `day`
  - admission 内相对进度 `relative_progress`
  - 每个 admission 内的线性斜率
  - V2 输出本身指标
  - V2 相对 base 的 improvement

## 全局相关性

### V2

| 指标 | abs day corr | relative progress corr | 解读 |
|---|---:|---:|---|
| V2 trajectory | -0.087 | 0.101 | 绝对 day 不升，admission 内后期略升 |
| base trajectory | -0.214 | -0.128 | base 后期明显变差 |
| trajectory improvement | 0.111 | 0.215 | V2 后期相对 base 提升更明显 |
| V2 quality sum | -0.193 | 0.012 | V2 绝对质量没有随后期稳定提升 |
| base quality sum | -0.344 | -0.223 | base 后期退化明显 |
| quality improvement | 0.190 | 0.282 | V2 后期更能抵消 base 退化 |
| V2 unsupported | 0.271 | 0.088 | unsupported 后期仍增加 |
| V2 missed | 0.113 | -0.025 | missed 没有明显后期恶化 |

### V2 judge-revise

| 指标 | abs day corr | relative progress corr | 解读 |
|---|---:|---:|---|
| V2 judge-revise trajectory | -0.010 | 0.115 | 绝对 day 基本持平，admission 内略升 |
| base trajectory | -0.216 | -0.147 | base 后期仍变差 |
| trajectory improvement | 0.175 | 0.232 | judge-revise 后期相对提升更强 |
| V2 judge-revise quality sum | -0.073 | 0.020 | 绝对质量较 V2 更稳，但仍非明显上升 |
| base quality sum | -0.321 | -0.258 | base 后期退化明显 |
| quality improvement | 0.274 | 0.299 | 后期相对 base 的提升更大 |
| V2 judge-revise unsupported | 0.144 | 0.111 | unsupported 仍然后期增加，但弱于 V2 |
| V2 judge-revise missed | 0.053 | -0.038 | missed 后期略好或基本持平 |

## 按 admission 相对进度分箱

### V2

| admission 阶段 | n | V2 quality | base quality | improvement | V2 win rate |
|---|---:|---:|---:|---:|---:|
| early | 199 | 6.83 | 8.50 | -1.67 | 28.1% |
| mid_early | 150 | 6.55 | 5.01 | +1.54 | 47.3% |
| mid_late | 123 | 6.20 | 4.68 | +1.52 | 48.0% |
| late | 181 | 6.83 | 5.02 | +1.81 | 49.2% |

V2 在 early 阶段明显输给 base，但从 mid_early 开始转为正收益，late 阶段 improvement 最大。这个现象支持“V2 对后期轨迹偏移有缓解”。

但是 V2 自身 quality 从 early 的 6.83 到 late 的 6.83，并没有持续上升。提升主要来自 base 后期变差，而不是 V2 后期越来越好。

### V2 judge-revise

| admission 阶段 | n | V2 judge-revise quality | base quality | improvement | V2 win rate |
|---|---:|---:|---:|---:|---:|
| early | 157 | 7.82 | 9.32 | -1.50 | 38.9% |
| mid_early | 123 | 8.10 | 6.36 | +1.74 | 50.4% |
| mid_late | 97 | 7.67 | 5.93 | +1.74 | 53.6% |
| late | 138 | 8.00 | 5.38 | +2.62 | 58.0% |

judge-revise 后的趋势更清楚：late 阶段 improvement 最大，win rate 达到 58.0%。这说明 revised 模块确实增强了后期相对优势。

但绝对质量仍只是从 7.82 到 8.00，提升有限；不能据此声称“同一 admission 越后期越好”。

## 按绝对 day 分箱

### V2

| day bin | n | V2 quality | base quality | improvement | V2 unsupported | V2 missed |
|---|---:|---:|---:|---:|---:|---:|
| day<=7 | 298 | 7.91 | 8.21 | -0.30 | 3.20 | 3.23 |
| 8-14 | 174 | 6.01 | 5.07 | +0.94 | 4.16 | 3.63 |
| 15-28 | 121 | 5.45 | 3.64 | +1.81 | 4.60 | 3.60 |
| >28 | 60 | 4.65 | 2.65 | +2.00 | 4.67 | 3.87 |

按绝对 day 看，V2 的相对提升越来越大，但 V2 自身 quality 反而下降，unsupported 和 missed 都偏高。这说明长住院后期仍然更难，V2 没有彻底解决累积性偏移。

### V2 judge-revise

| day bin | n | V2 judge-revise quality | base quality | improvement | V2 unsupported | V2 missed |
|---|---:|---:|---:|---:|---:|---:|
| day<=7 | 239 | 8.71 | 9.13 | -0.42 | 3.14 | 3.03 |
| 8-14 | 135 | 7.05 | 5.64 | +1.41 | 4.04 | 3.48 |
| 15-28 | 96 | 7.51 | 4.95 | +2.56 | 4.04 | 3.09 |
| >28 | 45 | 7.04 | 3.20 | +3.84 | 3.76 | 3.56 |

judge-revise 明显改善了后期绝对质量，尤其 >28 days 的 quality 从 V2 的 4.65 提到 7.04，unsupported 也从 4.67 降到 3.76。但 day<=7 仍然比 base 略差，且 >28 days 的 missed 仍高于早期。

## Admission 内斜率

### V2

| admission 内指标 | admission 数 | 平均斜率 | 中位斜率 | 正斜率比例 |
|---|---:|---:|---:|---:|
| V2 trajectory | 75 | +0.192 | +0.086 | 65.3% |
| base trajectory | 75 | -0.017 | -0.036 | 38.7% |
| trajectory improvement | 75 | +0.209 | +0.067 | 72.0% |
| V2 quality | 75 | +0.796 | +0.150 | 58.7% |
| base quality | 75 | -0.463 | -0.500 | 32.0% |
| quality improvement | 75 | +1.259 | +0.642 | 78.7% |
| V2 unsupported | 75 | +0.150 | +0.100 | 69.3% |
| V2 missed | 75 | -0.225 | 0.000 | 50.7% |

同 admission 内，V2 的 trajectory/quality 斜率多数为正，相对 base 的 improvement 斜率更明显为正。这是支持 V2 缓解轨迹偏移的最强证据。

但 unsupported 的正斜率比例也达到 69.3%，说明越到后期越容易积累不支持问题。V2 解决的是“遗漏/trajectory continuity”的一部分，而不是“unsupported hallucination”的全部。

### V2 judge-revise

| admission 内指标 | admission 数 | 平均斜率 | 中位斜率 | 正斜率比例 |
|---|---:|---:|---:|---:|
| V2 judge-revise trajectory | 59 | +0.210 | +0.056 | 67.8% |
| base trajectory | 59 | -0.006 | -0.008 | 42.4% |
| trajectory improvement | 59 | +0.216 | +0.071 | 71.2% |
| V2 judge-revise quality | 59 | +0.644 | +0.096 | 54.2% |
| base quality | 59 | -0.728 | -0.500 | 28.8% |
| quality improvement | 59 | +1.372 | +0.489 | 72.9% |
| V2 judge-revise unsupported | 59 | +0.170 | +0.107 | 74.6% |
| V2 judge-revise missed | 59 | -0.228 | 0.000 | 52.5% |

judge-revise 让相对 improvement 更强，但 unsupported 的 admission 内正斜率仍然存在。这说明当前 revised 模块能局部修正，但没有从 problem-list 层面阻止错误问题随时间被继承或放大。

## 可以写进论文的表述

可以说：

> V2 substantially mitigates longitudinal drift relative to the base generator. Within the same admission, the improvement over base increases in later notes, especially for trajectory capture and overall quality. However, V2 does not fully eliminate drift: absolute quality does not monotonically improve with hospital day, and unsupported problem counts still tend to increase in later notes.

不建议说：

> V2 solves trajectory drift.

更稳妥的中文表述是：

> V2 缓解了轨迹偏移，尤其体现在 admission 后期相对 base 的收益更大；但它尚未完全解决后期累积性幻觉和 problem-list 偏移。

## 方法层面的解释

V2 为什么会在后期相对 base 更好：

1. Scaffold/历史记忆让模型不完全依赖当日 note 的局部线索，因此能保留 longitudinal trajectory。
2. 后期病例中 base 更容易丢失或混淆历史问题，V2 的结构化上下文能减少这类退化。
3. judge-revise 对后期尤其有帮助，因为后期 A&P 更容易出现局部不一致，revision 能补一部分遗漏和 grounding。

V2 为什么还不能完全解决：

1. 后期 admission 的 active problem list 更复杂，历史问题和当前问题边界更模糊。
2. Scaffold 会带来继承风险：旧问题如果没有被明确下线，可能继续污染后续输出。
3. 当前 verifier/reviser 更偏 claim-level，没有足够强的 problem-level gating。
4. unsupported problem count 后期上升，说明错误问题不是简单的表述错误，而是 active problem selection 错误。

## 后续建议

为了让“解决轨迹偏移”的 claim 更 solid，下一步建议做三类实验：

1. Admission-level trajectory analysis：按 admission 画 base、V2、judge-revise 的 daily quality curve，并报告 within-admission slope 显著性。
2. Problem-list drift analysis：单独评估 active problem 的 carry-over、drop、reactivation、retirement，而不只评估最终 A&P 文本。
3. Problem-level verifier ablation：加入 active/inactive/uncertain problem gate，验证是否能降低后期 unsupported 的正斜率。

当前数据最支持的论文 claim 是：

> V2 improves longitudinal robustness and reduces relative drift, but remaining errors concentrate in problem-list hallucination and stale problem carry-over.
