# Paper-Style Trajectory Drift Figures

## Reproduction

Run from the repository root:

```bash
python scripts/plot_trajectory_drift_paper_figures.py
```

Input tables:

- `analysis/trajectory_drift_v2/V2_by_relative_progress_bin.csv`
- `analysis/trajectory_drift_v2/V2_judge_revise_by_relative_progress_bin.csv`
- `analysis/trajectory_drift_v2/V2_by_abs_day_bin.csv`
- `analysis/trajectory_drift_v2/V2_judge_revise_by_abs_day_bin.csv`
- `analysis/trajectory_drift_v2/V2_slope_summary.csv`
- `analysis/trajectory_drift_v2/V2_judge_revise_slope_summary.csv`

Output directory:

- `analysis/trajectory_drift_v2/paper_figures/`

Each figure is exported as `.pdf`, `.svg`, and `.png`. Use `.pdf` or `.svg` for paper submission and `.png` for quick preview.

## Figure 1. Longitudinal trajectory drift and V2 mitigation

Path:

- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_main.pdf`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_main.svg`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_main.png`

Caption:

**Figure 1. Longitudinal degradation of base A&P generation and mitigation by V2.**  
        (A) Base model performance across within-admission progress bins. The x-axis groups notes from early to late within the same admission (`Mid-E` = mid-early; `Mid-L` = mid-late). The y-axis reports the LLM-based evaluation score. `Base quality` is the aggregate A&P quality score; `Base trajectory` measures whether the generated A&P correctly captures the patient's longitudinal clinical trajectory. Both curves decline after the early stage, indicating longitudinal degradation.  
(B) Quality improvement over base, defined as `method quality - base quality`. Positive values indicate that the method outperforms base. Both V2 and V2 + judge-revise show larger relative gains in later admission stages, suggesting that the proposed scaffold/revision pipeline mitigates late-admission drift.  
(C) Absolute quality scores for base, V2, and V2 + judge-revise. V2 stabilizes late-stage quality relative to base, while judge-revise provides additional gains.

Suggested in-text description:

> We evaluate whether generation quality degrades as admissions progress. The base generator shows a clear drop in both aggregate A&P quality and trajectory capture from early to later admission stages. In contrast, the proposed V2 pipeline yields increasingly positive improvements over base in later stages, indicating mitigation of longitudinal trajectory drift rather than merely uniform performance gains.

## Figure 2. Absolute hospital-day analysis

Path:

- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_absolute_day.pdf`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_absolute_day.svg`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_absolute_day.png`

Caption:

**Figure 2. Trajectory drift measured by absolute hospital day.**  
(A) Quality scores are grouped by absolute hospital-day bins. The x-axis denotes hospital day ranges (`<=7`, `8-14`, `15-28`, `>28`), and the y-axis reports aggregate A&P quality. The base generator declines substantially for later hospital days, while V2 and V2 + judge-revise maintain higher quality.  
(B) Improvement over base increases for later hospital-day bins, especially for V2 + judge-revise, showing that the method is most beneficial in long-admission settings.

## Figure 3. Within-admission slope analysis

Path:

- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_within_admission_slopes.pdf`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_within_admission_slopes.svg`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_drift_within_admission_slopes.png`

Caption:

**Figure 3. Admission-level trend direction analysis.**  
For each admission, we fit a simple linear trend between hospital day and each evaluation metric, then report the percentage of admissions with a positive slope. The x-axis denotes the metric whose within-admission trend is measured, and the y-axis denotes the percentage of admissions where that metric increases over time. `Improvement over base` has a positive slope in most admissions, indicating that V2's advantage grows later within the same admission. However, `Unsupported problems` also frequently has a positive slope, showing that stale or unsupported problem carry-over remains a key failure mode.

## Metric definitions

- `Quality score`: aggregate A&P quality score derived from the LLM evaluation dimensions. Higher is better.
- `Trajectory capture`: score for whether the generated A&P follows the patient's longitudinal clinical trajectory. Higher is better.
- `Improvement over base`: `method quality - base quality`. Higher means the method improves more over the base generator.
- `Unsupported problems`: count of generated problems not supported by evidence. Lower is better.
- `Positive slope admissions (%)`: percentage of admissions where a metric increases as hospital day increases within that admission.

## Figure 4. Trajectory capture benefit

Path:

- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_capture_benefit.pdf`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_capture_benefit.svg`
- `analysis/trajectory_drift_v2/paper_figures/fig_trajectory_capture_benefit.png`

Caption:

**Figure 4. Direct comparison of trajectory capture across methods.**  
(A) Trajectory capture scores across within-admission progress bins. The x-axis groups notes from early to late within the same admission, and the y-axis reports the trajectory capture score. V2 and V2 + judge-revise outperform base after the early stage.  
(B) Trajectory improvement over base, defined as `method trajectory capture - base trajectory capture`, across within-admission progress bins. Positive values indicate improved longitudinal trajectory tracking. Gains are larger in later admission stages.  
(C) Trajectory capture scores by absolute hospital-day bins. V2 and V2 + judge-revise maintain higher trajectory capture than base in later hospital-day bins.  
(D) Trajectory improvement over base by absolute hospital-day bins. The benefit is strongest for later hospital days, particularly for V2 + judge-revise.

Suggested in-text description:

> To verify that the observed gains are not limited to aggregate quality, we separately evaluate trajectory capture. V2 improves trajectory capture over base after the early stage, and the improvement is larger in later admission stages and later hospital-day bins. This directly supports the claim that the scaffold/revision pipeline mitigates longitudinal trajectory drift.
