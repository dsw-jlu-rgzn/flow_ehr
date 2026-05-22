from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis" / "trajectory_drift_v2"
FIG_DIR = ANALYSIS_DIR / "paper_figures"


COLORS = {
    "base": "#4D4D4D",
    "v2": "#2B6CB0",
    "judge": "#C65F1A",
    "improve": "#2F855A",
    "trajectory": "#6B5FB5",
    "unsupported": "#B83232",
    "grid": "#D8D8D8",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 10,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "v2_rel": pd.read_csv(ANALYSIS_DIR / "V2_by_relative_progress_bin.csv"),
        "jr_rel": pd.read_csv(ANALYSIS_DIR / "V2_judge_revise_by_relative_progress_bin.csv"),
        "v2_abs": pd.read_csv(ANALYSIS_DIR / "V2_by_abs_day_bin.csv"),
        "jr_abs": pd.read_csv(ANALYSIS_DIR / "V2_judge_revise_by_abs_day_bin.csv"),
        "v2_slope": pd.read_csv(ANALYSIS_DIR / "V2_slope_summary.csv"),
        "jr_slope": pd.read_csv(ANALYSIS_DIR / "V2_judge_revise_slope_summary.csv"),
    }


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(FIG_DIR / f"{stem}.{suffix}", dpi=450)
    plt.close(fig)


def plot_line(ax: plt.Axes, x: np.ndarray, y: pd.Series, label: str, color: str, marker: str = "o") -> None:
    ax.plot(
        x,
        y,
        color=color,
        marker=marker,
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.2,
        linewidth=2.0,
        label=label,
    )


def main_figure(tables: dict[str, pd.DataFrame]) -> None:
    rel_labels = ["Early", "Mid-E", "Mid-L", "Late"]
    x = np.arange(len(rel_labels))
    v2 = tables["v2_rel"]
    jr = tables["jr_rel"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.35),
        gridspec_kw={"width_ratios": [1.0, 1.05, 1.05], "wspace": 0.38},
    )

    ax = axes[0]
    plot_line(ax, x, v2["base_quality"], "Base quality", COLORS["base"])
    plot_line(ax, x, v2["base_trajectory_capture"], "Base trajectory", COLORS["trajectory"], marker="s")
    ax.set_xticks(x, rel_labels)
    ax.set_ylabel("Evaluation score")
    ax.set_xlabel("Within-admission progress")
    ax.set_title("Base degrades")
    ax.set_ylim(1.7, 9.2)
    ax.legend(frameon=False, loc="upper right", handlelength=1.6)
    clean_axes(ax)
    panel_label(ax, "A")

    ax = axes[1]
    width = 0.36
    ax.axhline(0, color="#8A8A8A", linewidth=0.8)
    ax.bar(x - width / 2, v2["quality_improvement"], width=width, color=COLORS["v2"], alpha=0.85, label="V2")
    ax.bar(
        x + width / 2,
        jr["quality_improvement"],
        width=width,
        color=COLORS["judge"],
        alpha=0.85,
        label="V2 + judge-revise",
    )
    ax.set_xticks(x, rel_labels)
    ax.set_ylabel("Quality improvement over base")
    ax.set_xlabel("Within-admission progress")
    ax.set_title("Gains increase later")
    ax.legend(frameon=False, loc="upper left")
    clean_axes(ax)
    panel_label(ax, "B")

    ax = axes[2]
    plot_line(ax, x, v2["aug_quality"], "V2", COLORS["v2"])
    plot_line(ax, x, jr["aug_quality"], "V2 + judge-revise", COLORS["judge"])
    plot_line(ax, x, v2["base_quality"], "Base", COLORS["base"])
    ax.set_xticks(x, rel_labels)
    ax.set_ylabel("Quality score")
    ax.set_xlabel("Within-admission progress")
    ax.set_title("Quality is stabilized")
    ax.set_ylim(4.1, 9.8)
    ax.legend(frameon=False, loc="upper right", handlelength=1.6)
    clean_axes(ax)
    panel_label(ax, "C")

    save_all(fig, "fig_trajectory_drift_main")


def supplemental_absolute_day(tables: dict[str, pd.DataFrame]) -> None:
    labels = [r"$\leq$7", "8-14", "15-28", r"$>$28"]
    x = np.arange(len(labels))
    v2 = tables["v2_abs"]
    jr = tables["jr_abs"]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.4), gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    plot_line(ax, x, v2["base_quality"], "Base", COLORS["base"])
    plot_line(ax, x, v2["aug_quality"], "V2", COLORS["v2"])
    plot_line(ax, x, jr["aug_quality"], "V2 + judge-revise", COLORS["judge"])
    ax.set_xticks(x, labels)
    ax.set_xlabel("Hospital day bin")
    ax.set_ylabel("Quality score")
    ax.set_title("Quality by absolute hospital day")
    ax.legend(frameon=False, loc="upper right")
    clean_axes(ax)
    panel_label(ax, "A")

    ax = axes[1]
    width = 0.36
    ax.axhline(0, color="#8A8A8A", linewidth=0.8)
    ax.bar(x - width / 2, v2["quality_improvement"], width=width, color=COLORS["v2"], alpha=0.85, label="V2")
    ax.bar(
        x + width / 2,
        jr["quality_improvement"],
        width=width,
        color=COLORS["judge"],
        alpha=0.85,
        label="V2 + judge-revise",
    )
    ax.set_xticks(x, labels)
    ax.set_xlabel("Hospital day bin")
    ax.set_ylabel("Quality improvement over base")
    ax.set_title("Later notes show larger relative gains")
    ax.legend(frameon=False, loc="upper left")
    clean_axes(ax)
    panel_label(ax, "B")

    save_all(fig, "fig_trajectory_drift_absolute_day")


def supplemental_slopes(tables: dict[str, pd.DataFrame]) -> None:
    metrics = [
        ("quality_sum_base", "Base\nquality", COLORS["base"]),
        ("quality_sum_aug", "Output\nquality", COLORS["v2"]),
        ("quality_sum_improvement", "Improvement\nover base", COLORS["improve"]),
        ("augmented_unsupported_problem_count", "Unsupported\nproblems", COLORS["unsupported"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55), sharey=True, gridspec_kw={"wspace": 0.18})
    for ax, df, title, output_color, label in [
        (axes[0], tables["v2_slope"], "V2", COLORS["v2"], "A"),
        (axes[1], tables["jr_slope"], "V2 + judge-revise", COLORS["judge"], "B"),
    ]:
        lookup = df.set_index("slope_metric")
        values = [lookup.loc[m[0], "positive_frac"] * 100 for m in metrics]
        colors = [m[2] for m in metrics]
        colors[1] = output_color
        x = np.arange(len(metrics))
        ax.bar(x, values, color=colors, width=0.66)
        ax.axhline(50, color="#666666", linestyle=(0, (3, 2)), linewidth=0.8)
        ax.set_xticks(x, [m[1] for m in metrics])
        ax.set_ylim(0, 100)
        ax.set_title(title)
        ax.set_xlabel("Within-admission trend metric")
        for xi, yi in zip(x, values):
            ax.text(xi, yi + 2.0, f"{yi:.1f}", ha="center", va="bottom", fontsize=7.5)
        clean_axes(ax)
        panel_label(ax, label)

    axes[0].set_ylabel("Admissions with positive slope (%)")
    save_all(fig, "fig_trajectory_drift_within_admission_slopes")


def supplemental_trajectory_capture(tables: dict[str, pd.DataFrame]) -> None:
    rel_labels = ["Early", "Mid-E", "Mid-L", "Late"]
    abs_labels = [r"$\leq$7", "8-14", "15-28", r"$>$28"]
    rel_x = np.arange(len(rel_labels))
    abs_x = np.arange(len(abs_labels))
    v2_rel = tables["v2_rel"]
    jr_rel = tables["jr_rel"]
    v2_abs = tables["v2_abs"]
    jr_abs = tables["jr_abs"]

    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.05), gridspec_kw={"hspace": 0.72, "wspace": 0.46})

    ax = axes[0, 0]
    plot_line(ax, rel_x, v2_rel["base_trajectory_capture"], "Base", COLORS["base"])
    plot_line(ax, rel_x, v2_rel["aug_trajectory_capture"], "V2", COLORS["v2"])
    plot_line(ax, rel_x, jr_rel["aug_trajectory_capture"], "V2 + judge-revise", COLORS["judge"])
    ax.set_xticks(rel_x, rel_labels)
    ax.set_xlabel("Within-admission progress")
    ax.set_ylabel("Trajectory score")
    ax.set_title("Trajectory capture by admission progress")
    ax.set_ylim(2.0, 3.25)
    clean_axes(ax)
    panel_label(ax, "A")

    ax = axes[0, 1]
    width = 0.36
    ax.axhline(0, color="#8A8A8A", linewidth=0.8)
    ax.bar(
        rel_x - width / 2,
        v2_rel["imp_trajectory_capture"],
        width=width,
        color=COLORS["v2"],
        alpha=0.85,
        label="V2",
    )
    ax.bar(
        rel_x + width / 2,
        jr_rel["imp_trajectory_capture"],
        width=width,
        color=COLORS["judge"],
        alpha=0.85,
        label="V2 + judge-revise",
    )
    ax.set_xticks(rel_x, rel_labels)
    ax.set_xlabel("Within-admission progress")
    ax.set_ylabel("Improvement vs. base")
    ax.set_title("Trajectory gains increase later")
    clean_axes(ax)
    panel_label(ax, "B")

    ax = axes[1, 0]
    plot_line(ax, abs_x, v2_abs["base_trajectory_capture"], "Base", COLORS["base"])
    plot_line(ax, abs_x, v2_abs["aug_trajectory_capture"], "V2", COLORS["v2"])
    plot_line(ax, abs_x, jr_abs["aug_trajectory_capture"], "V2 + judge-revise", COLORS["judge"])
    ax.set_xticks(abs_x, abs_labels)
    ax.set_xlabel("Hospital day bin")
    ax.set_ylabel("Trajectory score")
    ax.set_title("Trajectory capture by hospital day")
    ax.set_ylim(2.0, 3.25)
    clean_axes(ax)
    panel_label(ax, "C")

    ax = axes[1, 1]
    ax.axhline(0, color="#8A8A8A", linewidth=0.8)
    ax.bar(
        abs_x - width / 2,
        v2_abs["imp_trajectory_capture"],
        width=width,
        color=COLORS["v2"],
        alpha=0.85,
        label="V2",
    )
    ax.bar(
        abs_x + width / 2,
        jr_abs["imp_trajectory_capture"],
        width=width,
        color=COLORS["judge"],
        alpha=0.85,
        label="V2 + judge-revise",
    )
    ax.set_xticks(abs_x, abs_labels)
    ax.set_xlabel("Hospital day bin")
    ax.set_ylabel("Improvement vs. base")
    ax.set_title("Later hospital days benefit more")
    clean_axes(ax)
    panel_label(ax, "D")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["base"],
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.2,
            linewidth=2.0,
            label="Base",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["v2"],
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.2,
            linewidth=2.0,
            label="V2",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["judge"],
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.2,
            linewidth=2.0,
            label="V2 + judge-revise",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        handlelength=1.7,
        columnspacing=1.8,
    )

    save_all(fig, "fig_trajectory_capture_benefit")


def write_caption_file() -> None:
    text = """# Paper-Style Trajectory Drift Figures

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
"""
    (FIG_DIR / "figure_captions_and_reproduction.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup_style()
    tables = load_tables()
    main_figure(tables)
    supplemental_absolute_day(tables)
    supplemental_slopes(tables)
    supplemental_trajectory_capture(tables)
    write_caption_file()
    print(f"Wrote paper figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
