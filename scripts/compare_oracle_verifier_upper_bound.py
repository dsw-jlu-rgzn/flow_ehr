from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_SELECTED = Path("outputs/oracle_claim_verifier_qwen653/selected_cases_with_failure_seed.json")
DEFAULT_V2_QWEN = Path(
    "outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2_eval_judge_qwen_siliconflow_detail.csv"
)
DEFAULT_JR_QWEN = Path(
    "outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2_judge_revise_eval_judge_qwen_siliconflow_detail.csv"
)
DEFAULT_ORACLE_JUDGE = Path(
    "outputs/oracle_claim_verifier_qwen653/llm_minimal_revision/llm_revise_eval_judge_detail.csv"
)
DEFAULT_OUTDIR = Path("outputs/oracle_claim_verifier_qwen653/llm_minimal_revision")


METRICS = [
    "active_problem_coverage",
    "trajectory_capture",
    "plan_specificity",
    "evidence_grounding",
    "disposition_context",
    "unsupported_problem_count",
    "missed_key_problem_count",
]


def key_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["case_id"] = out["admission_id"].astype(str) + "_day" + out["day"].astype(int).astype(str)
    return out


def load_method(path: Path, method_name: str, selected_cases: set[str]) -> pd.DataFrame:
    df = key_cols(pd.read_csv(path))
    df = df[df["case_id"].isin(selected_cases)].copy()
    rows = []
    for _, row in df.iterrows():
        item = {"case_id": row["case_id"], "method": method_name, "winner": row.get("winner", "")}
        for metric in METRICS:
            item[metric] = row.get(f"augmented_{metric}")
            item[f"baseline_{metric}"] = row.get(f"baseline_{metric}")
        rows.append(item)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in df.groupby("method", sort=False):
        row = {"method": method, "n": int(len(group))}
        for metric in METRICS:
            row[metric] = float(pd.to_numeric(group[metric], errors="coerce").mean())
        row["augmented_wins"] = int((group["winner"] == "augmented").sum())
        row["baseline_wins"] = int((group["winner"] == "baseline").sum())
        row["ties"] = int((group["winner"] == "tie").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown_table(df: pd.DataFrame) -> str:
    """Render a small markdown table without pandas' optional tabulate dependency."""
    cols = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        rendered = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                rendered.append(f"{value:.3f}")
            else:
                rendered.append(str(value))
        rows.append(rendered)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare oracle verifier upper-bound judge metrics with V2 baselines.")
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--v2-qwen", type=Path, default=DEFAULT_V2_QWEN)
    parser.add_argument("--judge-revise-qwen", type=Path, default=DEFAULT_JR_QWEN)
    parser.add_argument("--oracle-judge", type=Path, default=DEFAULT_ORACLE_JUDGE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    selected = json.loads(args.selected.read_text(encoding="utf-8"))
    selected_cases = {item["case_id"] for item in selected}

    frames = [load_method(args.v2_qwen, "V2", selected_cases)]
    if args.judge_revise_qwen.exists():
        frames.append(load_method(args.judge_revise_qwen, "V2 judge-revise", selected_cases))
    if args.oracle_judge.exists():
        frames.append(load_method(args.oracle_judge, "V2 + pseudo-oracle verifier + LLM revise", selected_cases))
    else:
        print(f"Oracle judge file not found yet: {args.oracle_judge}")

    detail = pd.concat(frames, ignore_index=True)
    summary = summarize(detail)

    args.outdir.mkdir(parents=True, exist_ok=True)
    detail_path = args.outdir / "upper_bound_comparison_detail.csv"
    summary_path = args.outdir / "upper_bound_comparison_summary.csv"
    report_path = args.outdir / "upper_bound_comparison_report.md"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    lines = ["# Oracle Verifier Upper-Bound Comparison", ""]
    lines.append(to_markdown_table(summary))
    lines.append("")
    lines.append("Lower is better for `unsupported_problem_count` and `missed_key_problem_count`; higher is better for the other metrics.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
