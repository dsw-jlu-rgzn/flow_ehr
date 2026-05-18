"""
Generate AP notes with problem-state scaffold augmentation.

This script keeps the direct-generation evidence surface intact: the LLM still
receives the same current-day EHR and previous-note context as the matching
direct baseline. The only additional input is the no-training problem-state
scaffold extracted by ap_problem_state_experiment.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from ap_problem_state_experiment import DEFAULT_CASES, parse_cases, rouge_l_f1
from deepseek_api_generation import AP_INSTRUCTION_1, AP_INSTRUCTION_2, DEFAULT_API_URL, call_deepseek, df2chron_str


def method_folder(method: str) -> str:
    return method


def read_previous_context(
    data_root: Path,
    baseline_run_name: str,
    baseline_setting: str,
    baseline_method: str,
    admission_id: str,
    day: int,
) -> str:
    if baseline_method == "method-1":
        return ""

    if baseline_setting == "gt":
        gold_df = pd.read_csv(data_root / "AP" / "gold" / f"gt_{admission_id}.csv")
        previous = gold_df[gold_df["DAY"].astype(int).lt(day)].sort_values("DAY")
        if previous.empty:
            return ""
        notes = previous["TEXT"].fillna("").astype(str).tolist()
    else:
        gen_df = pd.read_csv(
            data_root
            / "AP"
            / "generated"
            / "DG"
            / baseline_run_name
            / baseline_setting
            / baseline_method
            / f"genpns_{admission_id}.csv"
        )
        previous = gen_df[gen_df["DAY"].astype(int).lt(day)].sort_values("DAY")
        if previous.empty:
            return ""
        notes = previous["TEXT"].fillna("").astype(str).tolist()

    if baseline_method == "method1":
        return notes[-1]
    if baseline_method == "method2":
        return "\n".join(notes)
    raise ValueError(f"Unsupported method: {baseline_method}")


def read_case_inputs(args: argparse.Namespace, admission_id: str, day: int) -> dict:
    data_root = Path(args.data_root)
    input_df = pd.read_csv(data_root / "AP" / "input" / f"input_{admission_id}.csv")
    current_rows = input_df[(input_df["DAY"].astype(int).eq(day)) & (input_df["IS_NOTE"].astype(int).eq(0))]
    current_ehr = df2chron_str(current_rows)
    previous_context = read_previous_context(
        data_root=data_root,
        baseline_run_name=args.baseline_run_name,
        baseline_setting=args.baseline_setting,
        baseline_method=args.baseline_method,
        admission_id=admission_id,
        day=day,
    )
    update_path = (
        Path(args.problem_state_dir)
        / args.memory_source
        / args.baseline_method
        / f"{admission_id}_day{day}"
        / "problem_update.json"
    )
    problem_update = json.loads(update_path.read_text(encoding="utf-8"))

    gold_df = pd.read_csv(data_root / "AP" / "gold" / f"gt_{admission_id}.csv")
    gold_rows = gold_df[gold_df["DAY"].astype(int).eq(day)]
    gold_note = " ".join(gold_rows["TEXT"].fillna("").astype(str).tolist())

    baseline_df = pd.read_csv(
        data_root
        / "AP"
        / "generated"
        / "DG"
        / args.baseline_run_name
        / args.baseline_setting
        / args.baseline_method
        / f"genpns_{admission_id}.csv"
    )
    baseline_rows = baseline_df[baseline_df["DAY"].astype(int).eq(day)]
    baseline_note = " ".join(baseline_rows["TEXT"].fillna("").astype(str).tolist())

    return {
        "current_ehr": current_ehr,
        "previous_context": previous_context,
        "problem_update": problem_update,
        "gold_note": gold_note,
        "baseline_note": baseline_note,
    }


def build_prompt(case_inputs: dict) -> str:
    ehr_str = case_inputs["current_ehr"]
    if case_inputs["previous_context"]:
        ehr_str += "\n\nPrevious progress note context:\n" + case_inputs["previous_context"]

    scaffold = json.dumps(case_inputs["problem_update"], ensure_ascii=False, indent=2)
    return f"""{AP_INSTRUCTION_1}{ehr_str}

Problem-state scaffold:
The scaffold is a planning aid extracted from prior note context and today's EHR.
Use it to preserve active problems, daily trajectory, and plan constraints, but
do not copy unsupported items if the raw EHR contradicts them.

{scaffold}

{AP_INSTRUCTION_2}
"""


def generate_case(args: argparse.Namespace, admission_id: str, day: int) -> dict:
    out_dir = Path(args.output_dir) / args.config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    case_file = out_dir / f"{admission_id}_day{day}.txt"

    case_inputs = read_case_inputs(args, admission_id, day)
    if case_file.exists():
        generated = case_file.read_text(encoding="utf-8")
    else:
        prompt = build_prompt(case_inputs)
        generated = call_deepseek(
            prompt=prompt,
            model=args.model,
            api_url=args.api_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
        )
        case_file.write_text(generated, encoding="utf-8")

    return {
        "config": args.config_name,
        "admission_id": admission_id,
        "day": day,
        "baseline_run_name": args.baseline_run_name,
        "baseline_setting": args.baseline_setting,
        "baseline_method": args.baseline_method,
        "memory_source": args.memory_source,
        "baseline_rouge_l": rouge_l_f1(case_inputs["baseline_note"], case_inputs["gold_note"]),
        "augmented_rouge_l": rouge_l_f1(generated, case_inputs["gold_note"]),
        "baseline_words": len(case_inputs["baseline_note"].split()),
        "augmented_words": len(generated.split()),
        "gold_words": len(case_inputs["gold_note"].split()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AP with problem-state scaffold augmentation.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--problem-state-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/ap_problem_state_augmented")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--baseline-run-name", required=True)
    parser.add_argument("--baseline-setting", choices=["gt", "gen"], required=True)
    parser.add_argument("--baseline-method", choices=["method1", "method2", "method-1"], required=True)
    parser.add_argument("--memory-source", choices=["gold", "baseline_method", "none"], required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--sleep-seconds", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for admission_id, day in parse_cases(args.cases):
        print(f"Generating {args.config_name}: {admission_id} day {day}")
        rows.append(generate_case(args, admission_id, day))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.config_name}_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
