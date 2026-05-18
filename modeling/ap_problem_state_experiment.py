"""
No-training problem-state AP experiment.

This experiment targets a failure mode seen in low-scoring AP generations:
the model often loses the carried-forward active problem list and writes from
only the current day's sparse EHR rows. The pipeline is dataset-agnostic: it
uses an LLM to extract problem memory and daily evidence from generic
time-stamped clinical text instead of hard-coded disease or medication rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from deepseek_api_generation import DEFAULT_API_URL, call_deepseek, df2chron_str


DEFAULT_CASES = [
    "177997:3",
    "177997:4",
    "196033:4",
    "196033:5",
    "196033:6",
    "196033:7",
    "196033:8",
    "115691:3",
    "126929:3",
    "126929:4",
]


def parse_json_response(text: str) -> Any:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_json(prompt: str, args: argparse.Namespace, max_tokens: int | None = None) -> Any:
    response = call_deepseek(
        prompt=prompt,
        model=args.model,
        api_url=args.api_url,
        temperature=args.temperature,
        max_tokens=max_tokens or args.max_tokens,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
    )
    return parse_json_response(response)


def call_text(prompt: str, args: argparse.Namespace, max_tokens: int | None = None) -> str:
    return call_deepseek(
        prompt=prompt,
        model=args.model,
        api_url=args.api_url,
        temperature=args.temperature,
        max_tokens=max_tokens or args.generation_max_tokens,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
    )


def read_case(
    data_root: Path,
    admission_id: str,
    day: int,
    baseline_run_name: str,
    baseline_setting: str,
    baseline_method: str,
    memory_source: str,
) -> dict[str, str]:
    input_df = pd.read_csv(data_root / "AP" / "input" / f"input_{admission_id}.csv")
    gold_df = pd.read_csv(data_root / "AP" / "gold" / f"gt_{admission_id}.csv")
    baseline_df = pd.read_csv(
        data_root
        / "AP"
        / "generated"
        / "DG"
        / baseline_run_name
        / baseline_setting
        / baseline_method
        / f"genpns_{admission_id}.csv"
    )

    previous_note = ""
    if memory_source == "gold":
        previous_gold = gold_df[gold_df["DAY"].astype(int).lt(day)].sort_values("DAY")
        if not previous_gold.empty:
            previous_note = str(previous_gold.iloc[-1]["TEXT"])
    elif memory_source == "baseline_method":
        previous_generated = baseline_df[baseline_df["DAY"].astype(int).lt(day)].sort_values("DAY")
        if not previous_generated.empty:
            previous_note = str(previous_generated.iloc[-1]["TEXT"])
    elif memory_source == "none":
        previous_note = ""
    else:
        raise ValueError(f"Unknown memory source: {memory_source}")

    current_rows = input_df[(input_df["DAY"].astype(int).eq(day)) & (input_df["IS_NOTE"].astype(int).eq(0))]
    current_ehr = df2chron_str(current_rows)

    gold_rows = gold_df[gold_df["DAY"].astype(int).eq(day)]
    gold_note = " ".join(gold_rows["TEXT"].fillna("").astype(str).tolist())

    baseline_rows = baseline_df[baseline_df["DAY"].astype(int).eq(day)]
    baseline_note = " ".join(baseline_rows["TEXT"].fillna("").astype(str).tolist())

    return {
        "previous_note": previous_note,
        "current_ehr": current_ehr,
        "gold_note": gold_note,
        "baseline_note": baseline_note,
    }


def extract_problem_memory(previous_note: str, args: argparse.Namespace) -> dict:
    prompt = f"""Extract a portable clinical problem memory from the prior ICU progress note.

Do not assume any dataset-specific schema. Use only the note text. Capture active
problems, unresolved plans, disposition constraints, and uncertainties that
should carry forward to the next day's Assessment & Plan.

Return only valid JSON:
{{
  "active_problems": [
    {{
      "problem": "short clinical problem name",
      "status": "new|improving|worsening|stable|resolved|unclear",
      "evidence": ["brief quote or paraphrase from prior note"],
      "prior_plan": ["actionable prior plan items"],
      "confidence": "high|medium|low"
    }}
  ],
  "global_context": {{
    "disposition": "",
    "code_status": "",
    "major_events": []
  }},
  "uncertainties": []
}}

Prior note:
{previous_note}
"""
    return call_json(prompt, args)


def extract_daily_evidence(current_ehr: str, problem_memory: dict, args: argparse.Namespace) -> dict:
    prompt = f"""Extract evidence from today's EHR that can change an ICU Assessment & Plan.

This must be dataset-agnostic: do not rely on fixed disease names, medication
lists, or lab names. Select evidence because it changes diagnosis, severity,
treatment, monitoring, disposition, or uncertainty.

Use the problem memory as context, but allow new problems and contradictions.
Keep the extraction compact: at most 12 evidence_items; each evidence string
must be one concise sentence.

Return only valid JSON:
{{
  "evidence_items": [
    {{
      "time": "",
      "evidence": "concise evidence statement",
      "relevance": "new_problem|status_change|treatment_change|monitoring|disposition|contradiction|uncertainty",
      "linked_problem": "problem name or new problem",
      "confidence": "high|medium|low"
    }}
  ],
  "major_trajectory": ["up to 5 one-line trajectory statements"],
  "possible_new_problems": [],
  "missing_information": []
}}

Problem memory:
{json.dumps(problem_memory, ensure_ascii=False)}

Today's EHR:
{current_ehr}
"""
    return call_json(prompt, args, max_tokens=args.evidence_max_tokens)


def update_problem_state(problem_memory: dict, daily_evidence: dict, args: argparse.Namespace) -> dict:
    prompt = f"""Update the active problem state for today's ICU Assessment & Plan.

For each carried-forward or new problem, decide whether it is improving,
worsening, stable, resolved, or unclear. Every plan action must be grounded in
prior problem memory or today's evidence. If evidence is sparse, preserve
important unresolved prior problems rather than inventing a new primary problem.
Keep the representation compact: include at most 8 updated_problems; each
problem may have at most 2 supporting_evidence items and 2 plan_actions.

Return only valid JSON:
{{
  "updated_problems": [
    {{
      "problem": "short clinical problem name",
      "today_status": "new|improving|worsening|stable|resolved|unclear",
      "assessment": "one concise sentence",
      "supporting_evidence": ["evidence item"],
      "plan_actions": ["specific action or monitoring item"],
      "confidence": "high|medium|low"
    }}
  ],
  "global_plan": {{
    "disposition": "",
    "code_status": "",
    "monitoring_priorities": []
  }},
  "do_not_overstate": []
}}

Problem memory:
{json.dumps(problem_memory, ensure_ascii=False)}

Daily evidence:
{json.dumps(daily_evidence, ensure_ascii=False)}
"""
    return call_json(prompt, args, max_tokens=args.update_max_tokens)


def generate_ap(problem_update: dict, args: argparse.Namespace) -> str:
    prompt = f"""Write today's ICU Assessment & Plan from this structured problem-state update.

Requirements:
- Use only the provided problem-state update.
- Preserve carried-forward active problems when evidence is sparse.
- State today's trajectory when known.
- Make plans specific but do not invent unsupported treatments.
- Be concise and clinically grounded.

Problem-state update:
{json.dumps(problem_update, ensure_ascii=False)}

Output format:
Assessment:
...

Plan:
...
"""
    return call_text(prompt, args)


def judge_case(baseline_note: str, experiment_note: str, gold_note: str, args: argparse.Namespace) -> dict:
    prompt = f"""Compare two generated ICU Assessment & Plan notes against the gold progress note.

Score each candidate from 1 to 5 on:
- active_problem_coverage: covers the gold note's active clinical problems
- trajectory_capture: captures daily change/status and not just static diagnoses
- plan_specificity: gives concrete plan actions supported by evidence
- evidence_grounding: avoids unsupported diagnoses or invented treatments
- disposition_context: captures disposition/code/status/global care context when relevant

Also provide unsupported_problem_count and missed_key_problem_count. Judge
clinically; do not reward copying boilerplate sections that are not A&P content.

Return only valid JSON:
{{
  "baseline": {{
    "active_problem_coverage": 1,
    "trajectory_capture": 1,
    "plan_specificity": 1,
    "evidence_grounding": 1,
    "disposition_context": 1,
    "unsupported_problem_count": 0,
    "missed_key_problem_count": 0,
    "brief_rationale": ""
  }},
  "problem_state": {{
    "active_problem_coverage": 1,
    "trajectory_capture": 1,
    "plan_specificity": 1,
    "evidence_grounding": 1,
    "disposition_context": 1,
    "unsupported_problem_count": 0,
    "missed_key_problem_count": 0,
    "brief_rationale": ""
  }},
  "winner": "baseline|problem_state|tie"
}}

Gold note:
{gold_note}

Baseline generation:
{baseline_note}

Problem-state generation:
{experiment_note}
"""
    return call_json(prompt, args)


def rouge_l_f1(pred: str, gold: str) -> float:
    pred_tokens = str(pred).lower().split()
    gold_tokens = str(gold).lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    prev = [0] * (len(gold_tokens) + 1)
    for token in pred_tokens:
        curr = [0]
        for idx, gold_token in enumerate(gold_tokens, start=1):
            curr.append(prev[idx - 1] + 1 if token == gold_token else max(prev[idx], curr[-1]))
        prev = curr
    lcs = prev[-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def case_dir_name(admission_id: str, day: int) -> str:
    return f"{admission_id}_day{day}"


def load_or_run(path: Path, runner):
    if path.exists():
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return path.read_text(encoding="utf-8")
    try:
        result = runner()
    except Exception as exc:
        error_path = path.with_suffix(path.suffix + ".error.txt")
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(str(exc), encoding="utf-8")
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        path.write_text(str(result), encoding="utf-8")
    return result


def run_case(admission_id: str, day: int, args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir) / args.memory_source / args.baseline_method / case_dir_name(admission_id, day)
    data = read_case(
        Path(args.data_root),
        admission_id,
        day,
        args.baseline_run_name,
        args.baseline_setting,
        args.baseline_method,
        args.memory_source,
    )

    memory = load_or_run(out_dir / "problem_memory.json", lambda: extract_problem_memory(data["previous_note"], args))
    evidence = load_or_run(
        out_dir / "daily_evidence.json",
        lambda: extract_daily_evidence(data["current_ehr"], memory, args),
    )
    update = load_or_run(out_dir / "problem_update.json", lambda: update_problem_state(memory, evidence, args))
    generated = load_or_run(out_dir / "generated_ap.txt", lambda: generate_ap(update, args))
    judge = load_or_run(
        out_dir / "judge_metrics.json",
        lambda: judge_case(data["baseline_note"], generated, data["gold_note"], args),
    )

    metrics = {
        "admission_id": admission_id,
        "day": day,
        "baseline_method": args.baseline_method,
        "baseline_run_name": args.baseline_run_name,
        "baseline_setting": args.baseline_setting,
        "memory_source": args.memory_source,
        "baseline_rouge_l": rouge_l_f1(data["baseline_note"], data["gold_note"]),
        "problem_state_rouge_l": rouge_l_f1(generated, data["gold_note"]),
        "baseline_words": len(data["baseline_note"].split()),
        "problem_state_words": len(generated.split()),
        "gold_words": len(data["gold_note"].split()),
        "winner": judge.get("winner", ""),
    }
    for candidate in ["baseline", "problem_state"]:
        scores = judge.get(candidate, {})
        for key, value in scores.items():
            if isinstance(value, (int, float, str)):
                metrics[f"{candidate}_{key}"] = value
    return metrics


def parse_cases(values: list[str]) -> list[tuple[str, int]]:
    cases = []
    for value in values:
        admission_id, day = value.split(":", 1)
        cases.append((admission_id.strip(), int(day)))
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run no-training problem-state AP experiment.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs/ap_problem_state_experiment")
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES, help="Case specs like 177997:4")
    parser.add_argument("--baseline-method", default="method-1", choices=["method-1", "method1", "method2"])
    parser.add_argument("--baseline-run-name", default="deepseek_api_full")
    parser.add_argument("--baseline-setting", default="gt", choices=["gt", "gen"])
    parser.add_argument("--memory-source", default="gold", choices=["gold", "baseline_method", "none"])
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--evidence-max-tokens", type=int, default=2600)
    parser.add_argument("--update-max-tokens", type=int, default=3200)
    parser.add_argument("--generation-max-tokens", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--sleep-seconds", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for admission_id, day in parse_cases(args.cases):
        print(f"Running {admission_id} day {day}")
        rows.append(run_case(admission_id, day, args))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
