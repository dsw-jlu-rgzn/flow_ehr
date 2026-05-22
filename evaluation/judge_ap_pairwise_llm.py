"""Pairwise LLM judge for AP outputs against gold and current-day evidence."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.deepseek_api_generation import DEFAULT_API_URL, call_deepseek


def parse_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.S | re.I)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_gold(data_root: Path, hadm_id: str, day: int) -> str:
    df = pd.read_csv(data_root / "gold" / f"gt_{hadm_id}.csv")
    rows = df[df["DAY"].astype(int).eq(day)]
    return "\n".join(rows["TEXT"].fillna("").astype(str).tolist())


def load_input(data_root: Path, hadm_id: str, day: int, max_chars: int) -> str:
    df = pd.read_csv(data_root / "input" / f"input_{hadm_id}.csv")
    rows = df[df["DAY"].astype(int).eq(day)]
    lines = []
    for idx, row in rows.iterrows():
        text = str(row.get("TEXT", ""))
        if len(text) > 900:
            text = text[:900].rstrip() + " ...[truncated]"
        lines.append(
            f"[row_{idx:04d}] REL_TIME={row.get('REL_TIME','')} IS_NOTE={row.get('IS_NOTE','')}\n{text}"
        )
    out = "\n\n".join(lines)
    return out[:max_chars] + "\n...[truncated]" if len(out) > max_chars else out


def build_prompt(
    case_id: str,
    current_input: str,
    gold: str,
    output_a: str,
    output_b: str,
) -> str:
    return f"""You are an expert ICU Assessment & Plan evaluator.

Compare Output A and Output B against the gold A&P and current-day evidence.
Evaluate clinical quality, not lexical overlap. Reward correct active problem
coverage, trajectory capture, specific plans, evidence grounding, and disposition
context. Penalize unsupported diagnoses/treatments, stale carry-forward, missed
key active problems, and wrong problem trajectory.

Return exactly one valid JSON object. Use 1-5 scores where 5 is best.
unsupported_problem_count and missed_key_problem_count are nonnegative integers.

Schema:
{{
  "case_id": "{case_id}",
  "scores": {{
    "A": {{
      "active_problem_coverage": 0,
      "trajectory_capture": 0,
      "plan_specificity": 0,
      "evidence_grounding": 0,
      "disposition_context": 0,
      "unsupported_problem_count": 0,
      "missed_key_problem_count": 0,
      "overall_quality": 0
    }},
    "B": {{
      "active_problem_coverage": 0,
      "trajectory_capture": 0,
      "plan_specificity": 0,
      "evidence_grounding": 0,
      "disposition_context": 0,
      "unsupported_problem_count": 0,
      "missed_key_problem_count": 0,
      "overall_quality": 0
    }}
  }},
  "winner": "A|B|tie",
  "rationale": "one short sentence"
}}

Current-day evidence:
{current_input}

Gold A&P:
{gold}

Output A:
{output_a}

Output B:
{output_b}
"""


def numeric(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def judge_case(case: dict[str, Any], idx: int, args: argparse.Namespace) -> dict[str, Any]:
    case_id = case["case_id"]
    hadm_id = str(case["admission_id"])
    day = int(case["day"])
    a_file = Path(args.method_a_dir) / f"{case_id}.txt"
    b_file = Path(args.method_b_dir) / f"{case_id}.txt"
    if not a_file.exists():
        raise FileNotFoundError(a_file)
    if not b_file.exists():
        raise FileNotFoundError(b_file)

    output_a_text = read_text(a_file)
    output_b_text = read_text(b_file)
    label_a, label_b = args.method_a_name, args.method_b_name
    if idx % 2 == 0:
        output_a_text, output_b_text = output_b_text, output_a_text
        label_a, label_b = label_b, label_a

    prompt = build_prompt(
        case_id,
        load_input(Path(args.data_root), hadm_id, day, args.max_evidence_chars),
        load_gold(Path(args.data_root), hadm_id, day),
        output_a_text,
        output_b_text,
    )
    response = call_deepseek(
        prompt=prompt,
        model=args.model,
        api_url=args.api_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
        api_key_env=args.api_key_env,
    )
    repair_used = False
    raw_before_repair = ""
    try:
        parsed = parse_json(response)
    except Exception:
        repair_used = True
        raw_before_repair = response
        repair_prompt = f"""Repair the following AP judge response into exactly one valid JSON object.

Do not change scores or winner if present. If the response is empty or unusable,
return a conservative tie with all 1-5 scores set to 3 and count metrics set to 0.
Return JSON only.

Required case_id: {case_id}

Required schema:
{{
  "case_id": "{case_id}",
  "scores": {{
    "A": {{
      "active_problem_coverage": 3,
      "trajectory_capture": 3,
      "plan_specificity": 3,
      "evidence_grounding": 3,
      "disposition_context": 3,
      "unsupported_problem_count": 0,
      "missed_key_problem_count": 0,
      "overall_quality": 3
    }},
    "B": {{
      "active_problem_coverage": 3,
      "trajectory_capture": 3,
      "plan_specificity": 3,
      "evidence_grounding": 3,
      "disposition_context": 3,
      "unsupported_problem_count": 0,
      "missed_key_problem_count": 0,
      "overall_quality": 3
    }}
  }},
  "winner": "tie",
  "rationale": "fallback repair"
}}

Response:
{response}
"""
        response = call_deepseek(
            prompt=repair_prompt,
            model=args.repair_model or args.model,
            api_url=args.api_url,
            temperature=0.0,
            max_tokens=args.max_tokens,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
            api_key_env=args.api_key_env,
        )
        parsed = parse_json(response)
    raw_winner = str(parsed.get("winner", "tie")).strip().lower()
    if raw_winner == "a":
        winner = label_a
    elif raw_winner == "b":
        winner = label_b
    else:
        winner = "tie"
    row: dict[str, Any] = {
        "case_id": case_id,
        "admission_id": hadm_id,
        "day": day,
        "A_label": label_a,
        "B_label": label_b,
        "winner": winner,
        "raw_winner": raw_winner,
        "rationale": parsed.get("rationale", ""),
        "judge_repair_used": repair_used,
        "judge_raw_before_repair": raw_before_repair,
        "raw_response": response,
    }
    scores = parsed.get("scores", {})
    metrics = [
        "active_problem_coverage",
        "trajectory_capture",
        "plan_specificity",
        "evidence_grounding",
        "disposition_context",
        "unsupported_problem_count",
        "missed_key_problem_count",
        "overall_quality",
    ]
    for ab, label in [("A", label_a), ("B", label_b)]:
        score = scores.get(ab, {})
        for metric in metrics:
            row[f"{label}_{metric}"] = numeric(score.get(metric))
    return row


def write_summary(rows: list[dict[str, Any]], output_csv: Path, method_names: list[str]) -> None:
    df = pd.DataFrame(rows)
    summary_rows = []
    for method in method_names:
        item = {"method": method, "wins": int((df["winner"] == method).sum())}
        for metric in [
            "active_problem_coverage",
            "trajectory_capture",
            "plan_specificity",
            "evidence_grounding",
            "disposition_context",
            "unsupported_problem_count",
            "missed_key_problem_count",
            "overall_quality",
        ]:
            col = f"{method}_{metric}"
            item[metric] = float(df[col].mean()) if col in df else 0.0
        summary_rows.append(item)
    summary_rows.append({"method": "tie", "wins": int((df["winner"] == "tie").sum())})
    summary_csv = output_csv.with_name(output_csv.stem + "_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Summary: {summary_csv}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise AP LLM judge.")
    parser.add_argument("--selected", default=str(Path("outputs/oracle_claim_verifier_qwen653/selected_cases_with_failure_seed.json")))
    parser.add_argument("--method-a-dir", required=True)
    parser.add_argument("--method-b-dir", required=True)
    parser.add_argument("--method-a-name", default="v2")
    parser.add_argument("--method-b-name", default="truth_revised")
    parser.add_argument("--data-root", default="data_ap100_ap/AP")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-evidence-chars", type=int, default=14000)
    parser.add_argument("--model", default="deepseek-pro")
    parser.add_argument("--repair-model", default="")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    args = parser.parse_args()

    selected = json.loads(Path(args.selected).read_text(encoding="utf-8"))
    if args.limit:
        selected = selected[: args.limit]
    rows = []
    for idx, case in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] judging {case['case_id']}")
        rows.append(judge_case(case, idx, args))
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Detail: {output_csv}")
    write_summary(rows, output_csv, [args.method_a_name, args.method_b_name])


if __name__ == "__main__":
    main()
