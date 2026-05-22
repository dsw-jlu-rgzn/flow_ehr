"""Pairwise LLM judge for discharge summaries.

Compares two DS outputs against the gold DS and admission evidence. The two
methods are anonymized as A/B with deterministic alternation to reduce position
bias. The judge returns JSON scores and a preference.
"""

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

from modeling.deepseek_api_generation import call_deepseek
from scripts.run_ds_minimal_closed_loop import compact_chronology_for_prompt, read_chronology


def admission_id_from_output(path: Path) -> str:
    match = re.match(r"48h_all_abs_(\d+)\.txt$", path.name)
    if not match:
        raise ValueError(path.name)
    return match.group(1)


def strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S).strip()


def parse_json(text: str) -> dict[str, Any]:
    raw = strip_fences(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise


def build_prompt(
    admission_id: str,
    chronology: str,
    gold: str,
    output_a: str,
    output_b: str,
    max_evidence_words: int,
) -> str:
    evidence = compact_chronology_for_prompt(chronology, max_evidence_words)
    return f"""You are an expert clinical documentation evaluator.

Compare two generated discharge summaries against the gold discharge summary
and the admission evidence. The outputs are anonymized as Output A and Output B.

Evaluate clinical quality, not just wording. Reward summaries that are faithful,
complete, clinically specific, temporally coherent, and discharge-plan accurate.
Penalize unsupported diagnoses, unsupported medications, invented follow-up,
missed major diagnoses/procedures/events, wrong resolved/unresolved status, and
wrong discharge disposition/medications/instructions.

Use 1-5 scores where 5 is best. unsupported_claim_count and
missed_major_event_count are nonnegative integers.

Return one valid JSON object only. Do not use markdown fences.

Required JSON schema:
{{
  "admission_id": "{admission_id}",
  "scores": {{
    "A": {{
      "diagnosis_coverage": 0,
      "hospital_course_completeness": 0,
      "temporal_order_correctness": 0,
      "discharge_plan_correctness": 0,
      "evidence_grounding": 0,
      "unsupported_claim_count": 0,
      "missed_major_event_count": 0,
      "overall_quality": 0
    }},
    "B": {{
      "diagnosis_coverage": 0,
      "hospital_course_completeness": 0,
      "temporal_order_correctness": 0,
      "discharge_plan_correctness": 0,
      "evidence_grounding": 0,
      "unsupported_claim_count": 0,
      "missed_major_event_count": 0,
      "overall_quality": 0
    }}
  }},
  "winner": "A | B | tie",
  "rationale": "one short sentence without quotation marks or newlines"
}}

Admission evidence:
{evidence}

Gold discharge summary:
{gold}

Output A:
{output_a}

Output B:
{output_b}
"""


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def numeric(obj: Any, default: float = 0.0) -> float:
    try:
        return float(obj)
    except Exception:
        return default


def run_case(
    admission_id: str,
    args: argparse.Namespace,
    method_a_text: str,
    method_b_text: str,
    chronology: str,
    gold: str,
    a_is_method_a: bool,
) -> dict[str, Any]:
    if a_is_method_a:
        output_a, output_b = method_a_text, method_b_text
        label_a, label_b = args.method_a_name, args.method_b_name
    else:
        output_a, output_b = method_b_text, method_a_text
        label_a, label_b = args.method_b_name, args.method_a_name

    prompt = build_prompt(admission_id, chronology, gold, output_a, output_b, args.max_evidence_words)
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
    try:
        parsed = parse_json(response)
    except Exception:
        repair_prompt = f"""Repair the following text into one valid JSON object.

Do not change the scores or winner. Return JSON only. Remove invalid quotation
marks/newlines inside string values if needed.

Text:
{response}
"""
        response = call_deepseek(
            prompt=repair_prompt,
            model=args.model,
            api_url=args.api_url,
            temperature=0.0,
            max_tokens=args.max_tokens,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
            api_key_env=args.api_key_env,
        )
        parsed = parse_json(response)
    scores = parsed.get("scores", {})
    winner_ab = str(parsed.get("winner", "tie")).strip().lower()
    if winner_ab == "a":
        winner = label_a
    elif winner_ab == "b":
        winner = label_b
    else:
        winner = "tie"

    row: dict[str, Any] = {
        "admission_id": admission_id,
        "A_label": label_a,
        "B_label": label_b,
        "winner": winner,
        "raw_winner": winner_ab,
        "rationale": parsed.get("rationale", ""),
        "raw_response": response,
    }
    for ab, label in [("A", label_a), ("B", label_b)]:
        s = scores.get(ab, {})
        prefix = label
        for metric in [
            "diagnosis_coverage",
            "hospital_course_completeness",
            "temporal_order_correctness",
            "discharge_plan_correctness",
            "evidence_grounding",
            "unsupported_claim_count",
            "missed_major_event_count",
            "overall_quality",
        ]:
            row[f"{prefix}_{metric}"] = numeric(s.get(metric))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise LLM judge for DS outputs.")
    parser.add_argument("--method-a-dir", required=True)
    parser.add_argument("--method-b-dir", required=True)
    parser.add_argument("--method-a-name", default="base")
    parser.add_argument("--method-b-name", default="ours")
    parser.add_argument("--input-dir", default="data/DS_fixed_composed/full/input")
    parser.add_argument("--gold-dir", default="data/DS_fixed_composed/full/gold")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-evidence-words", type=int, default=12000)
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--api-url", default="https://api.siliconflow.cn/v1/chat/completions")
    parser.add_argument("--api-key-env", default="QWEN_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    args = parser.parse_args()

    method_a_dir = Path(args.method_a_dir)
    method_b_dir = Path(args.method_b_dir)
    input_dir = Path(args.input_dir)
    gold_dir = Path(args.gold_dir)
    files = sorted(method_a_dir.glob("48h_all_abs_*.txt"))
    if args.limit:
        files = files[: args.limit]

    rows = []
    for idx, a_file in enumerate(files, start=1):
        admission_id = admission_id_from_output(a_file)
        b_file = method_b_dir / a_file.name
        input_file = input_dir / f"full_both_{admission_id}.csv"
        gold_file = gold_dir / f"gtsummary_{admission_id}.txt"
        if not b_file.exists() or not input_file.exists() or not gold_file.exists():
            continue
        print(f"[{idx}/{len(files)}] HADM_ID={admission_id}")
        _, chronology = read_chronology(input_file)
        a_is_method_a = idx % 2 == 1
        row = run_case(
            admission_id,
            args,
            read_text(a_file),
            read_text(b_file),
            chronology,
            read_text(gold_file),
            a_is_method_a,
        )
        rows.append(row)
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_csv, index=False)

    print(f"Detail CSV: {args.output_csv}")
    if rows:
        df = pd.DataFrame(rows)
        print(df["winner"].value_counts(dropna=False).to_string())
        for method in [args.method_a_name, args.method_b_name]:
            cols = [c for c in df.columns if c.startswith(method + "_") and c != method + "_raw"]
            print(f"\n{method}:")
            for col in cols:
                print(f"  {col.removeprefix(method + '_')}: {df[col].mean():.2f}")


if __name__ == "__main__":
    main()
