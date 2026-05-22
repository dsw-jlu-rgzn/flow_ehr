"""Judge direct AP baseline vs augmented AP generations with an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling.deepseek_api_generation import DEFAULT_API_URL, call_deepseek


def parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def judge_pair(gold: str, baseline: str, augmented: str, args: argparse.Namespace) -> dict:
    prompt = f"""Compare two generated ICU Assessment & Plan notes against the gold progress note.

Score each candidate from 1 to 5 on:
- active_problem_coverage
- trajectory_capture
- plan_specificity
- evidence_grounding
- disposition_context

Also provide unsupported_problem_count and missed_key_problem_count. Judge
clinically; do not reward boilerplate unrelated to A&P.
Keep each brief_rationale to one short sentence.

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
  "augmented": {{
    "active_problem_coverage": 1,
    "trajectory_capture": 1,
    "plan_specificity": 1,
    "evidence_grounding": 1,
    "disposition_context": 1,
    "unsupported_problem_count": 0,
    "missed_key_problem_count": 0,
    "brief_rationale": ""
  }},
  "winner": "baseline|augmented|tie"
}}

Gold note:
{gold}

Baseline generation:
{baseline}

Augmented generation:
{augmented}
"""
    last_error = None
    last_response = ""
    for attempt in range(1, args.parse_retries + 1):
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
        last_response = response
        try:
            return parse_json_response(response)
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt < args.parse_retries:
                time.sleep(args.sleep_seconds * attempt)
    preview = last_response[:300].replace("\n", "\\n")
    raise RuntimeError(f"Judge JSON parse failed after {args.parse_retries} attempts: {last_error}; response={preview!r}")


def normalize_winner(value: object) -> str:
    winner = str(value or "").strip().lower()
    if winner in {"baseline", "augmented", "tie"}:
        return winner
    if "baseline" in winner and "augmented" not in winner:
        return "baseline"
    if "augmented" in winner and "baseline" not in winner:
        return "augmented"
    return "tie"


def judge_row(row: dict, args: argparse.Namespace) -> dict:
    data_root = Path(args.data_root)
    admission_id = str(row["admission_id"])
    day = int(row["day"])
    gold_df = pd.read_csv(data_root / "AP" / "gold" / f"gt_{admission_id}.csv")
    gold = " ".join(gold_df[gold_df["DAY"].astype(int).eq(day)]["TEXT"].fillna("").astype(str).tolist())

    baseline_df = pd.read_csv(
        data_root
        / "AP"
        / "generated"
        / "DG"
        / str(row["baseline_run_name"])
        / str(row["baseline_setting"])
        / str(row["baseline_method"])
        / f"genpns_{admission_id}.csv"
    )
    baseline = " ".join(
        baseline_df[baseline_df["DAY"].astype(int).eq(day)]["TEXT"].fillna("").astype(str).tolist()
    )
    augmented = (Path(args.augmented_dir) / str(row["config"]) / f"{admission_id}_day{day}.txt").read_text(
        encoding="utf-8"
    )

    judged = judge_pair(gold, baseline, augmented, args)
    flat = dict(row)
    flat["raw_winner"] = judged.get("winner", "")
    flat["winner"] = normalize_winner(judged.get("winner", ""))
    for candidate in ["baseline", "augmented"]:
        for metric, value in judged.get(candidate, {}).items():
            if isinstance(value, (str, int, float)):
                flat[f"{candidate}_{metric}"] = value
    return flat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail-csv", default="outputs/ap_problem_state_augmented/augmented_fair_detail.csv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--augmented-dir", default="outputs/ap_problem_state_augmented")
    parser.add_argument("--output-csv", default="outputs/ap_problem_state_augmented/augmented_judge_detail.csv")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument(
        "--api-key-env",
        default="DEEPSEEK_API_KEY",
        help="Environment variable containing the API key for the selected OpenAI-compatible endpoint.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--parse-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=6.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel judge API workers. Keep 1 for sequential runs.",
    )
    args = parser.parse_args()

    detail = pd.read_csv(args.detail_csv)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    if output_path.exists():
        existing = pd.read_csv(output_path)
        rows = existing.to_dict("records")
        done = {(str(r["config"]), str(r["admission_id"]), int(r["day"])) for r in rows}
    else:
        done = set()

    pending = []
    for _, row in detail.iterrows():
        key = (str(row["config"]), str(row["admission_id"]), int(row["day"]))
        if key not in done:
            pending.append(row.to_dict())

    if args.workers <= 1:
        for row in pending:
            print(f"Judging {row['config']} {row['admission_id']} day {row['day']}")
            rows.append(judge_row(row, args))
            pd.DataFrame(rows).to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    else:
        print(f"Judging {len(pending)} rows with {args.workers} workers")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_key = {
                executor.submit(judge_row, row, args): (row["config"], row["admission_id"], row["day"])
                for row in pending
            }
            completed = 0
            for future in as_completed(future_to_key):
                config, admission_id, day = future_to_key[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    print(f"Failed judging {config} {admission_id} day {day}: {exc}")
                    raise
                completed += 1
                print(f"Completed judge {completed}/{len(pending)}: {config} {admission_id} day {day}")
                pd.DataFrame(rows).to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
