"""Run a 10-case DS minimal closed-loop experiment.

The script supports a dry-run mode that verifies data loading, case selection,
chunking, prompts, and output layout without calling an API. Actual generation
uses the existing OpenAI-compatible DeepSeek helper.
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

from modeling.deepseek_api_generation import DEFAULT_API_URL, call_deepseek


DEFAULT_INPUT_DIR = Path("data/DS_fixed_composed/full/input")
DEFAULT_GOLD_DIR = Path("data/DS_fixed_composed/full/gold")
DEFAULT_OUTPUT_DIR = Path("outputs/ds_minimal_closed_loop_10")


def word_count(text: str) -> int:
    return len(str(text).split())


def admission_id_from_input(path: Path) -> str:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Cannot parse admission id from {path.name}")
    return match.group(1)


def read_chronology(csv_path: Path) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    if "TEXT" not in df.columns:
        raise ValueError(f"{csv_path} does not contain TEXT column")
    if "TIME" in df.columns:
        df = df.sort_values("TIME")
    rows = []
    for _, row in df.iterrows():
        time = str(row.get("TIME", "") or "")
        rel_time = str(row.get("REL_TIME", "") or "")
        text = str(row.get("TEXT", "") or "")
        prefix = time if time else rel_time
        rows.append(f"{prefix} | {text}")
    return df, "\n".join(rows)


def select_cases(input_dir: Path, limit: int, strategy: str) -> list[dict[str, Any]]:
    cases = []
    for csv_path in sorted(input_dir.glob("*.csv")):
        df, chronology = read_chronology(csv_path)
        cases.append(
            {
                "admission_id": admission_id_from_input(csv_path),
                "path": csv_path,
                "rows": len(df),
                "words": word_count(chronology),
            }
        )
    if strategy == "shortest":
        cases.sort(key=lambda item: (item["words"], item["admission_id"]))
    elif strategy == "first":
        cases.sort(key=lambda item: item["admission_id"])
    else:
        raise ValueError(strategy)
    return cases[:limit]


def split_chunks(chronology: str, max_chunk_words: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for line in chronology.splitlines():
        line_words = word_count(line)
        if current and current_words + line_words > max_chunk_words:
            chunks.append("\n".join(current))
            current = []
            current_words = 0
        current.append(line)
        current_words += line_words
    if current:
        chunks.append("\n".join(current))
    return chunks


def compact_chronology_for_prompt(chronology: str, max_words: int) -> str:
    words = chronology.split()
    if len(words) <= max_words:
        return chronology
    half = max_words // 2
    head = " ".join(words[:half])
    tail = " ".join(words[-(max_words - half) :])
    return (
        head
        + "\n\n[TRUNCATED MIDDLE FOR BASELINE CONTEXT LIMIT; HEAD AND TAIL PRESERVED]\n\n"
        + tail
    )


def llm_call(prompt: str, args: argparse.Namespace) -> str:
    if args.dry_run:
        return (
            "[DRY RUN PLACEHOLDER]\n"
            f"Prompt words: {word_count(prompt)}\n"
            "This file confirms the closed-loop stage executed without an API call."
        )
    return call_deepseek(
        prompt=prompt,
        model=args.model,
        api_url=args.api_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
        api_key_env=args.api_key_env,
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def base_prompt(chronology: str, max_words: int) -> str:
    context = compact_chronology_for_prompt(chronology, max_words)
    return f"""You are an experienced ICU clinician writing a discharge summary.

Use only the following full-admission chronology. Do not invent diagnoses,
procedures, medications, disposition, or follow-up plans that are not supported
by the chronology.

Full admission chronology:
{context}

Write exactly three sections using exactly these headings:

## 1. Diagnosis:
List the primary and secondary discharge diagnoses. Keep it concise.

## 2. Hospital Course Summary:
Summarize the major admission course, key treatments, procedures, complications,
resolved problems, unresolved problems at discharge, and clinical trajectory.

## 3. Discharge Instructions:
Summarize discharge medications, follow-up, diet/activity, precautions, and
ongoing issues when supported. If exact discharge instructions are not documented,
do not write "not documented" placeholders; instead provide only conservative
clinically necessary guidance supported by the documented diagnoses and course.

Do not add any extra sections after "## 3. Discharge Instructions:".
"""


def state_update_prompt(previous_state: str, chunk: str, chunk_index: int, total_chunks: int) -> str:
    return f"""You are maintaining a structured discharge-summary state for one hospital admission.

Update the existing state using only the new chronology chunk. Do not write the
final discharge summary. Preserve important resolved and unresolved problems.
Mark whether each problem is resolved, improved, worsened, ongoing, or uncertain.
Do not invent diagnoses, procedures, medications, discharge disposition, or
follow-up plans.

Return one compact valid JSON object only. Do not use markdown fences. Keep each
array to at most 8 concise strings. Escape internal quotation marks.

Existing state JSON:
{previous_state}

New chronology chunk {chunk_index}/{total_chunks}:
{chunk}

Required JSON schema:
{{
  "admission_reason": [],
  "principal_diagnoses": [],
  "secondary_diagnoses": [],
  "major_procedures": [],
  "hospital_course_timeline": [],
  "icu_course": [],
  "complications": [],
  "treatments": [],
  "resolved_problems": [],
  "unresolved_problems_at_discharge": [],
  "discharge_condition": [],
  "discharge_disposition": [],
  "discharge_medications": [],
  "follow_up": [],
  "diet_activity_instructions": [],
  "must_not_add": [],
  "uncertain_items": []
}}
"""


def scaffold_prompt(final_state: str) -> str:
    return f"""Create a discharge-summary scaffold from the final discharge state.

Do not write prose paragraphs. Return one compact valid JSON object only. Do not
use markdown fences. Include required sections, must-cover items, resolved problem
summary, unresolved problem summary, temporal course outline, must-not-add items,
and evidence gaps. Keep each array to at most 8 concise strings.

Final discharge state:
{final_state}
"""


def ours1_prompt(final_state: str, scaffold: str) -> str:
    return f"""You are an experienced ICU clinician writing the final discharge summary.

Use only the final discharge state and scaffold below. Do not invent unsupported
diagnoses, procedures, medications, disposition, follow-up, dates, or lab values.

Final discharge state:
{final_state}

DS scaffold:
{scaffold}

Write exactly three sections using exactly these headings:

## 1. Diagnosis:
List the primary and secondary discharge diagnoses. Keep it concise.

## 2. Hospital Course Summary:
Summarize the major admission course, key treatments, procedures, complications,
resolved problems, unresolved problems at discharge, and clinical trajectory.

## 3. Discharge Instructions:
Summarize discharge medications, follow-up, diet/activity, precautions, and
ongoing issues when supported by the final state. If exact discharge instructions
are not documented, do not write "not documented" placeholders; instead provide
only conservative clinically necessary guidance supported by the documented
diagnoses and course.

Do not add resolved-problem, unresolved-problem, temporal-outline, scaffold, or
evidence-gap sections after "## 3. Discharge Instructions:".
"""


def judge_prompt(initial_ds: str, final_state: str, scaffold: str) -> str:
    return f"""You are a global discharge-summary verifier.

Judge whether the initial discharge summary is faithful to the final discharge
state and scaffold. Do not compare two consecutive summaries. Your task is to
check whether the final DS covers the whole admission correctly.

Return one compact valid JSON object only. Do not use markdown fences. Keep each
array to at most 8 concise strings. Use these keys:
- unsupported_claims
- missed_major_events
- wrong_temporal_order
- missing_diagnoses
- missing_procedures
- missing_complications
- resolved_status_errors
- unresolved_status_errors
- discharge_medication_errors
- follow_up_or_disposition_errors
- stale_or_irrelevant_problem_carryover
- must_remove
- must_add
- do_not_change

Initial DS:
{initial_ds}

Final discharge state:
{final_state}

DS scaffold:
{scaffold}
"""


def revise_prompt(initial_ds: str, judge: str, final_state: str) -> str:
    return f"""Revise the initial discharge summary using the judge output.

Rules:
1. Remove only claims listed in must_remove or unsupported_claims.
2. Add only events listed in must_add or missed_major_events.
3. Correct resolved/unresolved status errors.
4. Preserve do_not_change content.
5. Do not introduce new diagnoses, procedures, medications, lab values, dates,
   disposition, or follow-up plans.
6. Do not mention judge, verifier, scaffold, oracle, or gold truth.
7. Return exactly three sections with these headings only:
   ## 1. Diagnosis:
   ## 2. Hospital Course Summary:
   ## 3. Discharge Instructions:
8. Do not add resolved-problem, unresolved-problem, temporal-outline, scaffold,
   or evidence-gap sections after the discharge instructions.

Initial DS:
{initial_ds}

Judge output:
{judge}

Final discharge state:
{final_state}

Return the revised discharge summary only.
"""


def run_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    admission_id = case["admission_id"]
    case_dir = Path(args.output_dir) / "cases" / admission_id
    _, chronology = read_chronology(case["path"])
    chunks = split_chunks(chronology, args.max_chunk_words)

    prompts_dir = case_dir / "prompts"
    write_text(prompts_dir / "base1_full_context_direct.md", base_prompt(chronology, args.base_max_words))

    base = llm_call(base_prompt(chronology, args.base_max_words), args)
    write_text(case_dir / "base1_full_context_direct.txt", base)
    write_text(
        Path(args.output_dir) / "method_outputs" / "base1_full_context_direct" / f"48h_all_abs_{admission_id}.txt",
        base,
    )

    state = json.dumps(
        {
            "admission_reason": [],
            "principal_diagnoses": [],
            "secondary_diagnoses": [],
            "major_procedures": [],
            "hospital_course_timeline": [],
            "icu_course": [],
            "complications": [],
            "treatments": [],
            "resolved_problems": [],
            "unresolved_problems_at_discharge": [],
            "discharge_condition": [],
            "discharge_disposition": [],
            "discharge_medications": [],
            "follow_up": [],
            "diet_activity_instructions": [],
            "must_not_add": [],
            "uncertain_items": [],
        },
        ensure_ascii=False,
        indent=2,
    )
    for idx, chunk in enumerate(chunks, start=1):
        prompt = state_update_prompt(state, chunk, idx, len(chunks))
        write_text(prompts_dir / f"state_update_chunk_{idx:02d}.md", prompt)
        state = llm_call(prompt, args)
        write_text(case_dir / f"state_after_chunk_{idx:02d}.json", state)

    write_text(case_dir / "final_discharge_state.json", state)

    prompt = scaffold_prompt(state)
    write_text(prompts_dir / "scaffold.md", prompt)
    scaffold = llm_call(prompt, args)
    write_text(case_dir / "ds_scaffold.json", scaffold)

    prompt = ours1_prompt(state, scaffold)
    write_text(prompts_dir / "ours1_generate.md", prompt)
    ours1 = llm_call(prompt, args)
    write_text(case_dir / "ours1_state_scaffold.txt", ours1)
    write_text(
        Path(args.output_dir) / "method_outputs" / "ours1_state_scaffold" / f"48h_all_abs_{admission_id}.txt",
        ours1,
    )

    prompt = judge_prompt(ours1, state, scaffold)
    write_text(prompts_dir / "ours2_global_judge.md", prompt)
    judge = llm_call(prompt, args)
    write_text(case_dir / "ours2_global_judge.json", judge)

    prompt = revise_prompt(ours1, judge, state)
    write_text(prompts_dir / "ours2_minimal_revise.md", prompt)
    ours2 = llm_call(prompt, args)
    write_text(case_dir / "ours2_global_judge_revise.txt", ours2)
    write_text(
        Path(args.output_dir) / "method_outputs" / "ours2_global_judge_revise" / f"48h_all_abs_{admission_id}.txt",
        ours2,
    )

    return {
        "admission_id": admission_id,
        "input_file": str(case["path"]),
        "rows": case["rows"],
        "input_words": case["words"],
        "chunks": len(chunks),
        "base_words": word_count(base),
        "state_words": word_count(state),
        "scaffold_words": word_count(scaffold),
        "ours1_words": word_count(ours1),
        "judge_words": word_count(judge),
        "ours2_words": word_count(ours2),
        "case_dir": str(case_dir),
    }


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", rows)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DS minimal closed-loop experiment.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--gold-dir", default=str(DEFAULT_GOLD_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--case-selection", choices=["shortest", "first"], default="shortest")
    parser.add_argument("--max-chunk-words", type=int, default=3500)
    parser.add_argument("--base-max-words", type=int, default=24000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    all_inputs = sorted(input_dir.glob("*.csv"))
    all_golds = sorted(Path(args.gold_dir).glob("*.txt"))
    cases = select_cases(input_dir, args.limit, args.case_selection)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "dataset_count.json",
        {
            "input_dir": str(input_dir),
            "gold_dir": str(args.gold_dir),
            "full_ds_input_count": len(all_inputs),
            "full_ds_gold_count": len(all_golds),
            "selected_limit": args.limit,
            "case_selection": args.case_selection,
            "dry_run": args.dry_run,
        },
    )

    rows = []
    for idx, case in enumerate(cases, start=1):
        print(
            f"[{idx}/{len(cases)}] HADM_ID={case['admission_id']} "
            f"rows={case['rows']} words={case['words']}"
        )
        rows.append(run_case(case, args))
        write_summary(output_dir, rows)

    print(f"Done. Summary: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
