"""Run DS Ours1-v2 / Ours2-v2 / Ours2-v3 variants from an existing closed-loop run."""

from __future__ import annotations

import argparse
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
from scripts.run_ds_minimal_closed_loop import (
    admission_id_from_input,
    compact_chronology_for_prompt,
    read_chronology,
    select_cases,
    word_count,
    write_json,
    write_text,
)


DEFAULT_INPUT_DIR = Path("data/DS_fixed_composed/full/input")
DEFAULT_SOURCE_RUN = Path("outputs/ds_minimal_closed_loop_10_format_aligned")
DEFAULT_OUTPUT_DIR = Path("outputs/ds_v2_variants_10")


def llm_call(prompt: str, args: argparse.Namespace) -> str:
    if args.dry_run:
        return (
            "[DRY RUN PLACEHOLDER]\n"
            f"Prompt words: {word_count(prompt)}\n"
            "This file confirms the variant stage executed without an API call."
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


def read_required(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8", errors="replace")


def discharge_plan_extractor_prompt(chronology: str, final_state: str, max_words: int) -> str:
    context = compact_chronology_for_prompt(chronology, max_words)
    return f"""You are extracting discharge-plan evidence for a discharge summary.

Use the full-admission chronology and final discharge state. Extract conservative
discharge-plan content that is explicit, strongly supported, or standard-care
inferred from documented diagnoses/procedures. Do not invent exact appointment
dates, new diagnoses, or unsupported medications.

Return one compact valid JSON object only. Do not use markdown fences. Keep each
array to at most 10 concise strings.

Evidence levels:
- explicit_discharge: directly documented as discharge medication, disposition, or follow-up
- strong_support: strongly implied by documented procedure/problem, e.g. antiplatelet therapy after PCI
- standard_care_inferred: conservative generic guidance implied by diagnosis/procedure
- unsupported: do not include in final DS

Full-admission chronology:
{context}

Final discharge state:
{final_state}

Required JSON schema:
{{
  "discharge_disposition_candidates": [],
  "discharge_medication_candidates": [],
  "follow_up_candidates": [],
  "diet_activity_candidates": [],
  "return_precautions": [],
  "monitoring_labs_or_tests": [],
  "unsupported_or_uncertain_items": []
}}
"""


def ours1_v2_prompt(final_state: str, scaffold: str, discharge_plan: str) -> str:
    return f"""You are an experienced ICU clinician writing a gold-compatible discharge summary.

Use the final discharge state, DS scaffold, and discharge-plan evidence below.
The final DS should be faithful but not overly sparse. You may include
conservative standard discharge guidance when it is explicit, strongly supported,
or standard-care inferred from documented diagnoses, procedures, or ongoing
problems. Do not invent exact appointment dates, new diagnoses, unsupported
medications, or unsupported doses.

Final discharge state:
{final_state}

DS scaffold:
{scaffold}

Discharge-plan evidence:
{discharge_plan}

Write exactly three sections using exactly these headings:

## 1. Diagnosis:
List 5-10 principal and secondary discharge diagnoses when supported.

## 2. Hospital Course Summary:
Write 220-320 words. Include admission reason, major procedures, complications,
treatments, resolved problems, unresolved problems at discharge, and clinical
trajectory.

## 3. Discharge Instructions:
Write 120-220 words. Include supported discharge medications/classes, follow-up,
monitoring labs/tests, activity/diet precautions, return precautions, and ongoing
issues. Avoid "not documented" placeholders.

Do not add extra sections.
"""


def additive_judge_prompt(initial_ds: str, final_state: str, scaffold: str, discharge_plan: str) -> str:
    return f"""You are a coverage-first global discharge-summary verifier.

Judge whether the DS is complete and faithful to the final state, scaffold, and
discharge-plan evidence. Your main job is not only to remove unsupported content,
but also to find supported missing content that should appear in a gold-style
discharge summary.

Return one compact valid JSON object only. Do not use markdown fences. Keep each
array to at most 10 concise strings.

Initial DS:
{initial_ds}

Final discharge state:
{final_state}

DS scaffold:
{scaffold}

Discharge-plan evidence:
{discharge_plan}

Required JSON schema:
{{
  "supported_missing_diagnoses": [],
  "supported_missing_hospital_course": [],
  "supported_missing_discharge_content": [],
  "supported_missing_medications": [],
  "supported_missing_followup": [],
  "supported_missing_disposition": [],
  "supported_missing_return_precautions": [],
  "unsupported_content_to_remove": [],
  "contradicted_content_to_correct": [],
  "do_not_remove": []
}}
"""


def additive_revise_prompt(initial_ds: str, judge: str, final_state: str, discharge_plan: str) -> str:
    return f"""Revise the discharge summary using the additive judge output.

Rules:
1. Add supported missing content first, especially discharge medications/classes,
   follow-up, disposition, monitoring, and return precautions.
2. Remove only clearly unsupported or contradicted content.
3. Preserve correct diagnosis and hospital course content.
4. Do not make the output shorter unless removing unsupported content.
5. Do not introduce exact appointment dates, new diagnoses, unsupported
   medications, unsupported doses, or unsupported procedures.
6. Return exactly three sections with these headings only:
   ## 1. Diagnosis:
   ## 2. Hospital Course Summary:
   ## 3. Discharge Instructions:

Initial DS:
{initial_ds}

Additive judge output:
{judge}

Final discharge state:
{final_state}

Discharge-plan evidence:
{discharge_plan}

Return the revised discharge summary only.
"""


def base_candidate_prompt(base_ds: str) -> str:
    return f"""Extract recall candidates from the Base full-context discharge summary.

These candidates are not automatically trusted. They will be evidence-verified
before use. Extract possible diagnoses, hospital-course events, procedures,
discharge medications/classes, disposition, follow-up, monitoring, diet/activity,
and return precautions.

Return one compact valid JSON object only. Do not use markdown fences.

Base DS:
{base_ds}

Required JSON schema:
{{
  "diagnosis_candidates": [],
  "hospital_course_candidates": [],
  "procedure_candidates": [],
  "discharge_medication_candidates": [],
  "disposition_candidates": [],
  "follow_up_candidates": [],
  "monitoring_candidates": [],
  "diet_activity_candidates": [],
  "return_precaution_candidates": []
}}
"""


def verify_base_candidates_prompt(base_candidates: str, final_state: str, discharge_plan: str, chronology: str, max_words: int) -> str:
    context = compact_chronology_for_prompt(chronology, max_words)
    return f"""Verify Base recall candidates against admission evidence.

Keep candidates that are explicit, strongly supported, or conservative
standard-care inferred from documented diagnoses/procedures. Reject unsupported,
over-specific, or contradicted candidates. Do not require exact wording in the
chronology for standard post-procedure precautions, but do require support for
specific medications, diagnoses, procedures, and disposition.

Return one compact valid JSON object only. Do not use markdown fences. Keep each
array to at most 10 concise strings.

Base recall candidates:
{base_candidates}

Final discharge state:
{final_state}

Discharge-plan evidence:
{discharge_plan}

Admission evidence:
{context}

Required JSON schema:
{{
  "verified_diagnoses": [],
  "verified_hospital_course": [],
  "verified_procedures": [],
  "verified_discharge_medications": [],
  "verified_disposition": [],
  "verified_follow_up": [],
  "verified_monitoring": [],
  "verified_diet_activity": [],
  "verified_return_precautions": [],
  "rejected_candidates": []
}}
"""


def base_recall_revise_prompt(ours1_v2: str, additive_judge: str, verified_base: str, final_state: str, discharge_plan: str) -> str:
    return f"""Create the final Ours2-v3 discharge summary.

Use Ours1-v2 as the base draft. Apply the additive judge output and the verified
Base recall candidates. The goal is to maximize clinically supported coverage
while avoiding unsupported hallucination.

Rules:
1. Add verified Base recall candidates when they improve diagnosis, hospital
   course, medication, disposition, follow-up, monitoring, or precautions.
2. Do not copy rejected or unverified Base candidates.
3. Keep gold-compatible discharge-summary style.
4. Return exactly three sections with these headings only:
   ## 1. Diagnosis:
   ## 2. Hospital Course Summary:
   ## 3. Discharge Instructions:

Ours1-v2 draft:
{ours1_v2}

Additive judge output:
{additive_judge}

Verified Base recall candidates:
{verified_base}

Final discharge state:
{final_state}

Discharge-plan evidence:
{discharge_plan}

Return the final DS only.
"""


def run_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    admission_id = case["admission_id"]
    source_case_dir = Path(args.source_run) / "cases" / admission_id
    out_case_dir = Path(args.output_dir) / "cases" / admission_id
    prompts_dir = out_case_dir / "prompts"

    _, chronology = read_chronology(case["path"])
    final_state = read_required(source_case_dir / "final_discharge_state.json")
    scaffold = read_required(source_case_dir / "ds_scaffold.json")
    base_ds = read_required(
        Path(args.source_run)
        / "method_outputs"
        / "base1_full_context_direct"
        / f"48h_all_abs_{admission_id}.txt"
    )

    prompt = discharge_plan_extractor_prompt(chronology, final_state, args.evidence_max_words)
    write_text(prompts_dir / "discharge_plan_extractor.md", prompt)
    discharge_plan = llm_call(prompt, args)
    write_text(out_case_dir / "discharge_plan_evidence.json", discharge_plan)

    prompt = ours1_v2_prompt(final_state, scaffold, discharge_plan)
    write_text(prompts_dir / "ours1_v2_generate.md", prompt)
    ours1_v2 = llm_call(prompt, args)
    write_text(out_case_dir / "ours1_v2.txt", ours1_v2)
    write_text(
        Path(args.output_dir) / "method_outputs" / "ours1_v2" / f"48h_all_abs_{admission_id}.txt",
        ours1_v2,
    )

    prompt = additive_judge_prompt(ours1_v2, final_state, scaffold, discharge_plan)
    write_text(prompts_dir / "ours2_v2_additive_judge.md", prompt)
    additive_judge = llm_call(prompt, args)
    write_text(out_case_dir / "ours2_v2_additive_judge.json", additive_judge)

    prompt = additive_revise_prompt(ours1_v2, additive_judge, final_state, discharge_plan)
    write_text(prompts_dir / "ours2_v2_revise.md", prompt)
    ours2_v2 = llm_call(prompt, args)
    write_text(out_case_dir / "ours2_v2_additive.txt", ours2_v2)
    write_text(
        Path(args.output_dir) / "method_outputs" / "ours2_v2_additive" / f"48h_all_abs_{admission_id}.txt",
        ours2_v2,
    )

    prompt = base_candidate_prompt(base_ds)
    write_text(prompts_dir / "base_recall_candidates.md", prompt)
    base_candidates = llm_call(prompt, args)
    write_text(out_case_dir / "base_recall_candidates.json", base_candidates)

    prompt = verify_base_candidates_prompt(
        base_candidates,
        final_state,
        discharge_plan,
        chronology,
        args.evidence_max_words,
    )
    write_text(prompts_dir / "verified_base_recall.md", prompt)
    verified_base = llm_call(prompt, args)
    write_text(out_case_dir / "verified_base_recall.json", verified_base)

    prompt = base_recall_revise_prompt(ours1_v2, additive_judge, verified_base, final_state, discharge_plan)
    write_text(prompts_dir / "ours2_v3_base_recall_revise.md", prompt)
    ours2_v3 = llm_call(prompt, args)
    write_text(out_case_dir / "ours2_v3_base_recall_verified.txt", ours2_v3)
    write_text(
        Path(args.output_dir)
        / "method_outputs"
        / "ours2_v3_base_recall_verified"
        / f"48h_all_abs_{admission_id}.txt",
        ours2_v3,
    )

    return {
        "admission_id": admission_id,
        "input_words": case["words"],
        "ours1_v2_words": word_count(ours1_v2),
        "ours2_v2_words": word_count(ours2_v2),
        "ours2_v3_words": word_count(ours2_v3),
        "case_dir": str(out_case_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DS v2 variant experiments.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--source-run", default=str(DEFAULT_SOURCE_RUN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--case-selection", choices=["shortest", "first"], default="shortest")
    parser.add_argument("--evidence-max-words", type=int, default=24000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = select_cases(Path(args.input_dir), args.limit, args.case_selection)
    rows = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] HADM_ID={case['admission_id']} words={case['words']}")
        rows.append(run_case(case, args))
        write_json(Path(args.output_dir) / "summary.json", rows)
        pd.DataFrame(rows).to_csv(Path(args.output_dir) / "summary.csv", index=False)
    print(f"Done. Summary: {Path(args.output_dir) / 'summary.csv'}")


if __name__ == "__main__":
    main()
