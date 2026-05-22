"""Run Ours2-v4-dx2 with role-based diagnosis classification.

This version avoids direct free-form diagnosis generation. It asks the LLM to
classify diagnosis candidates by clinical role, then a deterministic verbalizer
keeps only diagnosis-worthy candidates and composes the final Diagnosis section.
Hospital Course and Discharge Instructions are inherited from Ours2-v3.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.deepseek_api_generation import DEFAULT_API_URL, call_deepseek
from scripts.build_ds_ours2_v4_guarded import extract_sections
from scripts.run_ds_minimal_closed_loop import compact_chronology_for_prompt, read_chronology, select_cases, word_count, write_text


DEFAULT_INPUT_DIR = Path("data/DS_fixed_composed/full/input")
DEFAULT_SOURCE_RUN = Path("outputs/ds_minimal_closed_loop_10_format_aligned")
DEFAULT_VARIANT_RUN = Path("outputs/ds_v2_variants_10")
DEFAULT_OUTPUT_DIR = Path("outputs/ds_v2_variants_10/method_outputs/ours2_v4_dx2_role_classified_diagnosis")

INCLUDE_ROLES = {
    "principal_discharge_diagnosis",
    "major_secondary_diagnosis",
    "major_complication_affecting_course_or_discharge",
    "procedure_related_diagnosis",
}


def llm_call(prompt: str, args: argparse.Namespace) -> str:
    if args.dry_run:
        return json.dumps(
            {
                "diagnosis_candidates": [
                    {
                        "candidate": "dry run",
                        "role": "principal_discharge_diagnosis",
                        "include_in_diagnosis": True,
                        "reason": "dry run",
                        "final_phrase": "Dry run diagnosis",
                    }
                ]
            },
            indent=2,
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


def strip_fences(text: str) -> str:
    text = text.strip()
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()


def parse_json(text: str) -> dict[str, Any]:
    raw = strip_fences(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise


def normalize_candidate_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or re.match(r"^#{1,3}\s*", stripped):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def dx2_prompt(
    base_diagnosis: str,
    ours1_diagnosis: str,
    ours2v3_diagnosis: str,
    final_state: str,
    chronology: str,
    max_words: int,
) -> str:
    context = compact_chronology_for_prompt(chronology, max_words)
    return f"""You are a discharge-diagnosis role classifier.

Your task is NOT to maximize coverage and NOT to be artificially short. Your
task is to decide which candidate conditions are appropriate final discharge
diagnoses.

Use candidate diagnoses only as recall sources. Classify each clinically distinct
candidate by role and decide whether it belongs in the final Diagnosis section.

Include a condition in Diagnosis only if it explains the admission, required a
major treatment/procedure, caused a major complication, or remains relevant to
discharge monitoring/follow-up.

Usually exclude conditions that are only transient laboratory abnormalities,
minor resolved inpatient issues, symptoms, uncertain findings, or past history
that did not drive the admission/course/discharge plan. However, do include AKI,
anemia, electrolyte abnormalities, respiratory failure, infection, or bleeding
when they are major complications, require treatment, or affect discharge plan.

Allowed roles:
- principal_discharge_diagnosis
- major_secondary_diagnosis
- major_complication_affecting_course_or_discharge
- procedure_related_diagnosis
- chronic_comorbidity_relevant_to_discharge
- transient_lab_or_minor_resolved_issue
- symptom_or_uncertain_finding
- past_history_only
- duplicate_or_subsumed

Return one compact valid JSON object only. Do not use markdown fences. Target
3-7 included diagnoses unless the admission genuinely requires more.

JSON schema:
{{
  "diagnosis_candidates": [
    {{
      "candidate": "",
      "role": "",
      "include_in_diagnosis": true,
      "reason": "",
      "final_phrase": ""
    }}
  ]
}}

Base diagnosis candidates:
{normalize_candidate_lines(base_diagnosis)}

Ours1-v2 diagnosis candidates:
{normalize_candidate_lines(ours1_diagnosis)}

Ours2-v3 diagnosis candidates:
{normalize_candidate_lines(ours2v3_diagnosis)}

Final discharge state:
{final_state}

Admission evidence:
{context}
"""


def verbalize_diagnosis(classification_text: str) -> tuple[str, dict[str, Any]]:
    parsed = parse_json(classification_text)
    candidates = parsed.get("diagnosis_candidates", [])
    included = []
    seen = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        include = bool(item.get("include_in_diagnosis", False))
        phrase = str(item.get("final_phrase", "") or item.get("candidate", "")).strip()
        phrase = re.sub(r"^\s*[-*]\s*", "", phrase)
        phrase = phrase.rstrip(".")
        key = phrase.lower()
        if include and role in INCLUDE_ROLES and phrase and key not in seen:
            included.append(phrase)
            seen.add(key)

    if not included:
        # Fallback: use any explicitly included phrase, then any first candidates.
        for item in candidates:
            if not isinstance(item, dict):
                continue
            phrase = str(item.get("final_phrase", "") or item.get("candidate", "")).strip()
            phrase = re.sub(r"^\s*[-*]\s*", "", phrase).rstrip(".")
            key = phrase.lower()
            if phrase and key not in seen:
                included.append(phrase)
                seen.add(key)
            if len(included) >= 5:
                break

    bullets = "\n".join(f"- {phrase}" for phrase in included[:8])
    return bullets, parsed


def build_output(diagnosis_bullets: str, v3_text: str) -> str:
    v3_sections = extract_sections(v3_text)
    return (
        "## 1. Diagnosis:\n"
        f"{diagnosis_bullets.strip()}\n\n"
        "## 2. Hospital Course Summary:\n"
        f"{v3_sections['Hospital Course'].strip()}\n\n"
        "## 3. Discharge Instructions:\n"
        f"{v3_sections['Discharge Instructions'].strip()}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DS Ours2-v4-dx2 role-classified diagnosis.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--source-run", default=str(DEFAULT_SOURCE_RUN))
    parser.add_argument("--variant-run", default=str(DEFAULT_VARIANT_RUN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--case-selection", choices=["shortest", "first"], default="shortest")
    parser.add_argument("--evidence-max-words", type=int, default=24000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = select_cases(Path(args.input_dir), args.limit, args.case_selection)
    rows = []
    for idx, case in enumerate(cases, start=1):
        aid = case["admission_id"]
        print(f"[{idx}/{len(cases)}] HADM_ID={aid}")
        _, chronology = read_chronology(case["path"])
        source_run = Path(args.source_run)
        variant_run = Path(args.variant_run)

        base_text = (source_run / "method_outputs" / "base1_full_context_direct" / f"48h_all_abs_{aid}.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        ours1_text = (variant_run / "method_outputs" / "ours1_v2" / f"48h_all_abs_{aid}.txt").read_text(
            encoding="utf-8", errors="replace"
        )
        v3_text = (
            variant_run / "method_outputs" / "ours2_v3_base_recall_verified" / f"48h_all_abs_{aid}.txt"
        ).read_text(encoding="utf-8", errors="replace")
        final_state = (source_run / "cases" / aid / "final_discharge_state.json").read_text(
            encoding="utf-8", errors="replace"
        )

        prompt = dx2_prompt(
            extract_sections(base_text)["Diagnosis"],
            extract_sections(ours1_text)["Diagnosis"],
            extract_sections(v3_text)["Diagnosis"],
            final_state,
            chronology,
            args.evidence_max_words,
        )
        prompt_dir = variant_run / "cases" / aid / "prompts"
        write_text(prompt_dir / "ours2_v4_dx2_role_classifier.md", prompt)
        classification = llm_call(prompt, args)
        write_text(variant_run / "cases" / aid / "ours2_v4_dx2_classification.json", classification)
        diagnosis_bullets, parsed = verbalize_diagnosis(classification)
        write_text(variant_run / "cases" / aid / "ours2_v4_dx2_diagnosis.txt", diagnosis_bullets)
        output = build_output(diagnosis_bullets, v3_text)
        out_file = output_dir / f"48h_all_abs_{aid}.txt"
        out_file.write_text(output, encoding="utf-8")
        rows.append(
            {
                "admission_id": aid,
                "included_diagnoses": len(diagnosis_bullets.splitlines()),
                "words": len(output.split()),
                "output_file": str(out_file),
            }
        )

    summary = Path(args.variant_run) / "ours2_v4_dx2_summary.csv"
    summary.write_text(
        "admission_id,included_diagnoses,words,output_file\n"
        + "\n".join(
            f"{r['admission_id']},{r['included_diagnoses']},{r['words']},{r['output_file']}" for r in rows
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Built {len(rows)} outputs: {output_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
