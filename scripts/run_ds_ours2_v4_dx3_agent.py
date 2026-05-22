"""Run Ours2-v4-dx3 with an agentic diagnosis verbalizer.

This reuses dx2 role-classification outputs, then asks a Diagnosis Agent to
compose a compact gold-style Diagnosis section. Course/Instructions remain from
Ours2-v3.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.deepseek_api_generation import DEFAULT_API_URL, call_deepseek
from scripts.build_ds_ours2_v4_guarded import extract_sections
from scripts.run_ds_minimal_closed_loop import compact_chronology_for_prompt, read_chronology, select_cases, word_count, write_text


DEFAULT_INPUT_DIR = Path("data/DS_fixed_composed/full/input")
DEFAULT_SOURCE_RUN = Path("outputs/ds_minimal_closed_loop_10_format_aligned")
DEFAULT_VARIANT_RUN = Path("outputs/ds_v2_variants_10")
DEFAULT_OUTPUT_DIR = Path("outputs/ds_v2_variants_10/method_outputs/ours2_v4_dx3_agent_diagnosis")


def llm_call(prompt: str, args: argparse.Namespace) -> str:
    if args.dry_run:
        return "- Dry run diagnosis"
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


def agent_prompt(
    classification_json: str,
    base_diagnosis: str,
    ours1_diagnosis: str,
    v3_diagnosis: str,
    final_state: str,
    chronology: str,
    max_words: int,
) -> str:
    evidence = compact_chronology_for_prompt(chronology, max_words)
    return f"""You are the Diagnosis Agent for a discharge summary.

You are given diagnosis candidates and role-classification output. Your job is
to write the final Diagnosis section in compact MIMIC-style discharge diagnosis
language.

Important:
- Do not simply follow the rule classifier if it conflicts with discharge
  diagnosis style.
- Diagnosis is not a broad problem list.
- Prefer diagnoses likely to appear under "Discharge Diagnosis" in a real
  MIMIC discharge summary.
- Keep 3-6 bullet diagnoses unless the admission truly requires more.
- Preserve clinically important diagnoses from the candidates when supported.
- Include an uncertain diagnosis such as "possible pneumonia" if the course and
  treatment make it likely to be a discharge diagnosis.
- Exclude minor transient labs, symptoms, radiology findings, PMH-only problems,
  and small resolved issues unless central to admission or discharge plan.
- Use short gold-compatible phrases, not long explanatory sentences.

Return only bullet lines. Do not include a heading. Do not include rationales.

Role-classification JSON:
{classification_json}

Base diagnosis candidates:
{base_diagnosis}

Ours1-v2 diagnosis candidates:
{ours1_diagnosis}

Ours2-v3 diagnosis candidates:
{v3_diagnosis}

Final discharge state:
{final_state}

Admission evidence:
{evidence}
"""


def clean_diagnosis(text: str) -> str:
    text = re.sub(r"^```.*?\n|\n```$", "", text.strip(), flags=re.S)
    sections = extract_sections(text)
    if sections.get("Diagnosis"):
        text = sections["Diagnosis"]
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^#{1,3}\s*", stripped):
            continue
        if stripped.lower().startswith(("diagnosis:", "discharge diagnosis:")):
            continue
        if not stripped.startswith(("-", "*", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.")):
            stripped = "- " + stripped
        stripped = re.sub(r"^\d+\.\s*", "- ", stripped)
        stripped = re.sub(r"^\*\s*", "- ", stripped)
        lines.append(stripped)
    return "\n".join(lines[:8])


def build_output(diagnosis: str, v3_text: str) -> str:
    v3_sections = extract_sections(v3_text)
    return (
        "## 1. Diagnosis:\n"
        f"{clean_diagnosis(diagnosis)}\n\n"
        "## 2. Hospital Course Summary:\n"
        f"{v3_sections['Hospital Course'].strip()}\n\n"
        "## 3. Discharge Instructions:\n"
        f"{v3_sections['Discharge Instructions'].strip()}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DS Ours2-v4-dx3 Diagnosis Agent.")
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
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_run = Path(args.source_run)
    variant_run = Path(args.variant_run)
    rows = []
    for idx, case in enumerate(select_cases(Path(args.input_dir), args.limit, args.case_selection), start=1):
        aid = case["admission_id"]
        print(f"[{idx}/{args.limit}] HADM_ID={aid}")
        _, chronology = read_chronology(case["path"])
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
        classification = (variant_run / "cases" / aid / "ours2_v4_dx2_classification.json").read_text(
            encoding="utf-8", errors="replace"
        )
        prompt = agent_prompt(
            classification,
            extract_sections(base_text)["Diagnosis"],
            extract_sections(ours1_text)["Diagnosis"],
            extract_sections(v3_text)["Diagnosis"],
            final_state,
            chronology,
            args.evidence_max_words,
        )
        write_text(variant_run / "cases" / aid / "prompts" / "ours2_v4_dx3_agent.md", prompt)
        diagnosis = llm_call(prompt, args)
        write_text(variant_run / "cases" / aid / "ours2_v4_dx3_diagnosis.txt", diagnosis)
        output = build_output(diagnosis, v3_text)
        out_file = output_dir / f"48h_all_abs_{aid}.txt"
        out_file.write_text(output, encoding="utf-8")
        rows.append({"admission_id": aid, "words": len(output.split()), "diagnosis_words": len(clean_diagnosis(diagnosis).split())})

    summary = variant_run / "ours2_v4_dx3_summary.csv"
    summary.write_text(
        "admission_id,words,diagnosis_words\n"
        + "\n".join(f"{r['admission_id']},{r['words']},{r['diagnosis_words']}" for r in rows)
        + "\n",
        encoding="utf-8",
    )
    print(f"Built {len(rows)} outputs: {output_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
