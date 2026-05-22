"""Run formal Ours2-v4 with a verified compact diagnosis selector.

Unlike the guarded ablation, this script does not copy the Base diagnosis
section directly. It uses Base and Ours candidates as recall sources, then asks
an evidence-grounded selector to produce a compact diagnosis section from:
- Base candidate diagnoses
- Ours candidate diagnoses
- final discharge state
- admission evidence

Hospital Course and Discharge Instructions are inherited from Ours2-v3.
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
DEFAULT_OUTPUT_DIR = Path("outputs/ds_v2_variants_10/method_outputs/ours2_v4_final_verified_diagnosis")


def llm_call(prompt: str, args: argparse.Namespace) -> str:
    if args.dry_run:
        return (
            "- [DRY RUN] Verified diagnosis placeholder\n"
            f"- Prompt words: {word_count(prompt)}"
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


def normalize_candidate_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^#{1,3}\s*", stripped):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def verified_diagnosis_prompt(
    base_diagnosis: str,
    ours1_diagnosis: str,
    ours2v3_diagnosis: str,
    final_state: str,
    chronology: str,
    max_words: int,
) -> str:
    context = compact_chronology_for_prompt(chronology, max_words)
    return f"""You are selecting the final diagnosis section for a discharge summary.

You are given diagnosis candidates from multiple drafts, a structured final
discharge state, and admission evidence. Use the candidates only as recall
sources. Do not copy a candidate list blindly.

Goal: produce a compact, gold-compatible discharge diagnosis list.

Rules:
1. Prefer principal diagnoses and major secondary diagnoses supported by final
   discharge state or admission evidence.
2. Keep diagnoses that explain admission, procedures, major complications, or
   discharge plan.
3. Do not include transient lab abnormalities, symptoms, PMH, medications, or
   uncertain imaging findings as discharge diagnoses unless they are clearly
   clinically important and supported.
4. Avoid over-expanding the list. Target 4-9 diagnosis bullets.
5. Include concise specificity when supported, e.g. status post PCI, resolved
   pseudoaneurysm, acute kidney injury, pneumonia, heart failure.
6. Return only bullet lines for the Diagnosis section. Do not include a heading.

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


def build_output(diagnosis: str, v3_text: str) -> str:
    v3_sections = extract_sections(v3_text)
    diagnosis = diagnosis.strip()
    if "##" in diagnosis:
        diagnosis = extract_sections(diagnosis).get("Diagnosis", diagnosis).strip()
    return (
        "## 1. Diagnosis:\n"
        f"{diagnosis}\n\n"
        "## 2. Hospital Course Summary:\n"
        f"{v3_sections['Hospital Course'].strip()}\n\n"
        "## 3. Discharge Instructions:\n"
        f"{v3_sections['Discharge Instructions'].strip()}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal DS Ours2-v4 verified diagnosis selector.")
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
    prompt_dir = Path(args.variant_run) / "cases"

    cases = select_cases(Path(args.input_dir), args.limit, args.case_selection)
    rows = []
    for idx, case in enumerate(cases, start=1):
        aid = case["admission_id"]
        print(f"[{idx}/{len(cases)}] HADM_ID={aid}")
        _, chronology = read_chronology(case["path"])

        base_text = (
            Path(args.source_run)
            / "method_outputs"
            / "base1_full_context_direct"
            / f"48h_all_abs_{aid}.txt"
        ).read_text(encoding="utf-8", errors="replace")
        ours1_text = (
            Path(args.variant_run)
            / "method_outputs"
            / "ours1_v2"
            / f"48h_all_abs_{aid}.txt"
        ).read_text(encoding="utf-8", errors="replace")
        v3_text = (
            Path(args.variant_run)
            / "method_outputs"
            / "ours2_v3_base_recall_verified"
            / f"48h_all_abs_{aid}.txt"
        ).read_text(encoding="utf-8", errors="replace")
        final_state = (
            Path(args.source_run) / "cases" / aid / "final_discharge_state.json"
        ).read_text(encoding="utf-8", errors="replace")

        base_diag = extract_sections(base_text)["Diagnosis"]
        ours1_diag = extract_sections(ours1_text)["Diagnosis"]
        v3_diag = extract_sections(v3_text)["Diagnosis"]
        prompt = verified_diagnosis_prompt(
            base_diag,
            ours1_diag,
            v3_diag,
            final_state,
            chronology,
            args.evidence_max_words,
        )
        case_prompt_dir = prompt_dir / aid / "prompts"
        write_text(case_prompt_dir / "ours2_v4_final_verified_diagnosis.md", prompt)
        diagnosis = llm_call(prompt, args)
        write_text(Path(args.variant_run) / "cases" / aid / "ours2_v4_verified_diagnosis.txt", diagnosis)
        out_text = build_output(diagnosis, v3_text)
        out_file = output_dir / f"48h_all_abs_{aid}.txt"
        out_file.write_text(out_text, encoding="utf-8")
        rows.append({"admission_id": aid, "words": len(out_text.split()), "output_file": str(out_file)})

    summary = Path(args.variant_run) / "ours2_v4_final_summary.csv"
    summary.write_text(
        "admission_id,words,output_file\n"
        + "\n".join(f"{r['admission_id']},{r['words']},{r['output_file']}" for r in rows)
        + "\n",
        encoding="utf-8",
    )
    print(f"Built {len(rows)} outputs: {output_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
