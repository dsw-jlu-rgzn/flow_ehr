"""Build Ours2-v4 by preserving compact diagnosis and using Ours2-v3 course/instructions.

Ours2-v4 is a guarded section-wise composition:
- Diagnosis: compact diagnosis-preserving section from Base 1.
- Hospital Course Summary: Ours2-v3 enhanced section.
- Discharge Instructions: Ours2-v3 enhanced section.

This tests whether the v2/v3 gains in course/instructions can be retained while
removing the diagnosis precision loss caused by broader coverage-oriented
generation.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


HEADINGS = {
    "Diagnosis": re.compile(
        r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*\*)?(?:1\.\s*)?(?:Discharge\s+)?Diagnosis(?:es)?(?:\*\*)?:?\s*",
        re.I,
    ),
    "Hospital Course": re.compile(
        r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*\*)?(?:2\.\s*)?(?:Brief\s+)?Hospital\s+Course(?:\s+Summary)?(?:\*\*)?:?\s*",
        re.I,
    ),
    "Discharge Instructions": re.compile(
        r"(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*\*)?(?:3\.\s*)?(?:Discharge\s+)?(?:Medications\s+and\s+Follow-up\s+Instructions|Instructions)(?:\*\*)?:?\s*",
        re.I,
    ),
}


def extract_sections(text: str) -> dict[str, str]:
    starts = []
    for name, pattern in HEADINGS.items():
        match = pattern.search(text)
        if match:
            starts.append((match.start(), match.end(), name))
    starts.sort()
    sections = {name: "" for name in HEADINGS}
    for idx, (_, end, name) in enumerate(starts):
        next_start = starts[idx + 1][0] if idx + 1 < len(starts) else len(text)
        sections[name] = text[end:next_start].strip()
    return sections


def normalize_diagnosis_section(diagnosis: str) -> str:
    diagnosis = diagnosis.strip()
    diagnosis = re.sub(r"^\s*(?:[-*]\s*)?\*\*(?:Principal|Secondary) Diagnoses?:\*\*\s*", "", diagnosis, flags=re.I | re.M)
    return diagnosis


def build_v4(base_text: str, v3_text: str) -> str:
    base_sections = extract_sections(base_text)
    v3_sections = extract_sections(v3_text)
    diagnosis = normalize_diagnosis_section(base_sections["Diagnosis"])
    course = v3_sections["Hospital Course"].strip()
    instructions = v3_sections["Discharge Instructions"].strip()
    return (
        "## 1. Diagnosis:\n"
        f"{diagnosis}\n\n"
        "## 2. Hospital Course Summary:\n"
        f"{course}\n\n"
        "## 3. Discharge Instructions:\n"
        f"{instructions}\n"
    )


def admission_id(path: Path) -> str:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(path.name)
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DS Ours2-v4 guarded outputs.")
    parser.add_argument(
        "--base-dir",
        default="outputs/ds_minimal_closed_loop_10_format_aligned/method_outputs/base1_full_context_direct",
    )
    parser.add_argument(
        "--v3-dir",
        default="outputs/ds_v2_variants_10/method_outputs/ours2_v3_base_recall_verified",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ds_v2_variants_10/method_outputs/ours2_v4_diagnosis_guarded",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    v3_dir = Path(args.v3_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for base_file in sorted(base_dir.glob("48h_all_abs_*.txt")):
        aid = admission_id(base_file)
        v3_file = v3_dir / base_file.name
        if not v3_file.exists():
            continue
        base_text = base_file.read_text(encoding="utf-8", errors="replace")
        v3_text = v3_file.read_text(encoding="utf-8", errors="replace")
        out_text = build_v4(base_text, v3_text)
        out_file = output_dir / base_file.name
        out_file.write_text(out_text, encoding="utf-8")
        rows.append({"admission_id": aid, "output_file": str(out_file), "words": len(out_text.split())})

    summary = output_dir.parent.parent / "ours2_v4_summary.csv"
    summary.write_text(
        "admission_id,output_file,words\n"
        + "\n".join(f"{r['admission_id']},{r['output_file']},{r['words']}" for r in rows)
        + "\n",
        encoding="utf-8",
    )
    print(f"Built {len(rows)} Ours2-v4 outputs: {output_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
