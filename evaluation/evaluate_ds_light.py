"""
Lightweight DS section-level ROUGE-L evaluation.

This script avoids torch, transformers, rouge, and QuickUMLS. It is intended as
a smoke-test metric when the full paper evaluation environment is not ready.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


SECTIONS = ["Diagnosis", "Hospital Course", "Discharge Instructions"]


def tokenize(text: str) -> list[str]:
    return str(text).lower().split()


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            curr.append(prev[j - 1] + 1 if token_a == token_b else max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_len(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def extract_generated_sections(text: str) -> dict[str, str]:
    patterns = {
        "Diagnosis": re.compile(
            r"(?:\*\*)?(?:Part\s*1:\s*)?(?:#{1,3}\s*)?(?:1\.\s*)?(?:Discharge\s+)?Diagnosis(?:es)?(?:\*\*)?:?",
            re.I,
        ),
        "Hospital Course": re.compile(
            r"(?:\*\*)?(?:Part\s*2:\s*)?(?:#{1,3}\s*)?(?:3\.|2\.)?\s*(?:Brief\s+)?Hospital\s+Course(?:\s+Summary)?(?:\*\*)?:?",
            re.I,
        ),
        "Discharge Instructions": re.compile(
            r"(?:\*\*)?(?:Part\s*3:\s*)?(?:#{1,3}\s*)?(?:5\.|3\.)?\s*(?:Discharge\s+)?(?:Medications\s+and\s+Follow-up\s+Instructions|Instructions|Follow-up\s+Instructions)(?:\*\*)?:?",
            re.I,
        ),
    }
    starts = []
    for name, pattern in patterns.items():
        match = pattern.search(text)
        if match:
            starts.append((match.start(), match.end(), name))
    starts.sort()
    sections = {name: "" for name in SECTIONS}
    for idx, (_, end, name) in enumerate(starts):
        next_start = starts[idx + 1][0] if idx + 1 < len(starts) else len(text)
        sections[name] = text[end:next_start].strip()
    return sections


def extract_gold_sections(text: str) -> dict[str, str]:
    sections = {name: "" for name in SECTIONS}

    def between(start_pattern: str, end_pattern: str) -> str:
        start = re.search(start_pattern, text, re.I)
        if not start:
            return ""
        end = re.search(end_pattern, text[start.end() :], re.I)
        if end:
            return text[start.end() : start.end() + end.start()].strip()
        return text[start.end() :].strip()

    sections["Diagnosis"] = between(
        r"Discharge Diagnosis:|DISCHARGE DIAGNOSIS:|DISCHARGED DIAGNOSES:|FINAL DIAGNOSIS:",
        r"Discharge Condition:|DISCHARGE CONDITION:|Discharge Disposition:|DISCHARGE MEDICATIONS:",
    )
    sections["Hospital Course"] = between(
        r"Brief Hospital Course:|HOSPITAL COURSE:",
        r"Discharge Medications:|DISCHARGE MEDICATIONS:|Medications on Admission:|CONDITION ON DISCHARGE:|DISCHARGE CONDITION:",
    )
    instructions = between(
        r"Discharge Instructions:|DISCHARGE INSTRUCTIONS/FOLLOWUP:|DISCHARGE PLAN:|RECOMMENDED FOLLOWUP:|FOLLOWUP:",
        r"Followup Instructions:|RECOMMENDED FOLLOW-UP|\[\*\*First Name|\[\*\*Name",
    )
    meds = between(
        r"Discharge Medications:|DISCHARGE MEDICATIONS:|MEDICATIONS ON DISCHARGE:",
        r"Discharge Disposition:|DISCHARGE STATUS:|FOLLOW-UP:|FOLLOW-UP PLANS:",
    )
    sections["Discharge Instructions"] = (instructions + "\n" + meds).strip()
    return sections


def admission_id_from_generated(path: Path) -> str:
    match = re.match(r"48h_all_abs_(\d+)\.txt$", path.name)
    return match.group(1) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight DS ROUGE-L evaluator.")
    parser.add_argument("--gen-dir", required=True)
    parser.add_argument("--gt-dir", default="data/DS/gold")
    parser.add_argument("--output-csv", default="outputs/ds_light_eval.csv")
    args = parser.parse_args()

    rows = []
    for gen_file in sorted(Path(args.gen_dir).glob("48h_all_abs_*.txt")):
        admission_id = admission_id_from_generated(gen_file)
        gold_file = Path(args.gt_dir) / f"gtsummary_{admission_id}.txt"
        if not gold_file.exists():
            continue
        gen_sections = extract_generated_sections(gen_file.read_text(encoding="utf-8", errors="replace"))
        gold_sections = extract_gold_sections(gold_file.read_text(encoding="utf-8", errors="replace"))
        for section in SECTIONS:
            rows.append(
                {
                    "admission_id": admission_id,
                    "section": section,
                    "rouge_l_f1": rouge_l_f1(gen_sections[section], gold_sections[section]),
                    "gen_words": len(tokenize(gen_sections[section])),
                    "gold_words": len(tokenize(gold_sections[section])),
                }
            )

    if not rows:
        raise SystemExit("No matching generated/gold DS files found.")

    df = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("DS lightweight ROUGE-L:")
    for section, group in df.groupby("section", sort=False):
        print(f"  {section}: {group['rouge_l_f1'].mean() * 100:.2f} +/- {group['rouge_l_f1'].std(ddof=0) * 100:.2f} (n={len(group)})")
    print(f"Detail CSV: {output_csv}")


if __name__ == "__main__":
    main()
