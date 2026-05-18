#!/usr/bin/env python3
"""
Lightweight clinical semantic evaluation without UMLS, QuickUMLS, or SapBERT.

The script compares generated A&P text against gold A&P and input evidence using:
- ROUGE-L F1
- Clinical concept F1
- Problem-category F1
- Treatment/intervention F1
- Grounded concept rate
- Unsupported concept rate
- Numeric claim support rate
- Trend F1 against gold text
- Evidence trend accuracy when input events contain enough sequential values

Expected generation JSONL rows follow the ProblemFlow format:
  {"sample_id": "...", "generated_ap": "...", "gold_ap": "..."}

Samples JSONL is optional but recommended. When present, it supplies
ehr_events_before_cutoff for evidence grounding and trend checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


PROBLEM_PATTERNS: dict[str, list[str]] = {
    "respiratory_failure_copd": [
        r"\brespiratory failure\b",
        r"\bhypox(?:emia|emic)\b",
        r"\bhypercap(?:nia|nic)\b",
        r"\bcopd\b",
        r"\bbronchospasm\b",
        r"\bwheez",
        r"\bintubat",
        r"\bventilat",
        r"\bextubat",
        r"\bpeep\b",
        r"\bfi ?o2\b",
        r"\bpco2\b",
        r"\bpo2\b",
        r"\bsbt\b",
        r"\brsbi\b",
        r"\bpsv\b",
    ],
    "infection": [
        r"\binfection\b",
        r"\bsepsis\b",
        r"\bseptic\b",
        r"\bfever\b",
        r"\bculture",
        r"\bwbc\b",
        r"\bvancomycin\b",
        r"\blevofloxacin\b",
        r"\bcefepime\b",
        r"\bceftriaxone\b",
        r"\bazithromycin\b",
        r"\bantibiotic",
    ],
    "pneumonia": [r"\bpneumonia\b", r"\bpna\b", r"\binfiltrate\b"],
    "skin_wound": [r"\bcellulitis\b", r"\bwound\b", r"\bulcer\b", r"\berythema\b", r"\bskin\b"],
    "volume": [
        r"\bvolume\b",
        r"\bfluid\b",
        r"\bdiures",
        r"\bfurosemide\b",
        r"\blasix\b",
        r"\bedema\b",
        r"\bhypervolem",
        r"\beuvolem",
        r"\bi/os\b",
        r"\bnet negative\b",
    ],
    "renal_aki": [
        r"\baki\b",
        r"\brenal\b",
        r"\bkidney\b",
        r"\bcreatinine\b",
        r"\bcr\b",
        r"\bbun\b",
        r"\burea nitrogen\b",
    ],
    "glucose_diabetes": [
        r"\bdiabetes\b",
        r"\bdm\b",
        r"\bglucose\b",
        r"\bhyperglyc",
        r"\bhypoglyc",
        r"\binsulin\b",
        r"\bglargine\b",
        r"\bssi\b",
    ],
    "heme": [
        r"\banemia\b",
        r"\bpolycythemia\b",
        r"\bhemoglobin\b",
        r"\bhgb\b",
        r"\bhematocrit\b",
        r"\bhct\b",
        r"\bplatelet",
        r"\bthrombocyt",
        r"\bbleed",
    ],
    "neuro_pain_sedation": [
        r"\bsedat",
        r"\banxiety\b",
        r"\bagitation\b",
        r"\bpain\b",
        r"\bfentanyl\b",
        r"\bmidazolam\b",
        r"\bversed\b",
        r"\brass\b",
        r"\bmental status\b",
    ],
    "fen_nutrition": [
        r"\bnutrition\b",
        r"\btube feed",
        r"\benteral\b",
        r"\bnutren\b",
        r"\bfree water\b",
        r"\bf[e/]n\b",
        r"\bpotassium\b",
        r"\bmagnesium\b",
        r"\bphosphate\b",
        r"\bcalcium\b",
    ],
    "prophylaxis_access": [
        r"\bprophylaxis\b",
        r"\bdvt\b",
        r"\bvte\b",
        r"\bheparin\b",
        r"\bpantoprazole\b",
        r"\bprotonix\b",
        r"\bline\b",
        r"\baccess\b",
        r"\barterial line\b",
    ],
    "cardiovascular": [
        r"\bshock\b",
        r"\bpressor",
        r"\bnorepinephrine\b",
        r"\bhypotension\b",
        r"\bhypertension\b",
        r"\barrhythmia\b",
        r"\bheart failure\b",
        r"\blvef\b",
    ],
}

TREATMENT_PATTERNS: dict[str, list[str]] = {
    "vent_wean": [r"\bwean", r"\bsbt\b", r"\brsbi\b", r"\bpsv\b", r"\bextubat"],
    "bronchodilator_steroid": [r"\balbuterol\b", r"\bbronchodilator", r"\bsteroid", r"\bmethylpred", r"\bmedrol\b"],
    "antibiotics": [r"\bvancomycin\b", r"\blevofloxacin\b", r"\bcefepime\b", r"\bceftriaxone\b", r"\bantibiotic"],
    "diuresis": [r"\bfurosemide\b", r"\blasix\b", r"\bdiures"],
    "insulin": [r"\binsulin\b", r"\bglargine\b", r"\bssi\b"],
    "sedation_analgesia": [r"\bfentanyl\b", r"\bmidazolam\b", r"\bversed\b", r"\bsedat"],
    "dvt_prophylaxis": [r"\bheparin\b", r"\bdvt\b", r"\bvte\b"],
    "stress_ulcer": [r"\bpantoprazole\b", r"\bprotonix\b", r"\bstress ulcer\b"],
    "nutrition": [r"\bnutren\b", r"\btube feed", r"\benteral\b", r"\bnutrition\b"],
}

LAB_ALIASES: dict[str, list[str]] = {
    "glucose": [r"glucose(?: \(serum\))?"],
    "bun": [r"bun", r"urea nitrogen"],
    "creatinine": [r"creatinine", r"\bcr\b"],
    "pco2": [r"pco2", r"arterial co2 pressure"],
    "po2": [r"po2", r"arterial o2 pressure"],
    "bicarbonate": [r"bicarbonate", r"hco3", r"tco2", r"calculated total co2"],
    "hemoglobin": [r"hemoglobin", r"\bhgb\b"],
    "hematocrit": [r"hematocrit", r"\bhct\b"],
    "wbc": [r"\bwbc\b"],
    "platelets": [r"platelet count", r"\bplt\b", r"platelets"],
}

TREND_WORDS = {
    "up": [
        "increase",
        "increased",
        "increasing",
        "rising",
        "rise",
        "worse",
        "worsening",
        "higher",
        "elevated",
        "uptrending",
        "progressive",
    ],
    "down": [
        "decrease",
        "decreased",
        "decreasing",
        "falling",
        "lower",
        "improved",
        "improving",
        "downtrending",
        "wean",
        "weaned",
        "reduced",
    ],
    "stable": ["stable", "unchanged", "persistent", "persistently", "remains", "continued"],
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)?")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    text = text.replace("\uFFFD", " ")
    text = text.lower()
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    text = re.sub(r"[^a-z0-9%./+\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def rouge_l_f1(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    prev = [0] * (len(gold_tokens) + 1)
    for ptok in pred_tokens:
        curr = [0]
        for j, gtok in enumerate(gold_tokens, start=1):
            if ptok == gtok:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    lcs = prev[-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return f1(precision, recall)


def compile_match(patterns: Iterable[str], text: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def extract_from_patterns(text: str, patterns_by_name: dict[str, list[str]]) -> set[str]:
    norm = normalize_text(text)
    return {name for name, patterns in patterns_by_name.items() if compile_match(patterns, norm)}


def set_scores(pred_items: set[str], gold_items: set[str]) -> tuple[float, float, float]:
    if not pred_items and not gold_items:
        return 1.0, 1.0, 1.0
    if not pred_items:
        return 0.0, 0.0, 0.0
    if not gold_items:
        return 0.0, 0.0, 0.0
    overlap = len(pred_items & gold_items)
    precision = overlap / len(pred_items)
    recall = overlap / len(gold_items)
    return precision, recall, f1(precision, recall)


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evidence_text(sample: dict | None) -> str:
    if not sample:
        return ""
    parts = [sample.get("current_note_context", "")]
    for event in sample.get("ehr_events_before_cutoff", []) or []:
        parts.append(event.get("text", ""))
    return "\n".join(parts)


def extract_numbers(text: str) -> set[str]:
    numbers = set()
    for value in NUMBER_RE.findall(text):
        try:
            num = float(value)
        except ValueError:
            continue
        if math.isfinite(num):
            numbers.add(f"{num:g}")
    return numbers


def numeric_claim_support(pred: str, evidence: str) -> tuple[float, int, int]:
    pred_numbers = extract_numbers(pred)
    if not pred_numbers:
        return 1.0, 0, 0
    evidence_numbers = extract_numbers(evidence)
    supported = len(pred_numbers & evidence_numbers)
    return supported / len(pred_numbers), supported, len(pred_numbers)


def extract_lab_values(text: str) -> dict[str, list[float]]:
    norm = normalize_text(text)
    values: dict[str, list[float]] = defaultdict(list)
    for lab, aliases in LAB_ALIASES.items():
        alias_group = "|".join(f"(?:{a})" for a in aliases)
        patterns = [
            rf"\b(?:{alias_group})\s*(?:is|=|:)?\s*([-+]?\d+(?:\.\d+)?)",
            rf"([-+]?\d+(?:\.\d+)?)\s*(?:mg/dl|g/dl|k/ul|mm ?hg|meq/l|%)?\s*(?:of\s+)?(?:{alias_group})\b",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, norm):
                try:
                    values[lab].append(float(match.group(1)))
                except ValueError:
                    pass
    return values


def infer_evidence_trends(evidence: str) -> dict[str, str]:
    values = extract_lab_values(evidence)
    trends = {}
    for lab, nums in values.items():
        if len(nums) < 2:
            continue
        first, last = nums[0], nums[-1]
        span = max(abs(first), abs(last), 1.0)
        delta = last - first
        if abs(delta) / span < 0.05:
            trends[lab] = "stable"
        elif delta > 0:
            trends[lab] = "up"
        else:
            trends[lab] = "down"
    return trends


def extract_trend_claims(text: str) -> dict[str, str]:
    norm = normalize_text(text)
    claims: dict[str, str] = {}
    for lab, aliases in LAB_ALIASES.items():
        alias_group = "|".join(f"(?:{a})" for a in aliases)
        for direction, words in TREND_WORDS.items():
            word_group = "|".join(re.escape(w) for w in words)
            patterns = [
                rf"(?:{alias_group}).{{0,60}}\b(?:{word_group})\b",
                rf"\b(?:{word_group})\b.{{0,60}}(?:{alias_group})",
            ]
            if any(re.search(pattern, norm) for pattern in patterns):
                claims[lab] = direction
                break
    return claims


def trend_scores(pred_trends: dict[str, str], gold_trends: dict[str, str]) -> tuple[float, float, float]:
    pred_pairs = {f"{k}:{v}" for k, v in pred_trends.items()}
    gold_pairs = {f"{k}:{v}" for k, v in gold_trends.items()}
    return set_scores(pred_pairs, gold_pairs)


def evidence_trend_accuracy(pred_trends: dict[str, str], evidence_trends: dict[str, str]) -> tuple[float, int, int]:
    checked = 0
    correct = 0
    for lab, direction in pred_trends.items():
        if lab not in evidence_trends:
            continue
        checked += 1
        if evidence_trends[lab] == direction:
            correct += 1
    if checked == 0:
        return 1.0, 0, 0
    return correct / checked, correct, checked


def load_samples(path: Path | None) -> dict[str, dict]:
    if not path:
        return {}
    return {row["sample_id"]: row for row in read_jsonl(path)}


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    return name.strip(), Path(path)


def evaluate_row(method: str, row: dict, sample: dict | None) -> dict:
    pred = row.get("generated_ap") or row.get("prediction") or row.get("text") or ""
    gold = row.get("gold_ap") or (sample or {}).get("gold_ap", "")
    evidence = evidence_text(sample)

    pred_problems = extract_from_patterns(pred, PROBLEM_PATTERNS)
    gold_problems = extract_from_patterns(gold, PROBLEM_PATTERNS)
    evidence_problems = extract_from_patterns(evidence, PROBLEM_PATTERNS)

    pred_treatments = extract_from_patterns(pred, TREATMENT_PATTERNS)
    gold_treatments = extract_from_patterns(gold, TREATMENT_PATTERNS)

    concept_pred = pred_problems | pred_treatments
    concept_gold = gold_problems | gold_treatments
    concept_evidence = evidence_problems | extract_from_patterns(evidence, TREATMENT_PATTERNS)

    concept_p, concept_r, concept_f = set_scores(concept_pred, concept_gold)
    problem_p, problem_r, problem_f = set_scores(pred_problems, gold_problems)
    treatment_p, treatment_r, treatment_f = set_scores(pred_treatments, gold_treatments)

    grounded = concept_pred & (concept_gold | concept_evidence)
    unsupported = concept_pred - (concept_gold | concept_evidence)
    grounded_rate = len(grounded) / len(concept_pred) if concept_pred else 1.0
    unsupported_rate = len(unsupported) / len(concept_pred) if concept_pred else 0.0

    numeric_rate, numeric_supported, numeric_total = numeric_claim_support(pred, evidence)

    pred_trends = extract_trend_claims(pred)
    gold_trends = extract_trend_claims(gold)
    trend_p, trend_r, trend_f = trend_scores(pred_trends, gold_trends)

    evidence_trends = infer_evidence_trends(evidence)
    trend_acc, trend_correct, trend_checked = evidence_trend_accuracy(pred_trends, evidence_trends)

    return {
        "method": method,
        "sample_id": row.get("sample_id", ""),
        "hadm_id": row.get("hadm_id", (sample or {}).get("hadm_id", "")),
        "day": row.get("day", (sample or {}).get("day", "")),
        "rouge_l_f1": rouge_l_f1(pred, gold),
        "clinical_concept_precision": concept_p,
        "clinical_concept_recall": concept_r,
        "clinical_concept_f1": concept_f,
        "problem_precision": problem_p,
        "problem_recall": problem_r,
        "problem_f1": problem_f,
        "treatment_precision": treatment_p,
        "treatment_recall": treatment_r,
        "treatment_f1": treatment_f,
        "grounded_concept_rate": grounded_rate,
        "unsupported_concept_rate": unsupported_rate,
        "numeric_claim_support_rate": numeric_rate,
        "numeric_supported": numeric_supported,
        "numeric_total": numeric_total,
        "trend_precision": trend_p,
        "trend_recall": trend_r,
        "trend_f1": trend_f,
        "evidence_trend_accuracy": trend_acc,
        "trend_correct": trend_correct,
        "trend_checked": trend_checked,
        "pred_concepts": "|".join(sorted(concept_pred)),
        "gold_concepts": "|".join(sorted(concept_gold)),
        "unsupported_concepts": "|".join(sorted(unsupported)),
        "pred_trends": "|".join(f"{k}:{v}" for k, v in sorted(pred_trends.items())),
        "gold_trends": "|".join(f"{k}:{v}" for k, v in sorted(gold_trends.items())),
        "evidence_trends": "|".join(f"{k}:{v}" for k, v in sorted(evidence_trends.items())),
    }


def summarize(rows: list[dict]) -> list[dict]:
    metric_cols = [
        "rouge_l_f1",
        "clinical_concept_precision",
        "clinical_concept_recall",
        "clinical_concept_f1",
        "problem_precision",
        "problem_recall",
        "problem_f1",
        "treatment_precision",
        "treatment_recall",
        "treatment_f1",
        "grounded_concept_rate",
        "unsupported_concept_rate",
        "numeric_claim_support_rate",
        "trend_precision",
        "trend_recall",
        "trend_f1",
        "evidence_trend_accuracy",
    ]
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    summary = []
    for method, method_rows in sorted(by_method.items()):
        out = {"method": method, "n": len(method_rows)}
        for col in metric_cols:
            vals = [float(r[col]) for r in method_rows]
            out[f"{col}_mean"] = mean(vals) if vals else 0.0
            out[f"{col}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        out["numeric_claims_total"] = sum(int(r["numeric_total"]) for r in method_rows)
        out["trend_claims_checked"] = sum(int(r["trend_checked"]) for r in method_rows)
        summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: list[dict]) -> None:
    cols = [
        ("method", "Method"),
        ("n", "N"),
        ("rouge_l_f1_mean", "ROUGE-L"),
        ("clinical_concept_f1_mean", "Concept F1"),
        ("problem_f1_mean", "Problem F1"),
        ("treatment_f1_mean", "Treatment F1"),
        ("grounded_concept_rate_mean", "Grounded"),
        ("unsupported_concept_rate_mean", "Unsupported"),
        ("numeric_claim_support_rate_mean", "Numeric Support"),
        ("trend_f1_mean", "Trend F1"),
        ("evidence_trend_accuracy_mean", "Evidence Trend Acc"),
    ]
    lines = [
        "| " + " | ".join(label for _, label in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in summary:
        cells = []
        for key, _ in cols:
            value = row[key]
            if isinstance(value, float):
                cells.append(f"{value * 100:.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=None, help="ProblemFlow ap_samples.jsonl with evidence.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Generation JSONL. Use name=path or just path. Can be repeated.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("evaluation/clinical_semantics_outputs"))
    args = parser.parse_args()

    samples = load_samples(args.samples)
    detail_rows = []
    for run_arg in args.run:
        method, path = parse_run(run_arg)
        for row in read_jsonl(path):
            sample = samples.get(row.get("sample_id", ""))
            detail_rows.append(evaluate_row(method, row, sample))

    summary_rows = summarize(detail_rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "clinical_semantics_detail.csv", detail_rows)
    write_csv(args.outdir / "clinical_semantics_summary.csv", summary_rows)
    write_markdown(args.outdir / "clinical_semantics_summary.md", summary_rows)

    print(f"Wrote detail:  {args.outdir / 'clinical_semantics_detail.csv'}")
    print(f"Wrote summary: {args.outdir / 'clinical_semantics_summary.csv'}")
    print(f"Wrote report:  {args.outdir / 'clinical_semantics_summary.md'}")
    for row in summary_rows:
        print(
            f"{row['method']}: n={row['n']} "
            f"ROUGE-L={row['rouge_l_f1_mean']*100:.2f} "
            f"Concept-F1={row['clinical_concept_f1_mean']*100:.2f} "
            f"Problem-F1={row['problem_f1_mean']*100:.2f} "
            f"Grounded={row['grounded_concept_rate_mean']*100:.2f} "
            f"Unsupported={row['unsupported_concept_rate_mean']*100:.2f} "
            f"Trend-F1={row['trend_f1_mean']*100:.2f}"
        )


if __name__ == "__main__":
    main()
