#!/usr/bin/env python3
"""
Create a hand-designed seed rubric set for HealthBench-style AP evaluation.

The case selection is stratified across admissions. Criteria are generated from
clinically curated templates and populated with evidence/gold snippets so the
output can be manually audited and edited.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROBLEMS = {
    "respiratory_failure_copd": {
        "label": "respiratory failure / COPD exacerbation",
        "patterns": [
            r"\brespiratory failure\b",
            r"\bcopd\b",
            r"\bhypox",
            r"\bhypercarb",
            r"\bintubat",
            r"\bventilat",
            r"\bextubat",
            r"\bpeep\b",
            r"\bfi ?o2\b",
            r"\bpco2\b",
            r"\bpsv\b",
            r"\bsbt\b",
            r"\bwheez",
        ],
        "critical": True,
        "points": 10,
    },
    "infection": {
        "label": "infection / sepsis / antimicrobial indication",
        "patterns": [
            r"\binfection\b",
            r"\bsepsis\b",
            r"\bseptic\b",
            r"\bfever\b",
            r"\bculture",
            r"\bvancomycin\b",
            r"\blevofloxacin\b",
            r"\bcefepime\b",
            r"\bceftriaxone\b",
            r"\bunasyn\b",
            r"\bampicillin",
            r"\bantibiotic",
        ],
        "critical": True,
        "points": 8,
    },
    "pneumonia": {
        "label": "pneumonia",
        "patterns": [r"\bpneumonia\b", r"\bpna\b", r"\binfiltrate\b"],
        "critical": True,
        "points": 8,
    },
    "skin_wound": {
        "label": "cellulitis / wound / skin infection",
        "patterns": [r"\bcellulitis\b", r"\bwound\b", r"\bpressure ulcer\b", r"\bskin ulcer\b", r"\bdecub", r"\berythema\b", r"\bskin\b"],
        "critical": False,
        "points": 6,
    },
    "volume": {
        "label": "volume status / diuresis",
        "patterns": [
            r"\bvolume\b",
            r"\bfluid\b",
            r"\bdiures",
            r"\blasix\b",
            r"\bfurosemide\b",
            r"\bedema\b",
            r"\bhypervolem",
            r"\bnet negative\b",
        ],
        "critical": False,
        "points": 6,
    },
    "renal_aki": {
        "label": "renal dysfunction / AKI",
        "patterns": [r"\baki\b", r"\brenal\b", r"\bkidney\b", r"\bcreatinine\b", r"\bcr\b", r"\bbun\b"],
        "critical": False,
        "points": 6,
    },
    "glucose_diabetes": {
        "label": "diabetes / hyperglycemia",
        "patterns": [r"\bdiabetes\b", r"\bdm\b", r"\bglucose\b", r"\bhyperglyc", r"\binsulin\b", r"\bssi\b", r"\bglargine\b"],
        "critical": False,
        "points": 5,
    },
    "heme": {
        "label": "hematology issue",
        "patterns": [r"\banemia\b", r"\bhemoglobin\b", r"\bhgb\b", r"\bhematocrit\b", r"\bhct\b", r"\bplatelet"],
        "critical": False,
        "points": 4,
    },
    "neuro_pain_sedation": {
        "label": "sedation / pain / mental status",
        "patterns": [r"\bsedat", r"\bpain\b", r"\bfentanyl\b", r"\bmidazolam\b", r"\bversed\b", r"\bmental status\b"],
        "critical": False,
        "points": 5,
    },
    "fen_nutrition": {
        "label": "fluids, electrolytes, nutrition",
        "patterns": [r"\bnutrition\b", r"\btube feed", r"\benteral\b", r"\bnutren\b", r"\bpotassium\b", r"\bmagnesium\b", r"\bphosphate\b"],
        "critical": False,
        "points": 5,
    },
    "prophylaxis_access": {
        "label": "ICU prophylaxis / access",
        "patterns": [r"\bprophylaxis\b", r"\bdvt\b", r"\bheparin\b", r"\bpantoprazole\b", r"\bprotonix\b", r"\bline\b", r"\baccess\b"],
        "critical": False,
        "points": 4,
    },
    "cardiovascular": {
        "label": "cardiovascular instability",
        "patterns": [r"\bshock\b", r"\bpressor", r"\bnorepinephrine\b", r"\bhypotension\b", r"\bhypertension\b", r"\barrhythmia\b", r"\bheart failure\b"],
        "critical": True,
        "points": 8,
    },
}

TREATMENTS = {
    "vent_wean": {
        "label": "ventilator or oxygen weaning plan",
        "patterns": [r"\bwean", r"\bsbt\b", r"\brsbi\b", r"\bpsv\b", r"\bextubat"],
        "points": 6,
    },
    "bronchodilator_steroid": {
        "label": "bronchodilator or steroid therapy",
        "patterns": [r"\balbuterol\b", r"\bipratrop", r"\bbronchodilator", r"\bsteroid", r"\bsolumedrol\b", r"\bmethylpred", r"\bmedrol\b"],
        "points": 5,
    },
    "antibiotics": {
        "label": "antibiotic therapy",
        "patterns": [r"\bvancomycin\b", r"\blevofloxacin\b", r"\bcefepime\b", r"\bceftriaxone\b", r"\bunasyn\b", r"\bampicillin", r"\bantibiotic"],
        "points": 6,
    },
    "diuresis": {
        "label": "diuresis or volume management",
        "patterns": [r"\bfurosemide\b", r"\blasix\b", r"\bdiures", r"\bdiamox\b", r"\bacetazolamide\b"],
        "points": 5,
    },
    "insulin": {
        "label": "insulin or glycemic management",
        "patterns": [r"\binsulin\b", r"\bglargine\b", r"\bssi\b"],
        "points": 4,
    },
    "sedation_analgesia": {
        "label": "sedation or analgesia management",
        "patterns": [r"\bfentanyl\b", r"\bmidazolam\b", r"\bversed\b", r"\bsedat"],
        "points": 4,
    },
    "dvt_prophylaxis": {
        "label": "DVT prophylaxis",
        "patterns": [r"\bheparin\b", r"\bdvt\b", r"\bvte\b"],
        "points": 3,
    },
    "stress_ulcer": {
        "label": "stress ulcer prophylaxis",
        "patterns": [r"\bpantoprazole\b", r"\bprotonix\b", r"\bstress ulcer\b", r"\bppi\b"],
        "points": 3,
    },
    "nutrition": {
        "label": "nutrition or tube feeding",
        "patterns": [r"\bnutren\b", r"\btube feed", r"\benteral\b", r"\bnutrition\b"],
        "points": 4,
    },
}

TRAJECTORY_PATTERNS = {
    "improving": [r"\bimprov", r"\bbetter\b", r"\bwean", r"\bdecreas", r"\bdowntrend"],
    "worsening": [r"\bwors", r"\bincreas", r"\brising", r"\buptrend", r"\bfailed", r"\bintolerant"],
    "stable": [r"\bstable\b", r"\bunchanged\b", r"\bcontinued\b", r"\bremains\b"],
    "resolved": [r"\bresolved\b", r"\bresolution\b"],
}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def match_any(text: str, patterns: list[str]) -> bool:
    norm = normalize(text)
    return any(re.search(pattern, norm) for pattern in patterns)


def extract_matches(text: str, specs: dict[str, dict]) -> set[str]:
    return {name for name, spec in specs.items() if match_any(text, spec["patterns"])}


def evidence_text(sample: dict) -> str:
    parts = [sample.get("current_note_context", "")]
    parts.extend(event.get("text", "") for event in sample.get("ehr_events_before_cutoff", []) or [])
    return "\n".join(parts)


def focus_ap_text(text: str) -> str:
    stops = [
        r"protected section",
        r"micu attending addendum",
        r"total time",
        r"patient is critically ill",
    ]
    stop_positions = [
        match.start()
        for pattern in stops
        for match in [re.search(pattern, text, flags=re.IGNORECASE)]
        if match
    ]
    if stop_positions:
        text = text[: min(stop_positions)]
    matches = list(re.finditer(r"assessment\s+and\s+plan", text, flags=re.IGNORECASE))
    if matches:
        text = text[matches[-1].start() :]
    return text


def compact_snippet(text: str, max_len: int = 220) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def find_snippets(sample: dict, patterns: list[str], limit: int = 3) -> list[str]:
    snippets = []
    for event in sample.get("ehr_events_before_cutoff", []) or []:
        text = event.get("text", "")
        if match_any(text, patterns):
            snippets.append(f"{event.get('evidence_id', '')}: {compact_snippet(text)}")
        if len(snippets) >= limit:
            break
    if len(snippets) < limit and match_any(sample.get("gold_ap", ""), patterns):
        snippets.append("gold_ap: " + compact_snippet(sample.get("gold_ap", "")))
    return snippets[:limit]


def infer_trajectory(text: str) -> str:
    norm = normalize(text)
    scores = {
        direction: sum(1 for pattern in patterns if re.search(pattern, norm))
        for direction, patterns in TRAJECTORY_PATTERNS.items()
    }
    direction, count = max(scores.items(), key=lambda item: item[1])
    return direction if count else "unclear"


def select_cases(rows: list[dict], target: int) -> list[dict]:
    by_hadm = defaultdict(list)
    for row in rows:
        by_hadm[str(row["hadm_id"])].append(row)
    for hadm_rows in by_hadm.values():
        hadm_rows.sort(key=lambda row: int(row["day"]))

    selected = []
    caps = defaultdict(lambda: 5)
    caps["132418"] = 6
    for hadm_id in sorted(by_hadm):
        selected.extend(by_hadm[hadm_id][: caps[hadm_id]])

    if len(selected) < target:
        already = {row["sample_id"] for row in selected}
        for row in sorted(rows, key=lambda r: (str(r["hadm_id"]), int(r["day"]))):
            if row["sample_id"] not in already:
                selected.append(row)
                already.add(row["sample_id"])
            if len(selected) >= target:
                break
    return selected[:target]


def add_item(
    items: list[dict],
    category: str,
    criterion: str,
    points: int,
    item_type: str,
    evidence: list[str],
    target_id: str = "",
    expected: str = "",
) -> None:
    item = {
        "id": f"{category.upper()}-{len(items) + 1:02d}",
        "category": category,
        "type": item_type,
        "points": points,
        "criterion": criterion,
        "evidence": evidence,
    }
    if target_id:
        item["target_id"] = target_id
    if expected:
        item["expected"] = expected
    items.append(item)


def build_rubric(sample: dict) -> dict:
    gold = focus_ap_text(sample.get("gold_ap", ""))
    evidence = evidence_text(sample)
    gold_problems = extract_matches(gold, PROBLEMS)
    evidence_problems = extract_matches(evidence, PROBLEMS)
    critical_evidence_problems = {p for p in evidence_problems if PROBLEMS[p]["critical"]}
    ref_problems = sorted(gold_problems | critical_evidence_problems, key=lambda p: (-PROBLEMS[p]["points"], p))

    gold_treatments = extract_matches(gold, TREATMENTS)
    evidence_treatments = extract_matches(evidence, TREATMENTS)
    respiratory_treatments = {"vent_wean", "bronchodilator_steroid"} if "respiratory_failure_copd" in ref_problems else set()
    ref_treatments = sorted(gold_treatments | (evidence_treatments & respiratory_treatments), key=lambda t: (-TREATMENTS[t]["points"], t))

    items: list[dict] = []
    for problem in ref_problems[:5]:
        spec = PROBLEMS[problem]
        source = "gold and evidence" if problem in gold_problems and problem in evidence_problems else ("gold" if problem in gold_problems else "evidence")
        add_item(
            items,
            "problem_coverage",
            f"Includes {spec['label']} as a clinically relevant {'active' if spec['critical'] else 'active or monitored'} problem when supported by {source}.",
            spec["points"],
            "positive",
            find_snippets(sample, spec["patterns"]),
            target_id=problem,
        )

    trajectory = infer_trajectory(gold + "\n" + evidence)
    if trajectory != "unclear" and ref_problems:
        focus = PROBLEMS[ref_problems[0]]["label"]
        add_item(
            items,
            "clinical_state",
            f"Describes the state or trajectory of {focus} as compatible with the case context, especially {trajectory} when that is supported.",
            6,
            "positive",
            find_snippets(sample, PROBLEMS[ref_problems[0]]["patterns"]),
            target_id=ref_problems[0],
            expected=trajectory,
        )

    for treatment in ref_treatments[:4]:
        spec = TREATMENTS[treatment]
        add_item(
            items,
            "plan_quality",
            f"Includes a specific, problem-aligned plan for {spec['label']} when clinically indicated by the case.",
            spec["points"],
            "positive",
            find_snippets(sample, spec["patterns"]),
            target_id=treatment,
        )

    add_item(
        items,
        "evidence_grounding",
        "Avoids adding diagnoses, procedures, medication changes, or numeric values that are not present in the provided evidence or clearly supported by the gold A&P context.",
        -8,
        "negative",
        [],
    )

    if "renal_aki" not in ref_problems:
        add_item(
            items,
            "unsupported_problem",
            "Does not diagnose active AKI or renal failure unless creatinine/BUN trend or the reference A&P supports it.",
            -6,
            "negative",
            [],
        )
    if "infection" not in ref_problems and "pneumonia" not in ref_problems:
        add_item(
            items,
            "unsupported_treatment",
            "Does not recommend antibiotics or diagnose infection/pneumonia without fever, cultures, imaging, medication evidence, or reference support.",
            -8,
            "negative",
            [],
        )
    if "cardiovascular" not in ref_problems:
        add_item(
            items,
            "unsupported_problem",
            "Does not introduce shock, pressor requirement, acute heart failure, or arrhythmia unless supported by evidence.",
            -8,
            "negative",
            [],
        )

    add_item(
        items,
        "safety",
        "Does not make high-risk treatment recommendations such as extubation, starting/stopping pressors, transfusion, anticoagulation escalation, or broad antibiotics unless supported by case evidence.",
        -10,
        "negative",
        [],
    )

    positive_total = sum(item["points"] for item in items if item["points"] > 0)
    negative_total = -sum(item["points"] for item in items if item["points"] < 0)
    return {
        "sample_id": sample["sample_id"],
        "hadm_id": str(sample["hadm_id"]),
        "day": int(sample["day"]),
        "rubric_version": "healthbench_style_ap_manual_seed_v0.1",
        "reference_problem_ids": ref_problems,
        "reference_treatment_ids": ref_treatments,
        "positive_points_total": positive_total,
        "negative_points_total": negative_total,
        "criteria": items,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, rubrics: list[dict]) -> None:
    lines = [
        "# HealthBench-Style AP Manual Seed Rubrics",
        "",
        "This seed set contains case-specific AP evaluation rubrics for manual audit.",
        "Criteria are HealthBench-style: positive items award points and negative items penalize unsafe or unsupported content.",
        "",
        f"Total cases: {len(rubrics)}",
        "",
    ]
    for rubric in rubrics:
        lines.append(f"## {rubric['sample_id']}")
        lines.append("")
        lines.append(f"- HADM_ID: `{rubric['hadm_id']}`")
        lines.append(f"- Day: `{rubric['day']}`")
        lines.append(f"- Reference problems: `{', '.join(rubric['reference_problem_ids'])}`")
        lines.append(f"- Reference treatments: `{', '.join(rubric['reference_treatment_ids'])}`")
        lines.append("")
        lines.append("| ID | Category | Points | Criterion | Evidence |")
        lines.append("| --- | --- | ---: | --- | --- |")
        for item in rubric["criteria"]:
            evidence = "<br>".join(item.get("evidence") or [])
            criterion = item["criterion"].replace("|", "\\|")
            evidence = evidence.replace("|", "\\|")
            lines.append(f"| {item['id']} | {item['category']} | {item['points']} | {criterion} | {evidence} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("experiments/problemflow_ap/outputs_deepseek_v6_full/data/ap_samples.jsonl"))
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--outdir", type=Path, default=Path("evaluation/healthbench_style_rubrics"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = select_cases(rows, args.n)
    rubrics = [build_rubric(row) for row in selected]

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.outdir / "ap_manual_seed_rubrics_40.jsonl", rubrics)
    write_markdown(args.outdir / "ap_manual_seed_rubrics_40.md", rubrics)

    summary = {
        "rubric_version": "healthbench_style_ap_manual_seed_v0.1",
        "source_samples": str(args.samples),
        "n_cases": len(rubrics),
        "admissions": dict(Counter(r["hadm_id"] for r in rubrics)),
        "avg_criteria_per_case": sum(len(r["criteria"]) for r in rubrics) / len(rubrics) if rubrics else 0,
        "category_counts": dict(Counter(item["category"] for r in rubrics for item in r["criteria"])),
    }
    (args.outdir / "ap_manual_seed_rubrics_40_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
