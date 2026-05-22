from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


DEFAULT_V2_QWEN = Path(
    "outputs/ap_memory_gated_scaffold_ap100/"
    "ap100full_generated_method2_gen_v2_eval_judge_qwen_siliconflow_detail.csv"
)
DEFAULT_JUDGE_REVISE_QWEN = Path(
    "outputs/ap_memory_gated_scaffold_ap100/"
    "ap100full_generated_method2_gen_v2_judge_revise_eval_judge_qwen_siliconflow_detail.csv"
)
DEFAULT_OUTPUT = Path("outputs/oracle_claim_verifier_qwen653/selected_cases.json")


CASE_TYPE_KEYWORDS = {
    "respiratory": [
        "respiratory",
        "vent",
        "intubat",
        "extubat",
        "trach",
        "peep",
        "fio2",
        "ards",
        "pneumonia",
        "oxygen",
        "sbt",
    ],
    "renal_metabolic": [
        "renal",
        "aki",
        "kidney",
        "crrt",
        "cvvhd",
        "dialysis",
        "bun",
        "creatinine",
        "electrolyte",
        "sodium",
        "phosphate",
    ],
    "hemodynamic_cardiac": [
        "pressor",
        "vasopressor",
        "norepinephrine",
        "phenylephrine",
        "shock",
        "hypotension",
        "hypertension",
        "cardiac",
        "afib",
        "arrhythmia",
        "ekg",
        "heparin",
    ],
    "infection": [
        "infection",
        "sepsis",
        "antibiotic",
        "culture",
        "fever",
        "vap",
        "pseudomonas",
        "uti",
        "vancomycin",
        "cefepime",
    ],
    "surgical_bleeding": [
        "bleed",
        "hemorrhage",
        "hematoma",
        "hgb",
        "hemoglobin",
        "transfusion",
        "warfarin",
        "urology",
        "surgery",
        "post-op",
        "procedure",
    ],
    "disposition_context": [
        "disposition",
        "family",
        "goals",
        "code status",
        "full code",
        "dnr",
        "transfer",
        "rehab",
        "discharge",
        "tracheostomy discussion",
    ],
    "nutrition_gi": [
        "nutrition",
        "tube feed",
        "feeding",
        "enteral",
        "tfn",
        "tpn",
        "diet",
        "gi",
        "ileus",
    ],
}

HIGH_RISK_KEYWORDS = [
    "crrt",
    "cvvhd",
    "dialysis",
    "extubat",
    "failed extubation",
    "trach",
    "pressor",
    "norepinephrine",
    "phenylephrine",
    "abg",
    "peep",
    "fio2",
    "ett",
    "bleed",
    "hemorrhage",
    "transfusion",
    "code status",
    "family",
]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    if value == "":
        return default
    return float(value)


def case_id(row: dict[str, str]) -> str:
    return f"{row['admission_id']}_day{row['day']}"


def norm_text(*parts: str) -> str:
    return " ".join(p or "" for p in parts).lower()


def classify_case_type(text: str) -> list[str]:
    found: list[str] = []
    for case_type, keywords in CASE_TYPE_KEYWORDS.items():
        if any(k in text for k in keywords):
            found.append(case_type)
    return found or ["mixed_unclear"]


def failure_modes(v2: dict[str, str], jr: dict[str, str] | None, text: str) -> list[str]:
    modes: list[str] = []
    v2_unsup = f(v2, "augmented_unsupported_problem_count")
    base_unsup = f(v2, "baseline_unsupported_problem_count")
    if v2_unsup >= 4 or v2_unsup > base_unsup:
        modes.append("unsupported_fabrication")
    if f(v2, "augmented_evidence_grounding") <= 2:
        modes.append("weak_evidence_grounding")
    if f(v2, "augmented_trajectory_capture") <= 2 or "trajectory" in text or "contradict" in text:
        modes.append("trajectory_or_state_error")
    if f(v2, "augmented_active_problem_coverage") <= 2 or f(v2, "augmented_missed_key_problem_count") >= 4:
        modes.append("coverage_loss_or_missed_problem")
    if f(v2, "augmented_disposition_context") <= 2 or "disposition" in text or "family" in text or "goals" in text:
        modes.append("disposition_context_gap")
    if any(k in text for k in HIGH_RISK_KEYWORDS):
        modes.append("high_risk_claim_needs_verification")
    if jr:
        jr_unsup = f(jr, "augmented_unsupported_problem_count")
        if jr.get("winner") != "augmented":
            modes.append("judge_revise_not_better_than_base")
        if jr_unsup >= v2_unsup and jr_unsup >= 3:
            modes.append("judge_revise_does_not_reduce_unsupported")
        if f(jr, "augmented_active_problem_coverage") < f(v2, "augmented_active_problem_coverage"):
            modes.append("judge_revise_coverage_drop")
        if f(jr, "augmented_disposition_context") < f(v2, "augmented_disposition_context"):
            modes.append("judge_revise_disposition_drop")
    return sorted(set(modes))


def score_case(v2: dict[str, str], jr: dict[str, str] | None) -> float:
    text = norm_text(
        v2.get("baseline_brief_rationale", ""),
        v2.get("augmented_brief_rationale", ""),
        jr.get("augmented_brief_rationale", "") if jr else "",
    )
    score = 0.0
    v2_unsup = f(v2, "augmented_unsupported_problem_count")
    base_unsup = f(v2, "baseline_unsupported_problem_count")
    score += v2_unsup * 2.0
    score += max(0.0, v2_unsup - base_unsup) * 1.5
    score += (5.0 - f(v2, "augmented_evidence_grounding")) * 1.2
    score += f(v2, "augmented_missed_key_problem_count") * 0.8
    score += max(0.0, 3.0 - f(v2, "augmented_active_problem_coverage")) * 1.2
    score += max(0.0, 3.0 - f(v2, "augmented_trajectory_capture"))
    score += max(0.0, 3.0 - f(v2, "augmented_disposition_context"))
    if v2.get("winner") == "baseline":
        score += 2.0
    elif v2.get("winner") == "tie":
        score += 1.0
    if any(k in text for k in HIGH_RISK_KEYWORDS):
        score += 2.0
    if jr:
        if jr.get("winner") != "augmented":
            score += 1.5
        score += max(
            0.0,
            f(jr, "augmented_unsupported_problem_count") - f(v2, "augmented_unsupported_problem_count"),
        )
        if f(jr, "augmented_active_problem_coverage") < f(v2, "augmented_active_problem_coverage"):
            score += 1.0
        if f(jr, "augmented_disposition_context") < f(v2, "augmented_disposition_context"):
            score += 1.0
    return round(score, 3)


def reason(v2: dict[str, str], jr: dict[str, str] | None, modes: list[str]) -> str:
    bits = [
        f"Qwen653 V2 winner={v2.get('winner')}",
        f"unsupported {v2.get('augmented_unsupported_problem_count')} vs base {v2.get('baseline_unsupported_problem_count')}",
        f"grounding={v2.get('augmented_evidence_grounding')}",
        f"coverage={v2.get('augmented_active_problem_coverage')}",
        f"missed={v2.get('augmented_missed_key_problem_count')}",
    ]
    if jr:
        bits.append(
            "judge-revise "
            f"winner={jr.get('winner')}, unsupported={jr.get('augmented_unsupported_problem_count')}, "
            f"coverage={jr.get('augmented_active_problem_coverage')}, "
            f"disposition={jr.get('augmented_disposition_context')}"
        )
    if modes:
        bits.append("failure_modes=" + ",".join(modes[:4]))
    rationale = v2.get("augmented_brief_rationale", "").strip()
    if rationale:
        rationale = re.sub(r"\s+", " ", rationale)
        bits.append("Qwen rationale: " + rationale[:260])
    return "; ".join(bits)


def build_candidate(row: dict[str, str], jr: dict[str, str] | None) -> dict | None:
    cid = case_id(row)
    text = norm_text(
        row.get("baseline_brief_rationale", ""),
        row.get("augmented_brief_rationale", ""),
        jr.get("baseline_brief_rationale", "") if jr else "",
        jr.get("augmented_brief_rationale", "") if jr else "",
    )
    modes = failure_modes(row, jr, text)
    if not modes:
        return None
    return {
        "case_id": cid,
        "admission_id": row["admission_id"],
        "day": int(row["day"]),
        "selection_score": score_case(row, jr),
        "reason_for_selection": reason(row, jr, modes),
        "case_type": classify_case_type(text),
        "failure_modes": modes,
        "qwen653_v2": {
            "winner": row.get("winner"),
            "baseline_unsupported_problem_count": f(row, "baseline_unsupported_problem_count"),
            "v2_unsupported_problem_count": f(row, "augmented_unsupported_problem_count"),
            "v2_evidence_grounding": f(row, "augmented_evidence_grounding"),
            "v2_active_problem_coverage": f(row, "augmented_active_problem_coverage"),
            "v2_trajectory_capture": f(row, "augmented_trajectory_capture"),
            "v2_disposition_context": f(row, "augmented_disposition_context"),
            "v2_missed_key_problem_count": f(row, "augmented_missed_key_problem_count"),
        },
        "qwen_judge_revise": (
            {
                "winner": jr.get("winner"),
                "unsupported_problem_count": f(jr, "augmented_unsupported_problem_count"),
                "evidence_grounding": f(jr, "augmented_evidence_grounding"),
                "active_problem_coverage": f(jr, "augmented_active_problem_coverage"),
                "trajectory_capture": f(jr, "augmented_trajectory_capture"),
                "disposition_context": f(jr, "augmented_disposition_context"),
                "missed_key_problem_count": f(jr, "augmented_missed_key_problem_count"),
            }
            if jr
            else None
        ),
    }


def select_cases(
    v2_rows: list[dict[str, str]],
    jr_rows: list[dict[str, str]],
    limit: int,
    must_include: list[str] | None = None,
) -> list[dict]:
    jr_by_case = {case_id(row): row for row in jr_rows}
    must_include = must_include or []
    must_include_set = set(must_include)
    candidates = []
    for row in v2_rows:
        candidate = build_candidate(row, jr_by_case.get(case_id(row)))
        if candidate:
            candidates.append(candidate)

    candidates.sort(key=lambda x: x["selection_score"], reverse=True)
    by_case = {item["case_id"]: item for item in candidates}

    selected: list[dict] = []
    per_admission: dict[str, int] = {}
    type_counts: dict[str, int] = {}

    def can_add(item: dict, relaxed: bool = False) -> bool:
        cap = 3 if not relaxed else 5
        if per_admission.get(item["admission_id"], 0) >= cap:
            return False
        if not relaxed and all(type_counts.get(t, 0) >= 8 for t in item["case_type"]):
            return False
        return True

    def add(item: dict) -> None:
        selected.append(item)
        per_admission[item["admission_id"]] = per_admission.get(item["admission_id"], 0) + 1
        for t in item["case_type"]:
            type_counts[t] = type_counts.get(t, 0) + 1

    for cid in must_include:
        item = by_case.get(cid)
        if item and item not in selected and len(selected) < limit:
            add(item)

    for item in candidates:
        if len(selected) >= limit:
            break
        if item["case_id"] in must_include_set:
            continue
        if can_add(item):
            add(item)

    for item in candidates:
        if len(selected) >= limit:
            break
        if item in selected or item["case_id"] in must_include_set:
            continue
        if can_add(item, relaxed=True):
            add(item)

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select AP100 cases for oracle claim-verifier upper-bound validation using Qwen full-653 judge results."
    )
    parser.add_argument("--v2-qwen", type=Path, default=DEFAULT_V2_QWEN)
    parser.add_argument("--judge-revise-qwen", type=Path, default=DEFAULT_JUDGE_REVISE_QWEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--must-include",
        nargs="*",
        default=[],
        help="Case ids, such as 105351_day13, to force into the selected set when present in Qwen653.",
    )
    args = parser.parse_args()

    v2_rows = load_rows(args.v2_qwen)
    jr_rows = load_rows(args.judge_revise_qwen) if args.judge_revise_qwen.exists() else []
    selected = select_cases(v2_rows, jr_rows, args.limit, args.must_include)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {len(selected)} selected cases -> {args.output}")
    print("Top cases:")
    for item in selected[:10]:
        print(f"- {item['case_id']}: score={item['selection_score']} types={','.join(item['case_type'])}")


if __name__ == "__main__":
    main()
