#!/usr/bin/env python3
"""Score base vs V2 outputs with the HealthBench-style AP rubric seed."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.healthbench_rubrics_manual_seed import (  # noqa: E402
    PROBLEMS,
    TREATMENTS,
    TRAJECTORY_PATTERNS,
    build_rubric,
    extract_matches,
    normalize,
    select_cases,
)


DANGEROUS_PLAN_PATTERNS = [
    r"\bextubat",
    r"\bstart\b.{0,40}\bpressor",
    r"\bstop\b.{0,40}\bpressor",
    r"\btransfus",
    r"\btherapeutic anticoag",
    r"\bbroad[- ]spectrum antibiotic",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_day_text_csv(path: Path, day: int) -> str:
    df = pd.read_csv(path)
    if "DAY" in df.columns:
        day_df = df[df["DAY"].astype(str).eq(str(day))]
    else:
        day_df = df
    if "TEXT" not in day_df.columns:
        return ""
    return " ".join(day_df["TEXT"].fillna("").astype(str).tolist())


def build_sample_from_csv(data_root: Path, hadm_id: str, day: int) -> dict:
    gold_path = data_root / "AP" / "gold" / f"gt_{hadm_id}.csv"
    input_path = data_root / "AP" / "input" / f"input_{hadm_id}.csv"
    gold_ap = read_day_text_csv(gold_path, day)
    input_df = pd.read_csv(input_path)
    if "DAY" in input_df.columns:
        input_df = input_df[input_df["DAY"].astype(str).eq(str(day))]
    events = []
    for idx, row in input_df.iterrows():
        text = str(row.get("TEXT", ""))
        if text.strip():
            events.append(
                {
                    "evidence_id": f"{hadm_id}_day{day}_raw{len(events):04d}",
                    "rel_time": str(row.get("REL_TIME", "")),
                    "text": text,
                    "source_type": "ehr_event" if not bool(row.get("IS_NOTE", False)) else "note",
                }
            )
    return {
        "sample_id": f"{hadm_id}_day{day}",
        "hadm_id": hadm_id,
        "day": day,
        "current_note_context": "",
        "gold_ap": gold_ap,
        "ehr_events_before_cutoff": events,
    }


def load_base_output(base_dir: Path, hadm_id: str, day: int) -> str:
    return read_day_text_csv(base_dir / f"genpns_{hadm_id}.csv", day)


def load_txt_output(output_dir: Path, hadm_id: str, day: int) -> str:
    path = output_dir / f"{hadm_id}_day{day}.txt"
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def trajectory_matches(text: str, expected: str) -> bool:
    if not expected or expected == "unclear":
        return any(re.search(pattern, normalize(text)) for patterns in TRAJECTORY_PATTERNS.values() for pattern in patterns)
    return any(re.search(pattern, normalize(text)) for pattern in TRAJECTORY_PATTERNS.get(expected, []))


def score_negative(item: dict, text: str, rubric: dict) -> tuple[bool, str]:
    norm = normalize(text)
    refs = set(rubric.get("reference_problem_ids", []))
    treatments = set(rubric.get("reference_treatment_ids", []))
    pred_problems = extract_matches(text, PROBLEMS)
    pred_treatments = extract_matches(text, TREATMENTS)

    criterion = item["criterion"].lower()
    if "aki" in criterion or "renal failure" in criterion:
        violated = "renal_aki" in pred_problems and "renal_aki" not in refs
        return violated, "active renal/AKI introduced without reference support" if violated else ""
    if "shock" in criterion or "pressor" in criterion or "heart failure" in criterion:
        cv_terms = bool(re.search(r"\bshock\b|\bpressor|\bnorepinephrine\b|\bheart failure\b|\barrhythmia\b", norm))
        violated = cv_terms and "cardiovascular" not in refs
        return violated, "cardiovascular instability introduced without reference support" if violated else ""
    if "antibiotic" in criterion and "without" in criterion:
        violated = "antibiotics" in pred_treatments and "antibiotics" not in treatments
        return violated, "antibiotics recommended without reference support" if violated else ""
    if item["category"] == "safety":
        violated = any(re.search(pattern, norm) for pattern in DANGEROUS_PLAN_PATTERNS)
        return violated, "high-risk plan language detected" if violated else ""
    if item["category"] == "evidence_grounding":
        extra_problems = pred_problems - refs
        high_risk_extra = extra_problems & {"respiratory_failure_copd", "infection", "pneumonia", "cardiovascular", "renal_aki"}
        violated = bool(high_risk_extra)
        return violated, f"extra high-risk problems: {','.join(sorted(high_risk_extra))}" if violated else ""
    return False, ""


def score_item(item: dict, text: str, rubric: dict) -> dict:
    category = item["category"]
    target = item.get("target_id", "")
    met = False
    reason = ""
    if item["type"] == "positive":
        if category == "problem_coverage":
            met = target in extract_matches(text, PROBLEMS)
            reason = "target problem found" if met else "target problem missing"
        elif category == "plan_quality":
            met = target in extract_matches(text, TREATMENTS)
            reason = "target treatment/plan found" if met else "target treatment/plan missing"
        elif category == "clinical_state":
            problem_ok = target in extract_matches(text, PROBLEMS)
            traj_ok = trajectory_matches(text, item.get("expected", ""))
            met = problem_ok and traj_ok
            reason = f"problem_ok={problem_ok}; trajectory_ok={traj_ok}"
    else:
        violated, reason = score_negative(item, text, rubric)
        met = violated
    contribution = item["points"] if met else 0
    return {
        "criterion_id": item["id"],
        "category": category,
        "type": item["type"],
        "points": item["points"],
        "target_id": target,
        "met": met,
        "contribution": contribution,
        "reason": reason,
    }


def score_rubric(rubric: dict, text: str, method: str) -> tuple[dict, list[dict]]:
    rows = []
    for item in rubric["criteria"]:
        row = score_item(item, text, rubric)
        row.update({"sample_id": rubric["sample_id"], "hadm_id": rubric["hadm_id"], "day": rubric["day"], "method": method})
        rows.append(row)

    pos_total = sum(item["points"] for item in rubric["criteria"] if item["points"] > 0)
    neg_total = -sum(item["points"] for item in rubric["criteria"] if item["points"] < 0)
    positive_earned = sum(r["contribution"] for r in rows if r["points"] > 0)
    penalty = -sum(r["contribution"] for r in rows if r["points"] < 0)
    raw = positive_earned - penalty
    normalized = raw / pos_total if pos_total else 0.0
    summary = {
        "sample_id": rubric["sample_id"],
        "hadm_id": rubric["hadm_id"],
        "day": rubric["day"],
        "method": method,
        "positive_total": pos_total,
        "negative_total": neg_total,
        "positive_earned": positive_earned,
        "penalty": penalty,
        "raw_score": raw,
        "normalized_score": normalized,
    }
    by_cat = defaultdict(int)
    by_cat_total = defaultdict(int)
    for item in rubric["criteria"]:
        if item["points"] > 0:
            by_cat_total[item["category"]] += item["points"]
    for r in rows:
        if r["points"] > 0 and r["met"]:
            by_cat[r["category"]] += r["points"]
        if r["points"] < 0 and r["met"]:
            by_cat[r["category"] + "_penalty"] += -r["points"]
    for cat, total in by_cat_total.items():
        summary[f"{cat}_score"] = by_cat[cat] / total if total else 0.0
    for cat in ["evidence_grounding_penalty", "unsupported_problem_penalty", "unsupported_treatment_penalty", "safety_penalty"]:
        summary[cat] = by_cat[cat]
    return summary, rows


def select_ap100_cases(summary_csv: Path, n: int) -> list[tuple[str, int]]:
    df = pd.read_csv(summary_csv)
    rows = [{"sample_id": f"{r.admission_id}_day{int(r.day)}", "hadm_id": str(r.admission_id), "day": int(r.day)} for r in df.itertuples()]
    selected = select_cases(rows, n)
    return [(str(row["hadm_id"]), int(row["day"])) for row in selected]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data_ap100_ap"))
    parser.add_argument("--base-dir", type=Path, default=Path("data_ap100_ap/AP/generated/DG/deepseek_api_full_gen/gen/method2"))
    parser.add_argument("--v2-dir", type=Path, default=Path("outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2"))
    parser.add_argument("--case-summary", type=Path, default=Path("outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2_summary.csv"))
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--outdir", type=Path, default=Path("evaluation/healthbench_style_rubrics/ap100_base_vs_v2"))
    args = parser.parse_args()

    cases = select_ap100_cases(args.case_summary, args.n)
    rubrics = []
    case_rows = []
    detail_rows = []
    for hadm_id, day in cases:
        sample = build_sample_from_csv(args.data_root, hadm_id, day)
        rubric = build_rubric(sample)
        rubrics.append(rubric)
        outputs = {
            "base": load_base_output(args.base_dir, hadm_id, day),
            "v2": load_txt_output(args.v2_dir, hadm_id, day),
        }
        for method, text in outputs.items():
            summary, details = score_rubric(rubric, text, method)
            summary["output_words"] = len(normalize(text).split())
            case_rows.append(summary)
            detail_rows.extend(details)

    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "ap100_rubrics_40.jsonl").open("w", encoding="utf-8") as handle:
        for rubric in rubrics:
            handle.write(json.dumps(rubric, ensure_ascii=False) + "\n")
    write_csv(args.outdir / "rubric_case_scores.csv", case_rows)
    write_csv(args.outdir / "rubric_criterion_scores.csv", detail_rows)

    case_df = pd.DataFrame(case_rows)
    metric_cols = [c for c in case_df.columns if c.endswith("_score") or c.endswith("_penalty") or c in {"positive_earned", "penalty", "raw_score", "normalized_score", "output_words"}]
    summary_rows = []
    for method, group in case_df.groupby("method"):
        out = {"method": method, "n": len(group)}
        for col in metric_cols:
            out[col + "_mean"] = group[col].mean()
        summary_rows.append(out)
    summary_df = pd.DataFrame(summary_rows)

    pivot = case_df.pivot(index="sample_id", columns="method", values="normalized_score").dropna()
    delta = {
        "paired_n": len(pivot),
        "v2_minus_base_normalized": (pivot["v2"] - pivot["base"]).mean() if {"v2", "base"} <= set(pivot.columns) else 0.0,
        "v2_wins": int((pivot["v2"] > pivot["base"]).sum()) if {"v2", "base"} <= set(pivot.columns) else 0,
        "base_wins": int((pivot["base"] > pivot["v2"]).sum()) if {"v2", "base"} <= set(pivot.columns) else 0,
        "ties": int((pivot["base"] == pivot["v2"]).sum()) if {"v2", "base"} <= set(pivot.columns) else 0,
    }
    summary_df.to_csv(args.outdir / "rubric_model_summary.csv", index=False)
    (args.outdir / "rubric_paired_delta.json").write_text(json.dumps(delta, indent=2), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(json.dumps(delta, indent=2))
    print(f"Wrote {args.outdir}")


if __name__ == "__main__":
    main()
