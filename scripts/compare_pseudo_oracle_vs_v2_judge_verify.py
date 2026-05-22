from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


DEFAULT_SELECTED = Path("outputs/oracle_claim_verifier_qwen653/selected_cases_with_failure_seed.json")
DEFAULT_V2_JUDGE_DIR = Path(
    "outputs/ap_memory_gated_scaffold_ap100/generation_judges/ap100full_generated_method2_gen_v2_judge_revise"
)
DEFAULT_PSEUDO_LABELS = Path(
    "outputs/oracle_claim_verifier_qwen653/pseudo_verifier_oracle/claims_oracle_labeled_pseudo.jsonl"
)
DEFAULT_QWEN_DELTA = Path(
    "outputs/oracle_claim_verifier_qwen653/qwen25_selected30_upper_bound_comparison/qualitative_delta_cases.csv"
)
DEFAULT_OUTDIR = Path("outputs/oracle_claim_verifier_qwen653/verify_truth_diff")


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "today",
    "current",
    "continue",
    "monitor",
    "given",
    "patient",
    "plan",
    "section",
    "active",
    "problem",
    "problems",
}


def tokens(text: str) -> set[str]:
    return {tok for tok in TOKEN_RE.findall(str(text).lower()) if len(tok) > 2 and tok not in STOPWORDS}


def jaccard(a: str, b: str) -> float:
    aa = tokens(a)
    bb = tokens(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def short(text: str, limit: int = 150) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def load_pseudo(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def load_selected(path: Path) -> list[str]:
    selected = json.loads(path.read_text(encoding="utf-8"))
    return [item["case_id"] for item in selected]


def judge_text_items(judge: dict, key: str, text_fields: list[str]) -> list[dict]:
    out = []
    for item in judge.get(key, []) or []:
        parts = [str(item.get(field, "")) for field in text_fields if item.get(field)]
        out.append({"kind": key, "text": " | ".join(parts), "raw": item})
    return out


def classify_case(case_id: str, judge: dict, pseudo_case: pd.DataFrame) -> tuple[str, list[str]]:
    pseudo_fix = int((pseudo_case["binary_label"] == "FIX").sum())
    pseudo_delete = int((pseudo_case["action"] == "DELETE").sum())
    original_unsupported = len(judge.get("unsupported_changes", []) or [])
    forgotten = len(judge.get("forgotten_carried_problems", []) or [])
    missing = len(judge.get("missing_updates", []) or [])

    notes = []
    if pseudo_delete >= 15 and (forgotten + missing) >= 5:
        label = "pseudo_stricter_but_original_additive"
        notes.append("pseudo has many DELETE labels while original verifier asks to add/restore many items")
    elif pseudo_delete >= 15:
        label = "pseudo_stricter_delete_heavy"
        notes.append("pseudo delete pressure is high")
    elif (forgotten + missing) >= 5:
        label = "original_additive"
        notes.append("original verifier emphasizes missing updates/carried problems")
    elif original_unsupported > pseudo_fix + 3:
        label = "original_stricter"
        notes.append("original verifier flags more unsupported changes than pseudo")
    else:
        label = "roughly_aligned"
        notes.append("counts are broadly aligned")
    return label, notes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--v2-judge-dir", type=Path, default=DEFAULT_V2_JUDGE_DIR)
    parser.add_argument("--pseudo-labels", type=Path, default=DEFAULT_PSEUDO_LABELS)
    parser.add_argument("--qwen-delta", type=Path, default=DEFAULT_QWEN_DELTA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    case_ids = load_selected(args.selected)
    pseudo = load_pseudo(args.pseudo_labels)
    if "case_id" not in pseudo.columns:
        raise ValueError("pseudo labels must contain case_id")
    qwen_delta = pd.read_csv(args.qwen_delta) if args.qwen_delta.exists() else pd.DataFrame()

    rows = []
    case_reports = []
    for case_id in case_ids:
        judge_path = args.v2_judge_dir / f"{case_id}.json"
        judge = json.loads(judge_path.read_text(encoding="utf-8"))
        pc = pseudo[pseudo["case_id"].eq(case_id)].copy()
        label, notes = classify_case(case_id, judge, pc)

        original_supported = judge_text_items(judge, "supported_changes", ["change", "target_scaffold_update"])
        original_unsupported = judge_text_items(judge, "unsupported_changes", ["change", "reason", "suggested_revision"])
        original_forgotten = judge_text_items(judge, "forgotten_carried_problems", ["problem", "previous_evidence", "suggested_revision"])
        original_missing = judge_text_items(judge, "missing_updates", ["problem_or_event", "today_evidence", "suggested_revision"])
        additive = original_forgotten + original_missing

        fix_claims = pc[pc["binary_label"].eq("FIX")].copy()
        keep_claims = pc[pc["binary_label"].eq("KEEP")].copy()
        fix_texts = fix_claims["claim_text"].fillna("").astype(str).tolist()

        additive_hit = 0
        additive_missed_examples = []
        for item in additive:
            best = max([jaccard(item["text"], claim) for claim in fix_texts], default=0.0)
            if best >= 0.18:
                additive_hit += 1
            elif len(additive_missed_examples) < 4:
                additive_missed_examples.append(short(item["text"]))

        unsupported_aligned = 0
        unsupported_examples = []
        for item in original_unsupported:
            scored = sorted(((jaccard(item["text"], claim), claim) for claim in fix_texts), reverse=True)
            best, claim = scored[0] if scored else (0.0, "")
            if best >= 0.18:
                unsupported_aligned += 1
                if len(unsupported_examples) < 3:
                    unsupported_examples.append(f"{short(item['text'], 90)} ~= {short(claim, 90)}")

        pseudo_only_fixes = []
        original_all = original_supported + original_unsupported + original_forgotten + original_missing
        original_texts = [item["text"] for item in original_all]
        for claim in fix_texts:
            best = max([jaccard(claim, text) for text in original_texts], default=0.0)
            if best < 0.12 and len(pseudo_only_fixes) < 5:
                pseudo_only_fixes.append(short(claim))

        qrow = qwen_delta[qwen_delta["case_id"].eq(case_id)] if not qwen_delta.empty else pd.DataFrame()
        delta = float(qrow["delta_oracle_minus_jr"].iloc[0]) if not qrow.empty else None
        jr_rat = str(qrow["augmented_brief_rationale_jr"].iloc[0]) if not qrow.empty else ""
        oracle_rat = str(qrow["augmented_brief_rationale_oracle"].iloc[0]) if not qrow.empty else ""

        row = {
            "case_id": case_id,
            "category": label,
            "qwen_delta_oracle_minus_jr": delta,
            "original_overall_decision": (judge.get("summary") or {}).get("overall_decision", ""),
            "original_supported_changes": len(judge.get("supported_changes", []) or []),
            "original_unsupported_changes": len(judge.get("unsupported_changes", []) or []),
            "original_forgotten_carried": len(judge.get("forgotten_carried_problems", []) or []),
            "original_missing_updates": len(judge.get("missing_updates", []) or []),
            "pseudo_claims": int(len(pc)),
            "pseudo_keep": int((pc["binary_label"] == "KEEP").sum()),
            "pseudo_fix": int((pc["binary_label"] == "FIX").sum()),
            "pseudo_delete": int((pc["action"] == "DELETE").sum()),
            "pseudo_rewrite": int((pc["action"] == "REWRITE").sum()),
            "original_unsupported_aligned_with_pseudo_fix": unsupported_aligned,
            "original_additive_items": len(additive),
            "original_additive_matched_by_pseudo_fix": additive_hit,
            "notes": "; ".join(notes),
        }
        rows.append(row)

        case_reports.append(
            {
                **row,
                "unsupported_alignment_examples": unsupported_examples,
                "original_additive_not_targeted_examples": additive_missed_examples,
                "pseudo_only_fix_examples": pseudo_only_fixes,
                "jr_rationale": jr_rat,
                "oracle_rationale": oracle_rat,
            }
        )

    detail = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    detail_path = args.outdir / "verify_truth_diff_detail.csv"
    detail.to_csv(detail_path, index=False)

    report_lines = ["# V2 Judge-Revise Verify vs Pseudo-Oracle Verifier", ""]
    report_lines.append("## Category Counts")
    counts = Counter(detail["category"])
    for key, value in counts.most_common():
        report_lines.append(f"- {key}: {value}")
    report_lines.append("")
    report_lines.append("## Worst Oracle-vs-Judge-Revise Cases")
    worst = sorted(case_reports, key=lambda x: (999 if x["qwen_delta_oracle_minus_jr"] is None else x["qwen_delta_oracle_minus_jr"]))[:10]
    for item in worst:
        report_lines.append("")
        report_lines.append(f"### {item['case_id']} ({item['category']}, delta={item['qwen_delta_oracle_minus_jr']})")
        report_lines.append(
            "- counts: "
            f"original unsupported={item['original_unsupported_changes']}, forgotten={item['original_forgotten_carried']}, "
            f"missing={item['original_missing_updates']}; pseudo FIX={item['pseudo_fix']}, "
            f"DELETE={item['pseudo_delete']}, REWRITE={item['pseudo_rewrite']}"
        )
        if item["original_additive_not_targeted_examples"]:
            report_lines.append("- original additive items not targeted by pseudo:")
            for ex in item["original_additive_not_targeted_examples"]:
                report_lines.append(f"  - {ex}")
        if item["pseudo_only_fix_examples"]:
            report_lines.append("- pseudo-only FIX examples:")
            for ex in item["pseudo_only_fix_examples"]:
                report_lines.append(f"  - {ex}")
        if item["unsupported_alignment_examples"]:
            report_lines.append("- aligned unsupported examples:")
            for ex in item["unsupported_alignment_examples"]:
                report_lines.append(f"  - {ex}")
        report_lines.append(f"- JR rationale: {short(item['jr_rationale'], 260)}")
        report_lines.append(f"- oracle rationale: {short(item['oracle_rationale'], 260)}")

    report_path = args.outdir / "verify_truth_diff_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {detail_path}")
    print(f"Wrote {report_path}")
    print(detail.sort_values("qwen_delta_oracle_minus_jr").to_string(index=False))


if __name__ == "__main__":
    main()
