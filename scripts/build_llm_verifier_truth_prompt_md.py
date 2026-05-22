from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


DEFAULT_SELECTED = Path("outputs/oracle_claim_verifier_qwen653/selected_cases_with_failure_seed.json")
DEFAULT_DATA_ROOT = Path("data_ap100_ap/AP")
DEFAULT_V2_DIR = Path("outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2")
DEFAULT_PSEUDO_LABELS = Path(
    "outputs/oracle_claim_verifier_qwen653/pseudo_verifier_oracle/claims_oracle_labeled_pseudo.jsonl"
)
DEFAULT_PSEUDO_REVIEWS = Path("outputs/oracle_claim_verifier_qwen653/pseudo_verifier_oracle/per_case_review")
DEFAULT_OUT = Path("outputs/oracle_claim_verifier_qwen653/llm_truth_prompt/verifier_truth_annotation_prompt_30cases.md")
DEFAULT_SPLIT_DIR = Path("outputs/oracle_claim_verifier_qwen653/llm_truth_prompt/case_prompts")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def clean(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def md_code(text: str, lang: str = "text") -> str:
    text = clean(text).replace("```", "` ` `")
    return f"```{lang}\n{text}\n```"


def load_gold_by_day(data_root: Path, hadm_id: str) -> dict[int, str]:
    rows = read_csv_rows(data_root / "gold" / f"gt_{hadm_id}.csv")
    out: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        out[int(float(row["DAY"]))].append(row.get("TEXT", ""))
    return {day: "\n".join(parts).strip() for day, parts in out.items()}


def load_current_input(data_root: Path, hadm_id: str, day: int, max_non_note_chars: int = 700) -> str:
    path = data_root / "input" / f"input_{hadm_id}.csv"
    rows = read_csv_rows(path)
    lines = []
    for idx, row in enumerate(rows):
        if int(float(row.get("DAY", 0))) != int(day):
            continue
        text = clean(row.get("TEXT", ""))
        is_note = str(row.get("IS_NOTE", ""))
        if is_note != "1" and len(text) > max_non_note_chars:
            text = text[:max_non_note_chars].rstrip() + " ...[truncated]"
        lines.append(
            f"[input_{idx:04d}] TIME={row.get('TIME','')} REL_TIME={row.get('REL_TIME','')} IS_NOTE={is_note}\n{text}"
        )
    return "\n\n".join(lines)


def load_pseudo_labels(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def compact_pseudo_json(case_df: pd.DataFrame) -> str:
    fields = [
        "claim_id",
        "claim_type",
        "claim_text",
        "support_label",
        "binary_label",
        "action",
        "oracle_rewrite",
        "reason",
        "evidence_sources",
    ]
    rows = []
    for _, row in case_df.iterrows():
        item = {}
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float) and pd.isna(value):
                value = ""
            item[field] = value
        rows.append(item)
    return json.dumps(rows, ensure_ascii=False, indent=2)


def previous_gold_section(gold_by_day: dict[int, str], day: int, lookback_days: int) -> str:
    prior_days = [d for d in sorted(gold_by_day) if d < day]
    prior_days = prior_days[-lookback_days:]
    if not prior_days:
        return "No prior gold A&P is available before this day."
    parts = []
    for d in prior_days:
        parts.append(f"### Prior Gold A&P: day {d}\n\n{md_code(gold_by_day[d])}")
    return "\n\n".join(parts)


def build_case_block(
    case: dict,
    data_root: Path,
    v2_dir: Path,
    pseudo_df: pd.DataFrame,
    pseudo_reviews: Path,
    lookback_days: int,
) -> str:
    case_id = case["case_id"]
    hadm_id = str(case["admission_id"])
    day = int(case["day"])
    gold_by_day = load_gold_by_day(data_root, hadm_id)
    current_gold = gold_by_day.get(day, "")
    current_input = load_current_input(data_root, hadm_id, day)
    v2_ap = (v2_dir / f"{case_id}.txt").read_text(encoding="utf-8", errors="replace")
    cdf = pseudo_df[pseudo_df["case_id"].eq(case_id)].copy()
    pseudo_json = compact_pseudo_json(cdf)
    pseudo_review_path = pseudo_reviews / f"{case_id}.md"
    pseudo_review = pseudo_review_path.read_text(encoding="utf-8", errors="replace") if pseudo_review_path.exists() else ""

    expected_output = {
        "case_id": case_id,
        "verifier_truth": {
            "unsupported_claims_to_remove_or_rewrite": [
                {
                    "claim_text": "",
                    "support_label": "UNSUPPORTED|CONTRADICTED|PARTIALLY_SUPPORTED",
                    "action": "DELETE|REWRITE|DOWNGRADE",
                    "corrected_claim_or_rewrite": "",
                    "evidence": ["quote or paraphrase from current input / prior gold / current gold"],
                    "reason": "",
                }
            ],
            "supported_claims_to_keep": [
                {
                    "claim_text": "",
                    "support_label": "SUPPORTED|PARTIALLY_SUPPORTED",
                    "evidence": ["quote or paraphrase from current input / prior gold / current gold"],
                    "reason": "",
                }
            ],
            "missing_supported_claims_to_add": [
                {
                    "claim_text_to_add": "",
                    "target_section": "",
                    "evidence": ["quote or paraphrase from current input / prior gold / current gold"],
                    "reason": "",
                }
            ],
            "carried_forward_problems_to_restore": [
                {
                    "problem": "",
                    "claim_text_to_add": "",
                    "prior_gold_evidence": "",
                    "current_day_evidence_or_current_gold_support": "",
                    "reason": "",
                }
            ],
            "pseudo_verifier_corrections": [
                {
                    "pseudo_claim_id": "",
                    "pseudo_label_was": "",
                    "correct_label_should_be": "",
                    "reason": "",
                }
            ],
            "case_level_summary": {
                "major_errors_in_v2": [],
                "major_missing_items": [],
                "recommended_reviser_instruction": "",
            },
        },
    }

    lines = [
        f"## Case {case_id}",
        "",
        "### Case Metadata",
        "",
        md_code(
            json.dumps(
                {
                    "case_id": case_id,
                    "admission_id": hadm_id,
                    "day": day,
                    "selection_reason": case.get("reason_for_selection", ""),
                    "case_type": case.get("case_type", []),
                    "failure_modes": case.get("failure_modes", []),
                },
                ensure_ascii=False,
                indent=2,
            ),
            "json",
        ),
        "",
        "### Candidate A&P To Verify: V2 Output",
        "",
        md_code(v2_ap),
        "",
        "### Current Generated Verifier Result To Audit: Pseudo-Oracle Claim Labels",
        "",
        "The model should treat these labels as a draft verifier result. Correct them when they disagree with evidence or when they are too deletion-heavy.",
        "",
        md_code(pseudo_json, "json"),
        "",
        "### Current Generated Verifier Review Summary",
        "",
        md_code(pseudo_review),
        "",
        "### Historical Truth A&P",
        "",
        previous_gold_section(gold_by_day, day, lookback_days),
        "",
        "### Current-Day Input Evidence",
        "",
        md_code(current_input),
        "",
        "### Current-Day Truth A&P",
        "",
        md_code(current_gold),
        "",
        "### Required Output For This Case",
        "",
        "Return exactly one JSON object matching this schema. Do not include Markdown inside the JSON fields.",
        "",
        md_code(json.dumps(expected_output, ensure_ascii=False, indent=2), "json"),
    ]
    return "\n".join(lines).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2_DIR)
    parser.add_argument("--pseudo-labels", type=Path, default=DEFAULT_PSEUDO_LABELS)
    parser.add_argument("--pseudo-reviews", type=Path, default=DEFAULT_PSEUDO_REVIEWS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--write-split", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=3)
    args = parser.parse_args()

    selected = json.loads(args.selected.read_text(encoding="utf-8"))
    pseudo_df = load_pseudo_labels(args.pseudo_labels)

    header = [
        "# LLM Annotation Input: Generate Correct Claim-Level Verifier Truth",
        "",
        "## Background",
        "",
        "You are generating oracle-quality verifier truth for an ICU Assessment & Plan multi-agent pipeline.",
        "The candidate note is the V2 generated A&P. A draft pseudo-oracle verifier has already labeled its claims as KEEP/FIX with actions DELETE/REWRITE, but that draft may be too strict, too lenient, or incomplete.",
        "",
        "Your job is not to write a new A&P. Your job is to produce the correct verifier truth that a downstream minimal reviser should use.",
        "",
        "Use all provided evidence:",
        "",
        "- Historical truth A&P: previous gold notes, useful for carried-forward active problems.",
        "- Current-day input evidence: raw EHR events and notes for the target day.",
        "- Current-day truth A&P: gold reference for the target day. Use this as the strongest source for what the correct A&P should cover.",
        "- Current generated verifier result: draft pseudo labels. Audit them; do not copy them blindly.",
        "",
        "Important labeling policy:",
        "",
        "- Keep a claim when it is supported by current-day input, historical truth A&P, or current-day truth A&P and is clinically appropriate for the target day.",
        "- Mark a claim for DELETE when it is unsupported, contradicted, outdated, or clinically misleading.",
        "- Mark a claim for REWRITE/DOWNGRADE when part of it is correct but details, trajectory, numbers, device status, medication, or acuity are wrong.",
        "- Add `missing_supported_claims_to_add` when V2 omits key active problems, trajectory, plans, disposition, or high-risk facts present in the truth/evidence.",
        "- Add `carried_forward_problems_to_restore` when prior gold A&P contains an active problem that remains relevant and is also supported by current evidence or current gold.",
        "- Do not leak the phrase `gold note`, `oracle`, or `truth` into any claim text intended for the downstream clinical A&P.",
        "",
        "Output one JSON object per case, following the schema shown under each case. Be concise but include enough evidence for each decision.",
        "",
    ]

    case_blocks = []
    for case in selected:
        case_blocks.append(
            (
                case["case_id"],
                build_case_block(
                    case=case,
                    data_root=args.data_root,
                    v2_dir=args.v2_dir,
                    pseudo_df=pseudo_df,
                    pseudo_reviews=args.pseudo_reviews,
                    lookback_days=args.lookback_days,
                ),
            )
        )

    blocks = [block for _, block in case_blocks]
    if args.write_split:
        args.split_dir.mkdir(parents=True, exist_ok=True)
        index_lines = [
            "# Case Prompt Index",
            "",
            "Each file contains one case and the shared annotation instructions.",
            "",
        ]
        for case_id, block in case_blocks:
            out = args.split_dir / f"{case_id}_verifier_truth_prompt.md"
            out.write_text("\n\n---\n\n".join(["\n".join(header), block]) + "\n", encoding="utf-8", newline="\n")
            index_lines.append(f"- [{case_id}]({out.name})")
        (args.split_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8", newline="\n")

    # Keep the all-in-one artifact for archival/reproducibility.
    blocks = [
        block
        for _, block in case_blocks
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n\n---\n\n".join(["\n".join(header), *blocks]) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {args.out}")
    if args.write_split:
        print(f"Wrote split prompts to {args.split_dir}")
    print(f"Cases: {len(blocks)}")
    print(f"Size bytes: {args.out.stat().st_size}")


if __name__ == "__main__":
    main()
