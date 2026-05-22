from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling.deepseek_api_generation import DEFAULT_API_URL, call_deepseek


DEFAULT_SELECTED = Path("outputs/oracle_claim_verifier_qwen653/selected_cases_with_failure_seed.json")
DEFAULT_LABELS = Path(
    "outputs/oracle_claim_verifier_qwen653/pseudo_verifier_oracle/claims_oracle_labeled_pseudo.jsonl"
)
DEFAULT_V2_DIR = Path("outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2")
DEFAULT_V2_SUMMARY = Path("outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2_summary.csv")
DEFAULT_OUTDIR = Path("outputs/oracle_claim_verifier_qwen653/llm_minimal_revision")
DEFAULT_CONFIG = "ap100full_generated_method2_gen_v2_pseudo_oracle_verifier_llm_revise"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    fenced = re.search(r"```(?:text|markdown)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    return cleaned.strip() + "\n"


def compact_fix_items(labels: list[dict]) -> list[dict]:
    items = []
    for row in labels:
        if row.get("binary_label") != "FIX":
            continue
        item = {
            "claim_id": row["claim_id"],
            "claim_text": row["claim_text"],
            "claim_type": row["claim_type"],
            "support_label": row["support_label"],
            "action": row["action"],
            "oracle_rewrite": row.get("oracle_rewrite", ""),
            "reason": row.get("reason", ""),
        }
        items.append(item)
    return items


def build_reviser_prompt(original_ap: str, fix_items: list[dict]) -> str:
    fix_json = json.dumps(fix_items, ensure_ascii=False, indent=2)
    return f"""You are the Minimal Reviser Agent for ICU Assessment & Plan generation.

You are given:
1. The original generated ICU Assessment & Plan.
2. A claim-level verifier result listing only claims that must be fixed.

Revise the original A&P using minimal edits.

Rules:
- Preserve the original structure, sections, and clinically supported content.
- Do not rewrite the whole note.
- For action DELETE: remove only the unsupported claim or sentence.
- For action REWRITE: replace the unsupported claim with oracle_rewrite.
- Do not add diagnoses, procedures, medications, numbers, code status, disposition, or plans not present in the original or the provided rewrite.
- Keep the output as a fluent A&P note.
- Do not mention verifier, oracle, gold, labels, claims, or evidence.
- Return only the revised A&P text.

Claims to fix:
{fix_json}

Original generated A&P:
{original_ap}
"""


def llm_revise(original_ap: str, labels: list[dict], args: argparse.Namespace) -> str:
    fix_items = compact_fix_items(labels)
    if not fix_items:
        return original_ap.strip() + "\n"
    prompt = build_reviser_prompt(original_ap, fix_items)
    response = call_deepseek(
        prompt=prompt,
        model=args.model,
        api_url=args.api_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
        api_key_env=args.api_key_env,
    )
    return strip_code_fence(response)


def build_detail_csv(selected: list[dict], summary_csv: Path, config_name: str, out_csv: Path) -> None:
    summary = pd.read_csv(summary_csv)
    selected_keys = {(str(item["admission_id"]), int(item["day"])) for item in selected}
    rows = []
    for _, row in summary.iterrows():
        key = (str(row["admission_id"]), int(row["day"]))
        if key not in selected_keys:
            continue
        out = row.to_dict()
        out["config"] = config_name
        out["method"] = "memory_gated_scaffold_v2_pseudo_oracle_verifier_llm_revise"
        out["generation_time_judge_revise"] = True
        rows.append(out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM minimal revision using pseudo-oracle verifier labels.")
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2_DIR)
    parser.add_argument("--v2-summary", type=Path, default=DEFAULT_V2_SUMMARY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--sleep-seconds", type=float, default=6.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected = json.loads(args.selected.read_text(encoding="utf-8"))
    if args.limit:
        selected = selected[: args.limit]

    labels_by_case: dict[str, list[dict]] = {}
    for row in read_jsonl(args.labels):
        labels_by_case.setdefault(row["case_id"], []).append(row)

    config_dir = args.outdir / args.config_name
    review_dir = args.outdir / "case_reviews"
    config_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    output_rows = []
    for idx, item in enumerate(selected, start=1):
        case_id = item["case_id"]
        admission_id = str(item["admission_id"])
        day = int(item["day"])
        out_file = config_dir / f"{case_id}.txt"
        original = (args.v2_dir / f"{case_id}.txt").read_text(encoding="utf-8", errors="replace")
        labels = labels_by_case[case_id]
        fix_items = compact_fix_items(labels)

        if out_file.exists() and not args.force:
            revised = out_file.read_text(encoding="utf-8", errors="replace")
            print(f"[{idx}/{len(selected)}] skip existing {case_id}")
        else:
            print(f"[{idx}/{len(selected)}] LLM revising {case_id} fixes={len(fix_items)}")
            revised = llm_revise(original, labels, args)
            out_file.write_text(revised, encoding="utf-8", newline="\n")
            time.sleep(0.2)

        review = {
            "case_id": case_id,
            "admission_id": admission_id,
            "day": day,
            "fix_claim_count": len(fix_items),
            "fix_claim_ids": [row["claim_id"] for row in fix_items],
            "original_words": len(original.split()),
            "revised_words": len(revised.split()),
            "revised_path": str(out_file.as_posix()),
            "fix_items": fix_items,
        }
        (review_dir / f"{case_id}.json").write_text(
            json.dumps(review, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_rows.append(
            {
                "case_id": case_id,
                "source_output": "memory_gated_scaffold_v2",
                "revised_method": "v2_pseudo_oracle_claim_verifier_llm_revise",
                "original_output": original,
                "oracle_revised_output": revised,
                "fix_claim_ids": [row["claim_id"] for row in fix_items],
                "revised_path": str(out_file.as_posix()),
            }
        )

    write_jsonl(args.outdir / "oracle_revised_outputs_llm.jsonl", output_rows)
    detail_csv = args.outdir / "llm_revise_detail_for_judge.csv"
    build_detail_csv(selected, args.v2_summary, args.config_name, detail_csv)
    manifest = {
        "n_cases": len(selected),
        "case_ids": [item["case_id"] for item in selected],
        "config_name": args.config_name,
        "augmented_dir_for_judge": str(args.outdir.as_posix()),
        "detail_csv_for_judge": str(detail_csv.as_posix()),
        "outputs_jsonl": str((args.outdir / "oracle_revised_outputs_llm.jsonl").as_posix()),
        "model": args.model,
        "api_url": args.api_url,
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote LLM revision outputs -> {args.outdir}")
    print(f"Judge detail CSV -> {detail_csv}")


if __name__ == "__main__":
    main()
