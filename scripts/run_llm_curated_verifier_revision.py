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
DEFAULT_TRUTH = Path(
    "outputs/oracle_claim_verifier_qwen653/manual_verifier_truth_usable/"
    "verifier_truth_annotations_curated_usable.jsonl"
)
DEFAULT_V2_DIR = Path("outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2")
DEFAULT_V2_SUMMARY = Path("outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2_summary.csv")
DEFAULT_OUTDIR = Path("outputs/oracle_claim_verifier_qwen653/curated_verifier_llm_revision")
DEFAULT_CONFIG = "ap100full_generated_method2_gen_v2_curated_claim_verifier_llm_revise"


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


def compact_fix_items(verifier_truth: dict) -> list[dict]:
    items = []
    for row in verifier_truth.get("unsupported_claims_to_remove_or_rewrite", []):
        if not isinstance(row, dict):
            row = {"claim_text": str(row), "action": "DELETE", "reason": "Verifier item was provided as free text."}
        items.append(
            {
                "claim_text": row.get("claim_text", ""),
                "support_label": row.get("support_label", ""),
                "action": row.get("action", ""),
                "rewrite": row.get("corrected_claim_or_rewrite", ""),
                "reason": row.get("reason", ""),
            }
        )
    return items


def compact_add_items(verifier_truth: dict) -> list[dict]:
    items = []
    for row in verifier_truth.get("missing_supported_claims_to_add", []):
        if not isinstance(row, dict):
            row = {"claim_text_to_add": str(row), "target_section": "", "reason": "Verifier item was provided as free text."}
        items.append(
            {
                "claim_text_to_add": row.get("claim_text_to_add", ""),
                "target_section": row.get("target_section", ""),
                "reason": row.get("reason", ""),
            }
        )
    for row in verifier_truth.get("carried_forward_problems_to_restore", []):
        if not isinstance(row, dict):
            row = {"claim_text_to_add": str(row), "problem": "", "reason": "Verifier item was provided as free text."}
        items.append(
            {
                "claim_text_to_add": row.get("claim_text_to_add", ""),
                "target_section": row.get("problem", ""),
                "reason": row.get("reason", ""),
            }
        )
    return items


def build_reviser_prompt(original_ap: str, fix_items: list[dict], add_items: list[dict]) -> str:
    fix_json = json.dumps(fix_items, ensure_ascii=False, indent=2)
    add_json = json.dumps(add_items, ensure_ascii=False, indent=2)
    return f"""You are the Minimal Reviser Agent for ICU Assessment & Plan generation.

You are given:
1. The original generated ICU Assessment & Plan.
2. Claim-level verifier instructions identifying claims to delete or rewrite.
3. Evidence-grounded missing/carry-forward items that should be added.

Revise the original A&P using minimal edits.

Rules:
- Preserve the original structure and supported content as much as possible.
- For action DELETE: remove only the unsupported claim/sentence. Clean up empty sections.
- For action REWRITE: replace the claim with the provided rewrite.
- Add missing/carry-forward items only when listed below, placing them in the best matching section.
- If a listed add item is phrased as an instruction (for example "Add..." or "Mention..."), convert it into fluent clinical A&P wording.
- Do not add unsupported medications, numbers, diagnoses, code status, disposition, or plans beyond the instructions below.
- Keep the output as a coherent ICU A&P note with no empty headings.
- Do not mention verifier, oracle, gold, truth, labels, claims, or evidence.
- Return only the revised A&P text.

Claims to delete or rewrite:
{fix_json}

Missing or carried-forward items to add:
{add_json}

Original generated A&P:
{original_ap}
"""


def llm_revise(original_ap: str, verifier_truth: dict, args: argparse.Namespace) -> str:
    fix_items = compact_fix_items(verifier_truth)
    add_items = compact_add_items(verifier_truth)
    if not fix_items and not add_items:
        return original_ap.strip() + "\n"
    prompt = build_reviser_prompt(original_ap, fix_items, add_items)
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
    order = {(str(item["admission_id"]), int(item["day"])): idx for idx, item in enumerate(selected)}
    rows = []
    for _, row in summary.iterrows():
        key = (str(row["admission_id"]), int(row["day"]))
        if key not in selected_keys:
            continue
        out = row.to_dict()
        out["config"] = config_name
        out["method"] = "memory_gated_scaffold_v2_curated_claim_verifier_llm_revise"
        out["generation_time_judge_revise"] = True
        out["_order"] = order[key]
        rows.append(out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("_order").drop(columns=["_order"])
    df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM revision using curated claim-level verifier truth.")
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2_DIR)
    parser.add_argument("--v2-summary", type=Path, default=DEFAULT_V2_SUMMARY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--sleep-seconds", type=float, default=6.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected = json.loads(args.selected.read_text(encoding="utf-8"))
    if args.limit:
        selected = selected[: args.limit]

    truth_by_case = {row["case_id"]: row["verifier_truth"] for row in read_jsonl(args.truth)}
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
        verifier_truth = truth_by_case[case_id]
        fix_items = compact_fix_items(verifier_truth)
        add_items = compact_add_items(verifier_truth)

        if out_file.exists() and not args.force:
            revised = out_file.read_text(encoding="utf-8", errors="replace")
            print(f"[{idx}/{len(selected)}] skip existing {case_id}")
        else:
            print(f"[{idx}/{len(selected)}] LLM revising {case_id} fixes={len(fix_items)} adds={len(add_items)}")
            revised = llm_revise(original, verifier_truth, args)
            out_file.write_text(revised, encoding="utf-8", newline="\n")
            time.sleep(0.2)

        review = {
            "case_id": case_id,
            "admission_id": admission_id,
            "day": day,
            "fix_item_count": len(fix_items),
            "add_item_count": len(add_items),
            "original_words": len(original.split()),
            "revised_words": len(revised.split()),
            "revised_path": str(out_file.as_posix()),
            "fix_items": fix_items,
            "add_items": add_items,
        }
        (review_dir / f"{case_id}.json").write_text(
            json.dumps(review, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_rows.append(
            {
                "case_id": case_id,
                "source_output": "memory_gated_scaffold_v2",
                "revised_method": "v2_curated_claim_verifier_llm_revise",
                "original_output": original,
                "curated_revised_output": revised,
                "revised_path": str(out_file.as_posix()),
                "fix_item_count": len(fix_items),
                "add_item_count": len(add_items),
            }
        )

    write_jsonl(args.outdir / "curated_revised_outputs_llm.jsonl", output_rows)
    detail_csv = args.outdir / "curated_llm_revise_detail_for_judge.csv"
    build_detail_csv(selected, args.v2_summary, args.config_name, detail_csv)
    manifest = {
        "n_cases": len(selected),
        "case_ids": [item["case_id"] for item in selected],
        "config_name": args.config_name,
        "augmented_dir_for_judge": str(args.outdir.as_posix()),
        "detail_csv_for_judge": str(detail_csv.as_posix()),
        "outputs_jsonl": str((args.outdir / "curated_revised_outputs_llm.jsonl").as_posix()),
        "truth_jsonl": str(args.truth.as_posix()),
        "model": args.model,
        "api_url": args.api_url,
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote curated LLM revision outputs -> {args.outdir}")
    print(f"Judge detail CSV -> {detail_csv}")


if __name__ == "__main__":
    main()
