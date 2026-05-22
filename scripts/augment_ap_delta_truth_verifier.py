"""Populate missing verifier_truth from trajectory truth and candidate AP."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.deepseek_api_generation import DEFAULT_API_URL, call_deepseek


def parse_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.S | re.I)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def verifier_empty(vt: dict[str, Any]) -> bool:
    return not any(
        vt.get(key)
        for key in [
            "unsupported_claims_to_remove_or_rewrite",
            "missing_supported_claims_to_add",
            "carried_forward_problems_to_restore",
        ]
    )


def build_prompt(case_id: str, trajectory_truth: dict[str, Any], candidate_ap: str) -> str:
    schema = {
        "verifier_truth": {
            "unsupported_claims_to_remove_or_rewrite": [
                {
                    "claim_text": "",
                    "support_label": "UNSUPPORTED|CONTRADICTED|PARTIALLY_SUPPORTED",
                    "action": "DELETE|REWRITE|DOWNGRADE",
                    "corrected_claim_or_rewrite": "",
                    "evidence": [],
                    "reason": "",
                }
            ],
            "supported_claims_to_keep": [],
            "missing_supported_claims_to_add": [
                {
                    "claim_text_to_add": "",
                    "target_section": "",
                    "evidence": [],
                    "reason": "",
                }
            ],
            "carried_forward_problems_to_restore": [],
            "case_level_summary": {
                "major_gold_delta": [],
                "major_errors_in_candidate_v2": [],
                "major_missing_items": [],
                "recommended_reviser_instruction": "",
            },
        }
    }
    return f"""You are converting trajectory truth into verifier truth for a Minimal Reviser Agent.

Given:
1. Gold-derived trajectory delta truth for case {case_id}.
2. Candidate V2 A&P.

Audit the candidate against the trajectory truth. Your goal is not only to remove unsupported claims. Your goal is to make the final A&P match the current active problem trajectory while preserving useful supported content.

Populate verifier_truth:
- unsupported_claims_to_remove_or_rewrite: candidate claims contradicted by trajectory truth, stale carry-forward claims, wrong trajectory/state claims, or unsupported treatment/diagnosis details.
- supported_claims_to_keep: candidate claims that correctly cover active current problems or stable carry-forward problems.
- missing_supported_claims_to_add: clinically important current active problems, plans, trajectory states, disposition, nutrition, or prophylaxis items that are absent or too vague in candidate.
- carried_forward_problems_to_restore: active prior/current problems that should remain documented but are absent from candidate.

Rules:
- Be concrete: quote the exact candidate claim_text when asking to DELETE/REWRITE.
- corrected_claim_or_rewrite must be fluent A&P wording and must not mention gold/truth/verifier.
- Do not include empty placeholder rows.
- Produce balanced verifier truth. Do not only delete. For each active trajectory thread, decide KEEP, ADD, RESTORE, or REWRITE.
- If a trajectory thread has document_decision.include_in_current_ap=true:
  * If candidate covers it accurately, add a supported_claims_to_keep item.
  * If candidate misses it, add a missing_supported_claims_to_add item with fluent A&P wording.
  * If candidate includes it with wrong trajectory, outdated plan, unsupported medication, or wrong severity, add a REWRITE item rather than deleting the entire problem.
  * If it is a stable carry-forward problem and candidate omits it, add it to carried_forward_problems_to_restore.
- If a trajectory thread has document_decision.remove_if_present=true:
  * Add a DELETE or REWRITE item only if the candidate carries that resolved problem forward.
- Prefer REWRITE over DELETE when part of the candidate claim is clinically supported.
- Do not remove broad active problems just because one detail is unsupported; rewrite the detail and preserve the problem.
- Missing active problems are as important as unsupported hallucinations. If there are active trajectory threads absent from candidate, missing_supported_claims_to_add must not be empty.
- case_level_summary.major_missing_items should explicitly list active trajectory threads missing from candidate.
- Return JSON only.

Trajectory truth:
{json.dumps(trajectory_truth, ensure_ascii=False, indent=2)}

Candidate V2 A&P:
{candidate_ap}

Required JSON:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def augment_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    vt = row.get("verifier_truth", {})
    if not verifier_empty(vt):
        return row

    case_id = row["case_id"]
    print(f"augmenting verifier_truth for {case_id}", flush=True)
    candidate = (Path(args.v2_dir) / f"{case_id}.txt").read_text(encoding="utf-8", errors="replace")
    prompt = build_prompt(case_id, row.get("trajectory_delta_truth", {}), candidate)
    try:
        response = call_deepseek(
            prompt=prompt,
            model=args.model,
            api_url=args.api_url,
            temperature=0.0,
            max_tokens=args.max_tokens,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
            api_key_env=args.api_key_env,
        )
        try:
            parsed = parse_json(response)
        except Exception:
            repair = call_deepseek(
                prompt="Repair this text into valid JSON only:\n" + response,
                model=args.repair_model,
                api_url=args.api_url,
                temperature=0.0,
                max_tokens=args.max_tokens,
                retries=args.retries,
                sleep_seconds=args.sleep_seconds,
                api_key_env=args.api_key_env,
            )
            parsed = parse_json(repair)
        row["verifier_truth"] = parsed.get("verifier_truth", parsed)
        row.pop("augment_error", None)
    except Exception as exc:
        row["augment_error"] = str(exc)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-jsonl", required=True)
    parser.add_argument("--v2-dir", default="outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2")
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--repair-model", default="deepseek-v4-flash")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.truth_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.workers <= 1:
        out_rows = [augment_row(row, args) for row in rows]
    else:
        out_rows: list[dict[str, Any] | None] = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(augment_row, row, args): idx for idx, row in enumerate(rows)}
            for future in as_completed(futures):
                idx = futures[future]
                out_rows[idx] = future.result()
                print(f"completed {idx + 1}/{len(rows)}", flush=True)
        out_rows = [row for row in out_rows if row is not None]

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
