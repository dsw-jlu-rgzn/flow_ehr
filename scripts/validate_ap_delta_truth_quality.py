from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ACTIVE_STATUSES = {"active", "uncertain"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def problem_threads(row: dict[str, Any]) -> list[dict[str, Any]]:
    return as_list((row.get("trajectory_delta_truth") or {}).get("problem_threads"))


def verifier_counts(row: dict[str, Any]) -> dict[str, int]:
    vt = row.get("verifier_truth") or {}
    return {
        "remove_rewrite": len(as_list(vt.get("unsupported_claims_to_remove_or_rewrite"))),
        "keep": len(as_list(vt.get("supported_claims_to_keep"))),
        "add": len(as_list(vt.get("missing_supported_claims_to_add"))),
        "restore": len(as_list(vt.get("carried_forward_problems_to_restore"))),
    }


def active_threads(row: dict[str, Any]) -> list[dict[str, Any]]:
    active = []
    for thread in problem_threads(row):
        decision = thread.get("document_decision") or {}
        current_status = str(thread.get("current_gold_status", "")).lower()
        if decision.get("include_in_current_ap") is True or current_status in ACTIVE_STATUSES:
            active.append(thread)
    return active


def validate_row(row: dict[str, Any]) -> dict[str, Any]:
    counts = verifier_counts(row)
    threads = problem_threads(row)
    active = active_threads(row)
    total_actions = counts["remove_rewrite"] + counts["add"] + counts["restore"]
    issues = []

    if not threads:
        issues.append("no_trajectory_threads")
    if total_actions == 0:
        issues.append("empty_verifier_actions")
    if active and counts["add"] + counts["restore"] == 0:
        issues.append("no_missing_or_restore_actions_for_active_threads")
    if counts["remove_rewrite"] > 0 and counts["add"] + counts["restore"] == 0 and len(active) >= 3:
        issues.append("delete_heavy_no_add_or_restore")
    if row.get("augment_error"):
        issues.append("augment_error")

    return {
        "case_id": row.get("case_id", ""),
        "admission_id": row.get("admission_id", ""),
        "day": row.get("day", ""),
        "n_threads": len(threads),
        "n_active_threads": len(active),
        **counts,
        "n_total_revision_actions": total_actions,
        "issues": ";".join(issues),
        "needs_retry": bool(issues),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "admission_id",
        "day",
        "n_threads",
        "n_active_threads",
        "remove_rewrite",
        "keep",
        "add",
        "restore",
        "n_total_revision_actions",
        "issues",
        "needs_retry",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AP delta trajectory/verifier truth quality.")
    parser.add_argument("--truth-jsonl", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--retry-cases", type=Path, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.truth_jsonl)
    checks = [validate_row(row) for row in rows]
    write_csv(args.out_csv, checks)

    retry_ids = [row["case_id"] for row in checks if row["needs_retry"]]
    if args.retry_cases:
        args.retry_cases.parent.mkdir(parents=True, exist_ok=True)
        args.retry_cases.write_text("\n".join(retry_ids) + ("\n" if retry_ids else ""), encoding="utf-8")

    print(f"Validated {len(checks)} cases")
    print(f"Needs retry: {len(retry_ids)}")
    if retry_ids:
        print("\n".join(retry_ids))


if __name__ == "__main__":
    main()
