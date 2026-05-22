"""Generate AP gold-delta trajectory/verifier truth with DeepSeek.

For each selected AP day, this script compares the previous gold A&P with the
current gold A&P, uses previous/current EHR input as evidence, and emits a JSONL
truth file that is compatible with the existing curated verifier reviser.

The current gold A&P is used only for offline label construction. It must not be
included in verifier inference prompts at deployment time.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.deepseek_api_generation import DEFAULT_API_URL, call_deepseek


DEFAULT_SELECTED = Path("outputs/oracle_claim_verifier_qwen653/selected_cases_with_failure_seed.json")
DEFAULT_DATA_ROOT = Path("data_ap100_ap/AP")
DEFAULT_V2_DIR = Path("outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2")
DEFAULT_OUTDIR = Path("outputs/ap_delta_trajectory_truth_deepseek")


def clean(text: object, max_chars: int = 0) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\n{4,}", "\n\n\n", value).strip()
    if max_chars and len(value) > max_chars:
        return value[:max_chars].rstrip() + "\n...[truncated]"
    return value


def strip_fences(text: str) -> str:
    raw = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.S | re.I)
    if fenced:
        raw = fenced.group(1).strip()
    return raw


def parse_json_response(text: str) -> dict[str, Any]:
    raw = strip_fences(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_gold_by_day(data_root: Path, hadm_id: str) -> dict[int, str]:
    rows = read_csv_rows(data_root / "gold" / f"gt_{hadm_id}.csv")
    by_day: dict[int, list[str]] = {}
    for row in rows:
        day = int(float(row["DAY"]))
        by_day.setdefault(day, []).append(row.get("TEXT", ""))
    return {day: clean("\n".join(parts)) for day, parts in by_day.items()}


def load_input_day(data_root: Path, hadm_id: str, day: int, max_row_chars: int, max_total_chars: int) -> str:
    rows = read_csv_rows(data_root / "input" / f"input_{hadm_id}.csv")
    lines = []
    for idx, row in enumerate(rows):
        if int(float(row.get("DAY", 0))) != int(day):
            continue
        text = clean(row.get("TEXT", ""), max_row_chars)
        lines.append(
            f"[row_{idx:04d}] TIME={row.get('TIME','')} REL_TIME={row.get('REL_TIME','')} "
            f"IS_NOTE={row.get('IS_NOTE','')}\n{text}"
        )
    return clean("\n\n".join(lines), max_total_chars)


def previous_gold_day(gold_by_day: dict[int, str], current_day: int) -> int | None:
    prior = [day for day in sorted(gold_by_day) if day < current_day]
    return prior[-1] if prior else None


def expected_schema(case_id: str, previous_day: int | None, current_day: int) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "previous_day": previous_day,
        "current_day": current_day,
        "trajectory_delta_truth": {
            "problem_threads": [
                {
                    "problem": "",
                    "delta_type": "continued_active|new_active|resolved_removed|improved_active|worsened_active|stable_active|plan_changed|renamed_or_merged|uncertain_or_weak_evidence",
                    "previous_gold_status": "active|resolved|not_present|uncertain",
                    "current_gold_status": "active|resolved|not_present|uncertain",
                    "trajectory_direction": "new|improving|worsening|stable|resolved|uncertain",
                    "evidence_chain": [
                        {
                            "source": "previous_input|previous_gold_ap|current_input|current_gold_ap",
                            "time": "",
                            "evidence": "",
                            "supports": "",
                        }
                    ],
                    "document_decision": {
                        "include_in_current_ap": True,
                        "carry_forward_from_previous": True,
                        "remove_if_present": False,
                        "revise_if_present": False,
                    },
                    "revision_instruction": "",
                }
            ],
            "must_not_carry_forward": [],
            "missing_if_absent": [],
        },
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
            "supported_claims_to_keep": [
                {
                    "claim_text": "",
                    "support_label": "SUPPORTED|PARTIALLY_SUPPORTED",
                    "evidence": [],
                    "reason": "",
                }
            ],
            "missing_supported_claims_to_add": [
                {
                    "claim_text_to_add": "",
                    "target_section": "",
                    "evidence": [],
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
            "case_level_summary": {
                "major_gold_delta": [],
                "major_errors_in_candidate_v2": [],
                "major_missing_items": [],
                "recommended_reviser_instruction": "",
            },
        },
    }


def build_prompt(
    case: dict[str, Any],
    previous_day: int | None,
    previous_input: str,
    previous_gold: str,
    current_input: str,
    current_gold: str,
    candidate_v2: str,
) -> str:
    case_id = case["case_id"]
    day = int(case["day"])
    schema = expected_schema(case_id, previous_day, day)
    return f"""You are generating offline pseudo-gold labels for ICU longitudinal A&P trajectory modeling and verifier training.

Goal:
Compare the previous gold A&P with the current gold A&P. Produce a complete problem-level trajectory truth first. Then, if possible, audit the candidate V2 A&P and produce verifier truth that can guide a minimal reviser.

Important constraints:
- Previous and current gold A&P are authoritative for offline label construction.
- Do not judge whether the gold A&P is correct. Explain how it changed and what evidence supports the change.
- Use only the provided previous/current inputs and gold A&P.
- Treat the current gold A&P as the note-time target. If current-day input contains events that appear later than, or inconsistent with, the current gold A&P, do not use those later events as truth targets. Current gold A&P wins over conflicting later input.
- The downstream revised clinical note must not mention "gold", "truth", "oracle", "verifier", or "label".
- If current gold contains an item but evidence in current input is weak, still record the delta but mark evidence as from current_gold_ap and note weak input support.
- Keep claim_text_to_add/rewrite clinically fluent and suitable for an A&P note.
- Return exactly one valid JSON object and no markdown.

Trajectory completeness requirements:
- Cover every clinically meaningful problem in the current gold A&P, including stable chronic/active problems, disposition, nutrition/prophylaxis if explicitly present, and resolved problems that disappeared from the current gold.
- Do not focus only on changed or dramatic problems.
- For each current active problem, set document_decision.include_in_current_ap=true.
- For each resolved/removed problem, set include_in_current_ap=false and remove_if_present=true.
- For each problem thread, write a concise revision_instruction that says whether the current A&P should KEEP, ADD, REWRITE, or REMOVE that problem.
- Prefer current_gold_ap evidence over speculative current_input evidence when current input support is weak.

Verifier truth requirements:
- The verifier truth must be problem-level and balanced: it should remove/rewrite unsupported claims, but it must also add missing active problems and preserve supported carry-forward problems.
- For every trajectory thread with include_in_current_ap=true:
  1. If the candidate already covers it correctly, add one supported_claims_to_keep item.
  2. If the candidate covers it but with wrong state, outdated plan, or unsupported details, add one unsupported_claims_to_remove_or_rewrite item.
  3. If the candidate misses it, add one missing_supported_claims_to_add item.
- For every trajectory thread with remove_if_present=true, add a remove/rewrite item only if the candidate carries it forward.
- Do not delete a whole problem when a narrower rewrite can preserve a supported active problem.
- Avoid over-deletion: generic but clinically harmless context should usually be kept unless it contradicts the trajectory truth or would mislead the plan.
- If verifier_truth cannot be fully generated because of length, still return complete trajectory_delta_truth and put a concise case_level_summary explaining what verifier actions are needed.

Case metadata:
{json.dumps(case, ensure_ascii=False, indent=2)}

Previous day: {previous_day}
Previous-day input evidence:
{previous_input or "(none)"}

Previous-day gold A&P:
{previous_gold or "(none)"}

Current day: {day}
Current-day input evidence:
{current_input}

Current-day gold A&P:
{current_gold}

Candidate V2 A&P to verify/revise:
{candidate_v2}

Required output schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def call_json(prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    last_error: Exception | None = None
    last_response = ""
    for attempt in range(1, args.parse_retries + 1):
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
        last_response = response
        try:
            return parse_json_response(response)
        except Exception as exc:
            last_error = exc
            repair_prompt = f"""Repair the following text into one strictly valid JSON object.

Rules:
- Do not change the clinical meaning.
- Do not add markdown fences.
- Escape internal quotation marks inside string values.
- Remove trailing commas and invalid control characters.
- Return JSON only.

Text to repair:
{response}
"""
            try:
                repaired = call_deepseek(
                    prompt=repair_prompt,
                    model=args.repair_model or args.model,
                    api_url=args.api_url,
                    temperature=0.0,
                    max_tokens=args.repair_max_tokens,
                    retries=args.retries,
                    sleep_seconds=args.sleep_seconds,
                    api_key_env=args.api_key_env,
                )
                last_response = repaired
                return parse_json_response(repaired)
            except Exception as repair_exc:
                last_error = repair_exc
            if attempt < args.parse_retries:
                time.sleep(args.sleep_seconds * attempt)
    preview = last_response[:500].replace("\n", "\\n")
    raise RuntimeError(f"JSON parse failed: {last_error}; response={preview!r}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_case(
    case: dict[str, Any],
    idx: int,
    total: int,
    args: argparse.Namespace,
    prompt_dir: Path,
    case_dir: Path,
) -> dict[str, Any]:
    case_id = case["case_id"]
    hadm_id = str(case["admission_id"])
    day = int(case["day"])
    out_path = case_dir / f"{case_id}.json"

    if out_path.exists() and not args.force:
        truth = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"[{idx}/{total}] skip existing {case_id}", flush=True)
    else:
        gold_by_day = load_gold_by_day(args.data_root, hadm_id)
        prev_day = previous_gold_day(gold_by_day, day)
        prev_input = (
            load_input_day(args.data_root, hadm_id, prev_day, args.max_row_chars, args.max_day_input_chars)
            if prev_day
            else ""
        )
        curr_input = load_input_day(args.data_root, hadm_id, day, args.max_row_chars, args.max_day_input_chars)
        prev_gold = gold_by_day.get(prev_day, "") if prev_day else ""
        curr_gold = gold_by_day.get(day, "")
        candidate_v2 = (args.v2_dir / f"{case_id}.txt").read_text(encoding="utf-8", errors="replace")
        prompt = build_prompt(case, prev_day, prev_input, prev_gold, curr_input, curr_gold, candidate_v2)
        (prompt_dir / f"{case_id}.md").write_text(prompt, encoding="utf-8")
        print(f"[{idx}/{total}] generating truth {case_id} prev_day={prev_day}", flush=True)
        truth = call_json(prompt, args)
        out_path.write_text(json.dumps(truth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        time.sleep(0.2)

    return {
        "case_id": case_id,
        "admission_id": hadm_id,
        "day": day,
        "previous_day": truth.get("previous_day"),
        "trajectory_delta_truth": truth.get("trajectory_delta_truth", {}),
        "verifier_truth": truth.get("verifier_truth", {}),
        "truth_path": str(out_path.as_posix()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AP delta trajectory/verifier truth with DeepSeek.")
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--model", default="deepseek-pro")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--repair-model", default="")
    parser.add_argument("--repair-max-tokens", type=int, default=12000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--max-row-chars", type=int, default=900)
    parser.add_argument("--max-day-input-chars", type=int, default=14000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected = json.loads(args.selected.read_text(encoding="utf-8"))
    if args.limit:
        selected = selected[: args.limit]

    args.outdir.mkdir(parents=True, exist_ok=True)
    prompt_dir = args.outdir / "prompts"
    case_dir = args.outdir / "case_truth"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    rows_by_case: dict[str, dict[str, Any]] = {}
    total = len(selected)
    if args.workers <= 1:
        for idx, case in enumerate(selected, start=1):
            row = generate_case(case, idx, total, args, prompt_dir, case_dir)
            rows_by_case[row["case_id"]] = row
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(generate_case, case, idx, total, args, prompt_dir, case_dir): case["case_id"]
                for idx, case in enumerate(selected, start=1)
            }
            for future in as_completed(futures):
                case_id = futures[future]
                row = future.result()
                rows_by_case[case_id] = row
                print(f"completed {case_id} ({len(rows_by_case)}/{total})", flush=True)

    rows = [rows_by_case[case["case_id"]] for case in selected]

    truth_jsonl = args.outdir / "ap_delta_trajectory_verifier_truth.jsonl"
    write_jsonl(truth_jsonl, rows)
    manifest = {
        "n_cases": len(rows),
        "model": args.model,
        "truth_jsonl": str(truth_jsonl.as_posix()),
        "case_truth_dir": str(case_dir.as_posix()),
        "prompt_dir": str(prompt_dir.as_posix()),
        "selected": str(args.selected.as_posix()),
        "workers": args.workers,
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote truth JSONL: {truth_jsonl}")


if __name__ == "__main__":
    main()
