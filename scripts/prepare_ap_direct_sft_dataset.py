from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path


SYSTEM_PROMPT = (
    "You are an experienced ICU clinician. Generate a concise, clinically "
    "grounded Assessment and Plan for the current day. Use only the provided "
    "EHR input. Return the Assessment and Plan text only."
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def resolve_ap_root(data_root: Path) -> Path:
    if (data_root / "input").exists() and (data_root / "gold").exists():
        return data_root
    if (data_root / "AP" / "input").exists() and (data_root / "AP" / "gold").exists():
        return data_root / "AP"
    raise FileNotFoundError(f"Could not find AP/input and AP/gold under {data_root}")


def day_key(value: str | int) -> int:
    return int(float(str(value).strip()))


def compact_day_input(rows: list[dict[str, str]], include_note_rows: bool, max_chars: int) -> str:
    lines: list[str] = []
    for idx, row in enumerate(rows):
        is_note = str(row.get("IS_NOTE", "")).strip()
        if not include_note_rows and is_note == "1":
            continue
        text = re.sub(r"\s+", " ", str(row.get("TEXT", "")).strip())
        if not text:
            continue
        rel_time = str(row.get("REL_TIME", "")).strip()
        time = str(row.get("TIME", "")).strip()
        lines.append(f"[row_{idx:04d}] TIME={time} REL_TIME={rel_time} IS_NOTE={is_note}\n{text}")
    joined = "\n\n".join(lines)
    if max_chars > 0 and len(joined) > max_chars:
        return joined[:max_chars].rstrip() + "\n...[truncated]"
    return joined


def extract_assessment_plan(text: str) -> str:
    raw = text.strip()
    matches = list(re.finditer(r"(?im)^\s*assessment\s+and\s+plan\.?\s*$", raw))
    if matches:
        return raw[matches[-1].start() :].strip()
    matches = list(re.finditer(r"(?im)^\s*(?:a\s*/?\s*p|assessment\s*/\s*plan)\s*:?\s*$", raw))
    if matches:
        return raw[matches[-1].start() :].strip()
    return raw


def build_user_prompt(ehr_input: str) -> str:
    return (
        "Current-day EHR input:\n"
        "```text\n"
        f"{ehr_input}\n"
        "```\n\n"
        "Write the current day's ICU Assessment and Plan."
    )


def make_example(
    case_id: str,
    admission_id: str,
    day: int,
    ehr_input: str,
    target: str,
) -> dict:
    return {
        "id": case_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(ehr_input)},
            {"role": "assistant", "content": target.strip() + "\n"},
        ],
        "metadata": {
            "admission_id": admission_id,
            "day": day,
            "task": "ap_direct_input_to_gold",
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a simple AP SFT dataset: current-day input -> gold A&P."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_ap100_ap/AP"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/ap_direct_sft_dataset"))
    parser.add_argument("--target-mode", choices=["ap_section", "full_gold"], default="ap_section")
    parser.add_argument("--include-note-rows", action="store_true")
    parser.add_argument(
        "--exclude-first-gold-day",
        action="store_true",
        help="Skip the earliest gold day per admission to match AP generation settings that require prior context.",
    )
    parser.add_argument("--max-input-chars", type=int, default=24000)
    parser.add_argument("--min-target-chars", type=int, default=80)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    ap_root = resolve_ap_root(args.data_root)
    input_dir = ap_root / "input"
    gold_dir = ap_root / "gold"

    examples: list[dict] = []
    for gold_path in sorted(gold_dir.glob("gt_*.csv")):
        admission_id = gold_path.stem.removeprefix("gt_")
        input_path = input_dir / f"input_{admission_id}.csv"
        if not input_path.exists():
            continue
        input_rows = read_csv_rows(input_path)
        input_by_day: dict[int, list[dict[str, str]]] = {}
        for row in input_rows:
            if not str(row.get("DAY", "")).strip():
                continue
            input_by_day.setdefault(day_key(row["DAY"]), []).append(row)

        gold_rows = read_csv_rows(gold_path)
        first_gold_day = min(
            [day_key(row["DAY"]) for row in gold_rows if str(row.get("DAY", "")).strip()],
            default=None,
        )
        for gold_row in gold_rows:
            if not str(gold_row.get("DAY", "")).strip():
                continue
            day = day_key(gold_row["DAY"])
            if args.exclude_first_gold_day and first_gold_day is not None and day == first_gold_day:
                continue
            day_rows = input_by_day.get(day, [])
            if not day_rows:
                continue
            ehr_input = compact_day_input(day_rows, args.include_note_rows, args.max_input_chars)
            if not ehr_input.strip():
                continue
            gold_text = str(gold_row.get("TEXT", "")).strip()
            target = extract_assessment_plan(gold_text) if args.target_mode == "ap_section" else gold_text
            if len(target.strip()) < args.min_target_chars:
                continue
            case_id = f"{admission_id}_day{day}"
            examples.append(make_example(case_id, admission_id, day, ehr_input, target))

    examples.sort(key=lambda x: (x["metadata"]["admission_id"], x["metadata"]["day"]))
    if args.limit:
        examples = examples[: args.limit]

    rng = random.Random(args.seed)
    admissions = sorted({ex["metadata"]["admission_id"] for ex in examples})
    rng.shuffle(admissions)
    n_val = int(round(len(admissions) * args.val_ratio))
    n_test = int(round(len(admissions) * args.test_ratio))
    test_ids = set(admissions[:n_test])
    val_ids = set(admissions[n_test : n_test + n_val])

    train, val, test = [], [], []
    for ex in examples:
        admission_id = ex["metadata"]["admission_id"]
        if admission_id in test_ids:
            test.append(ex)
        elif admission_id in val_ids:
            val.append(ex)
        else:
            train.append(ex)

    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "val.jsonl", val)
    write_jsonl(args.out_dir / "test.jsonl", test)
    manifest = {
        "data_root": str(ap_root.as_posix()),
        "target_mode": args.target_mode,
        "include_note_rows": args.include_note_rows,
        "exclude_first_gold_day": args.exclude_first_gold_day,
        "max_input_chars": args.max_input_chars,
        "num_examples": len(examples),
        "num_train": len(train),
        "num_val": len(val),
        "num_test": len(test),
        "num_admissions": len(admissions),
        "seed": args.seed,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
