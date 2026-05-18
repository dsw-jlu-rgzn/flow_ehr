"""
Validate AP/DS task files generated from MIMIC-III.

This is a lightweight sanity checker for the outputs of prepare_mimic3_tasks.py
and the original chronology scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


AP_PROGRESS_PATTERNS = [
    "assessment",
    "plan",
    "subjective",
    "objective",
    "24 hour events",
    "physical exam",
]

RADIOLOGY_PATTERNS = [
    "examination:",
    "indication:",
    "impression:",
    "findings:",
    "portable chest",
    "radiograph",
]

DS_PATTERNS = [
    "history of present illness",
    "brief hospital course",
    "discharge diagnosis",
    "discharge condition",
]


def classify_ap_gold(text: str) -> str:
    lower = str(text).lower()
    if any(pattern in lower for pattern in RADIOLOGY_PATTERNS):
        return "radiology_like"
    if any(pattern in lower for pattern in AP_PROGRESS_PATTERNS):
        return "progress_like"
    return "other"


def validate_ap(data_root: Path) -> None:
    input_dir = data_root / "AP" / "input"
    gold_dir = data_root / "AP" / "gold"
    allowed_ids = load_current_hadm_ids(data_root)
    rows = []

    for gold_file in sorted(gold_dir.glob("gt_*.csv")):
        hadm_id = gold_file.stem.replace("gt_", "")
        if allowed_ids and hadm_id not in allowed_ids:
            continue
        input_file = input_dir / f"input_{hadm_id}.csv"
        gold = pd.read_csv(gold_file)
        inp = pd.read_csv(input_file) if input_file.exists() else pd.DataFrame()

        for _, row in gold.iterrows():
            day = row.get("DAY")
            text = str(row.get("TEXT", ""))
            day_inp = inp[inp["DAY"] == day] if "DAY" in inp.columns else pd.DataFrame()
            rows.append(
                {
                    "hadm_id": hadm_id,
                    "day": day,
                    "gold_type": classify_ap_gold(text),
                    "gold_words": len(text.split()),
                    "data_rows": int((day_inp.get("IS_NOTE", pd.Series(dtype=int)) == 0).sum()),
                    "note_rows": int((day_inp.get("IS_NOTE", pd.Series(dtype=int)) == 1).sum()),
                }
            )

    if not rows:
        print("AP: no gold files found")
        return

    df = pd.DataFrame(rows)
    print("\nAP task summary")
    print(f"  admissions: {df['hadm_id'].nunique()}")
    print(f"  gold days:  {len(df)}")
    print(f"  avg gold words/day: {df['gold_words'].mean():.1f}")
    print(f"  avg non-note rows/day: {df['data_rows'].mean():.1f}")
    print("  gold type counts:")
    for name, count in df["gold_type"].value_counts().items():
        print(f"    {name}: {count}")
    no_context = df[df["data_rows"] == 0]
    if not no_context.empty:
        print(f"  warning: {len(no_context)} gold days have no non-note EHR context")


def validate_ds(data_root: Path) -> None:
    input_dir = data_root / "DS" / "input"
    gold_dir = data_root / "DS" / "gold"
    allowed_ids = load_current_hadm_ids(data_root)
    rows = []

    for gold_file in sorted(gold_dir.glob("gtsummary_*.txt")):
        hadm_id = gold_file.stem.replace("gtsummary_", "")
        if allowed_ids and hadm_id not in allowed_ids:
            continue
        text = gold_file.read_text(encoding="utf-8", errors="replace")
        input_files = sorted(input_dir.glob(f"*_{hadm_id}.csv"))
        input_rows = 0
        input_words = 0
        if input_files:
            inp = pd.read_csv(input_files[0])
            input_rows = len(inp)
            input_words = int(inp["TEXT"].fillna("").astype(str).str.split().str.len().sum())
        rows.append(
            {
                "hadm_id": hadm_id,
                "gold_words": len(text.split()),
                "input_rows": input_rows,
                "input_words": input_words,
                "has_course": "hospital course" in text.lower(),
                "has_diagnosis": "discharge diagnosis" in text.lower(),
                "has_instructions": "discharge instructions" in text.lower(),
                "has_core_sections": all(pattern in text.lower() for pattern in DS_PATTERNS),
            }
        )

    if not rows:
        print("DS: no gold files found")
        return

    df = pd.DataFrame(rows)
    print("\nDS task summary")
    print(f"  admissions: {len(df)}")
    print(f"  avg gold words: {df['gold_words'].mean():.1f}")
    print(f"  avg input rows: {df['input_rows'].mean():.1f}")
    print(f"  avg input words: {df['input_words'].mean():.1f}")
    print(f"  has hospital course: {int(df['has_course'].sum())}/{len(df)}")
    print(f"  has discharge diagnosis: {int(df['has_diagnosis'].sum())}/{len(df)}")
    print(f"  has discharge instructions: {int(df['has_instructions'].sum())}/{len(df)}")
    missing = df[~df["has_core_sections"]]
    if not missing.empty:
        print("  warning: some summaries are missing one or more core sections:")
        print("   ", ", ".join(missing["hadm_id"].astype(str).tolist()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated MIMIC-III AP/DS task files.")
    parser.add_argument("--data-root", default="data")
    return parser.parse_args()


def load_current_hadm_ids(data_root: Path) -> set[str]:
    icu_file = data_root / "target_population" / "filtered" / "filtered_ICUSTAYS.csv"
    if not icu_file.exists():
        return set()
    icu = pd.read_csv(icu_file, usecols=["HADM_ID"])
    return set(icu["HADM_ID"].dropna().astype(int).astype(str))


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    validate_ap(data_root)
    validate_ds(data_root)


if __name__ == "__main__":
    main()
