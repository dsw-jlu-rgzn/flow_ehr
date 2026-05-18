"""Regenerate MIMIC-III DS inputs/gold with a discharge-time anchor.

The original DS chronology script anchors the input window on the last
available event for an admission and picks the first discharge summary note.
Both choices can produce mismatched pairs when an admission has multiple
discharge summaries or post-summary events. This script instead:

1. composes all discharge-summary notes for each HADM_ID, ordered by CHARTDATE
   then ROW_ID, so a short addendum cannot replace the main summary;
2. anchors the input chronology on ADMISSIONS.DISCHTIME;
3. keeps only events at or before DISCHTIME;
4. optionally restricts inputs to the last N hours before DISCHTIME.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from get_chronologies_DS import (
    format_result,
    get_rel_times,
    get_structured,
    temporal_order_note,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate fixed MIMIC-III DS task files.")
    parser.add_argument("--target-path", default="data/target_population/filtered")
    parser.add_argument("--lookup-path", default="data/MIMIC-III")
    parser.add_argument("--output-root", default="data/DS_fixed")
    parser.add_argument("--modality", choices=["both", "notes", "tab"], default="both")
    parser.add_argument(
        "--window-hours",
        type=int,
        default=24,
        help="Input window before DISCHTIME. Use 0 to keep the full admission chronology.",
    )
    parser.add_argument("--summary-csv", default=None)
    return parser.parse_args()


def normalize_hadm(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def parse_datetime(value: object) -> pd.Timestamp | pd.NaT:
    return pd.to_datetime(value, errors="coerce")


def prepare_discharge_summaries(notes: pd.DataFrame) -> pd.DataFrame:
    ds_notes = notes[notes["CATEGORY"].eq("Discharge summary")].copy()
    ds_notes["HADM_ID_NUM"] = normalize_hadm(ds_notes["HADM_ID"])
    ds_notes["CHARTDATE_TS"] = pd.to_datetime(ds_notes["CHARTDATE"], errors="coerce")
    ds_notes["ROW_ID_NUM"] = pd.to_numeric(ds_notes["ROW_ID"], errors="coerce").fillna(-1)
    ds_notes = ds_notes.dropna(subset=["HADM_ID_NUM"])
    return ds_notes.sort_values(["HADM_ID_NUM", "CHARTDATE_TS", "ROW_ID_NUM"])


def compose_discharge_record(rows: pd.DataFrame) -> str:
    parts = []
    for idx, (_, row) in enumerate(rows.iterrows(), start=1):
        description = str(row.get("DESCRIPTION", "")).strip() or "Report"
        chartdate = str(row.get("CHARTDATE", "")).strip()
        row_id = str(row.get("ROW_ID", "")).strip()
        header = f"===== Discharge summary note {idx}: {description}; chartdate={chartdate}; row_id={row_id} ====="
        parts.append(f"{header}\n{row['TEXT']}")
    return "\n\n".join(parts)


def build_note_timeline(hadmid: int, notes: pd.DataFrame) -> pd.DataFrame:
    patient_notes = notes[normalize_hadm(notes["HADM_ID"]).eq(hadmid)].copy()
    patient_notes = patient_notes[~patient_notes["CATEGORY"].eq("Discharge summary")]
    notes_tl = temporal_order_note(patient_notes)
    return notes_tl.drop_duplicates(subset=["TIME"], keep="last")


def apply_discharge_anchor(
    chronology: pd.DataFrame,
    discharge_time: pd.Timestamp,
    window_hours: int,
) -> pd.DataFrame:
    df = chronology.copy()
    df["TIME_TS"] = pd.to_datetime(df["TIME"], errors="coerce")
    df = df.dropna(subset=["TIME_TS"])
    df = df[df["TIME_TS"] <= discharge_time]
    if window_hours > 0:
        start_time = discharge_time - pd.Timedelta(hours=window_hours)
        df = df[df["TIME_TS"] >= start_time]
    df = df.sort_values("TIME_TS").drop(columns=["TIME_TS"]).reset_index(drop=True)
    return df


def parse_note_discharge_date(text: str) -> str:
    match = re.search(r"Discharge Date:\s*\[\*\*(.*?)\*\*\]", str(text))
    if not match:
        return ""
    date_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", match.group(1))
    if not date_match:
        return ""
    year, month, day = date_match.groups()
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text)))


def load_inputs(target_path: Path, lookup_path: Path) -> dict[str, pd.DataFrame]:
    return {
        "icu": pd.read_csv(target_path / "filtered_ICUSTAYS.csv", low_memory=False),
        "admissions": pd.read_csv(target_path / "filtered_ADMISSIONS.csv", low_memory=False),
        "notes": pd.read_csv(target_path / "filtered_NOTEEVENTS.csv", low_memory=False),
        "input_cv": pd.read_csv(target_path / "filtered_INPUTEVENTS_CV.csv", low_memory=False),
        "input_mv": pd.read_csv(target_path / "filtered_INPUTEVENTS_MV.csv", low_memory=False),
        "labs": pd.read_csv(target_path / "filtered_LABEVENTS.csv", low_memory=False),
        "charts": pd.read_csv(target_path / "filtered_CHARTEVENTS.csv", low_memory=False),
        "meds": pd.read_csv(target_path / "filtered_PRESCRIPTIONS.csv", low_memory=False),
        "lab_items": pd.read_csv(lookup_path / "D_LABITEMS.csv", low_memory=False),
        "chart_items": pd.read_csv(lookup_path / "D_ITEMS.csv", low_memory=False),
    }


def main() -> None:
    pd.options.mode.chained_assignment = None
    args = parse_args()

    target_path = Path(args.target_path)
    lookup_path = Path(args.lookup_path)
    output_root = Path(args.output_root)
    label = f"{args.window_hours}h" if args.window_hours > 0 else "full"
    output_dir = output_root / label / "input"
    gold_dir = output_root / label / "gold"
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    data = load_inputs(target_path, lookup_path)
    ds_notes = prepare_discharge_summaries(data["notes"])
    ds_by_hadm = {int(hadmid): group.copy() for hadmid, group in ds_notes.groupby("HADM_ID_NUM")}

    admissions = data["admissions"].copy()
    admissions["HADM_ID_NUM"] = normalize_hadm(admissions["HADM_ID"])
    admissions["DISCHTIME_TS"] = pd.to_datetime(admissions["DISCHTIME"], errors="coerce")
    dischtime_by_hadm = dict(zip(admissions["HADM_ID_NUM"].astype(int), admissions["DISCHTIME_TS"]))
    dischdate_by_hadm = dict(zip(admissions["HADM_ID_NUM"].astype(int), admissions["DISCHTIME"].astype(str).str[:10]))

    summary_rows: list[dict[str, object]] = []
    admission_ids = [int(x) for x in data["icu"]["HADM_ID"].dropna().astype(int).tolist()]

    for admission_id in tqdm(admission_ids, desc=f"DS {label}"):
        discharge_rows = ds_by_hadm.get(admission_id)
        gold_txt = compose_discharge_record(discharge_rows) if discharge_rows is not None else ""
        discharge_time = dischtime_by_hadm.get(admission_id)
        if discharge_rows is None or not gold_txt or pd.isna(discharge_time):
            summary_rows.append(
                {
                    "hadm_id": admission_id,
                    "status": "skipped_missing_gold_or_dischtime",
                }
            )
            continue

        dbsource = data["icu"][data["icu"]["HADM_ID"].astype(int).eq(admission_id)]["DBSOURCE"].values[0]
        input_df = data["input_cv"] if dbsource == "carevue" else data["input_mv"]
        tab_data = (data["labs"], data["charts"], input_df, data["meds"])
        dictionaries = (data["lab_items"], data["chart_items"])

        structured_data = get_structured(admission_id, tab_data, dictionaries)
        note_data = build_note_timeline(admission_id, data["notes"])

        if args.modality == "both":
            combined_tl = pd.concat((structured_data, note_data), ignore_index=True)
        elif args.modality == "notes":
            combined_tl = note_data
        else:
            combined_tl = structured_data

        combined_tl = combined_tl.sort_values(by="TIME").reset_index(drop=True)
        anchored = apply_discharge_anchor(combined_tl, discharge_time, args.window_hours)
        anchored = get_rel_times(anchored) if not anchored.empty else pd.DataFrame(columns=["TIME", "TEXT", "REL_TIME"])
        formatted = format_result(anchored)

        input_name = f"{label}_{args.modality}_{admission_id}.csv"
        gold_name = f"gtsummary_{admission_id}.txt"
        formatted.to_csv(output_dir / input_name, index=False, quoting=csv.QUOTE_MINIMAL)
        (gold_dir / gold_name).write_text(str(gold_txt), encoding="utf-8")

        event_times = pd.to_datetime(formatted["TIME"], errors="coerce") if not formatted.empty else pd.Series(dtype="datetime64[ns]")
        summary_rows.append(
            {
                "hadm_id": admission_id,
                "status": "ok",
                "dbsource": dbsource,
                "window_hours": args.window_hours,
                "input_file": input_name,
                "gold_file": gold_name,
                "input_rows": len(formatted),
                "input_words": word_count(" ".join(formatted.get("TEXT", pd.Series(dtype=str)).astype(str).tolist())),
                "gold_words": word_count(gold_txt),
                "admissions_dischdate": dischdate_by_hadm.get(admission_id, ""),
                "gold_note_count": len(discharge_rows),
                "gold_note_discharge_dates": "|".join(
                    date for date in (parse_note_discharge_date(text) for text in discharge_rows["TEXT"].astype(str)) if date
                ),
                "gold_note_row_ids": "|".join(discharge_rows["ROW_ID"].astype(str).tolist()),
                "first_input_time": "" if event_times.empty else str(event_times.min()),
                "last_input_time": "" if event_times.empty else str(event_times.max()),
                "events_after_discharge": int((event_times > discharge_time).sum()) if not event_times.empty else 0,
            }
        )

    summary_csv = Path(args.summary_csv) if args.summary_csv else output_root / label / "ds_regeneration_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Wrote input files to {output_dir}")
    print(f"Wrote gold files to {gold_dir}")
    print(f"Wrote summary to {summary_csv}")


if __name__ == "__main__":
    main()
