from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from processing import get_chronologies_AP as ap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate AP chronology files while skipping completed admissions and logging failures."
    )
    parser.add_argument("--target-path", default="data/target_population/filtered")
    parser.add_argument("--mimic-dir", default="data/MIMIC-III")
    parser.add_argument("--output-dir", default="data/AP/input")
    parser.add_argument("--gt-dir", default="data/AP/gold")
    parser.add_argument("--failure-log", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_path = Path(args.target_path)
    output_dir = Path(args.output_dir)
    gt_dir = Path(args.gt_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    failure_log = Path(args.failure_log) if args.failure_log else output_dir.parent / "ap_chronology_failures.jsonl"

    pd.options.mode.chained_assignment = None

    icu_df = pd.read_csv(target_path / "filtered_ICUSTAYS.csv", low_memory=False)
    notes = pd.read_csv(target_path / "filtered_NOTEEVENTS.csv", low_memory=False)
    phys = notes[notes["CATEGORY"] == "Physician "]
    prog = phys[phys["DESCRIPTION"].astype(str).str.contains(r"\bProgress Note\b", case=False, na=False)]

    input_cv = pd.read_csv(target_path / "filtered_INPUTEVENTS_CV.csv", low_memory=False)
    input_mv = pd.read_csv(target_path / "filtered_INPUTEVENTS_MV.csv", low_memory=False)
    input_mv = input_mv.rename(columns={"STARTTIME": "CHARTTIME"})
    lab_df = pd.read_csv(target_path / "filtered_LABEVENTS.csv", low_memory=False)
    chart_df = pd.read_csv(target_path / "filtered_CHARTEVENTS.csv", low_memory=False)
    meds_df = pd.read_csv(target_path / "filtered_PRESCRIPTIONS.csv", low_memory=False)

    mimic_dir = Path(args.mimic_dir)
    lab_items = pd.read_csv(mimic_dir / "D_LABITEMS.csv", low_memory=False)
    chart_items = pd.read_csv(mimic_dir / "D_ITEMS.csv", low_memory=False)

    prog_ids = set(pd.to_numeric(prog["HADM_ID"], errors="coerce").dropna().astype(int))
    icu_prog = icu_df[icu_df["HADM_ID"].isin(prog_ids)]
    admission_id_list = icu_prog["HADM_ID"].dropna().astype(int).tolist()

    failures = 0
    generated = 0
    skipped = 0
    with failure_log.open("a", encoding="utf-8", newline="\n") as log:
        for admission_id in tqdm(admission_id_list):
            input_path = output_dir / f"input_{admission_id}.csv"
            gt_path = gt_dir / f"gt_{admission_id}.csv"
            if not args.force and input_path.exists() and gt_path.exists():
                skipped += 1
                continue
            try:
                dbsource = icu_df.loc[icu_df["HADM_ID"] == admission_id, "DBSOURCE"].values[0]
                input_df = input_cv if dbsource == "carevue" else input_mv

                struc_tl = ap.get_structured(
                    admission_id,
                    (lab_df, chart_df, input_df, meds_df),
                    (lab_items, chart_items),
                )
                tl_prog = ap.get_prog_notes(admission_id, prog)
                notes_tl = ap.get_ehr_notes(admission_id, notes)

                combined = pd.concat([struc_tl, tl_prog, notes_tl])
                combined = combined.dropna(subset="TIME")
                combined = combined.sort_values(by="TIME").reset_index(drop=True)
                combined["IS_NOTE"] = combined["IS_NOTE"].apply(lambda x: 1 if x == 1 else 0)

                combined_w_days = ap.day_count(combined)
                combined_tl_rel = ap.get_rel_times(combined_w_days)
                gold_notes = ap.get_gold(combined_w_days)

                combined_tl_rel.to_csv(input_path, index=False)
                gold_notes.to_csv(gt_path, index=False)
                generated += 1
            except Exception as exc:  # noqa: BLE001 - this is a batch data-generation guard.
                failures += 1
                log.write(
                    json.dumps(
                        {
                            "admission_id": int(admission_id),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                log.flush()

    print(
        json.dumps(
            {
                "num_admissions": len(admission_id_list),
                "generated": generated,
                "skipped_existing": skipped,
                "failures": failures,
                "failure_log": str(failure_log),
                "output_dir": str(output_dir),
                "gt_dir": str(gt_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
