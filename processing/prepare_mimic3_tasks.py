"""
Prepare MIMIC-III raw data for the original AP/DS tasks.

This script fixes the path/file-name mismatch between the original code and a
raw MIMIC-III directory. It reads source files such as *.csv.gz, the local
CHARTEVENTS.csv-003.gz shard, and NOTEEVENTS-002.csv, then writes the filtered
files expected by the original chronology scripts:

  data/target_population/filtered/filtered_*.csv
  data/MIMIC-III/D_ITEMS.csv
  data/MIMIC-III/D_LABITEMS.csv

After this script finishes you can run:
  python processing/get_chronologies_AP.py
  python processing/get_chronologies_DS.py --modality both --window 24

Or pass --make-tasks to run both chronology scripts automatically.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


CORE_FILES = ["ICUSTAYS", "ADMISSIONS", "PATIENTS"]
LOOKUP_FILES = ["D_ITEMS", "D_LABITEMS"]
EVENT_FILES = [
    "INPUTEVENTS_CV",
    "INPUTEVENTS_MV",
    "LABEVENTS",
    "CHARTEVENTS",
    "PRESCRIPTIONS",
    "NOTEEVENTS",
]


def resolve_mimic_file(raw_dir: Path, stem: str) -> Path:
    candidates = [
        raw_dir / f"{stem}.csv",
        raw_dir / f"{stem}.csv.gz",
        raw_dir / f"{stem}-002.csv",
        raw_dir / f"{stem}-002.csv.gz",
        raw_dir / f"{stem}.csv-002.gz",
        raw_dir / f"{stem}.csv-003.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(raw_dir.glob(f"{stem}*.csv*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {stem} under {raw_dir}")


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "filtered": output_root / "target_population" / "filtered",
        "mimic3": output_root / "MIMIC-III",
        "ap_input": output_root / "AP" / "input",
        "ap_gold": output_root / "AP" / "gold",
        "ds_input": output_root / "DS" / "input",
        "ds_gold": output_root / "DS" / "gold",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def compute_age_years(dob: pd.Series, intime: pd.Series) -> pd.Series:
    # MIMIC-III date shifting can place dates outside pandas' nanosecond
    # timestamp range, so avoid datetime64 arithmetic here.
    dob_parts = dob.astype(str).str.extract(r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})").astype(float)
    intime_parts = intime.astype(str).str.extract(r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})").astype(float)

    age = intime_parts["year"] - dob_parts["year"]
    birthday_passed = (
        (intime_parts["month"] > dob_parts["month"])
        | ((intime_parts["month"] == dob_parts["month"]) & (intime_parts["day"] >= dob_parts["day"]))
    )
    age = age - (~birthday_passed).astype(float)
    return age.astype("float")


def sample_target_icus(
    raw_dir: Path,
    sample_size: int,
    seed: int,
    min_age: int,
    min_los_days: float,
    require_alive: bool,
    require_ap_progress_notes: bool,
) -> tuple[pd.DataFrame, list[int]]:
    icu = read_csv(resolve_mimic_file(raw_dir, "ICUSTAYS"))
    admissions = read_csv(resolve_mimic_file(raw_dir, "ADMISSIONS"))
    patients = read_csv(resolve_mimic_file(raw_dir, "PATIENTS"))

    matched = icu.merge(patients[["SUBJECT_ID", "DOB"]], on="SUBJECT_ID", how="left")
    matched["age"] = compute_age_years(matched["DOB"], matched["INTIME"])
    matched = matched[(matched["age"] >= min_age) & (matched["LOS"] > min_los_days)]

    if require_alive:
        alive_hadm = set(admissions[admissions["DEATHTIME"].isna()]["HADM_ID"].dropna().astype(int))
        matched = matched[matched["HADM_ID"].isin(alive_hadm)]

    if require_ap_progress_notes:
        note_path = resolve_mimic_file(raw_dir, "NOTEEVENTS")
        ap_hadmids: set[int] = set()
        usecols = ["HADM_ID", "CATEGORY", "DESCRIPTION"]
        for chunk in pd.read_csv(note_path, usecols=usecols, chunksize=500_000, low_memory=False):
            hadm = pd.to_numeric(chunk["HADM_ID"], errors="coerce")
            mask = (
                chunk["CATEGORY"].astype(str).eq("Physician ")
                & chunk["DESCRIPTION"].astype(str).str.contains(r"\bProgress Note\b", case=False, na=False)
                & hadm.notna()
            )
            if mask.any():
                ap_hadmids.update(hadm[mask].astype(int).tolist())
        matched = matched[matched["HADM_ID"].isin(ap_hadmids)]
        print(f"Found {len(ap_hadmids)} admissions with physician progress notes")

    matched = matched.dropna(subset=["HADM_ID"]).drop_duplicates("HADM_ID")
    if matched.empty:
        raise ValueError("No ICU stays matched the target criteria.")

    n = min(sample_size, len(matched))
    sampled = matched.sample(n=n, random_state=seed).sort_values("HADM_ID").reset_index(drop=True)
    hadmids = sampled["HADM_ID"].astype(int).tolist()
    print(f"Selected {len(hadmids)} admissions from {len(matched)} eligible ICU stays")
    return sampled, hadmids


def copy_lookup_files(raw_dir: Path, mimic3_dir: Path) -> None:
    for stem in LOOKUP_FILES:
        src = resolve_mimic_file(raw_dir, stem)
        dst = mimic3_dir / f"{stem}.csv"
        print(f"Writing lookup {dst}")
        df = read_csv(src)
        df.to_csv(dst, index=False)


def filter_event_file(
    raw_dir: Path,
    stem: str,
    hadmids: set[int],
    output_file: Path,
    chunksize: int,
) -> int:
    src = resolve_mimic_file(raw_dir, stem)
    print(f"Filtering {stem}: {src.name}")
    output_file.unlink(missing_ok=True)

    total = 0
    wrote_header = False
    for chunk in tqdm(pd.read_csv(src, chunksize=chunksize, low_memory=False), desc=stem):
        if "HADM_ID" not in chunk.columns:
            print(f"  Warning: {stem} has no HADM_ID column, skipping")
            break

        hadm = pd.to_numeric(chunk["HADM_ID"], errors="coerce")
        filtered = chunk[hadm.isin(hadmids)]
        if filtered.empty:
            continue

        filtered.to_csv(output_file, mode="a", index=False, header=not wrote_header)
        wrote_header = True
        total += len(filtered)

    if not wrote_header:
        header = pd.read_csv(src, nrows=0)
        header.to_csv(output_file, index=False)
    print(f"  Wrote {total} rows -> {output_file}")
    return total


def write_core_filtered(raw_dir: Path, filtered_dir: Path, sampled_icu: pd.DataFrame, hadmids: set[int]) -> None:
    sampled_icu.to_csv(filtered_dir / "filtered_ICUSTAYS.csv", index=False)

    for stem in ["ADMISSIONS", "PATIENTS"]:
        src = resolve_mimic_file(raw_dir, stem)
        df = read_csv(src)
        if stem == "ADMISSIONS":
            df = df[df["HADM_ID"].isin(hadmids)]
        elif stem == "PATIENTS":
            subject_ids = set(sampled_icu["SUBJECT_ID"].dropna().astype(int))
            df = df[df["SUBJECT_ID"].isin(subject_ids)]
        df.to_csv(filtered_dir / f"filtered_{stem}.csv", index=False)


def make_filtered_dataset(args) -> None:
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    output_root = Path(args.output_root).resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw MIMIC-III directory not found: {raw_dir}")

    dirs = ensure_dirs(output_root)
    sampled_icu, hadmid_list = sample_target_icus(
        raw_dir=raw_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        min_age=args.min_age,
        min_los_days=args.min_los_days,
        require_alive=not args.include_deceased,
        require_ap_progress_notes=args.require_ap_progress_notes,
    )
    hadmids = set(hadmid_list)

    copy_lookup_files(raw_dir, dirs["mimic3"])
    write_core_filtered(raw_dir, dirs["filtered"], sampled_icu, hadmids)

    for stem in EVENT_FILES:
        out = dirs["filtered"] / f"filtered_{stem}.csv"
        filter_event_file(raw_dir, stem, hadmids, out, chunksize=args.chunksize)


def run_module(script_path: Path, argv: list[str] | None = None) -> None:
    import sys

    argv = argv or []
    old_argv = sys.argv[:]
    sys.argv = [str(script_path)] + argv
    try:
        spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    finally:
        sys.argv = old_argv


def make_tasks(args) -> None:
    repo_root = Path.cwd()
    ap_script = repo_root / "processing" / "get_chronologies_AP.py"
    ds_script = repo_root / "processing" / "get_chronologies_DS.py"

    print("Generating AP task files...")
    run_module(ap_script)

    print("Generating DS task files...")
    run_module(ds_script, ["--modality", args.ds_modality, "--window", str(args.ds_window)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MIMIC-III raw data for AP/DS task generation.")
    parser.add_argument(
        "--raw-dir",
        default=r"C:\Users\dsw54\Desktop\MIMIC_related\mimic-iii-20260513T124356Z-3-001\mimic-iii",
        help="Directory containing raw MIMIC-III CSV/CSV.GZ files.",
    )
    parser.add_argument("--output-root", default="data", help="Output root inside this repo.")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of admissions to sample.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-age", type=int, default=65)
    parser.add_argument("--min-los-days", type=float, default=3.0)
    parser.add_argument("--include-deceased", action="store_true")
    parser.add_argument(
        "--require-ap-progress-notes",
        action="store_true",
        help="Sample only admissions that have physician progress notes usable for AP.",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--make-tasks", action="store_true", help="Run AP and DS chronology scripts after filtering.")
    parser.add_argument("--ds-modality", choices=["both", "notes", "tab"], default="both")
    parser.add_argument("--ds-window", type=int, choices=[24, 48], default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_filtered_dataset(args)
    if args.make_tasks:
        make_tasks(args)


if __name__ == "__main__":
    main()
