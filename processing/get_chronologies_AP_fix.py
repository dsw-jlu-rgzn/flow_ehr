import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build AP-style chronologies from MIMIC-IV target tables. "
            "This script requires a progress-note source; radiology reports are "
            "not used as AP targets unless explicitly requested."
        )
    )
    parser.add_argument("--target-path", default="data/MIMIC-IV/target")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--gt-dir", default=None)
    parser.add_argument(
        "--notes-file",
        default="",
        help=(
            "Optional note CSV to use for AP targets. Expected columns can be "
            "MIMIC-III style (HADM_ID, CHARTTIME, CATEGORY, DESCRIPTION, TEXT) "
            "or lower-case equivalents."
        ),
    )
    parser.add_argument(
        "--allow-radiology-target",
        action="store_true",
        help=(
            "Explicitly build a radiology-report target dataset from "
            "target_radiology.csv. This is not an AP dataset."
        ),
    )
    return parser.parse_args()


def normalize_note_columns(notes: pd.DataFrame) -> pd.DataFrame:
    """Normalize common MIMIC note schemas to lower-case names used below."""

    rename_map = {
        "HADM_ID": "hadm_id",
        "CHARTTIME": "charttime",
        "STORETIME": "storetime",
        "CATEGORY": "category",
        "DESCRIPTION": "description",
        "TEXT": "text",
        "ROW_ID": "row_id",
        "SUBJECT_ID": "subject_id",
    }
    notes = notes.rename(columns={key: value for key, value in rename_map.items() if key in notes.columns})
    required = {"hadm_id", "charttime", "text"}
    missing = required.difference(notes.columns)
    if missing:
        raise ValueError(f"Note file is missing required columns: {sorted(missing)}")

    if "note_id" not in notes.columns:
        notes["note_id"] = notes.index.astype(str)
    if "category" not in notes.columns:
        notes["category"] = ""
    if "description" not in notes.columns:
        notes["description"] = ""
    if "note_type" not in notes.columns:
        notes["note_type"] = ""
    return notes


def select_progress_notes(notes: pd.DataFrame) -> pd.DataFrame:
    """Match the original MIMIC-III AP target rule as closely as possible."""

    category = notes["category"].fillna("").astype(str)
    description = notes["description"].fillna("").astype(str)
    note_type = notes["note_type"].fillna("").astype(str)

    mimic3_rule = category.str.strip().str.lower().eq("physician") & description.str.contains(
        r"\bProgress Note\b", case=False, na=False
    )
    generic_progress_rule = (
        description.str.contains(r"\bProgress Note\b", case=False, na=False)
        | note_type.str.contains(r"\bProgress\b", case=False, na=False)
    )
    return notes[mimic3_rule | generic_progress_rule].copy()


def load_note_frames(args) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return target notes, context notes, and a target label."""

    target_path = args.target_path
    candidates = []
    if args.notes_file:
        candidates.append(args.notes_file)
    candidates.extend(
        [
            f"{target_path}/target_noteevents.csv",
            f"{target_path}/target_NOTEEVENTS.csv",
            f"{target_path}/target_notes.csv",
            f"{target_path}/target_progress_notes.csv",
        ]
    )

    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        notes = normalize_note_columns(pd.read_csv(path))
        progress = select_progress_notes(notes)
        if progress.empty:
            continue
        progress_ids = set(progress["note_id"].astype(str))
        context = notes[~notes["note_id"].astype(str).isin(progress_ids)].copy()
        context = context[~context["note_type"].fillna("").astype(str).str.upper().eq("DS")]
        print(f"Using progress-note targets from {path}: {len(progress)} rows")
        return progress, context, "ap"

    if args.allow_radiology_target:
        radiology_path = f"{target_path}/target_radiology.csv"
        if not os.path.exists(radiology_path):
            raise FileNotFoundError(f"--allow-radiology-target was set, but {radiology_path} does not exist.")
        notes = normalize_note_columns(pd.read_csv(radiology_path))
        target = notes[notes["note_type"].fillna("").astype(str).str.upper().eq("RR")].copy()
        context = notes[~notes["note_type"].fillna("").astype(str).str.upper().eq("RR")].copy()
        if target.empty:
            raise ValueError(f"No RR radiology reports found in {radiology_path}.")
        print(
            "Warning: building a radiology-report target dataset, not AP. "
            "Use RAD output directories or rename the task accordingly."
        )
        return target, context, "radiology"

    raise FileNotFoundError(
        "No progress-note source was found. To build AP data aligned with the "
        "MIMIC-III preprocessing, provide a note file with physician progress "
        "notes via --notes-file or place target_noteevents.csv/target_notes.csv "
        "under the target path. Refusing to use target_radiology.csv as AP gold."
    )


def filter(labs: pd.DataFrame, charts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # remove duplicates and keep only abnormal lab values
    # filtering by warning in charts - remove 0 values    

    # MIMIC-IV: use lowercase column names
    duplicates = pd.merge(charts, labs, on=['charttime', 'item_desc'], how='inner')
    duplicate_indices = duplicates.index.get_level_values(0) if isinstance(duplicates.index, pd.MultiIndex) else []

    if len(duplicate_indices) > 0:
        charts = charts.drop(duplicate_indices, errors='ignore')

    # MIMIC-IV: warning column is lowercase
    charts = charts[charts['warning'] != 0].reset_index(drop=True)
    labs = labs[labs['flag'] == 'abnormal']
    return labs, charts


def remove_duplicates_struc(df: pd.DataFrame) -> pd.DataFrame:
    meds = df[df['is_med'] == 1]
    df = df[df['is_med'] == 0]
    df['charttime'] = pd.to_datetime(df['charttime'])

    # Group by VALUE, UNIT, and ITEM
    result = []
    for _, group in df.groupby(['value', 'valueuom', 'item_desc']):
        # Sort by TIME within each group (already sorted globally)
        group = group.sort_values(by='charttime')
        keep_indices = []
        last_time = None

        for index, row in group.iterrows():
            if last_time is None or (row['charttime'] - last_time).total_seconds() > 3600:
                keep_indices.append(index)
            last_time = row['charttime']
        
        result.append(group.loc[keep_indices])

    if not result:
        return df
    
    # Combine all groups back into a single dataframe
    filtered_df = pd.concat(result).sort_index()
    filtered_df['charttime'] = filtered_df['charttime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    filtered_df = pd.concat((filtered_df, meds))

    filtered_df = filtered_df.sort_values(by='charttime')
    return filtered_df


def temporal_order_struc(tab_df: pd.DataFrame) -> pd.DataFrame:
    # get unique timestamps, group by those. throw out empty values. convert to narrative format - different format if is_input == 1
    # truncate floats after two decimals

    timestamps = tab_df['charttime'].unique()
    text = []
    for time in timestamps:
        output_str = ''
        time_df = tab_df[tab_df['charttime'] == time]
        for i in time_df.index:
            value = time_df.loc[i, 'value'] if not pd.isna(time_df.loc[i, 'value']) else ''
            uom = time_df.loc[i, 'valueuom'] if not pd.isna(time_df.loc[i, 'valueuom']) else ''
            desc = time_df.loc[i, 'item_desc']
            # if value can be converted to float round to two decimal places
            if '.' in str(value):
                try:
                    value = f'{float(value):.2f}'
                except:
                    ValueError
            # different phrasing if input event
            if time_df.loc[i, 'is_input'] == 0 and value != '':
                output_str += f'{desc} is {value} {uom}. '
            elif time_df.loc[i, 'is_input'] == 1:
                if value == '':    
                    output_str += f'{desc} is administered. '
                else:
                    output_str += f'{value} {uom} of {desc} is administered. '
            elif time_df.loc[i, 'is_med'] == 1:
                drug = time_df.loc[i, 'drug'] if not pd.isna(time_df.loc[i, 'drug']) else ''
                value = time_df.loc[i, 'prod_strength'] if not pd.isna(time_df.loc[i, 'prod_strength']) else ''
                if value == '':    
                    output_str += f'{drug} is administered. '
                else:
                    output_str += f'{value} of {drug} is administered. '
        text.append(output_str)
    return pd.DataFrame(zip(timestamps, text), columns=['TIME', 'TEXT'])


def get_structured(hadmid: int, tab_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], lookup: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    # takes hospital admission and returns dataframe of chronologically ordered structured data
    lab_df, chart_df, input_df, meds_df = tab_data
    lab_items, chart_items = lookup
    
    # get structured data (lab, chart, input events for relevant hospital admission)
    # MIMIC-IV target data may have hadm_id as string with '.0' suffix (e.g., '23056393.0')
    # Convert hadmid to match the dtype of the target column
    patient_lab = lab_df[lab_df['hadm_id'].astype(str).str.replace(r'\.0$', '', regex=True) == str(hadmid)]
    patient_chart = chart_df[chart_df['hadm_id'].astype(str).str.replace(r'\.0$', '', regex=True) == str(hadmid)]
    patient_input = input_df[input_df['hadm_id'].astype(str).str.replace(r'\.0$', '', regex=True) == str(hadmid)]
    patient_meds = meds_df[meds_df['hadm_id'].astype(str).str.replace(r'\.0$', '', regex=True) == str(hadmid)]

    # convert itemid to natural language description
    # MIMIC-IV: labevents itemid may be string, d_labitems itemid is int64
    # Convert both to int for comparison
    patient_lab['itemid_int'] = pd.to_numeric(patient_lab['itemid'], errors='coerce')
    lab_items['itemid_int'] = pd.to_numeric(lab_items['itemid'], errors='coerce')
    patient_lab['item_desc'] = patient_lab['itemid_int'].apply(
        lambda x: lab_items.loc[lab_items['itemid_int'] == x]['label'].values[0]
        if x in lab_items['itemid_int'].values else f'Unknown Lab Item {int(x) if pd.notna(x) else x}'
    )

    # MIMIC-IV: chartevents and d_items itemid are both int64
    patient_chart['itemid_int'] = pd.to_numeric(patient_chart['itemid'], errors='coerce')
    chart_items['itemid_int'] = pd.to_numeric(chart_items['itemid'], errors='coerce')
    patient_chart['item_desc'] = patient_chart['itemid_int'].apply(
        lambda x: chart_items.loc[chart_items['itemid_int'] == x]['label'].values[0]
        if x in chart_items['itemid_int'].values else f'Unknown Chart Item {int(x) if pd.notna(x) else x}'
    )

    patient_input['itemid_int'] = pd.to_numeric(patient_input['itemid'], errors='coerce')
    patient_input['item_desc'] = patient_input['itemid_int'].apply(
        lambda x: chart_items.loc[chart_items['itemid_int'] == x]['label'].values[0]
        if x in chart_items['itemid_int'].values else f'Unknown Input Item {int(x) if pd.notna(x) else x}'
    )

    # MIMIC-IV: inputevents uses 'amount' and 'amountuom' (lowercase)
    patient_input = patient_input.rename(columns={'amount': 'value', 'amountuom': 'valueuom'})
    patient_input['is_input'] = 1

    # MIMIC-IV: prescriptions uses 'starttime' instead of 'STARTDATE'
    patient_meds['is_med'] = 1
    patient_meds = patient_meds.rename(columns={'starttime': 'charttime'})

    patient_lab, patient_chart = filter(patient_lab, patient_chart)

    # consolidate all structured data into one df, order by timestamp
    patient_struc = pd.concat((patient_lab, patient_chart, patient_input, patient_meds))
    patient_struc['is_input'] = patient_struc['is_input'].apply(lambda x: 1 if x == 1 else 0)
    patient_struc['is_med'] = patient_struc['is_med'].apply(lambda x: 1 if x == 1 else 0)
    patient_struc = patient_struc.sort_values(by='charttime')
    patient_struc = patient_struc.reset_index(drop=True)

    # remove duplicate entries in structured data here
    patient_struc = remove_duplicates_struc(patient_struc)

    struc_tl = temporal_order_struc(patient_struc)
    return struc_tl


def get_prog_notes(hadmid: int, notes: pd.DataFrame) -> pd.DataFrame:
    patient_notes = notes[notes['hadm_id'] == hadmid]
    patient_notes = patient_notes.sort_values(by='charttime')
    patient_notes = patient_notes.reset_index(drop=True)
    tl_prog = patient_notes[["charttime", 'text']]
    tl_prog = tl_prog.rename(columns={'charttime': 'TIME', 'text': 'TEXT'})
    tl_prog['IS_NOTE'] = 1
    return tl_prog


def temporal_order_note(note_df: pd.DataFrame) -> pd.DataFrame:
    timestamps = []
    text = []

    for i in note_df.index:
        ts = note_df.loc[i, 'charttime']
        note_type = note_df.loc[i, 'note_type'] if "note_type" in note_df.columns else ""
        if not note_type and "category" in note_df.columns:
            note_type = note_df.loc[i, "category"]
        note_text = note_df.loc[i, 'text']
        timestamps.append(ts)
        text.append(f'{note_type} note: \n{note_text}')

    notes_tl = pd.DataFrame(zip(timestamps, text), columns=['TIME', 'TEXT'])
    notes_tl = notes_tl.sort_values(by='TIME')
    return notes_tl


def get_ehr_notes(hadmid: int, notes: pd.DataFrame) -> pd.DataFrame:
    patient_notes = notes[notes['hadm_id'] == hadmid]
    ehr_notes = patient_notes
    ehr_notes = ehr_notes.dropna(subset='charttime')
    notes_tl = temporal_order_note(ehr_notes)

    # remove notes at same timestamp
    notes_tl = notes_tl.drop_duplicates(subset=['TIME'], keep='last')
    return notes_tl


def get_rel_times(df):
    # convert absolute timestamps to relative ones (w.r.t first entry)
    format_str = r'%Y-%m-%d %H:%M:%S'
    rel_times = []
    for i in df.index:
        if i == 0:
            rel_times.append('First entry: ')
        else:
            previous_ts = str(df.iloc[i-1]['TIME'])
            current_ts = str(df.iloc[i]['TIME'])
            if current_ts != 'nan':
                previous_ts = datetime.strptime(previous_ts, format_str)
                current_ts = datetime.strptime(current_ts, format_str)
                difference = current_ts - previous_ts
                
                days = difference.days
                hours, remainder = divmod(difference.seconds, 3600)
                minutes = remainder // 60
                time_lst = []
                if days > 0:
                    time_lst.append(f"{days} day{'s' if days > 1 else ''}")
                if hours > 0:
                    time_lst.append(f"{hours} hour{'s' if hours > 1 else ''}")
                if minutes > 0:
                    time_lst.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
                if not time_lst:
                    # Same timestamp as previous entry
                    rel_times.append('Same time: ')
                else:
                    rel_times.append(" ".join(time_lst) + " later: ")
            else:
                rel_times.append(None)
    df['REL_TIME'] = rel_times
    return df


def day_count(df: pd.DataFrame) -> pd.DataFrame:
    format_str = r'%Y-%m-%d %H:%M:%S'
    days = []
    for i in df.index:
        if i == 0:
            first_entry = datetime.strptime(df.iloc[i]['TIME'], format_str)
            days.append(1)
        else:
            current_entry = datetime.strptime(df.iloc[i]['TIME'], format_str)
            diff_days = (current_entry - first_entry).days
            days.append(diff_days + 1)
    df['DAY'] = days
    return df


def get_gold(df: pd.DataFrame) -> pd.DataFrame:
    day_groups = {day: group for day, group in df.groupby('DAY')}

    days = []
    pns = []

    for day, group in day_groups.items():
        if len(group[group['IS_NOTE'] != 0]):
            progress_note = group[group['IS_NOTE'] == 1].iloc[-1]['TEXT']
            days.append(day)
            pns.append(progress_note)
    return pd.DataFrame(zip(days, pns), columns=['DAY', 'TEXT'])


def main():
    args = parse_args()
    pd.options.mode.chained_assignment = None  # Turns off the warning

    # MIMIC-IV target path (output from get_target_population_fix.py)
    target_path = args.target_path

    icu_df = pd.read_csv(f'{target_path}/target_ICUSTAYS.csv')

    target_notes, context_notes, target_kind = load_note_frames(args)

    # MIMIC-IV: single inputevents.csv (no CV/MV split)
    input_df = pd.read_csv(f'{target_path}/target_inputevents.csv')

    lab_df = pd.read_csv(f'{target_path}/target_labevents.csv')
    chart_df = pd.read_csv(f'{target_path}/target_chartevents.csv')
    meds_df = pd.read_csv(f'{target_path}/target_prescriptions.csv')

    # MIMIC-IV lookup tables
    lab_items = pd.read_csv('long_data_related/longitudinal_clinical_summarization/mimic-iv-3.1/hosp/d_labitems.csv.gz')
    chart_items = pd.read_csv('long_data_related/longitudinal_clinical_summarization/mimic-iv-3.1/icu/d_items.csv.gz')

    if target_kind == "radiology":
        output_dir = args.output_dir or "data/RAD/input"
        gt_dir = args.gt_dir or "data/RAD/gold"
    else:
        output_dir = args.output_dir or "data/AP/input"
        gt_dir = args.gt_dir or "data/AP/gold"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # Match the original MIMIC-III AP logic: only admissions with target notes.
    prog_ids = target_notes['hadm_id'].unique()
    icu_prog = icu_df[icu_df['hadm_id'].isin(prog_ids)]
    admission_id_list = icu_prog['hadm_id'].to_list()
    print(f'Found {len(admission_id_list)} admissions with both ICUSTAYS and {target_kind} target notes')

    for admission_id in tqdm(admission_id_list):
        # MIMIC-IV: no DBSOURCE column, single inputevents.csv
        tab_data = (lab_df, chart_df, input_df, meds_df)
        dictionaries = (lab_items, chart_items)

        struc_tl = get_structured(admission_id, tab_data, dictionaries)
        tl_prog = get_prog_notes(admission_id, target_notes)
        notes_tl = get_ehr_notes(admission_id, context_notes)

        combined = pd.concat([struc_tl, tl_prog, notes_tl])
        combined = combined.dropna(subset='TIME')
        combined = combined.sort_values(by='TIME').reset_index(drop=True)
        combined['IS_NOTE'] = combined['IS_NOTE'].apply(lambda x: 1 if x == 1 else 0)

        combined_w_days = day_count(combined)
        combined_tl_rel = get_rel_times(combined_w_days)
        gold_notes = get_gold(combined_w_days)

        combined_tl_rel.to_csv(f'{output_dir}/input_{admission_id}.csv', index=False)
        gold_notes.to_csv(f'{gt_dir}/gt_{admission_id}.csv', index=False)


if __name__ == '__main__':
    main()
