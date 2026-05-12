import pandas as pd
from datetime import datetime
from tqdm import tqdm
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default='both', help='modality - both, notes or tab')
    parser.add_argument('--window', type=int, default=24, help='temporal context window - 24 or 48')
    args = parser.parse_args()
    return args


def filter(labs: pd.DataFrame, charts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes duplicate instances that are present in both chart and lab events from charts.
    Measurements in chart events that have warning == 0 are discarded. warning == 1 or NaN are kept.
    Measurements in lab events with the 'abnormal' flag are kept, others discarded.

    Args:
        labs (pd.DataFrame): DataFrame containing lab events
        charts (pd.DataFrame): DataFrame containing chart events

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: modified lab and chart event DataFrames
    """
    # MIMIC-IV: use lowercase column names
    duplicates = pd.merge(charts, labs, on=['charttime', 'item_desc'], how='inner')
    # Use index-based dedup since MIMIC-IV doesn't have ROW_ID
    # Instead, we use the index of the charts DataFrame
    duplicate_indices = duplicates.index.get_level_values(0) if isinstance(duplicates.index, pd.MultiIndex) else []

    if len(duplicate_indices) > 0:
        charts = charts.drop(duplicate_indices, errors='ignore')

    # MIMIC-IV: warning column is lowercase
    charts = charts[charts['warning'] != 0].reset_index(drop=True)
    labs = labs[labs['flag'] == 'abnormal']
    return labs, charts


def remove_duplicates_struc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate measurements (with the same value) in structured/tabular data 
    if they are present multiple times at the same timestamp or within one hour of each other. 

    Args:
        df (pd.DataFrame): DataFrame containing combined structured data

    Returns:
        pd.DataFrame: modified structured data DataFrame
    """
    meds = df[df['is_med'] == 1]
    new_df = df[df['is_med'] == 0]
    new_df['charttime'] = pd.to_datetime(new_df['charttime'])

    result = []
    for _, group in new_df.groupby(['value', 'valueuom', 'item_desc']):
        group = group.sort_values(by='charttime')
        keep_indices = []
        last_time = None

        for index, row in group.iterrows():
            if last_time is None or (row['charttime'] - last_time).total_seconds() > 3600:
                keep_indices.append(index)
            last_time = row['charttime']
        
        result.append(group.loc[keep_indices])
    
    # if no duplicates, return original input DataFrame
    if not result:
        return df

    # Combine all groups back into a single dataframe
    filtered_df = pd.concat(result).sort_index()
    filtered_df['charttime'] = filtered_df['charttime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    filtered_df = pd.concat((filtered_df, meds))

    filtered_df = filtered_df.sort_values(by='charttime')
    return filtered_df


def temporal_order_struc(tab_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct temporally ordered structured data. Measurements are grouped by timestamp and
    converted to a narrative format. Measurements without timestamps are removed.

    Args:
        tab_df (pd.DataFrame): DataFrame containing combined structured data

    Returns:
        pd.DataFrame: DataFrame containing temporally ordered structured data in narrative format
    """
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
            # different phrasing if input event or medication
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
    """
    For a given hadmid, gathers all structured data (lab, chart, input, medications) and combines it into
    one DataFrame. Applies filtering, converting to narrative format, and removal of duplicates.

    Args:
        hadmid (int): unique hospital admission ID
        tab_data (tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]): tuple containing DataFrames for lab, chart, input events + medications
        lookup (tuple[pd.DataFrame, pd.DataFrame]): tuple of DataFrames used as dictionaries to map item IDs to natural language descriptions

    Returns:
        pd.DataFrame: DataFrame of combined structured data, ordered by timestamp
    """
    lab_df, chart_df, input_df, meds_df = tab_data
    lab_items, chart_items = lookup
    
    # get structured data for relevant hospital admission
    patient_lab = lab_df[lab_df['hadm_id'] == hadmid]
    patient_chart = chart_df[chart_df['hadm_id'] == hadmid]
    patient_input = input_df[input_df['hadm_id'] == hadmid]
    patient_meds = meds_df[meds_df['hadm_id'] == hadmid]

    # convert itemid to natural language description
    # MIMIC-IV: d_labitems uses 'itemid' (lowercase), d_items uses 'itemid' (lowercase)
    patient_lab['item_desc'] = patient_lab['itemid'].apply(
        lambda x: lab_items.loc[lab_items['itemid'] == x]['label'].values[0]
        if x in lab_items['itemid'].values else f'Unknown Lab Item {x}'
    )
    patient_chart['item_desc'] = patient_chart['itemid'].apply(
        lambda x: chart_items.loc[chart_items['itemid'] == x]['label'].values[0]
        if x in chart_items['itemid'].values else f'Unknown Chart Item {x}'
    )
    patient_input['item_desc'] = patient_input['itemid'].apply(
        lambda x: chart_items.loc[chart_items['itemid'] == x]['label'].values[0]
        if x in chart_items['itemid'].values else f'Unknown Input Item {x}'
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

    # remove duplicate entries in structured data
    patient_struc = remove_duplicates_struc(patient_struc)

    struc_tl = temporal_order_struc(patient_struc)
    return struc_tl


def temporal_order_note(note_df: pd.DataFrame) -> pd.DataFrame:
    """
    Order note data by timestamp

    Args:
        note_df (pd.DataFrame): collected notes for one hospital admission

    Returns:
        pd.DataFrame: ordered notes 
    """
    timestamps = []
    text = []

    for i in note_df.index:
        ts = note_df.loc[i, 'charttime']
        # MIMIC-IV: discharge.csv uses 'note_type' instead of 'CATEGORY'
        note_type = note_df.loc[i, 'note_type']
        note_text = note_df.loc[i, 'text']
        timestamps.append(ts)
        text.append(f'{note_type} note: \n{note_text}')

    notes_tl = pd.DataFrame(zip(timestamps, text), columns=['TIME', 'TEXT'])
    notes_tl = notes_tl.sort_values(by='TIME')
    return notes_tl


def get_notes(hadmid: int, notes: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Extracts all notes for a given hadmid except discharge summaries. Removes duplicate notes at same timestamp.

    Args:
        hadmid (int): unique hospital admission ID
        notes (pd.DataFrame): DataFrame containing all notes in the dataset

    Returns:
        pd.DataFrame: DataFrame of all notes for the given hadmid
    """
    # MIMIC-IV: discharge.csv uses 'note_type' column, 'DS' = Discharge Summary
    patient_notes = notes[notes['hadm_id'] == hadmid]
    patient_notes = patient_notes[patient_notes['note_type'] != 'DS']
    notes_tl = temporal_order_note(patient_notes)

    # remove notes at same timestamp
    notes_tl = notes_tl.drop_duplicates(subset=['TIME'], keep='last')

    # get ground truth discharge summary
    patient_notes = notes[notes['hadm_id'] == int(hadmid)]
    discharge = patient_notes[patient_notes['note_type'] == 'DS']
    if discharge.empty:
        print(f'{hadmid} has no discharge summary')
        return notes_tl, None
    else:
        discharge_txt = discharge['text'].values[0]
        return notes_tl, discharge_txt


def get_rel_times(df):
    """
    Convert absolute timestamps to relative ones (w.r.t first entry)

    Args:
        df (pd.DataFrame): DataFrame with TIME column

    Returns:
        pd.DataFrame: DataFrame with added REL_TIME column
    """
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
                    time_lst.append(f"{minutes} minute{'s' if minutes > 1 else ''} later: ")
                rel_times.append(" ".join(time_lst))
            else:
                rel_times.append(None)
    df['REL_TIME'] = rel_times
    return df


def get_last_day(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Get data from last 24h or 48h before discharge/last entry

    Args:
        df (pd.DataFrame): DataFrame with TIME column
        window (int): 24 or 48 hour window

    Returns:
        pd.DataFrame: filtered DataFrame
    """
    format_str = r'%Y-%m-%d %H:%M:%S'

    for i in df.index[::-1]:
        if i == len(df)-1:
            last_entry = datetime.strptime(df.iloc[i]['TIME'], format_str)
        else:
            current_entry = datetime.strptime(df.iloc[i]['TIME'], format_str)
            diff_days = (last_entry - current_entry).days
            if window == 24:
                if diff_days != 0:
                    return df[i+1:].reset_index(drop=True)
            elif window == 48:
                if diff_days >= 2:
                    return df[i+1:].reset_index(drop=True)
    return df.reset_index(drop=True)


def format_result(df: pd.DataFrame) -> pd.DataFrame:
    # move relative time column to front
    col = df.pop('REL_TIME')
    df.insert(0, 'REL_TIME', col)
    return df


def main():
    pd.options.mode.chained_assignment = None  # Turns off the warning

    args = parse_args()
    modality = args.modality
    window = args.window

    # MIMIC-IV target path (output from get_target_population_fix.py)
    target_path = 'data/MIMIC-IV/target'

    icu_df = pd.read_csv(f'{target_path}/target_ICUSTAYS.csv')
    notes = pd.read_csv(f'{target_path}/target_discharge.csv')
    
    # MIMIC-IV: INPUTEVENTS_CV and INPUTEVENTS_MV are merged into a single inputevents.csv
    input_df = pd.read_csv(f'{target_path}/target_inputevents.csv')
    
    lab_df = pd.read_csv(f'{target_path}/target_labevents.csv')
    chart_df = pd.read_csv(f'{target_path}/target_chartevents.csv')
    meds_df = pd.read_csv(f'{target_path}/target_prescriptions.csv')
    
    # MIMIC-IV lookup tables
    lab_items = pd.read_csv('long_data_related/longitudinal_clinical_summarization/mimic-iv-3.1/hosp/d_labitems.csv.gz')
    chart_items = pd.read_csv('long_data_related/longitudinal_clinical_summarization/mimic-iv-3.1/icu/d_items.csv.gz')

    admission_ids = icu_df['hadm_id'].to_list()

    output_dir = 'data/DS/input'
    gt_dir = 'data/DS/gold'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for admission_id in tqdm(admission_ids):
        admission_id = int(admission_id)
        
        # MIMIC-IV: no DBSOURCE column, single inputevents.csv
        tab_data = (lab_df, chart_df, input_df, meds_df)
        dictionaries = (lab_items, chart_items)

        structured_data = get_structured(admission_id, tab_data, dictionaries)
        note_data, gold_txt = get_notes(admission_id, notes)

        if gold_txt is None:
            print(f'Skipping {admission_id}: no discharge summary found')
            continue

        if modality == 'both':
            combined_tl = pd.concat((structured_data, note_data))
        elif modality == 'notes':
            combined_tl = note_data
        else:           # modality == tab
            combined_tl = structured_data
        
        combined_tl = combined_tl.sort_values(by='TIME').reset_index(drop=True)

        combined_tl_rel = get_rel_times(combined_tl)
        combined_tl_rel = combined_tl_rel.dropna(subset=['TIME'])

        last_day = get_last_day(combined_tl_rel, window)

        formatted_last_day = format_result(last_day)

        # write gold discharge summary to text file
        with open(f'{gt_dir}/gtsummary_{admission_id}.txt', 'w') as text_file:
            text_file.write(gold_txt)

        formatted_last_day.to_csv(f'{output_dir}/{window}_{modality}_{admission_id}.csv', index=False)


if __name__ == '__main__':
    main()
