import pandas as pd
from tqdm import tqdm
import os


def main():
    """
    Get target population data based on existing target_discharge.csv.

    Pipeline:
    1. Read target_discharge.csv to get hadm_ids of interest
    2. Filter ICUSTAYS by those hadm_ids
    3. Filter all other data files (inputevents, labevents, chartevents,
       prescriptions, radiology) by those hadm_ids
    """
    # Paths
    base_path = 'long_data_related/longitudinal_clinical_summarization/mimic-iv-3.1'
    target_path = 'data/MIMIC-IV/target'

    # Step 1: Read target_discharge.csv to get hadm_ids
    discharge_path = f'{target_path}/target_discharge.csv'
    if not os.path.exists(discharge_path):
        print(f'Error: {discharge_path} not found. Run get_target_discharge.py first.')
        return

    discharge_df = pd.read_csv(discharge_path)
    hadmids = discharge_df['hadm_id'].unique()
    print(f'Found {len(hadmids)} unique hadm_ids from target_discharge.csv: {sorted(hadmids)}')

    # Step 2: Filter ICUSTAYS by hadm_ids
    print('Processing ICUSTAYS...')
    icu_path = f'{base_path}/icu/icustays.csv.gz'
    icu_df = pd.read_csv(icu_path, compression='gzip')
    target_icu = icu_df[icu_df['hadm_id'].isin(hadmids)]
    print(f'  Found {len(target_icu)} ICU stays matching the target hadm_ids')
    target_icu.to_csv(f'{target_path}/target_ICUSTAYS.csv', index=False)

    # Step 3: Filter all other data files by hadm_ids
    # MIMIC-IV file mapping: (output_name, input_path, hadm_id_column)
    files = [
        ('inputevents', 'icu/inputevents.csv.gz', 'hadm_id'),
        ('labevents', 'hosp/labevents.csv.gz', 'hadm_id'),
        ('chartevents', 'icu/chartevents.csv.gz', 'hadm_id'),
        ('prescriptions', 'hosp/prescriptions.csv.gz', 'hadm_id'),
        ('radiology', 'note/radiology.csv', 'hadm_id'),
    ]
    chunk_size = 10 ** 6

    for output_name, input_path, id_col in files:
        print(f'Processing {output_name}...')
        full_input_path = f'{base_path}/{input_path}'

        if not os.path.exists(full_input_path):
            print(f'  Warning: {full_input_path} not found, skipping...')
            continue

        header_written = False
        total_rows = 0

        # Process the file in chunks
        for chunk in tqdm(pd.read_csv(full_input_path, chunksize=chunk_size, low_memory=False)):
            # Check which column name is used for hadm_id
            if id_col in chunk.columns:
                target_file = chunk[chunk[id_col].isin(hadmids)]
            elif 'HADM_ID' in chunk.columns:
                target_file = chunk[chunk['HADM_ID'].isin(hadmids)]
            else:
                print(f'  Warning: {input_path} has no hadm_id column, skipping...')
                break

            if not target_file.empty:
                target_file.to_csv(
                    f'{target_path}/target_{output_name}.csv',
                    mode='a',
                    index=False,
                    header=not header_written
                )
                header_written = True
                total_rows += len(target_file)

        print(f'  Saved {total_rows} rows to target_{output_name}.csv')

    print('\nDone! All target files have been generated.')


if __name__ == '__main__':
    main()
