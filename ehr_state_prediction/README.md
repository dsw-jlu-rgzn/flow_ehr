# MIMIC-III Disease State Prediction MVP

This first-pass task builder creates longitudinal ICU disease-state prediction
samples from the current filtered MIMIC-III tables in this repository.

Default data sources:

```text
data/target_population/filtered/filtered_*.csv
data/MIMIC-III/D_ITEMS.csv
data/MIMIC-III/D_LABITEMS.csv
```

The MVP uses structured EHR only. `NOTEEVENTS` is intentionally excluded from
the model input and labels in this first version.

Run:

```powershell
python ehr_state_prediction/scripts/build_mimic3_task_mvp.py `
  --filtered-dir data/target_population/filtered `
  --lookup-dir data/MIMIC-III `
  --output-dir outputs/ehr_state_prediction_mimic3_mvp_v2 `
  --max-samples 20
```

Outputs:

```text
outputs/ehr_state_prediction_mimic3_mvp_v2/ehr_state_prediction_full.jsonl
outputs/ehr_state_prediction_mimic3_mvp_v2/ehr_state_prediction_text.jsonl
outputs/ehr_state_prediction_mimic3_mvp_v2/train.jsonl
outputs/ehr_state_prediction_mimic3_mvp_v2/valid.jsonl
outputs/ehr_state_prediction_mimic3_mvp_v2/test.jsonl
outputs/ehr_state_prediction_mimic3_mvp_v2/preview_zh.md
outputs/ehr_state_prediction_mimic3_mvp_v2/dataset_summary.json
```

Each sample is one `ICUSTAY_ID + anchor_hour`. The model input is the
observation window from ICU admission to the anchor hour. Labels are generated
from the future target window, normally the next 24 hours.

Default anchors are `24, 48, 72`; default horizon is `24`.

MIMIC-III deidentifies ages above 89 as values near 300. The builder reports
these as `age=89` with `age_is_deidentified_over_89=true`.
