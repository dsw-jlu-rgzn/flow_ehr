"""Build a small MIMIC-III disease-state prediction MVP dataset.

The script is intentionally conservative: it uses only structured filtered
MIMIC-III tables, builds 6-hour timeline bins, and creates weak rule labels for
future organ trajectories plus risk events. Notes are not used in this MVP.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ORGANS = [
    "cardiovascular",
    "respiratory",
    "renal",
    "infectious",
    "hematologic",
    "neurological",
]

STATE_DIRECTION = {"improving": 1, "stable": 0, "worsening": -1}

LAB_ITEMIDS = {
    "creatinine": {50912},
    "bun": {51006},
    "lactate": {50813},
    "wbc": {51300, 51301},
    "platelets": {51265},
    "hemoglobin": {50811, 51222},
    "inr": {51237},
    "ptt": {51275},
}

CHART_ITEMIDS = {
    "heart_rate": {211, 220045},
    "map": {52, 456, 220052, 220181},
    "spo2": {646, 220277},
    "resp_rate": {618, 220210},
    "fio2": {189, 190, 191, 3420, 3422, 223835},
    "temperature_c": {676, 677, 223762},
    "temperature_f": {678, 679, 223761},
}

VASOPRESSOR_TERMS = [
    "norepinephrine",
    "levophed",
    "phenylephrine",
    "neosynephrine",
    "vasopressin",
    "epinephrine",
    "dopamine",
]

ANTIBIOTIC_TERMS = [
    "vancomycin",
    "piperacillin",
    "tazobactam",
    "zosyn",
    "cefepime",
    "ceftriaxone",
    "cefazolin",
    "meropenem",
    "imipenem",
    "levofloxacin",
    "ciprofloxacin",
    "metronidazole",
    "azithromycin",
    "gentamicin",
    "linezolid",
]

VENT_TERMS = ["ventilator", "ventilation", "intubat", "ett", "endotracheal"]
RRT_TERMS = ["dialysis", "hemodialysis", "cvvh", "crrt", "rrt"]


def clean_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def median(values: list[float]) -> float | None:
    values = sorted(v for v in values if v is not None and not math.isnan(v))
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2)


def min_value(values: list[float]) -> float | None:
    values = [v for v in values if v is not None and not math.isnan(v)]
    return float(min(values)) if values else None


def max_value(values: list[float]) -> float | None:
    values = [v for v in values if v is not None and not math.isnan(v)]
    return float(max(values)) if values else None


def last_value(df: pd.DataFrame, concept: str, end_hour: float) -> float | None:
    if df.empty:
        return None
    sub = df[(df["concept"] == concept) & (df["hour"] <= end_hour)].sort_values("hour")
    if sub.empty:
        return None
    return clean_float(sub.iloc[-1]["value"])


def window_values(df: pd.DataFrame, concept: str, start_hour: float, end_hour: float) -> list[float]:
    if df.empty:
        return []
    sub = df[(df["concept"] == concept) & (df["hour"] >= start_hour) & (df["hour"] < end_hour)]
    return [v for v in sub["value"].map(clean_float).tolist() if v is not None]


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def make_item_concepts(items: pd.DataFrame) -> dict[int, str]:
    item_concepts: dict[int, str] = {}
    labels = {
        int(row.ITEMID): normalize_text(getattr(row, "LABEL", ""))
        for row in items.itertuples()
        if pd.notna(row.ITEMID)
    }

    for concept, itemids in CHART_ITEMIDS.items():
        for itemid in itemids:
            item_concepts[itemid] = concept

    for itemid, label in labels.items():
        if itemid in item_concepts:
            continue
        if label == "heart rate":
            item_concepts[itemid] = "heart_rate"
        elif "blood pressure mean" in label or label in {"map", "mean bp"}:
            item_concepts[itemid] = "map"
        elif "o2 saturation" in label or "spo2" in label:
            item_concepts[itemid] = "spo2"
        elif label == "respiratory rate":
            item_concepts[itemid] = "resp_rate"
        elif "fio2" in label or "inspired o2" in label:
            item_concepts[itemid] = "fio2"
        elif "temperature c" in label:
            item_concepts[itemid] = "temperature_c"
        elif "temperature f" in label:
            item_concepts[itemid] = "temperature_f"
    return item_concepts


def load_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def add_int_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def hours_since(series: pd.Series, intime: pd.Timestamp) -> pd.Series:
    return (pd.to_datetime(series, errors="coerce") - intime).dt.total_seconds() / 3600.0


def rows_for_stay(df: pd.DataFrame, hadm_id: int, icustay_id: int | None = None) -> pd.DataFrame:
    if df.empty or "HADM_ID" not in df.columns:
        return df.iloc[0:0].copy()
    sub = df[pd.to_numeric(df["HADM_ID"], errors="coerce").eq(hadm_id)]
    if icustay_id is not None and "ICUSTAY_ID" in sub.columns:
        icu_mask = pd.to_numeric(sub["ICUSTAY_ID"], errors="coerce").eq(icustay_id)
        if icu_mask.any():
            sub = sub[icu_mask]
    return sub.copy()


def concept_events_from_chart(chart: pd.DataFrame, item_concepts: dict[int, str], intime: pd.Timestamp) -> pd.DataFrame:
    if chart.empty:
        return pd.DataFrame(columns=["hour", "concept", "value", "label"])
    chart = chart.copy()
    chart["ITEMID"] = pd.to_numeric(chart["ITEMID"], errors="coerce")
    chart = chart[chart["ITEMID"].isin(item_concepts)]
    if chart.empty:
        return pd.DataFrame(columns=["hour", "concept", "value", "label"])
    chart["hour"] = hours_since(chart["CHARTTIME"], intime)
    chart["concept"] = chart["ITEMID"].astype(int).map(item_concepts)
    chart["value"] = pd.to_numeric(chart["VALUENUM"], errors="coerce")
    return chart.dropna(subset=["hour", "concept", "value"])[["hour", "concept", "value"]]


def concept_events_from_labs(labs: pd.DataFrame, intime: pd.Timestamp) -> pd.DataFrame:
    if labs.empty:
        return pd.DataFrame(columns=["hour", "concept", "value"])
    lab_concepts = {}
    for concept, itemids in LAB_ITEMIDS.items():
        for itemid in itemids:
            lab_concepts[itemid] = concept
    labs = labs.copy()
    labs["ITEMID"] = pd.to_numeric(labs["ITEMID"], errors="coerce")
    labs = labs[labs["ITEMID"].isin(lab_concepts)]
    if labs.empty:
        return pd.DataFrame(columns=["hour", "concept", "value"])
    labs["hour"] = hours_since(labs["CHARTTIME"], intime)
    labs["concept"] = labs["ITEMID"].astype(int).map(lab_concepts)
    labs["value"] = pd.to_numeric(labs["VALUENUM"], errors="coerce")
    return labs.dropna(subset=["hour", "concept", "value"])[["hour", "concept", "value"]]


def medication_events(
    input_cv: pd.DataFrame,
    input_mv: pd.DataFrame,
    prescriptions: pd.DataFrame,
    d_items: pd.DataFrame,
    intime: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = {
        int(row.ITEMID): str(getattr(row, "LABEL", ""))
        for row in d_items.itertuples()
        if pd.notna(row.ITEMID)
    }

    for source_name, df, time_col in [
        ("input_cv", input_cv, "CHARTTIME"),
        ("input_mv", input_mv, "STARTTIME"),
    ]:
        if df.empty or time_col not in df.columns:
            continue
        tmp = df.copy()
        tmp["hour"] = hours_since(tmp[time_col], intime)
        tmp["ITEMID"] = pd.to_numeric(tmp["ITEMID"], errors="coerce")
        for row in tmp.dropna(subset=["hour", "ITEMID"]).itertuples():
            itemid = int(row.ITEMID)
            label = labels.get(itemid, "")
            text = normalize_text(label)
            rate = clean_float(getattr(row, "RATE", None))
            amount = clean_float(getattr(row, "AMOUNT", None))
            concept = classify_med_text(text)
            if concept:
                rows.append(
                    {
                        "hour": float(row.hour),
                        "concept": concept,
                        "name": label or str(itemid),
                        "rate": rate,
                        "amount": amount,
                        "source": source_name,
                    }
                )

    if not prescriptions.empty and "STARTDATE" in prescriptions.columns:
        tmp = prescriptions.copy()
        tmp["hour"] = hours_since(tmp["STARTDATE"], intime)
        for row in tmp.dropna(subset=["hour"]).itertuples():
            drug = " ".join(
                str(getattr(row, field, "") or "")
                for field in ["DRUG", "DRUG_NAME_POE", "DRUG_NAME_GENERIC"]
            )
            concept = classify_med_text(normalize_text(drug))
            if concept:
                rows.append(
                    {
                        "hour": float(row.hour),
                        "concept": concept,
                        "name": drug.strip(),
                        "rate": None,
                        "amount": clean_float(getattr(row, "DOSE_VAL_RX", None)),
                        "source": "prescriptions",
                    }
                )

    return pd.DataFrame(rows, columns=["hour", "concept", "name", "rate", "amount", "source"])


def classify_med_text(text: str) -> str | None:
    if any(term in text for term in VASOPRESSOR_TERMS):
        return "vasopressor"
    if any(term in text for term in ANTIBIOTIC_TERMS):
        return "antibiotic"
    if any(term in text for term in RRT_TERMS):
        return "rrt"
    return None


def procedure_events(chart: pd.DataFrame, d_items: pd.DataFrame, intime: pd.Timestamp) -> pd.DataFrame:
    if chart.empty:
        return pd.DataFrame(columns=["hour", "concept", "name"])
    labels = {
        int(row.ITEMID): str(getattr(row, "LABEL", ""))
        for row in d_items.itertuples()
        if pd.notna(row.ITEMID)
    }
    tmp = chart.copy()
    tmp["ITEMID"] = pd.to_numeric(tmp["ITEMID"], errors="coerce")
    tmp["hour"] = hours_since(tmp["CHARTTIME"], intime)
    rows = []
    for row in tmp.dropna(subset=["hour", "ITEMID"]).itertuples():
        itemid = int(row.ITEMID)
        label = labels.get(itemid, "")
        text = normalize_text(label)
        value = normalize_text(getattr(row, "VALUE", ""))
        combined = f"{text} {value}"
        if any(term in combined for term in VENT_TERMS):
            rows.append({"hour": float(row.hour), "concept": "mechanical_ventilation", "name": label})
        if any(term in combined for term in RRT_TERMS):
            rows.append({"hour": float(row.hour), "concept": "rrt", "name": label})
    return pd.DataFrame(rows, columns=["hour", "concept", "name"])


def has_event(df: pd.DataFrame, concept: str, start_hour: float, end_hour: float) -> bool:
    if df.empty:
        return False
    sub = df[(df["concept"] == concept) & (df["hour"] >= start_hour) & (df["hour"] < end_hour)]
    return not sub.empty


def max_rate(df: pd.DataFrame, concept: str, start_hour: float, end_hour: float) -> float | None:
    if df.empty or "rate" not in df.columns:
        return None
    sub = df[(df["concept"] == concept) & (df["hour"] >= start_hour) & (df["hour"] < end_hour)]
    vals = [v for v in sub["rate"].map(clean_float).tolist() if v is not None]
    return max(vals) if vals else None


def trend_state(old: float | None, new: float | None, worse_high: bool, abs_delta: float, rel_delta: float) -> str:
    if old is None or new is None:
        return "unknown"
    delta = new - old
    if old != 0:
        rel = abs(delta) / abs(old)
    else:
        rel = 0
    threshold_met = abs(delta) >= abs_delta or rel >= rel_delta
    if not threshold_met:
        return "stable"
    if worse_high:
        return "worsening" if delta > 0 else "improving"
    return "worsening" if delta < 0 else "improving"


def relative_change(old: float | None, new: float | None) -> float | None:
    if old is None or new is None or old == 0:
        return None
    return (new - old) / abs(old)


def clinically_low_wbc(value: float | None) -> bool:
    return value is not None and value < 4.0


def clinically_high_wbc(value: float | None) -> bool:
    return value is not None and value > 12.0


def build_timeline(
    concept_df: pd.DataFrame,
    meds: pd.DataFrame,
    procedures: pd.DataFrame,
    anchor_hour: int,
    bin_hours: int,
) -> list[dict[str, Any]]:
    timeline = []
    for start in range(0, anchor_hour, bin_hours):
        end = min(start + bin_hours, anchor_hour)
        bin_obj: dict[str, Any] = {
            "start_hour": start,
            "end_hour": end,
            "vitals": {},
            "labs": {},
            "medications": [],
            "procedures": [],
            "outputs": {},
        }
        for concept in ["heart_rate", "map", "spo2", "resp_rate", "fio2", "temperature_c", "temperature_f"]:
            vals = window_values(concept_df, concept, start, end)
            if vals:
                bin_obj["vitals"][f"{concept}_median"] = round(median(vals), 3)
                bin_obj["vitals"][f"{concept}_min"] = round(min_value(vals), 3)
                bin_obj["vitals"][f"{concept}_max"] = round(max_value(vals), 3)
        for concept in ["creatinine", "bun", "lactate", "wbc", "platelets", "hemoglobin", "inr", "ptt"]:
            vals = window_values(concept_df, concept, start, end)
            if vals:
                bin_obj["labs"][f"{concept}_last"] = round(vals[-1], 3)
                bin_obj["labs"][f"{concept}_median"] = round(median(vals), 3)
        if not meds.empty:
            med_sub = meds[(meds["hour"] >= start) & (meds["hour"] < end)]
            for row in med_sub.head(12).itertuples():
                bin_obj["medications"].append(
                    {
                        "concept": row.concept,
                        "name": str(row.name)[:100],
                        "rate": clean_float(row.rate),
                        "amount": clean_float(row.amount),
                    }
                )
        if not procedures.empty:
            proc_sub = procedures[(procedures["hour"] >= start) & (procedures["hour"] < end)]
            for row in proc_sub.head(8).itertuples():
                bin_obj["procedures"].append({"concept": row.concept, "name": str(row.name)[:100]})
        timeline.append(bin_obj)
    return timeline


def label_sample(
    concept_df: pd.DataFrame,
    meds: pd.DataFrame,
    procedures: pd.DataFrame,
    admission: dict[str, Any],
    anchor_hour: int,
    horizon_hour: int,
    outtime_hour: float,
) -> dict[str, Any]:
    target_start = anchor_hour
    target_end = anchor_hour + horizon_hour
    base_start = max(0, anchor_hour - 6)
    evidence: dict[str, list[str]] = defaultdict(list)
    debug: dict[str, Any] = {}

    base_map = median(window_values(concept_df, "map", base_start, anchor_hour))
    future_min_map = min_value(window_values(concept_df, "map", target_start, target_end))
    base_lactate = last_value(concept_df, "lactate", anchor_hour)
    future_lactate = max_value(window_values(concept_df, "lactate", target_start, target_end))
    vaso_before = has_event(meds, "vasopressor", 0, anchor_hour)
    vaso_future = has_event(meds, "vasopressor", target_start, target_end)
    vaso_base_rate = max_rate(meds, "vasopressor", max(0, anchor_hour - 24), anchor_hour)
    vaso_future_rate = max_rate(meds, "vasopressor", target_start, target_end)
    cv_state = "stable"
    cv_rules = []
    if vaso_future and not vaso_before:
        cv_state = "worsening"
        cv_rules.append("vasopressor_start")
        evidence["cardiovascular"].append("vasopressor was started in the target window")
    elif vaso_future_rate is not None and vaso_base_rate is not None and vaso_future_rate > vaso_base_rate * 1.3:
        cv_state = "worsening"
        cv_rules.append("vasopressor_escalation")
        evidence["cardiovascular"].append(
            f"vasopressor rate increased from {round(vaso_base_rate, 3)} to {round(vaso_future_rate, 3)}"
        )
    elif vaso_before and not vaso_future:
        cv_state = "improving"
        cv_rules.append("vasopressor_discontinued")
        evidence["cardiovascular"].append("vasopressor was discontinued in the target window")
    if future_min_map is not None and future_min_map < 65 and (base_map is None or future_min_map <= base_map - 5):
        cv_state = "worsening"
        cv_rules.append("map_lt_65")
        evidence["cardiovascular"].append(f"MAP dropped to {round(future_min_map, 3)} mmHg")
    lactate_state = trend_state(base_lactate, future_lactate, True, 2.0, 0.3)
    if lactate_state == "worsening":
        cv_state = "worsening"
        cv_rules.append("lactate_increase")
        evidence["cardiovascular"].append(
            f"lactate increased from {round(base_lactate, 3)} to {round(future_lactate, 3)} mmol/L"
        )
    elif cv_state == "stable" and lactate_state == "improving":
        cv_state = "improving"
        cv_rules.append("lactate_decrease")
        evidence["cardiovascular"].append(
            f"lactate decreased from {round(base_lactate, 3)} to {round(future_lactate, 3)} mmol/L"
        )
    if cv_state == "worsening" and "vasopressor_discontinued" in cv_rules:
        cv_rules = [rule for rule in cv_rules if rule != "vasopressor_discontinued"]
        evidence["cardiovascular"] = [
            item for item in evidence["cardiovascular"] if "vasopressor was discontinued" not in item
        ]
    debug["cardiovascular"] = {
        "baseline_map": base_map,
        "future_min_map": future_min_map,
        "baseline_lactate": base_lactate,
        "future_worst_lactate": future_lactate,
        "vasopressor_before": vaso_before,
        "vasopressor_future": vaso_future,
        "triggered_rules": cv_rules,
    }

    base_fio2 = last_value(concept_df, "fio2", anchor_hour)
    future_fio2 = max_value(window_values(concept_df, "fio2", target_start, target_end))
    base_spo2 = median(window_values(concept_df, "spo2", base_start, anchor_hour))
    future_spo2 = min_value(window_values(concept_df, "spo2", target_start, target_end))
    vent_before = has_event(procedures, "mechanical_ventilation", 0, anchor_hour)
    vent_future = has_event(procedures, "mechanical_ventilation", target_start, target_end)
    resp_state = "stable"
    resp_rules = []
    if vent_future and not vent_before:
        resp_state = "worsening"
        resp_rules.append("mechanical_ventilation_start")
        evidence["respiratory"].append("mechanical ventilation started in the target window")
    fio2_state = trend_state(base_fio2, future_fio2, True, 10.0, 0.25)
    if fio2_state == "worsening":
        resp_state = "worsening"
        resp_rules.append("fio2_increase")
        evidence["respiratory"].append(f"FiO2 increased from {base_fio2} to {future_fio2}")
    spo2_state = trend_state(base_spo2, future_spo2, False, 5.0, 0.06)
    if spo2_state == "worsening":
        resp_state = "worsening"
        resp_rules.append("spo2_decrease")
        evidence["respiratory"].append(f"SpO2 decreased from {base_spo2} to {future_spo2}")
    if resp_state == "stable" and fio2_state == "improving":
        resp_state = "improving"
        resp_rules.append("fio2_decrease")
        evidence["respiratory"].append(f"FiO2 decreased from {base_fio2} to {future_fio2}")
    debug["respiratory"] = {
        "baseline_fio2": base_fio2,
        "future_max_fio2": future_fio2,
        "baseline_spo2": base_spo2,
        "future_min_spo2": future_spo2,
        "vent_before": vent_before,
        "vent_future": vent_future,
        "triggered_rules": resp_rules,
    }

    base_cr = last_value(concept_df, "creatinine", anchor_hour)
    future_cr = max_value(window_values(concept_df, "creatinine", target_start, target_end))
    rrt_before = has_event(meds, "rrt", 0, anchor_hour) or has_event(procedures, "rrt", 0, anchor_hour)
    rrt_future = has_event(meds, "rrt", target_start, target_end) or has_event(procedures, "rrt", target_start, target_end)
    renal_state = trend_state(base_cr, future_cr, True, 0.3, 0.5)
    renal_rules = []
    if rrt_future and not rrt_before:
        renal_state = "worsening"
        renal_rules.append("rrt_start")
        evidence["renal"].append("dialysis/RRT started in the target window")
    if renal_state == "worsening":
        renal_rules.append("creatinine_increase")
        if base_cr is not None and future_cr is not None:
            evidence["renal"].append(f"creatinine increased from {round(base_cr, 3)} to {round(future_cr, 3)} mg/dL")
    elif renal_state == "improving" and base_cr is not None and future_cr is not None:
        renal_rules.append("creatinine_decrease")
        evidence["renal"].append(f"creatinine decreased from {round(base_cr, 3)} to {round(future_cr, 3)} mg/dL")
    debug["renal"] = {
        "baseline_creatinine": base_cr,
        "future_worst_creatinine": future_cr,
        "rrt_before": rrt_before,
        "rrt_future": rrt_future,
        "triggered_rules": renal_rules,
    }

    base_wbc = last_value(concept_df, "wbc", anchor_hour)
    future_wbc = max_value(window_values(concept_df, "wbc", target_start, target_end))
    base_temp_c = last_value(concept_df, "temperature_c", anchor_hour)
    future_temp_c_vals = window_values(concept_df, "temperature_c", target_start, target_end)
    future_temp_f_vals = window_values(concept_df, "temperature_f", target_start, target_end)
    future_temp_c = max_value(future_temp_c_vals)
    future_temp_f = max_value(future_temp_f_vals)
    abx_before_count = len(meds[(meds["concept"] == "antibiotic") & (meds["hour"] >= max(0, anchor_hour - 24)) & (meds["hour"] < anchor_hour)]) if not meds.empty else 0
    abx_future_count = len(meds[(meds["concept"] == "antibiotic") & (meds["hour"] >= target_start) & (meds["hour"] < target_end)]) if not meds.empty else 0
    inf_state = "stable"
    inf_rules = []
    wbc_rel = relative_change(base_wbc, future_wbc)
    if (
        base_wbc is not None
        and future_wbc is not None
        and wbc_rel is not None
        and wbc_rel >= 0.3
        and (clinically_high_wbc(future_wbc) or clinically_high_wbc(base_wbc))
    ):
        inf_state = "worsening"
        inf_rules.append("wbc_increase")
        evidence["infectious"].append(f"WBC increased from {round(base_wbc, 3)} to {round(future_wbc, 3)} K/uL")
    elif (
        base_wbc is not None
        and future_wbc is not None
        and wbc_rel is not None
        and wbc_rel <= -0.2
        and clinically_high_wbc(base_wbc)
        and future_wbc >= 4.0
    ):
        inf_state = "improving"
        inf_rules.append("wbc_decrease")
        evidence["infectious"].append(f"WBC decreased from {round(base_wbc, 3)} to {round(future_wbc, 3)} K/uL")
    elif (
        base_wbc is not None
        and future_wbc is not None
        and clinically_low_wbc(base_wbc)
        and future_wbc >= 4.0
    ):
        inf_state = "improving"
        inf_rules.append("leukopenia_recovery")
        evidence["infectious"].append(f"WBC recovered from {round(base_wbc, 3)} to {round(future_wbc, 3)} K/uL")
    elif (
        base_wbc is not None
        and future_wbc is not None
        and base_wbc >= 4.0
        and clinically_low_wbc(future_wbc)
    ):
        inf_state = "worsening"
        inf_rules.append("new_leukopenia")
        evidence["infectious"].append(f"WBC decreased into leukopenic range from {round(base_wbc, 3)} to {round(future_wbc, 3)} K/uL")
    if (future_temp_c is not None and future_temp_c >= 38.3) or (future_temp_f is not None and future_temp_f >= 101):
        inf_state = "worsening"
        inf_rules.append("fever")
        evidence["infectious"].append("fever occurred in the target window")
    if abx_future_count > abx_before_count + 1:
        inf_state = "worsening"
        inf_rules.append("antibiotic_escalation")
        evidence["infectious"].append("antibiotic exposure increased in the target window")
    debug["infectious"] = {
        "baseline_wbc": base_wbc,
        "future_worst_wbc": future_wbc,
        "future_temp_c": future_temp_c,
        "future_temp_f": future_temp_f,
        "antibiotic_before_count": abx_before_count,
        "antibiotic_future_count": abx_future_count,
        "triggered_rules": inf_rules,
    }

    base_platelets = last_value(concept_df, "platelets", anchor_hour)
    future_platelets = min_value(window_values(concept_df, "platelets", target_start, target_end))
    base_hgb = last_value(concept_df, "hemoglobin", anchor_hour)
    future_hgb = min_value(window_values(concept_df, "hemoglobin", target_start, target_end))
    heme_state = "stable"
    heme_rules = []
    platelet_rel = relative_change(base_platelets, future_platelets)
    hgb_delta = None if base_hgb is None or future_hgb is None else future_hgb - base_hgb
    platelet_worsening = (
        base_platelets is not None
        and future_platelets is not None
        and (
            (platelet_rel is not None and platelet_rel <= -0.3)
            or (base_platelets >= 100 and future_platelets < 50)
        )
    )
    hgb_worsening = hgb_delta is not None and hgb_delta <= -2.0
    platelet_improving = (
        base_platelets is not None
        and future_platelets is not None
        and (
            (base_platelets < 100 and future_platelets >= 100)
            or (platelet_rel is not None and platelet_rel >= 0.3 and future_platelets > base_platelets)
        )
    )
    hgb_improving = hgb_delta is not None and hgb_delta >= 2.0
    if platelet_worsening:
        heme_state = "worsening"
        heme_rules.append("platelet_drop")
        evidence["hematologic"].append(f"platelets decreased from {round(base_platelets, 3)} to {round(future_platelets, 3)} K/uL")
    if hgb_worsening:
        heme_state = "worsening"
        heme_rules.append("hemoglobin_drop")
        evidence["hematologic"].append(f"hemoglobin decreased from {round(base_hgb, 3)} to {round(future_hgb, 3)} g/dL")
    if heme_state == "stable" and (platelet_improving or hgb_improving):
        heme_state = "improving"
        heme_rules.append("hematologic_recovery")
    debug["hematologic"] = {
        "baseline_platelets": base_platelets,
        "future_min_platelets": future_platelets,
        "baseline_hemoglobin": base_hgb,
        "future_min_hemoglobin": future_hgb,
        "triggered_rules": heme_rules,
    }

    organ_states = {
        "cardiovascular": cv_state,
        "respiratory": resp_state,
        "renal": renal_state if renal_state != "unknown" else "stable",
        "infectious": inf_state,
        "hematologic": heme_state,
        "neurological": "unknown",
    }

    death_time = pd.to_datetime(admission.get("DEATHTIME"), errors="coerce")
    intime = admission["INTIME"]
    death_hour = None
    if pd.notna(death_time):
        death_hour = (death_time - intime).total_seconds() / 3600.0
    mortality = int(death_hour is not None and target_start <= death_hour < target_end)
    icu_transfer_out = int(target_start <= outtime_hour < target_end and not mortality)
    risk_events = {
        "mortality": mortality,
        "icu_transfer_out": icu_transfer_out,
        "mechanical_ventilation_start": int(vent_future and not vent_before),
        "vasopressor_start_or_escalation": int(
            (vaso_future and not vaso_before)
            or (vaso_future_rate is not None and vaso_base_rate is not None and vaso_future_rate > vaso_base_rate * 1.3)
        ),
        "rrt_start": int(rrt_future and not rrt_before),
        "sofa_increase_ge_2": None,
    }

    worsening_count = sum(1 for state in organ_states.values() if state == "worsening")
    improving_count = sum(1 for state in organ_states.values() if state == "improving")
    if mortality:
        global_state = "critical"
    elif worsening_count >= 2:
        global_state = "worsening"
    elif improving_count >= 2 and worsening_count == 0:
        global_state = "improving"
    elif improving_count >= 1 and worsening_count >= 1:
        global_state = "mixed"
    elif icu_transfer_out and worsening_count == 0:
        global_state = "recovering"
    else:
        global_state = "stable"

    return {
        "global_state": global_state,
        "organ_states": organ_states,
        "risk_events": risk_events,
        "evidence": dict(evidence),
        "label_debug": debug,
    }


def build_input_text(sample: dict[str, Any]) -> str:
    demo = sample["input"]["demographics"]
    age_text = "89+" if demo.get("age_is_deidentified_over_89") else str(demo.get("age", "unknown"))
    lines = [
        f"患者 {sample['subject_id']}，{age_text} 岁，性别 {demo.get('gender', 'unknown')}。",
        f"观察窗口：ICU 入科后 {sample['observation_window'][0]}-{sample['observation_window'][1]} 小时；预测未来 {sample['prediction_horizon'][0]}-{sample['prediction_horizon'][1]} 小时。",
    ]
    for bin_obj in sample["input"]["timeline"]:
        parts = []
        for group in ["vitals", "labs"]:
            for key, value in bin_obj[group].items():
                parts.append(f"{key}={value}")
        if bin_obj["medications"]:
            meds = sorted({m["concept"] for m in bin_obj["medications"]})
            parts.append("medications=" + ",".join(meds))
        if bin_obj["procedures"]:
            procs = sorted({p["concept"] for p in bin_obj["procedures"]})
            parts.append("procedures=" + ",".join(procs))
        if parts:
            lines.append(f"{bin_obj['start_hour']}-{bin_obj['end_hour']} 小时：" + "；".join(parts))
    return "\n".join(lines)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def split_samples(samples: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    by_subject: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_subject[int(sample["subject_id"])].append(sample)
    subjects = sorted(by_subject)
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    train_subjects = set(subjects[: max(1, int(n * 0.7))])
    valid_subjects = set(subjects[max(1, int(n * 0.7)) : max(1, int(n * 0.8))])
    splits = {"train": [], "valid": [], "test": []}
    for subject, rows in by_subject.items():
        if subject in train_subjects:
            splits["train"].extend(rows)
        elif subject in valid_subjects:
            splits["valid"].extend(rows)
        else:
            splits["test"].extend(rows)
    return splits


def make_preview(samples: list[dict[str, Any]], path: Path) -> None:
    lines = ["# MIMIC-III 病程状态预测 MVP 预览", ""]
    for idx, sample in enumerate(samples[:20], start=1):
        label = sample["label"]
        lines.extend(
            [
                f"## {idx}. {sample['sample_id']}",
                "",
                f"- 输入窗口：{sample['observation_window'][0]}-{sample['observation_window'][1]} 小时",
                f"- 真值窗口：{sample['prediction_horizon'][0]}-{sample['prediction_horizon'][1]} 小时",
                f"- global_state：`{label['global_state']}`",
                f"- organ_states：`{json.dumps(label['organ_states'], ensure_ascii=False)}`",
                f"- risk_events：`{json.dumps(label['risk_events'], ensure_ascii=False)}`",
                "",
                "输入摘要：",
                "",
                "```text",
                sample["input_text"][:1800],
                "```",
                "",
                "规则证据：",
                "",
                "```json",
                json.dumps(label["evidence"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MIMIC-III disease-state prediction MVP samples.")
    parser.add_argument("--filtered-dir", default="data/target_population/filtered")
    parser.add_argument("--lookup-dir", default="data/MIMIC-III")
    parser.add_argument("--output-dir", default="outputs/ehr_state_prediction_mimic3_mvp_v2")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--anchor-hours", type=int, nargs="+", default=[24, 48, 72])
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--bin-hours", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    filtered = Path(args.filtered_dir)
    lookup = Path(args.lookup_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    icu = load_csv(filtered / "filtered_ICUSTAYS.csv")
    admissions = load_csv(filtered / "filtered_ADMISSIONS.csv")
    patients = load_csv(filtered / "filtered_PATIENTS.csv")
    d_items = load_csv(lookup / "D_ITEMS.csv")

    chart = load_csv(
        filtered / "filtered_CHARTEVENTS.csv",
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUE", "VALUENUM"],
    )
    labs = load_csv(
        filtered / "filtered_LABEVENTS.csv",
        usecols=["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM", "VALUEUOM"],
    )
    input_cv = load_csv(
        filtered / "filtered_INPUTEVENTS_CV.csv",
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID", "AMOUNT", "RATE"],
    )
    input_mv = load_csv(
        filtered / "filtered_INPUTEVENTS_MV.csv",
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTTIME", "ENDTIME", "ITEMID", "AMOUNT", "RATE"],
    )
    prescriptions = load_csv(
        filtered / "filtered_PRESCRIPTIONS.csv",
        usecols=[
            "SUBJECT_ID",
            "HADM_ID",
            "ICUSTAY_ID",
            "STARTDATE",
            "DRUG",
            "DRUG_NAME_POE",
            "DRUG_NAME_GENERIC",
            "DOSE_VAL_RX",
        ],
    )

    item_concepts = make_item_concepts(d_items)
    admissions_by_hadm = {
        int(row.HADM_ID): row._asdict()
        for row in admissions.itertuples()
        if pd.notna(row.HADM_ID)
    }
    patient_by_subject = {
        int(row.SUBJECT_ID): row._asdict()
        for row in patients.itertuples()
        if pd.notna(row.SUBJECT_ID)
    }

    samples: list[dict[str, Any]] = []
    icu = icu.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    for stay in icu.itertuples():
        if len(samples) >= args.max_samples:
            break
        subject_id = int(stay.SUBJECT_ID)
        hadm_id = int(stay.HADM_ID)
        icustay_id = int(stay.ICUSTAY_ID)
        intime = pd.to_datetime(stay.INTIME, errors="coerce")
        outtime = pd.to_datetime(stay.OUTTIME, errors="coerce")
        if pd.isna(intime) or pd.isna(outtime):
            continue
        outtime_hour = (outtime - intime).total_seconds() / 3600.0
        if outtime_hour < min(args.anchor_hours) + args.horizon_hours:
            continue

        stay_chart = rows_for_stay(chart, hadm_id, icustay_id)
        stay_labs = rows_for_stay(labs, hadm_id, None)
        if stay_chart.empty and stay_labs.empty:
            continue
        stay_input_cv = rows_for_stay(input_cv, hadm_id, icustay_id)
        stay_input_mv = rows_for_stay(input_mv, hadm_id, icustay_id)
        stay_prescriptions = rows_for_stay(prescriptions, hadm_id, icustay_id)

        concept_df = pd.concat(
            [
                concept_events_from_chart(stay_chart, item_concepts, intime),
                concept_events_from_labs(stay_labs, intime),
            ],
            ignore_index=True,
        )
        meds = medication_events(stay_input_cv, stay_input_mv, stay_prescriptions, d_items, intime)
        procs = procedure_events(stay_chart, d_items, intime)
        admission = admissions_by_hadm.get(hadm_id, {})
        admission["INTIME"] = intime
        patient = patient_by_subject.get(subject_id, {})

        for anchor_hour in args.anchor_hours:
            if len(samples) >= args.max_samples:
                break
            if outtime_hour < anchor_hour + args.horizon_hours:
                continue
            obs_has_data = (
                not concept_df[(concept_df["hour"] >= 0) & (concept_df["hour"] < anchor_hour)].empty
                or not meds[(meds["hour"] >= 0) & (meds["hour"] < anchor_hour)].empty
            )
            target_has_data = (
                not concept_df[(concept_df["hour"] >= anchor_hour) & (concept_df["hour"] < anchor_hour + args.horizon_hours)].empty
                or not meds[(meds["hour"] >= anchor_hour) & (meds["hour"] < anchor_hour + args.horizon_hours)].empty
            )
            if not obs_has_data or not target_has_data:
                continue
            sample_id = f"{subject_id}_{hadm_id}_{icustay_id}_t{anchor_hour}_h{args.horizon_hours}"
            timeline = build_timeline(concept_df, meds, procs, anchor_hour, args.bin_hours)
            raw_age = clean_float(getattr(stay, "age", None))
            age_is_deidentified = raw_age is not None and raw_age >= 120
            display_age = 89.0 if age_is_deidentified else raw_age
            sample = {
                "sample_id": sample_id,
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "icustay_id": icustay_id,
                "prediction_time_hours": anchor_hour,
                "observation_window": [0, anchor_hour],
                "prediction_horizon": [anchor_hour, anchor_hour + args.horizon_hours],
                "input": {
                    "demographics": {
                        "age": display_age,
                        "age_is_deidentified_over_89": age_is_deidentified,
                        "raw_age_value": raw_age if age_is_deidentified else None,
                        "gender": patient.get("GENDER"),
                        "first_careunit": getattr(stay, "FIRST_CAREUNIT", None),
                        "admission_diagnosis": admission.get("DIAGNOSIS"),
                    },
                    "timeline": timeline,
                },
                "label": label_sample(
                    concept_df=concept_df,
                    meds=meds,
                    procedures=procs,
                    admission=admission,
                    anchor_hour=anchor_hour,
                    horizon_hour=args.horizon_hours,
                    outtime_hour=outtime_hour,
                ),
            }
            sample["input_text"] = build_input_text(sample)
            samples.append(sample)

    if not samples:
        raise RuntimeError("No samples were generated. Check the filtered MIMIC-III data paths.")

    write_jsonl(out_dir / "ehr_state_prediction_full.jsonl", samples)
    text_rows = [
        {"sample_id": s["sample_id"], "input_text": s["input_text"], "target_json": s["label"]}
        for s in samples
    ]
    write_jsonl(out_dir / "ehr_state_prediction_text.jsonl", text_rows)
    splits = split_samples(samples, args.seed)
    for split, rows in splits.items():
        write_jsonl(out_dir / f"{split}.jsonl", rows)
    make_preview(samples, out_dir / "preview_zh.md")
    global_counts = pd.Series([s["label"]["global_state"] for s in samples]).value_counts()
    summary = {
        "sample_count": len(samples),
        "subjects": len({s["subject_id"] for s in samples}),
        "anchors": sorted({s["prediction_time_hours"] for s in samples}),
        "global_state_counts": {str(k): int(v) for k, v in global_counts.items()},
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "notes_used": False,
        "source": "MIMIC-III filtered structured tables",
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
