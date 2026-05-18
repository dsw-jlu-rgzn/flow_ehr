"""
ProblemFlow-AP MVP experiment runner.

This script builds leakage-aware A&P completion samples and runs a small
multi-agent-style AP generation experiment with rule/mock agents. It is meant
to validate the experimental design before swapping the mock writer for a real
LLM.

Commands:
  build      Build day-level AP samples from data/AP/input and data/AP/gold.
  generate   Generate AP outputs for direct/trend/problemflow baselines.
  evaluate   Evaluate generated AP outputs against gold A&P.
  run-all    Run build, generate, and evaluate.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import math
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_OUTDIR = Path("experiments/problemflow_ap/outputs")
DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
PROBLEMFLOW_METHODS = [
    "problemflow",
    "problemflow_v2",
    "problemflow_v3",
    "problemflow_v4",
    "problemflow_v5",
    "problemflow_v6",
]
GATED_PROBLEMFLOW_METHODS = {
    "problemflow_v2",
    "problemflow_v3",
    "problemflow_v4",
    "problemflow_v5",
    "problemflow_v6",
}
REVISION_PROBLEMFLOW_METHODS = {"problemflow_v5", "problemflow_v6"}
ALL_METHODS = ["direct", "trend", *PROBLEMFLOW_METHODS]


PROBLEM_TAXONOMY = {
    "respiratory_failure_copd": {
        "label": "Respiratory failure / COPD",
        "keywords": [
            "respiratory",
            "copd",
            "hypercarb",
            "hypox",
            "intubat",
            "extubat",
            "vent",
            "peep",
            "fio2",
            "sbt",
            "rsbi",
            "pco2",
            "po2",
            "bronchodilator",
            "albuterol",
            "ipratropium",
            "steroid",
            "prednisone",
            "methylpred",
        ],
    },
    "infection": {
        "label": "Infection / antimicrobial coverage",
        "keywords": [
            "infection",
            "pneumonia",
            "sepsis",
            "fever",
            "wbc",
            "culture",
            "sputum",
            "vancomycin",
            "zosyn",
            "levofloxacin",
            "cef",
            "antibiotic",
        ],
    },
    "pneumonia": {
        "label": "Pneumonia",
        "keywords": ["pneumonia", "infiltrate", "consolidation", "cxr", "sputum culture", "purulent sputum"],
    },
    "renal_aki": {
        "label": "AKI / renal dysfunction",
        "keywords": ["aki", "renal", "creatinine", "bun", "urea nitrogen", "dialysis", "urine"],
    },
    "glucose_diabetes": {
        "label": "Diabetes / hyperglycemia",
        "keywords": ["glucose", "diabetes", "insulin", "hypergly", "hypogly", "steroid-induced"],
    },
    "volume": {
        "label": "Volume / diuresis",
        "keywords": ["fluid", "lasix", "furosemide", "diures", "edema", "i/o", "volume"],
    },
    "fen_nutrition": {
        "label": "FEN / nutrition",
        "keywords": [
            "sodium",
            "potassium",
            "magnesium",
            "phosphate",
            "calcium",
            "nutrition",
            "tube feed",
            "diet",
            "free water",
        ],
    },
    "heme": {
        "label": "Heme / anemia",
        "keywords": ["hemoglobin", "hematocrit", "platelet", "anemia", "thrombocytopenia", "transfusion"],
    },
    "skin_wound": {
        "label": "Skin / wound / cellulitis",
        "keywords": ["cellulitis", "wound", "erythema", "drainage", "skin", "ulcer"],
    },
    "neuro_pain_sedation": {
        "label": "Neuro / pain / sedation",
        "keywords": ["sedation", "versed", "midazolam", "fentanyl", "pain", "delirium", "mental status"],
    },
    "prophylaxis_access": {
        "label": "ICU prophylaxis / access",
        "keywords": ["heparin", "prophylaxis", "ppi", "pantoprazole", "line", "foley", "access"],
    },
}


A_AND_P_PATTERNS = [
    r"\bAssessment\s+and\s+Plan\b",
    r"\bAssessment\s*/\s*Plan\b",
    r"\bA\s*/\s*P\b",
    r"\bA&P\b",
    r"\bAssessment:\b",
    r"\bPlan:\b",
]


@dataclass
class ProblemState:
    problem_id: str
    label: str
    status: str = "active"
    trajectory_summary: str = ""
    current_assessment: str = ""
    current_plan: list[str] = field(default_factory=list)
    supporting_evidence_ids: list[str] = field(default_factory=list)
    days_seen: list[int] = field(default_factory=list)


def normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def call_deepseek(
    prompt: str,
    model: str = "deepseek-chat",
    api_url: str = DEFAULT_API_URL,
    temperature: float = 0.0,
    max_tokens: int = 1200,
    retries: int = 8,
    sleep_seconds: float = 3.0,
    system: str = "You are an experienced ICU clinician. Be concise, evidence-grounded, and clinically precise.",
) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"].strip()
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            ssl.SSLError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)
    raise RuntimeError(f"DeepSeek API request failed after {retries} attempts: {last_error}")


def split_ap_section(note_text: str) -> tuple[str, str, str]:
    text = str(note_text or "")
    matches = []
    for pattern in A_AND_P_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            matches.append(match)
    if not matches:
        return "", text.strip(), "missing_ap_heading"

    match = min(matches, key=lambda m: m.start())
    context = text[: match.start()].strip()
    gold_ap = text[match.start() :].strip()
    if len(gold_ap.split()) < 10:
        return context, gold_ap, "short_ap"
    return context, gold_ap, "ok"


def build_samples(args: argparse.Namespace) -> Path:
    input_dir = Path(args.inputdir)
    gold_dir = Path(args.golddir)
    out_path = Path(args.output)
    fail_path = Path(args.failed_log)

    samples = []
    failures = []
    for gold_path in sorted(gold_dir.glob("gt_*.csv")):
        hadm_id = gold_path.stem.replace("gt_", "")
        input_path = input_dir / f"input_{hadm_id}.csv"
        if not input_path.exists():
            failures.append({"hadm_id": hadm_id, "reason": "missing_input"})
            continue

        input_df = pd.read_csv(input_path)
        gold_df = pd.read_csv(gold_path)
        if not {"DAY", "TEXT"}.issubset(gold_df.columns) or not {"DAY", "TEXT", "IS_NOTE"}.issubset(input_df.columns):
            failures.append({"hadm_id": hadm_id, "reason": "missing_required_columns"})
            continue

        note_days = [int(day) for day in gold_df["DAY"].tolist()]
        first_note_day = min(note_days) if note_days else None
        for _, row in gold_df.iterrows():
            day = int(row["DAY"])
            full_note = str(row["TEXT"])
            context, gold_ap, status = split_ap_section(full_note)
            if day == first_note_day and not args.include_first_note:
                failures.append({"hadm_id": hadm_id, "day": day, "reason": "first_note_context_only"})
                continue

            day_input = input_df[input_df["DAY"].astype(int) == day].copy()
            ehr_rows = day_input[day_input["IS_NOTE"].astype(int) == 0].copy()
            ehr_events = []
            for idx, event in enumerate(ehr_rows.itertuples(index=False)):
                text = normalize_text(getattr(event, "TEXT", ""))
                if not text:
                    continue
                ehr_events.append(
                    {
                        "evidence_id": f"{hadm_id}_day{day}_raw{idx:04d}",
                        "rel_time": normalize_text(getattr(event, "REL_TIME", "")),
                        "text": text,
                        "source_type": "ehr_event",
                    }
                )

            ehr_blob = "\n".join(event["text"] for event in ehr_events)
            leakage_reasons = []
            if normalize_text(full_note) and normalize_text(full_note) in normalize_text(ehr_blob):
                leakage_reasons.append("full_note_in_ehr_events")
            if normalize_text(gold_ap) and normalize_text(gold_ap) in normalize_text(context):
                leakage_reasons.append("gold_ap_in_context")
            if normalize_text(gold_ap) and normalize_text(gold_ap) in normalize_text(ehr_blob):
                leakage_reasons.append("gold_ap_in_ehr_events")

            sample = {
                "sample_id": f"{hadm_id}_day{day}",
                "hadm_id": hadm_id,
                "day": day,
                "current_note_context": context,
                "gold_ap": gold_ap,
                "full_gold_note": full_note if args.keep_full_note else "",
                "ehr_events_before_cutoff": ehr_events,
                "ap_extraction_status": status,
                "leakage_check_pass": len(leakage_reasons) == 0,
                "leakage_reasons": leakage_reasons,
            }
            if status != "ok" or leakage_reasons:
                failures.append(
                    {
                        "sample_id": sample["sample_id"],
                        "hadm_id": hadm_id,
                        "day": day,
                        "ap_extraction_status": status,
                        "leakage_reasons": leakage_reasons,
                    }
                )
            samples.append(sample)

    write_jsonl(out_path, samples)
    write_jsonl(fail_path, failures)
    print(f"Built {len(samples)} AP samples -> {out_path}")
    print(f"Logged {len(failures)} skipped/warning rows -> {fail_path}")
    return out_path


def detect_problem_ids(text: str) -> list[str]:
    lower = text.lower()
    problem_ids = []
    for problem_id, spec in PROBLEM_TAXONOMY.items():
        if any(keyword in lower for keyword in spec["keywords"]):
            problem_ids.append(problem_id)
    return problem_ids


def evidence_agent(sample: dict) -> list[dict]:
    evidence = []
    by_name_values: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for idx, raw in enumerate(sample["ehr_events_before_cutoff"]):
        text = normalize_text(raw["text"])
        problem_ids = detect_problem_ids(text)
        if not problem_ids:
            continue
        evidence_id = f"{sample['sample_id']}_ev{idx:04d}"
        item = {
            "evidence_id": evidence_id,
            "text": text,
            "rel_time": raw.get("rel_time", ""),
            "problem_ids": problem_ids,
            "trend": "unknown",
            "source_type": raw.get("source_type", "ehr_event"),
        }
        evidence.append(item)
        for name, value in extract_numeric_observations(text):
            by_name_values[name].append((idx, value, evidence_id))

    trend_by_evidence = {}
    for name, values in by_name_values.items():
        if len(values) < 2:
            continue
        values = sorted(values)
        first = values[0][1]
        last = values[-1][1]
        delta = last - first
        if abs(delta) < 1e-6:
            trend = "stable"
        elif delta > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        for _, _, evidence_id in values:
            trend_by_evidence[evidence_id] = f"{name} {trend} ({first:g} -> {last:g})"

    for item in evidence:
        if item["evidence_id"] in trend_by_evidence:
            item["trend"] = trend_by_evidence[item["evidence_id"]]
    return evidence


def extract_numeric_observations(text: str) -> list[tuple[str, float]]:
    observations = []
    patterns = [
        ("glucose", r"glucose is ([0-9.]+)"),
        ("creatinine", r"creatinine is ([0-9.]+)"),
        ("bun", r"(?:urea nitrogen|bun) is ([0-9.]+)"),
        ("hemoglobin", r"hemoglobin is ([0-9.]+)"),
        ("platelet", r"platelet count is ([0-9.]+)"),
        ("pco2", r"pco2 is ([0-9.]+)"),
        ("po2", r"po2 is ([0-9.]+)"),
        ("calcium", r"calcium, total is ([0-9.]+)"),
    ]
    lower = text.lower()
    for name, pattern in patterns:
        for match in re.finditer(pattern, lower):
            try:
                observations.append((name, float(match.group(1))))
            except ValueError:
                pass
    return observations


def problem_detector_agent(sample: dict, evidence: list[dict], memory: dict[str, ProblemState]) -> list[str]:
    detected = set()
    for item in evidence:
        detected.update(item["problem_ids"])
    detected.update(detect_problem_ids(sample.get("current_note_context", "")))
    for problem_id, state in memory.items():
        if state.status.startswith("active"):
            detected.add(problem_id)
    return sorted(detected)


def problem_gate_agent(sample: dict, evidence: list[dict], problem_ids: list[str], version: str = "v2") -> dict[str, list[str] | dict]:
    """Separate real A&P problems from supporting evidence-only findings.

    V2/V3 use coarse buckets. V4 adds a general clinical certainty gate so
    medications, monitoring labs, and weak single-source clues are not promoted
    into standalone diagnoses.
    """
    evidence_text_by_problem: dict[str, str] = defaultdict(str)
    counts: Counter[str] = Counter()
    for item in evidence:
        for problem_id in item.get("problem_ids", []):
            evidence_text_by_problem[problem_id] += " " + item.get("text", "")
            counts[problem_id] += 1
    context = normalize_text(sample.get("current_note_context", "")).lower()
    gated = {"primary_active": [], "secondary_active": [], "evidence_only": []}
    if version == "v4":
        detailed = {}
        for problem_id in problem_ids:
            text = (evidence_text_by_problem.get(problem_id, "") + " " + context).lower()
            decision = certainty_gate_decision(problem_id, text, counts[problem_id])
            detailed[problem_id] = decision
            level = decision["allowed_output_level"]
            if level == "standalone_section":
                gated["primary_active"].append(problem_id)
            elif level == "supportive_care":
                gated["secondary_active"].append(problem_id)
            else:
                gated["evidence_only"].append(problem_id)
        gated["certainty"] = detailed
        return gated

    for problem_id in problem_ids:
        text = (evidence_text_by_problem.get(problem_id, "") + " " + context).lower()
        count = counts[problem_id]
        bucket = "evidence_only"

        if problem_id == "respiratory_failure_copd":
            if any(term in text for term in ["intubat", "ventilat", "pco2", "fio2", "peep", "sbt", "rsbi", "wheeze", "hypox"]):
                bucket = "primary_active"
        elif problem_id == "infection":
            if version == "v2":
                if any(term in text for term in ["cellulitis", "pneumonia", "sepsis", "fever", "culture", "vancomycin", "levofloxacin", "zosyn", "antibiotic"]):
                    bucket = "primary_active" if any(term in text for term in ["cellulitis", "pneumonia", "sepsis"]) else "secondary_active"
            else:
                if any(term in text for term in ["sepsis", "fever", "culture", "vancomycin", "levofloxacin", "zosyn", "antibiotic"]):
                    bucket = "primary_active" if any(term in text for term in ["sepsis", "bacteremia"]) else "secondary_active"
        elif problem_id == "pneumonia":
            if has_pneumonia_evidence(text):
                bucket = "primary_active"
            else:
                bucket = "evidence_only"
        elif problem_id == "volume":
            if any(term in text for term in ["furosemide", "lasix", "diures", "volume", "edema", "cor pulmonale"]):
                bucket = "primary_active"
        elif problem_id == "skin_wound":
            if any(term in text for term in ["cellulitis", "wound", "erythema", "drainage"]):
                bucket = "primary_active"
        elif problem_id == "glucose_diabetes":
            if any(term in text for term in ["glucose", "insulin", "diabetes", "hypergly"]):
                bucket = "secondary_active"
        elif problem_id == "neuro_pain_sedation":
            if any(term in text for term in ["sedation", "midazolam", "versed", "fentanyl", "anxiety", "pain"]):
                bucket = "secondary_active"
        elif problem_id in {"fen_nutrition", "prophylaxis_access"}:
            if count:
                bucket = "secondary_active"
        elif problem_id == "renal_aki":
            # Do not create AKI unless kidney injury is directly supported.
            if any(term in text for term in ["aki", "renal failure", "dialysis"]) or has_high_creatinine(text):
                bucket = "primary_active"
            else:
                bucket = "evidence_only"
        elif problem_id == "heme":
            if any(term in text for term in ["anemia", "bleeding", "transfusion", "thrombocytopenia"]):
                bucket = "primary_active"
            else:
                bucket = "evidence_only"

        gated[bucket].append(problem_id)

    return gated


def certainty_gate_decision(problem_id: str, text: str, evidence_count: int) -> dict:
    """General evidence-strength gate for AP problem promotion."""
    explicit_terms = {
        "respiratory_failure_copd": ["respiratory failure", "copd exacerbation", "intubat", "mechanical ventilation", "hypercarb", "hypox"],
        "infection": ["sepsis", "bacteremia", "infection", "cellulitis"],
        "pneumonia": ["pneumonia", "infiltrate", "consolidation"],
        "renal_aki": ["aki", "acute kidney injury", "renal failure", "dialysis"],
        "glucose_diabetes": ["diabetes", "hyperglycemia", "hypoglycemia"],
        "volume": ["volume overload", "hypervolemia", "diuresis", "cor pulmonale"],
        "fen_nutrition": ["nutrition", "tube feed", "diet", "electrolyte"],
        "heme": ["anemia", "bleeding", "transfusion", "thrombocytopenia"],
        "skin_wound": ["cellulitis", "wound", "erythema", "drainage"],
        "neuro_pain_sedation": ["anxiety", "delirium", "sedation", "pain"],
        "prophylaxis_access": ["prophylaxis", "line", "foley", "access"],
    }
    weak_treatment_terms = {
        "infection": ["vancomycin", "levofloxacin", "zosyn", "cef", "antibiotic", "culture"],
        "pneumonia": ["sputum", "vancomycin", "levofloxacin", "antibiotic"],
        "renal_aki": ["bun", "urea nitrogen", "creatinine"],
        "glucose_diabetes": ["glucose", "insulin"],
        "volume": ["furosemide", "lasix"],
        "heme": ["hemoglobin", "hematocrit", "platelet"],
        "fen_nutrition": ["sodium", "potassium", "magnesium", "phosphate", "calcium", "free water", "nutren"],
        "neuro_pain_sedation": ["fentanyl", "midazolam", "versed"],
        "prophylaxis_access": ["heparin", "pantoprazole", "ppi"],
    }
    monitoring_only_problems = {"prophylaxis_access"}

    explicit_hit = any(term in text for term in explicit_terms.get(problem_id, []))
    weak_hit = any(term in text for term in weak_treatment_terms.get(problem_id, []))
    multi_evidence = evidence_count >= 2

    if problem_id == "pneumonia" and not has_pneumonia_evidence(text):
        return {
            "evidence_strength": "weak" if weak_hit else "evidence_only",
            "allowed_output_level": "mention_only" if weak_hit else "suppress",
            "reason": "No direct pneumonia evidence; respiratory antibiotics/sputum alone should not become pneumonia diagnosis.",
        }

    if problem_id == "renal_aki" and not (explicit_hit or has_high_creatinine(text)):
        return {
            "evidence_strength": "monitoring_only" if weak_hit else "evidence_only",
            "allowed_output_level": "mention_only" if weak_hit else "suppress",
            "reason": "Renal labs without AKI/high creatinine support monitoring, not AKI diagnosis.",
        }

    if problem_id == "heme" and not explicit_hit:
        return {
            "evidence_strength": "monitoring_only" if weak_hit else "evidence_only",
            "allowed_output_level": "mention_only" if weak_hit else "suppress",
            "reason": "CBC values alone should not become an active hematology diagnosis.",
        }

    if problem_id in monitoring_only_problems:
        return {
            "evidence_strength": "monitoring_only",
            "allowed_output_level": "supportive_care",
            "reason": "Routine ICU prophylaxis/access should be summarized under supportive care.",
        }

    if explicit_hit:
        return {
            "evidence_strength": "confirmed" if multi_evidence else "probable",
            "allowed_output_level": "standalone_section",
            "reason": "Direct problem terms or strong clinical context support standalone A&P problem.",
        }

    if weak_hit and problem_id in {"glucose_diabetes", "fen_nutrition", "neuro_pain_sedation", "infection", "volume"}:
        return {
            "evidence_strength": "weak" if not multi_evidence else "probable",
            "allowed_output_level": "supportive_care" if problem_id != "volume" else "standalone_section",
            "reason": "Treatment/lab evidence supports management language, but diagnostic certainty is limited.",
        }

    return {
        "evidence_strength": "evidence_only",
        "allowed_output_level": "suppress",
        "reason": "Insufficient evidence to include in generated A&P.",
    }


def has_high_creatinine(text: str, threshold: float = 1.5) -> bool:
    for match in re.finditer(r"creatinine is ([0-9.]+)", text):
        try:
            if float(match.group(1)) >= threshold:
                return True
        except ValueError:
            pass
    return False


def has_pneumonia_evidence(text: str) -> bool:
    direct = any(term in text for term in ["pneumonia", "infiltrate", "consolidation"])
    cxr_with_finding = "cxr" in text and any(term in text for term in ["opacity", "infiltrate", "consolidation", "pneumonia"])
    sputum_with_infection = any(term in text for term in ["sputum culture", "purulent sputum"])
    return direct or cxr_with_finding or sputum_with_infection


def route_evidence(evidence: list[dict], problem_ids: list[str]) -> dict[str, list[dict]]:
    routed = {problem_id: [] for problem_id in problem_ids}
    for item in evidence:
        for problem_id in item["problem_ids"]:
            if problem_id in routed:
                routed[problem_id].append(item)
    return routed


def state_updater_agent(
    sample: dict,
    problem_ids: list[str],
    routed: dict[str, list[dict]],
    memory: dict[str, ProblemState],
) -> dict[str, ProblemState]:
    updated = dict(memory)
    for problem_id in problem_ids:
        spec = PROBLEM_TAXONOMY[problem_id]
        evidence = routed.get(problem_id, [])
        prev = updated.get(problem_id, ProblemState(problem_id=problem_id, label=spec["label"]))
        has_new = bool(evidence)
        trend_words = [item["trend"] for item in evidence if item.get("trend") and item["trend"] != "unknown"]
        if trend_words:
            if any("decreasing" in word and problem_id in {"renal_aki", "glucose_diabetes", "infection"} for word in trend_words):
                status = "active_improving"
            elif any("increasing" in word for word in trend_words):
                status = "active_worsening"
            else:
                status = "active_stable"
        elif has_new:
            status = "active"
        else:
            status = "inactive_uncertain"

        snippets = [item["text"] for item in evidence[:3]]
        assessment = f"{spec['label']} remains relevant today"
        if snippets:
            assessment += " based on " + "; ".join(snippets[:2])
        plan = make_problem_plan(problem_id, status, evidence)
        prev.status = status
        prev.current_assessment = assessment + "."
        prev.current_plan = plan
        prev.supporting_evidence_ids = [item["evidence_id"] for item in evidence[:8]]
        if sample["day"] not in prev.days_seen:
            prev.days_seen.append(sample["day"])
        if snippets:
            prev.trajectory_summary = f"Day {sample['day']}: " + "; ".join(snippets[:2])
        updated[problem_id] = prev
    return updated


def make_problem_plan(problem_id: str, status: str, evidence: list[dict]) -> list[str]:
    if problem_id == "respiratory_failure_copd":
        return ["Continue respiratory monitoring and ventilatory/oxygen support as indicated.", "Reassess readiness to wean support daily."]
    if problem_id == "infection":
        return ["Continue or reassess antimicrobial therapy based on cultures and clinical trajectory.", "Monitor fever curve and leukocytosis."]
    if problem_id == "renal_aki":
        return ["Trend creatinine, BUN, electrolytes, and urine output.", "Avoid nephrotoxins and adjust medication dosing for renal function."]
    if problem_id == "glucose_diabetes":
        return ["Continue glucose monitoring and insulin adjustment.", "Reassess insulin needs with nutrition and steroid changes."]
    if problem_id == "volume":
        return ["Monitor fluid balance and response to diuresis.", "Adjust diuretics based on renal function and volume status."]
    if problem_id == "fen_nutrition":
        return ["Replete electrolytes as needed and monitor nutrition plan.", "Continue enteral/diet support as tolerated."]
    if problem_id == "heme":
        return ["Trend CBC and monitor for bleeding or transfusion needs."]
    if problem_id == "skin_wound":
        return ["Continue wound care and monitor skin findings.", "Continue infection coverage if clinically indicated."]
    if problem_id == "neuro_pain_sedation":
        return ["Titrate sedation and analgesia to clinical goals.", "Monitor mental status and delirium risk."]
    if problem_id == "prophylaxis_access":
        return ["Continue ICU prophylaxis and reassess lines/tubes daily."]
    return ["Continue monitoring and update plan as new evidence becomes available."]


def ap_writer_agent(sample: dict, states: dict[str, ProblemState], method: str) -> str:
    active_states = [state for state in states.values() if state.status.startswith("active")]
    if not active_states:
        active_states = list(states.values())[:3]
    lines = ["Assessment and Plan"]
    context = normalize_text(sample.get("current_note_context", ""))
    if context:
        lines.append("")
        lines.append("Context: " + context[:350])
    for state in sorted(active_states, key=lambda s: s.label):
        lines.append("")
        lines.append(f"# {state.label}")
        lines.append(f"Assessment: {state.current_assessment}")
        if method in {"trend", "problemflow"} and state.trajectory_summary:
            lines.append(f"Trajectory: {state.trajectory_summary}")
        lines.append("Plan:")
        for plan in state.current_plan:
            lines.append(f"- {plan}")
    return "\n".join(lines).strip()


def compact_evidence(evidence: list[dict], max_items: int = 30) -> str:
    lines = []
    for item in evidence[:max_items]:
        trend = item.get("trend", "unknown")
        trend_text = "" if trend == "unknown" else f" | trend: {trend}"
        labels = ", ".join(PROBLEM_TAXONOMY[pid]["label"] for pid in item.get("problem_ids", []) if pid in PROBLEM_TAXONOMY)
        lines.append(f"- [{item['evidence_id']}] {item['text']} | problems: {labels}{trend_text}")
    return "\n".join(lines)


def build_coverage_guard(problem_gate: dict | None, routed_evidence: dict[str, list[dict]]) -> dict:
    """Compact V6 guardrail that tells the reviser what coverage to preserve."""
    gate = problem_gate or {"primary_active": [], "secondary_active": [], "evidence_only": []}
    guard = {"primary_active": [], "secondary_active": [], "evidence_only": gate.get("evidence_only", [])}
    for bucket in ["primary_active", "secondary_active"]:
        for problem_id in gate.get(bucket, []):
            if problem_id not in PROBLEM_TAXONOMY:
                continue
            guard[bucket].append(
                {
                    "problem_id": problem_id,
                    "label": PROBLEM_TAXONOMY[problem_id]["label"],
                    "evidence": [
                        {
                            "evidence_id": item["evidence_id"],
                            "text": item["text"][:220],
                            "trend": item.get("trend", "unknown"),
                        }
                        for item in routed_evidence.get(problem_id, [])[:3]
                    ],
                }
            )
    return guard


def llm_direct_writer(sample: dict, evidence: list[dict], args: argparse.Namespace) -> str:
    prompt = f"""Write the Assessment and Plan section for today's ICU progress note.

Use only the provided current pre-A&P context and EHR evidence. Do not invent unsupported diagnoses, procedures, or medications.

Current pre-A&P context:
{sample.get('current_note_context', '')[:2500]}

Today's EHR evidence:
{compact_evidence(evidence, max_items=args.max_evidence_items)}

Output only the final Assessment and Plan text."""
    return call_deepseek(
        prompt,
        model=args.model,
        api_url=args.api_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
    )


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def llm_state_updater_agent(
    sample: dict,
    evidence: list[dict],
    problem_ids: list[str],
    memory: dict[str, ProblemState],
    args: argparse.Namespace,
    problem_gate: dict[str, list[str]] | None = None,
) -> dict[str, ProblemState]:
    previous_states = [
        {
            "problem_id": state.problem_id,
            "label": state.label,
            "status": state.status,
            "trajectory_summary": state.trajectory_summary,
            "current_assessment": state.current_assessment,
            "current_plan": state.current_plan,
        }
        for state in memory.values()
    ]
    candidate_problems = [
        {"problem_id": pid, "label": PROBLEM_TAXONOMY[pid]["label"]}
        for pid in problem_ids
        if pid in PROBLEM_TAXONOMY
    ]
    gate_payload = problem_gate or {
        "primary_active": problem_ids,
        "secondary_active": [],
        "evidence_only": [],
    }
    prompt = f"""You are the Problem State Agent for ICU A&P generation.

Update today's structured problem memory using previous memory and today's evidence.
Return strict JSON only with this schema:
{{
  "problems": [
    {{
      "problem_id": "respiratory_failure_copd",
      "label": "Respiratory failure / COPD",
      "status": "active_improving|active_worsening|active_stable|active|inactive_uncertain",
      "trajectory_summary": "one sentence longitudinal trajectory",
      "current_assessment": "one concise assessment sentence",
      "current_plan": ["plan item 1", "plan item 2"],
      "supporting_evidence_ids": ["sample_day_ev0001"]
    }}
  ]
}}

Rules:
- Use only listed evidence IDs.
- Respect the problem gate:
  - primary_active can become standalone A&P sections.
  - secondary_active should be supportive ICU-care items unless clinically central.
  - evidence_only must not become standalone diagnoses or active problems.
- If a certainty map is present, obey allowed_output_level:
  - standalone_section -> can be a problem state.
  - supportive_care -> keep concise, avoid diagnosis escalation.
  - mention_only -> monitoring language only.
  - suppress -> omit.
- Preserve important active previous problems unless clearly resolved.
- Prefer problem-specific plans supported by evidence.
- Do not label AKI, pneumonia, anemia, or worsening status unless directly supported.
- If pneumonia is not in primary_active, do not use the word pneumonia. Use "antimicrobial coverage" or "infection coverage" instead.
- Do not include explanations outside JSON.

Patient/day: {sample['sample_id']}

Current pre-A&P context:
{sample.get('current_note_context', '')[:1800]}

Candidate active problems:
{json.dumps(candidate_problems, ensure_ascii=False)}

Problem gate:
{json.dumps(gate_payload, ensure_ascii=False)}

Previous problem memory:
{json.dumps(previous_states, ensure_ascii=False)}

Today's evidence:
{compact_evidence(evidence, max_items=args.max_evidence_items)}
"""
    text = call_deepseek(
        prompt,
        model=args.model,
        api_url=args.api_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
    )
    try:
        payload = parse_json_object(text)
    except Exception:
        routed = route_evidence(evidence, problem_ids)
        return state_updater_agent(sample, problem_ids, routed, memory)

    updated = dict(memory)
    for item in payload.get("problems", []):
        problem_id = item.get("problem_id")
        if problem_id not in PROBLEM_TAXONOMY:
            continue
        spec = PROBLEM_TAXONOMY[problem_id]
        state = updated.get(problem_id, ProblemState(problem_id=problem_id, label=spec["label"]))
        state.label = item.get("label") or spec["label"]
        state.status = item.get("status") or "active"
        state.trajectory_summary = item.get("trajectory_summary") or state.trajectory_summary
        state.current_assessment = item.get("current_assessment") or ""
        state.current_plan = [str(x) for x in item.get("current_plan", []) if str(x).strip()]
        state.supporting_evidence_ids = [str(x) for x in item.get("supporting_evidence_ids", []) if str(x).strip()]
        if sample["day"] not in state.days_seen:
            state.days_seen.append(sample["day"])
        updated[problem_id] = state
    return updated


def llm_problemflow_writer(sample: dict, states: dict[str, ProblemState], args: argparse.Namespace) -> str:
    state_payload = [
        {
            "problem_id": state.problem_id,
            "label": state.label,
            "status": state.status,
            "trajectory_summary": state.trajectory_summary,
            "current_assessment": state.current_assessment,
            "current_plan": state.current_plan,
            "supporting_evidence_ids": state.supporting_evidence_ids,
        }
        for state in states.values()
        if state.status.startswith("active")
    ]
    routed_evidence = getattr(args, "_current_routed_evidence", {})
    gate_payload = getattr(args, "_current_problem_gate", {})
    method = getattr(args, "_current_method", "")
    v6_rules = ""
    if method == "problemflow_v6":
        v6_rules = """
V6 coverage rules:
- Cover every primary_active problem with direct evidence, even if briefly.
- Preserve likely gold A&P categories by grouping secondary_active ICU-care items under one supportive care section instead of deleting them.
- Keep uncertainty explicit: use "concern for", "coverage for", "monitoring", or "risk of" when evidence is suggestive but not diagnostic.
- Do not let the revision-oriented grounding constraints erase supported respiratory, infectious, renal, volume, glucose, nutrition, hematology, sedation, skin, or access issues."""
    evidence_payload = {
        problem_id: [
            {
                "evidence_id": item["evidence_id"],
                "text": item["text"][:350],
                "trend": item.get("trend", "unknown"),
            }
            for item in items[:6]
        ]
        for problem_id, items in routed_evidence.items()
    }
    prompt = f"""You are the A&P Writer Agent.

Write the final Assessment and Plan using the structured problem states below.

Rules:
- Organize by clinical problem.
- Only primary_active problems should receive standalone sections.
- Put secondary_active items under a concise ICU care/supportive care section.
- Do not write evidence_only items as diagnoses.
- If a certainty map is present, follow allowed_output_level exactly.
- Preserve evidence-grounded assessments and plans.
- Do not add new unsupported diagnoses, medications, procedures, or outcomes.
- Be concise but clinically useful.
- If evidence supports a lab value only, mention it as monitoring rather than a diagnosis.
- If pneumonia is not in primary_active, do not write pneumonia. Use "antimicrobial coverage" or "infection coverage" instead.
- Use a gold-style ICU A&P voice: short problem headings, concise actions, avoid broad textbook recommendations.
- Avoid generic plans unless directly supported by evidence.
{v6_rules}

Current pre-A&P context:
{sample.get('current_note_context', '')[:1800]}

Problem gate:
{json.dumps(gate_payload, ensure_ascii=False)}

Structured problem states:
{json.dumps(state_payload, ensure_ascii=False)}

Problem-specific evidence:
{json.dumps(evidence_payload, ensure_ascii=False)}

Output only the final Assessment and Plan text."""
    return call_deepseek(
        prompt,
        model=args.model,
        api_url=args.api_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
    )


def llm_reviser_agent(
    sample: dict,
    draft_ap: str,
    verification: dict,
    evidence: list[dict],
    args: argparse.Namespace,
) -> str:
    unsupported = [
        {
            "claim": item["claim"],
            "status": item["support_status"],
            "keywords": item.get("keywords", []),
        }
        for item in verification.get("claims", [])
        if item.get("support_status") == "unsupported"
    ][:20]
    method = getattr(args, "_current_method", "")
    v6_rules = ""
    if method == "problemflow_v6":
        v6_rules = """
V6 revision rules:
- Treat the draft as a high-coverage direct baseline. Perform minimal edits, not a full rewrite.
- Preserve the draft's supported headings, section order, wording, numbers, medications, and plans whenever they are supported by evidence.
- Fix unsupported wording with narrower, evidence-grounded language before deleting an entire supported problem.
- Only delete a sentence when it is unsupported and cannot be downgraded safely.
- Preserve all primary_active problems unless every claim for that problem is unsupported.
- Keep secondary_active issues as concise supportive-care bullets when there is evidence for monitoring or management.
- Prefer shorter local edits over dropping supported coverage.
- The revised A&P should remain close in style and content to the draft while improving grounding."""
    prompt = f"""You are the Verifier-Reviser Agent for ICU A&P generation.

Revise the draft A&P to improve evidence grounding while preserving supported active problem coverage.

Rules:
- This is a minimal-revision task. Do not rewrite the whole note if local edits are enough.
- Remove unsupported claims when they are not necessary.
- Downgrade weak diagnoses to monitoring or coverage language.
- Preserve supported primary problems and concise gold-style ICU A&P formatting.
- Do not add new diagnoses, medications, procedures, culture results, imaging findings, or goals not present in evidence.
- Avoid generic textbook recommendations.
- If a plan is clinically reasonable but unsupported by evidence, either delete it or phrase it as monitoring only.
- Output only the revised final A&P text.
{v6_rules}

Patient/day: {sample['sample_id']}

Problem gate:
{json.dumps(getattr(args, "_current_problem_gate", {}), ensure_ascii=False)}

Coverage guard:
{json.dumps(getattr(args, "_current_coverage_guard", {}), ensure_ascii=False)}

Draft A&P:
{draft_ap}

Unsupported claims to fix:
{json.dumps(unsupported, ensure_ascii=False)}

Available evidence:
{compact_evidence(evidence, max_items=args.max_evidence_items)}
"""
    return call_deepseek(
        prompt,
        model=args.model,
        api_url=args.api_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
    )


def direct_writer(sample: dict) -> str:
    evidence = evidence_agent(sample)
    detected = sorted({pid for item in evidence for pid in item["problem_ids"]})
    states = {}
    for pid in detected[:6]:
        spec = PROBLEM_TAXONOMY[pid]
        states[pid] = ProblemState(
            problem_id=pid,
            label=spec["label"],
            current_assessment=f"{spec['label']} is suggested by current EHR evidence.",
            current_plan=make_problem_plan(pid, "active", []),
        )
    return ap_writer_agent(sample, states, method="direct")


def trend_writer(sample: dict) -> str:
    evidence = evidence_agent(sample)
    pids = sorted({pid for item in evidence for pid in item["problem_ids"]})
    routed = route_evidence(evidence, pids)
    states = state_updater_agent(sample, pids, routed, {})
    return ap_writer_agent(sample, states, method="trend")


def verifier_agent(generated_ap: str, evidence: list[dict]) -> dict:
    evidence_blob = normalize_text(" ".join(item["text"] for item in evidence)).lower()
    claims = split_claims(generated_ap)
    verified = []
    for claim in claims:
        keywords = [word for word in re.findall(r"[A-Za-z][A-Za-z0-9/-]{3,}", claim.lower()) if word not in STOPWORDS]
        support = sum(1 for word in keywords if word in evidence_blob)
        if not keywords:
            status = "unknown"
        elif support >= max(1, math.ceil(len(keywords) * 0.2)):
            status = "supported"
        else:
            status = "unsupported"
        verified.append({"claim": claim, "support_status": status, "keyword_support": support, "keywords": keywords[:12]})
    counts = Counter(item["support_status"] for item in verified)
    total = len(verified) or 1
    return {
        "claims": verified,
        "summary": {
            "num_claims": len(verified),
            "num_supported": counts["supported"],
            "num_unsupported": counts["unsupported"],
            "grounded_claim_rate": counts["supported"] / total,
            "unsupported_claim_rate": counts["unsupported"] / total,
        },
    }


STOPWORDS = {
    "assessment",
    "plan",
    "continue",
    "monitor",
    "today",
    "current",
    "based",
    "with",
    "this",
    "that",
    "from",
    "patient",
    "daily",
    "needed",
    "clinical",
}


def split_claims(text: str) -> list[str]:
    candidates = []
    for line in text.splitlines():
        line = line.strip(" -\t")
        if not line or line.startswith("#") or line.lower() in {"assessment and plan", "plan:"}:
            continue
        parts = re.split(r"(?<=[.;])\s+", line)
        candidates.extend(part.strip() for part in parts if len(part.split()) >= 4)
    return candidates


def generate(args: argparse.Namespace) -> Path:
    samples = read_jsonl(Path(args.samples))
    out_dir = Path(args.outdir)
    generation_dir = out_dir / "generations"
    memory_dir = out_dir / "memories"
    verification_dir = out_dir / "verification"
    generation_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    verification_dir.mkdir(parents=True, exist_ok=True)

    methods = ALL_METHODS if args.method == "all" else [args.method]
    for method in methods:
        outputs = []
        verification_outputs = []
        memories: dict[str, dict[str, ProblemState]] = defaultdict(dict)
        for sample in sorted(samples, key=lambda s: (s["hadm_id"], int(s["day"]))):
            if args.limit and len(outputs) >= args.limit:
                break
            if not sample.get("leakage_check_pass", False) and not args.allow_leakage_warnings:
                continue
            evidence = evidence_agent(sample)
            if method == "direct":
                generated_ap = llm_direct_writer(sample, evidence, args) if args.llm == "deepseek" else direct_writer(sample)
                states = {}
            elif method == "trend":
                generated_ap = trend_writer(sample)
                states = {}
            elif method in PROBLEMFLOW_METHODS:
                memory = memories[sample["hadm_id"]]
                problem_ids = problem_detector_agent(sample, evidence, memory)
                problem_gate = (
                    problem_gate_agent(
                        sample,
                        evidence,
                        problem_ids,
                        version="v4" if method in {"problemflow_v4", "problemflow_v5", "problemflow_v6"} else ("v3" if method == "problemflow_v3" else "v2"),
                    )
                    if method in GATED_PROBLEMFLOW_METHODS
                    else None
                )
                gated_problem_ids = (
                    problem_gate["primary_active"] + problem_gate["secondary_active"]
                    if problem_gate
                    else problem_ids
                )
                routed = route_evidence(evidence, gated_problem_ids)
                coverage_guard = build_coverage_guard(problem_gate, routed)
                if method == "problemflow_v6":
                    # V6 deliberately skips the Problem State Agent. Its hypothesis is
                    # that a high-coverage direct draft plus evidence-constrained
                    # minimal revision can preserve ROUGE while improving grounding.
                    states = dict(memory)
                elif args.llm == "deepseek":
                    states = llm_state_updater_agent(sample, evidence, gated_problem_ids, memory, args, problem_gate=problem_gate)
                else:
                    states = state_updater_agent(sample, gated_problem_ids, routed, memory)
                if method != "problemflow_v6":
                    memories[sample["hadm_id"]] = states
                setattr(args, "_current_problem_gate", problem_gate or {"primary_active": problem_ids, "secondary_active": [], "evidence_only": []})
                setattr(args, "_current_routed_evidence", routed)
                setattr(args, "_current_coverage_guard", coverage_guard)
                setattr(args, "_current_method", method)
                if method == "problemflow_v6":
                    # V6 is a draft-preserving pipeline: start from the high-coverage
                    # direct draft, then use ProblemFlow gate/verifier signals for
                    # minimal evidence-constrained revision.
                    generated_ap = llm_direct_writer(sample, evidence, args) if args.llm == "deepseek" else direct_writer(sample)
                else:
                    generated_ap = llm_problemflow_writer(sample, states, args) if args.llm == "deepseek" else ap_writer_agent(sample, states, method=method)
                draft_ap = generated_ap
                draft_verification = verifier_agent(draft_ap, evidence)
                if method in REVISION_PROBLEMFLOW_METHODS and args.llm == "deepseek":
                    generated_ap = llm_reviser_agent(sample, draft_ap, draft_verification, evidence, args)
            else:
                raise ValueError(method)

            verification = verifier_agent(generated_ap, evidence)
            row = {
                "sample_id": sample["sample_id"],
                "hadm_id": sample["hadm_id"],
                "day": sample["day"],
                "method": method,
                "generated_ap": generated_ap,
                "draft_ap": draft_ap if method in REVISION_PROBLEMFLOW_METHODS else "",
                "gold_ap": sample["gold_ap"],
                "detected_problems": sorted(detect_problem_ids(generated_ap)),
                "gold_problems": sorted(detect_problem_ids(sample["gold_ap"])),
                "num_evidence": len(evidence),
                "problem_gate": problem_gate if method in GATED_PROBLEMFLOW_METHODS else {},
                "draft_verification_summary": draft_verification["summary"] if method in REVISION_PROBLEMFLOW_METHODS else {},
                "verification_summary": verification["summary"],
            }
            outputs.append(row)
            verification_outputs.append({"sample_id": sample["sample_id"], **verification})

        write_jsonl(generation_dir / f"{method}.jsonl", outputs)
        write_jsonl(verification_dir / f"{method}_verification.jsonl", verification_outputs)
        if method in PROBLEMFLOW_METHODS:
            memory_rows = []
            for hadm_id, memory in memories.items():
                for state in memory.values():
                    memory_rows.append(
                        {
                            "hadm_id": hadm_id,
                            "problem_id": state.problem_id,
                            "label": state.label,
                            "status": state.status,
                            "days_seen": state.days_seen,
                            "trajectory_summary": state.trajectory_summary,
                            "supporting_evidence_ids": state.supporting_evidence_ids,
                        }
                    )
            write_jsonl(memory_dir / "problemflow_memory.jsonl", memory_rows)
        print(f"Generated {len(outputs)} rows for {method} -> {generation_dir / (method + '.jsonl')}")
    return generation_dir


def rouge_l_f1(pred: str, gold: str) -> float:
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_len(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def lcs_len(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def f1_sets(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(args: argparse.Namespace) -> Path:
    generation_dir = Path(args.generation_dir)
    out_path = Path(args.output)
    rows = []
    for gen_path in sorted(generation_dir.glob("*.jsonl")):
        method_rows = read_jsonl(gen_path)
        for row in method_rows:
            pred = normalize_text(row["generated_ap"])
            gold = normalize_text(row["gold_ap"])
            pred_probs = set(row.get("detected_problems") or detect_problem_ids(pred))
            gold_probs = set(row.get("gold_problems") or detect_problem_ids(gold))
            p, r, f1 = f1_sets(pred_probs, gold_probs)
            verification = row.get("verification_summary", {})
            rows.append(
                {
                    "method": row["method"],
                    "sample_id": row["sample_id"],
                    "hadm_id": row["hadm_id"],
                    "day": row["day"],
                    "rouge_l_f1": rouge_l_f1(pred, gold),
                    "problem_precision": p,
                    "problem_recall": r,
                    "problem_f1": f1,
                    "num_evidence": row.get("num_evidence", 0),
                    "grounded_claim_rate": verification.get("grounded_claim_rate", 0.0),
                    "unsupported_claim_rate": verification.get("unsupported_claim_rate", 0.0),
                    "pred_problem_count": len(pred_probs),
                    "gold_problem_count": len(gold_probs),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["method"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote per-sample metrics -> {out_path}")
    print_summary(rows)
    return out_path


def print_summary(rows: list[dict]) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    print("\nSummary")
    for method, method_rows in sorted(grouped.items()):
        def mean(key: str) -> float:
            return sum(float(row[key]) for row in method_rows) / len(method_rows) if method_rows else 0.0

        print(
            f"{method}: n={len(method_rows)} "
            f"rouge={mean('rouge_l_f1')*100:.2f} "
            f"problem_f1={mean('problem_f1')*100:.2f} "
            f"grounded={mean('grounded_claim_rate')*100:.2f} "
            f"unsupported={mean('unsupported_claim_rate')*100:.2f}"
        )


def run_all(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    samples_path = outdir / "data" / "ap_samples.jsonl"
    failed_log = outdir / "logs" / "failed_ap_extraction.jsonl"
    build_args = argparse.Namespace(
        inputdir=args.inputdir,
        golddir=args.golddir,
        output=str(samples_path),
        failed_log=str(failed_log),
        include_first_note=args.include_first_note,
        keep_full_note=False,
    )
    build_samples(build_args)
    gen_args = argparse.Namespace(
        samples=str(samples_path),
        outdir=str(outdir),
        method=args.method,
        allow_leakage_warnings=args.allow_leakage_warnings,
        llm=args.llm,
        model=args.model,
        api_url=args.api_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
        max_evidence_items=args.max_evidence_items,
        limit=args.limit,
    )
    generate(gen_args)
    eval_args = argparse.Namespace(
        generation_dir=str(outdir / "generations"),
        output=str(outdir / "metrics" / "metrics.csv"),
    )
    evaluate(eval_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProblemFlow-AP multi-agent experiment scripts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--inputdir", default="data/AP/input")
    build.add_argument("--golddir", default="data/AP/gold")
    build.add_argument("--output", default=str(DEFAULT_OUTDIR / "data" / "ap_samples.jsonl"))
    build.add_argument("--failed-log", default=str(DEFAULT_OUTDIR / "logs" / "failed_ap_extraction.jsonl"))
    build.add_argument("--include-first-note", action="store_true")
    build.add_argument("--keep-full-note", action="store_true")

    gen = subparsers.add_parser("generate")
    gen.add_argument("--samples", default=str(DEFAULT_OUTDIR / "data" / "ap_samples.jsonl"))
    gen.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    gen.add_argument("--method", choices=[*ALL_METHODS, "all"], default="all")
    gen.add_argument("--allow-leakage-warnings", action="store_true")
    gen.add_argument("--llm", choices=["mock", "deepseek"], default="mock")
    gen.add_argument("--model", default="deepseek-chat")
    gen.add_argument("--api-url", default=DEFAULT_API_URL)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--max-tokens", type=int, default=1200)
    gen.add_argument("--retries", type=int, default=8)
    gen.add_argument("--sleep-seconds", type=float, default=3.0)
    gen.add_argument("--max-evidence-items", type=int, default=30)
    gen.add_argument("--limit", type=int, default=0)

    ev = subparsers.add_parser("evaluate")
    ev.add_argument("--generation-dir", default=str(DEFAULT_OUTDIR / "generations"))
    ev.add_argument("--output", default=str(DEFAULT_OUTDIR / "metrics" / "metrics.csv"))

    run = subparsers.add_parser("run-all")
    run.add_argument("--inputdir", default="data/AP/input")
    run.add_argument("--golddir", default="data/AP/gold")
    run.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    run.add_argument("--include-first-note", action="store_true")
    run.add_argument("--allow-leakage-warnings", action="store_true")
    run.add_argument("--method", choices=[*ALL_METHODS, "all"], default="all")
    run.add_argument("--llm", choices=["mock", "deepseek"], default="mock")
    run.add_argument("--model", default="deepseek-chat")
    run.add_argument("--api-url", default=DEFAULT_API_URL)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-tokens", type=int, default=1200)
    run.add_argument("--retries", type=int, default=8)
    run.add_argument("--sleep-seconds", type=float, default=3.0)
    run.add_argument("--max-evidence-items", type=int, default=30)
    run.add_argument("--limit", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build_samples(args)
    elif args.command == "generate":
        generate(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "run-all":
        run_all(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
