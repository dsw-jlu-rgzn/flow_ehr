"""
Generate AP notes with memory-gated problem-state scaffold augmentation.

This is a no-training experiment. It keeps the direct-generation evidence
surface intact: the LLM receives the same current-day EHR and matched history
context as the corresponding direct baseline. The only added input is a
compact AP-Memory-V2-style scaffold with active problems separated from
watchlist and supportive care.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from ap_problem_state_experiment import DEFAULT_CASES, parse_cases, rouge_l_f1
from ap_problem_state_augmented_generation import read_previous_context
from deepseek_api_generation import AP_INSTRUCTION_1, AP_INSTRUCTION_2, DEFAULT_API_URL, call_deepseek
from deepseek_api_generation import df2chron_str


def parse_json_response(text: str) -> Any:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_json(prompt: str, args: argparse.Namespace, max_tokens: int | None = None) -> Any:
    last_error = None
    last_response = ""
    for attempt in range(1, args.parse_retries + 1):
        response = call_deepseek(
            prompt=prompt,
            model=args.model,
            api_url=args.api_url,
            temperature=args.temperature,
            max_tokens=max_tokens or args.scaffold_max_tokens,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
            api_key_env=args.api_key_env,
        )
        last_response = response
        try:
            return parse_json_response(response)
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt < args.parse_retries:
                time.sleep(args.sleep_seconds * attempt)
    preview = last_response[:300].replace("\n", "\\n")
    raise RuntimeError(f"JSON parse failed after {args.parse_retries} attempts: {last_error}; response={preview!r}")


def load_or_run_json(path: Path, runner) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    result = runner()
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def read_case_inputs(args: argparse.Namespace, admission_id: str, day: int) -> dict:
    data_root = Path(args.data_root)
    input_df = pd.read_csv(data_root / "AP" / "input" / f"input_{admission_id}.csv")
    current_rows = input_df[(input_df["DAY"].astype(int).eq(day)) & (input_df["IS_NOTE"].astype(int).eq(0))]
    current_ehr = df2chron_str(current_rows)
    previous_context = read_previous_context(
        data_root=data_root,
        baseline_run_name=args.baseline_run_name,
        baseline_setting=args.baseline_setting,
        baseline_method=args.baseline_method,
        admission_id=admission_id,
        day=day,
    )

    gold_df = pd.read_csv(data_root / "AP" / "gold" / f"gt_{admission_id}.csv")
    gold_rows = gold_df[gold_df["DAY"].astype(int).eq(day)]
    gold_note = " ".join(gold_rows["TEXT"].fillna("").astype(str).tolist())

    baseline_df = pd.read_csv(
        data_root
        / "AP"
        / "generated"
        / "DG"
        / args.baseline_run_name
        / args.baseline_setting
        / args.baseline_method
        / f"genpns_{admission_id}.csv"
    )
    baseline_rows = baseline_df[baseline_df["DAY"].astype(int).eq(day)]
    baseline_note = " ".join(baseline_rows["TEXT"].fillna("").astype(str).tolist())

    return {
        "current_ehr": current_ehr,
        "previous_context": previous_context,
        "previous_latest_note": read_latest_previous_note(
            data_root=data_root,
            baseline_run_name=args.baseline_run_name,
            baseline_setting=args.baseline_setting,
            baseline_method=args.baseline_method,
            admission_id=admission_id,
            day=day,
        ),
        "gold_note": gold_note,
        "baseline_note": baseline_note,
    }


def read_latest_previous_note(
    data_root: Path,
    baseline_run_name: str,
    baseline_setting: str,
    baseline_method: str,
    admission_id: str,
    day: int,
) -> str:
    if baseline_method == "method-1":
        return ""

    if baseline_setting == "gt":
        note_df = pd.read_csv(data_root / "AP" / "gold" / f"gt_{admission_id}.csv")
    else:
        note_df = pd.read_csv(
            data_root
            / "AP"
            / "generated"
            / "DG"
            / baseline_run_name
            / baseline_setting
            / baseline_method
            / f"genpns_{admission_id}.csv"
        )
    previous = note_df[note_df["DAY"].astype(int).lt(day)].sort_values("DAY")
    if previous.empty:
        return ""
    return str(previous.iloc[-1]["TEXT"])


def build_scaffold_prompt_v1(case_inputs: dict) -> str:
    previous_context = case_inputs["previous_context"] or "(No previous note context is available.)"
    return f"""Build a memory-gated scaffold for today's ICU Assessment & Plan.

Use only the previous note context and today's EHR. This scaffold is a planning
aid, not the final note. It must be portable across EHR datasets: do not rely
on hard-coded disease lists, medication lists, lab names, or MIMIC-specific
schemas.

Problem-promotion gate:
- Put a problem in active_ap_problems only if it is an unresolved major problem
  from prior A&P, or today's evidence changes diagnosis, severity, treatment
  decision, disposition, or a major complication.
- Isolated lab abnormalities, single medication administrations, prophylaxis,
  nutrition, access, routine monitoring, generic risk, and low-confidence
  issues should usually go to watchlist or supportive_care, not their own A&P
  section.
- Keep at most 6 active_ap_problems.
- If the raw EHR and previous context conflict, cite the uncertainty instead of
  inventing a resolution.

Return only valid JSON with this schema:
{{
  "global_status": {{
    "overall_trajectory": "improving|worsening|stable|mixed|unclear",
    "current_severity": "critical|serious|stable|unclear",
    "one_sentence_summary": ""
  }},
  "active_ap_problems": [
    {{
      "problem": "",
      "today_status": "new|improving|worsening|stable|resolved|unclear",
      "why_active_today": "",
      "supporting_evidence": ["up to 3 concise evidence statements"],
      "plan_actions": ["up to 3 concrete actions or monitoring items"],
      "confidence": "high|medium|low"
    }}
  ],
  "watchlist": [
    {{
      "item": "",
      "reason_not_active_ap_problem": "",
      "monitoring_plan": ""
    }}
  ],
  "supportive_care": [
    {{
      "item": "",
      "reason_not_active_ap_problem": "",
      "routine_plan": ""
    }}
  ],
  "resolved_problems": [],
  "uncertainties": [],
  "promotion_gate_notes": []
}}

Previous note context:
{previous_context}

Today's EHR:
{case_inputs["current_ehr"]}
"""


def build_scaffold_prompt_v2(case_inputs: dict) -> str:
    previous_context = case_inputs["previous_context"] or "(No previous note context is available.)"
    return f"""Build a memory-gated scaffold for today's ICU Assessment & Plan.

Use only the previous note context and today's EHR. This scaffold is a planning
aid, not the final note. It must be portable across EHR datasets: do not rely
on hard-coded disease lists, medication lists, lab names, or MIMIC-specific
schemas.

The goal is to preserve the recall of a broad problem scaffold while using a
general promotion gate to avoid turning routine care, isolated labs, or generic
risks into standalone A&P sections.

Step 1. Extract carry_forward_major_headings:
- Identify the major A&P section headings or major problems from the previous
  note context.
- Keep the original clinical granularity when possible.
- Do not create disease-specific headings that were not implied by the prior
  A&P or today's EHR.

Step 2. Build candidate_problem_pool:
- Include prior major problems, possible new problems, important status changes,
  disposition blockers, and clinically relevant uncertainties.
- This pool is a recall backup; not every candidate should become its own A&P
  section.

Step 3. Apply the promotion gate:
- Promote a candidate to active_ap_problems only if it is an unresolved major
  prior A&P problem, or today's evidence changes diagnosis, severity, treatment
  decision, disposition, or a major complication.
- An isolated abnormal lab or single medication administration cannot be a
  standalone active problem unless one of these is true:
  1) prior A&P already treated it as a major problem;
  2) today's clinician note explicitly treats it as a diagnosis or treatment target;
  3) it changes treatment, consults, disposition, or monitoring intensity.
- Generic risk, prophylaxis, nutrition, access, routine monitoring, and
  low-confidence issues should usually go to watchlist or supportive_care.
- Keep at most 6 active_ap_problems.

For each active problem, assign section_role:
- primary_section: should receive its own A&P subsection.
- merged_into_existing_section: clinically important but should be merged into
  a broader major heading.
- brief_monitoring: should only appear as a short monitoring/update line.

Return only valid JSON with this schema:
{{
  "global_status": {{
    "overall_trajectory": "improving|worsening|stable|mixed|unclear",
    "current_severity": "critical|serious|stable|unclear",
    "one_sentence_summary": ""
  }},
  "carry_forward_major_headings": [
    {{
      "heading": "",
      "prior_status": "active|improving|worsening|stable|resolved|unclear",
      "carry_forward_reason": ""
    }}
  ],
  "candidate_problem_pool": [
    {{
      "problem": "",
      "source": "prior_major_heading|today_status_change|new_possible_problem|disposition|uncertainty",
      "should_promote": true,
      "promotion_reason": "",
      "if_not_promoted_where": "active_ap_problems|watchlist|supportive_care|resolved_problems"
    }}
  ],
  "active_ap_problems": [
    {{
      "problem": "",
      "today_status": "new|improving|worsening|stable|resolved|unclear",
      "section_role": "primary_section|merged_into_existing_section|brief_monitoring",
      "merge_target": "",
      "why_active_today": "",
      "supporting_evidence": ["up to 3 concise evidence statements"],
      "plan_actions": ["up to 3 concrete actions or monitoring items"],
      "confidence": "high|medium|low"
    }}
  ],
  "watchlist": [
    {{
      "item": "",
      "reason_not_active_ap_problem": "",
      "monitoring_plan": ""
    }}
  ],
  "supportive_care": [
    {{
      "item": "",
      "reason_not_active_ap_problem": "",
      "routine_plan": ""
    }}
  ],
  "resolved_problems": [],
  "uncertainties": [],
  "promotion_gate_notes": []
}}

Previous note context:
{previous_context}

Today's EHR:
{case_inputs["current_ehr"]}
"""


def build_scaffold_prompt_v2_cui_recall(case_inputs: dict) -> str:
    previous_context = case_inputs["previous_context"] or "(No previous note context is available.)"
    return f"""Build a CUI-recall-oriented V2 scaffold for today's ICU Assessment & Plan.

Use only the previous note context and today's EHR. This scaffold is a planning
aid, not the final note. It must improve clinical concept coverage without
carrying forward stale or unsupported problems.

Optimization target:
- Preserve specific clinical concepts that should appear in the final A&P:
  diagnoses, active problems, medications/treatments, procedures/devices,
  lab/physiology abnormalities, microbiology/infection findings, and major
  goals-of-care or disposition state changes.
- Avoid replacing specific concepts with generic labels. For example, keep
  "sustained VT on amiodarone infusion" rather than "arrhythmia"; keep
  "neutropenic fever on vancomycin/cefepime/metronidazole/voriconazole" rather
  than "infection".
- Do not carry forward pressors, ventilation, dialysis/CRRT, antibiotics,
  anticoagulation, nutrition route, procedures, or comfort-measures/death
  status unless today's EHR supports the current state.

Step 1. Extract must_cover_concepts:
- Include high-confidence concepts from today's EHR and unresolved prior major
  A&P headings that are still supported today.
- Each concept must have a specific evidence quote and a today_status.
- Prefer concrete medical terms over broad organ-system categories.

Step 2. Assign active-state labels to prior and candidate problems:
- active_today: should usually appear in the final A&P.
- worsening_today or improving_today: should appear with its status change.
- resolved_today: may be mentioned briefly as resolved/off/discontinued, but
  must not receive an active treatment plan.
- historical_context_only: may appear only in the assessment background.
- unsupported_today: must not appear in the final A&P.

Step 3. Build active_ap_problems:
- Promote concrete active_today, worsening_today, and improving_today problems
  into active_ap_problems.
- Use specific problem names that contain the core clinical concept.
- Keep at most 8 active_ap_problems, but do not collapse distinct high-priority
  concepts into generic headings.
- Routine prophylaxis, access, isolated labs, single medication administrations,
  and generic monitoring should not become standalone sections unless today's
  clinician text makes them a treatment target or they cause a major state
  change.

Return only valid JSON with this schema:
{{
  "global_status": {{
    "overall_trajectory": "improving|worsening|stable|mixed|unclear",
    "current_severity": "critical|serious|stable|unclear",
    "one_sentence_summary": ""
  }},
  "must_cover_concepts": [
    {{
      "concept": "",
      "category": "active_problem|medication_treatment|procedure_device|lab_physiology|infection_microbiology|goals_disposition_status",
      "today_status": "active_today|worsening_today|improving_today|resolved_today|historical_context_only|unsupported_today|unclear",
      "priority": "must_cover|should_cover|background_only|do_not_include",
      "evidence": "",
      "target_section": ""
    }}
  ],
  "active_state_audit": [
    {{
      "item": "",
      "prior_or_candidate_claim": "",
      "today_status": "active_today|worsening_today|improving_today|resolved_today|historical_context_only|unsupported_today|unclear",
      "today_evidence": "",
      "output_rule": "standalone_section|merge_into_related_section|brief_resolved_update|background_only|exclude"
    }}
  ],
  "carry_forward_major_headings": [
    {{
      "heading": "",
      "prior_status": "active|improving|worsening|stable|resolved|unclear",
      "today_status": "active_today|worsening_today|improving_today|resolved_today|historical_context_only|unsupported_today|unclear",
      "carry_forward_reason": ""
    }}
  ],
  "candidate_problem_pool": [
    {{
      "problem": "",
      "source": "prior_major_heading|today_status_change|new_possible_problem|disposition|uncertainty|must_cover_concept",
      "specific_concepts_to_preserve": [],
      "should_promote": true,
      "promotion_reason": "",
      "if_not_promoted_where": "active_ap_problems|watchlist|supportive_care|resolved_problems|exclude"
    }}
  ],
  "active_ap_problems": [
    {{
      "problem": "",
      "today_status": "new|continued|improving|worsening|stable|resolved|unclear",
      "section_role": "primary_section|merged_into_existing_section|brief_monitoring",
      "merge_target": "",
      "must_cover_concepts_used": [],
      "why_active_today": "",
      "supporting_evidence": ["up to 4 concise evidence statements"],
      "plan_actions": ["up to 4 concrete actions or monitoring items"],
      "confidence": "high|medium|low"
    }}
  ],
  "watchlist": [
    {{
      "item": "",
      "reason_not_active_ap_problem": "",
      "monitoring_plan": ""
    }}
  ],
  "supportive_care": [
    {{
      "item": "",
      "reason_not_active_ap_problem": "",
      "routine_plan": ""
    }}
  ],
  "resolved_problems": [],
  "unsupported_or_stale_concepts": [
    {{
      "concept": "",
      "reason_to_exclude_or_downgrade": "",
      "today_evidence": ""
    }}
  ],
  "uncertainties": [],
  "promotion_gate_notes": []
}}

Previous note context:
{previous_context}

Today's EHR:
{case_inputs["current_ehr"]}
"""


def build_scaffold_prompt_v3(case_inputs: dict) -> str:
    previous_context = case_inputs["previous_context"] or "(No previous note context is available.)"
    return f"""Build an Evidence-Hierarchy Gated Scaffold V3 for today's ICU Assessment & Plan.

Use only the previous note context and today's EHR. This is a no-training,
dataset-agnostic evidence gate. Do not rely on disease-specific, medication-
specific, lab-specific, admission-specific, or MIMIC-specific rules. Use source
strength and documented state changes to decide what may appear in the final
A&P.

The goal is to preserve longitudinal trajectory while preventing weak evidence
from becoming unsupported active problems or unsupported plan actions.

Step 1. Extract evidence_events:
- Extract concise events from today's EHR without expanding them into new
  diagnoses or plans.
- Medication administration only proves that a medication was administered.
- An isolated lab only proves the lab value; do not infer a diagnosis unless
  clinician text or a treatment decision explicitly supports it.
- Case-management/social-work text may inform disposition but cannot alone
  create a medical active problem.

Step 2. Track prior_problem_states:
- Identify major active problems or A&P headings from the previous context.
- Decide whether each is continued, improving, worsening, resolved, or unclear
  using today's EHR.
- If the previous generated note conflicts with today's EHR, mark the conflict
  and revise/drop the prior claim instead of carrying it forward.
- Preserve note continuity without overfitting: if a prior major A&P heading is
  not clearly resolved and not contradicted, keep it in must_carry_forward_sections
  even when today's evidence is thin. It may be written as a short continued/
  uncertain update rather than promoted as a primary active problem.

Step 3. Apply the hard promotion gate:
Promote a candidate to active_ap_problems only if at least one condition holds:
1. prior_active_with_today_support: prior A&P treated it as active and today's
   EHR corroborates continuation or a status change;
2. clinician_assessment_plan: today's clinician assessment/plan explicitly
   discusses the problem as a diagnosis, treatment target, or major management
   issue;
3. procedure_or_imaging_decision: procedure, imaging, or consult creates a new
   diagnosis or concrete treatment/disposition decision;
4. major_state_change: documented intubation/extubation, pressor start/stop,
   antibiotic start/stop, dialysis/CRRT start/stop, NPO/tube-feed/PO diet
   change, surgery/procedure, bleeding event, code-status/CMO/hospice change,
   ICU/floor transfer, or another major care-state transition.

Do NOT promote if the only support is:
- a single medication administration;
- an isolated abnormal lab;
- routine ICU care, prophylaxis, nutrition, access, or generic monitoring;
- case-management/social-work text without clinical corroboration;
- a plausible but unstated diagnosis;
- a previous generated-note claim without today's corroborating evidence.

Plan-action gate:
- allowed_plan_actions must be directly supported by today's EHR or a carried
  forward prior active plan with today's support.
- Do not invent consults, imaging, antibiotics, anticoagulation, dialysis,
  ventilator changes, or medication titrations merely because they seem
  clinically reasonable.

Contradiction audit:
- Explicitly audit any high-impact care state when present, using the evidence
  wording from the chart rather than fixed vocab: respiratory support state,
  hemodynamic support state, anti-infective treatment state, nutrition route,
  renal replacement status, procedure/surgery status, bleeding status,
  infection status, mobility/disposition status, and goals-of-care/code-status
  trajectory.

Return only valid JSON with this schema:
{{
  "global_status": {{
    "overall_trajectory": "improving|worsening|stable|mixed|unclear",
    "current_severity": "critical|serious|stable|unclear",
    "one_sentence_summary": ""
  }},
  "evidence_events": [
    {{
      "event": "",
      "source_type": "clinician_note|radiology|lab|med_admin|respiratory_note|nursing|case_management|other",
      "evidence_text": "",
      "inference_level": "direct|weak_inference|unsupported",
      "confidence": "high|medium|low"
    }}
  ],
  "prior_problem_states": [
    {{
      "problem": "",
      "today_status": "continued|improving|worsening|resolved|unclear",
      "supporting_evidence": [],
      "contradicting_evidence": [],
      "carry_forward_decision": "carry_forward|revise|drop|unclear"
    }}
  ],
  "must_carry_forward_sections": [
    {{
      "heading": "",
      "reason_to_preserve": "prior_major_heading_not_resolved|prior_major_heading_uncertain|documented_chronic_active_issue|today_mentions_related_management",
      "today_update": "continued|improving|worsening|resolved|unclear",
      "supporting_or_limiting_evidence": [],
      "final_note_role": "primary_section|secondary_update|one_line_monitoring"
    }}
  ],
  "disposition_and_goals": {{
    "current_location_or_level_of_care": "",
    "transfer_or_discharge_trajectory": "",
    "goals_or_code_status_context": "",
    "family_or_team_communication": "",
    "evidence": [],
    "confidence": "high|medium|low|unclear"
  }},
  "active_ap_problems": [
    {{
      "problem": "",
      "today_status": "new|continued|improving|worsening|resolved|unclear",
      "section_role": "primary_section|merged_into_existing_section|brief_monitoring",
      "merge_target": "",
      "promotion_reason": "",
      "promotion_evidence_type": "prior_active_with_today_support|clinician_assessment_plan|procedure_or_imaging_decision|major_state_change",
      "supporting_evidence": ["up to 3 concise evidence statements"],
      "allowed_plan_actions": ["up to 3 concrete actions supported by evidence"],
      "confidence": "high|medium"
    }}
  ],
  "watchlist": [
    {{
      "item": "",
      "reason_not_active_ap_problem": "",
      "monitoring_plan": ""
    }}
  ],
  "supportive_care": [
    {{
      "item": "",
      "reason_not_active_ap_problem": "",
      "routine_plan": ""
    }}
  ],
  "rejected_candidates": [
    {{
      "candidate": "",
      "rejection_reason": "",
      "evidence_type": "med_admin_only|isolated_lab_only|case_management_only|routine_care_only|previous_claim_without_today_support|plausible_but_unstated|other",
      "unsafe_if_promoted": true
    }}
  ],
  "contradiction_audit": [
    {{
      "field": "",
      "candidate_or_prior_claim": "",
      "today_evidence": "",
      "decision": "remove|revise|keep_uncertain"
    }}
  ],
  "resolved_problems": [],
  "uncertainties": []
}}

Previous note context:
{previous_context}

Today's EHR:
{case_inputs["current_ehr"]}
"""


def build_scaffold_prompt(case_inputs: dict, args: argparse.Namespace) -> str:
    if args.prompt_version == "v3":
        return build_scaffold_prompt_v3(case_inputs)
    if args.prompt_version == "v2_cui_recall":
        return build_scaffold_prompt_v2_cui_recall(case_inputs)
    if args.prompt_version == "v2_judge_cui_guard":
        return build_scaffold_prompt_v2(case_inputs)
    if args.prompt_version == "v2":
        return build_scaffold_prompt_v2(case_inputs)
    return build_scaffold_prompt_v1(case_inputs)


def build_generation_prompt_v1(case_inputs: dict, scaffold: dict) -> str:
    ehr_str = case_inputs["current_ehr"]
    if case_inputs["previous_context"]:
        ehr_str += "\n\nPrevious progress note context:\n" + case_inputs["previous_context"]

    scaffold_str = json.dumps(scaffold, ensure_ascii=False, indent=2)
    return f"""{AP_INSTRUCTION_1}{ehr_str}

Memory-gated problem-state scaffold:
The scaffold separates problems that should receive their own A&P section from
watchlist and supportive-care items. Use it to preserve unresolved active
problems, daily trajectory, and plan constraints.

Rules:
- Write main A&P subsections primarily for active_ap_problems.
- Mention watchlist items only as concise monitoring or uncertainty context.
- Fold supportive_care into ICU care/prophylaxis/nutrition when appropriate;
  do not turn routine care into a main diagnosis.
- If the scaffold conflicts with the raw EHR, trust the raw EHR and explicit
  evidence.

{scaffold_str}

{AP_INSTRUCTION_2}
"""


def build_generation_prompt_v2(case_inputs: dict, scaffold: dict) -> str:
    ehr_str = case_inputs["current_ehr"]
    if case_inputs["previous_context"]:
        ehr_str += "\n\nPrevious progress note context:\n" + case_inputs["previous_context"]

    scaffold_str = json.dumps(scaffold, ensure_ascii=False, indent=2)
    return f"""{AP_INSTRUCTION_1}{ehr_str}

Memory-gated problem-state scaffold:
This scaffold is a clinical planning aid. It should improve continuity and
grounding without replacing the raw EHR or forcing a checklist style.

Use the scaffold this way:
- Preserve the major A&P heading granularity from carry_forward_major_headings
  unless today's raw EHR clearly resolves or supersedes the problem.
- Use active_ap_problems with section_role=primary_section as main A&P
  subsections.
- Merge section_role=merged_into_existing_section items into their merge_target
  or the closest broader organ-system/problem subsection.
- Mention section_role=brief_monitoring and watchlist items only as concise
  monitoring or uncertainty lines, not standalone sections.
- Fold supportive_care into ICU care/prophylaxis/nutrition when appropriate;
  do not turn routine care into a main diagnosis.
- Use candidate_problem_pool as a recall backup: if a candidate is clearly
  supported by the raw EHR and fits the prior A&P style, include it; otherwise
  do not promote it.
- If the scaffold conflicts with the raw EHR, trust the raw EHR and explicit
  evidence.
- Do not create a standalone diagnosis from an isolated abnormal lab or single
  medication administration unless the prior A&P or today's clinician text
  clearly treats it as a major problem.

{scaffold_str}

{AP_INSTRUCTION_2}
"""


def build_generation_prompt_v2_cui_recall(case_inputs: dict, scaffold: dict) -> str:
    ehr_str = case_inputs["current_ehr"]
    if case_inputs["previous_context"]:
        ehr_str += "\n\nPrevious progress note context:\n" + case_inputs["previous_context"]

    scaffold_str = json.dumps(scaffold, ensure_ascii=False, indent=2)
    return f"""{AP_INSTRUCTION_1}{ehr_str}

CUI-recall-oriented V2 scaffold:
This scaffold is a clinical planning aid. Use it to preserve specific clinical
concepts while avoiding stale or unsupported carry-forward.

Generation rules:
- Cover all must_cover_concepts with priority=must_cover unless today's raw EHR
  contradicts them.
- Use specific problem headings that include the concrete clinical concept.
  Do not replace VT, AFib, pneumonia, PE/DVT, AKI, vancomycin, IABP, dialysis,
  pressors, CMO/death, or other specific charted concepts with generic labels
  such as arrhythmia, infection, organ failure, or complex ICU course.
- For active_state_audit items, follow output_rule exactly:
  standalone_section may be its own section; merge_into_related_section must be
  folded into a related section; brief_resolved_update must be stated only as
  resolved/off/discontinued; background_only belongs only in the assessment;
  exclude must not appear.
- Use active_ap_problems with section_role=primary_section as main A&P
  subsections. Merge section_role=merged_into_existing_section items into their
  merge_target or closest concrete section.
- Use candidate_problem_pool as recall backup only when the raw EHR supports the
  specific_concepts_to_preserve today.
- Include medication/treatment, procedure/device, lab/physiology, infection/
  microbiology, and goals/disposition concepts when they are high-confidence and
  affect today's assessment or plan.
- Do not carry forward pressors, ventilation, dialysis/CRRT, antibiotics,
  anticoagulation, nutrition route, procedures, or comfort-measures/death status
  as active unless today's raw EHR supports the current status.
- Before finalizing, silently check that high-priority concepts are covered and
  stale/unsupported concepts are excluded. Output only the final A&P.

{scaffold_str}

{AP_INSTRUCTION_2}
"""


def build_generation_prompt_v3(case_inputs: dict, scaffold: dict) -> str:
    ehr_str = case_inputs["current_ehr"]
    if case_inputs["previous_context"]:
        ehr_str += "\n\nPrevious progress note context:\n" + case_inputs["previous_context"]

    scaffold_str = json.dumps(scaffold, ensure_ascii=False, indent=2)
    return f"""{AP_INSTRUCTION_1}{ehr_str}

Evidence-Hierarchy Gated Scaffold V3:
This scaffold has already separated approved active A&P problems from
watchlist, supportive care, rejected candidates, and contradiction audits.

Conservative generation rules:
- Only active_ap_problems may become major A&P sections.
- Also preserve must_carry_forward_sections according to final_note_role:
  primary_section may be a section, secondary_update should be a short
  subsection or paragraph, and one_line_monitoring should be a concise line
  under a related approved section. Do not drop unresolved prior major headings
  solely because today's evidence is sparse.
- Use section_role=primary_section as standalone sections. Merge
  section_role=merged_into_existing_section into its merge_target or the
  closest approved active section. Mention section_role=brief_monitoring only
  briefly.
- Do not create new diagnoses, active problems, severity changes, or plan
  actions that are not in active_ap_problems.allowed_plan_actions.
- Include disposition_and_goals when evidence is present, even if it is not an
  active medical problem. Keep it concise and do not invent a destination,
  prognosis, code status, or family discussion beyond the scaffold evidence.
- Watchlist items may appear only as one-line monitoring/uncertainty within a
  related approved section. Do not make them standalone sections.
- Supportive_care may appear only under ICU care/prophylaxis/nutrition or as
  routine care; do not make it a main diagnosis.
- Rejected candidates must not appear in the final A&P.
- If contradiction_audit says remove or revise a prior/candidate claim, follow
  that decision and do not carry the contradicted claim forward.
- If evidence is uncertain, state uncertainty rather than resolving it.
- Do not infer disease identity from a medication, isolated lab, or social/case
  management note unless the approved scaffold promoted it with a valid
  promotion_evidence_type.
- Keep the final note concise and close to a physician A&P, not an evidence
  dump.

{scaffold_str}

{AP_INSTRUCTION_2}
"""


def build_generation_prompt(case_inputs: dict, scaffold: dict, args: argparse.Namespace) -> str:
    if args.prompt_version == "v3":
        return build_generation_prompt_v3(case_inputs, scaffold)
    if args.prompt_version == "v2_cui_recall":
        return build_generation_prompt_v2_cui_recall(case_inputs, scaffold)
    if args.prompt_version == "v2_judge_cui_guard":
        return build_generation_prompt_v2_cui_guard(case_inputs, scaffold)
    if args.prompt_version == "v2":
        return build_generation_prompt_v2(case_inputs, scaffold)
    return build_generation_prompt_v1(case_inputs, scaffold)


def build_generation_prompt_v2_cui_guard(case_inputs: dict, scaffold: dict) -> str:
    ehr_str = case_inputs["current_ehr"]
    if case_inputs["previous_context"]:
        ehr_str += "\n\nPrevious progress note context:\n" + case_inputs["previous_context"]

    scaffold_str = json.dumps(scaffold, ensure_ascii=False, indent=2)
    return f"""{AP_INSTRUCTION_1}{ehr_str}

Coverage-preserving memory-gated scaffold:
Use this scaffold to write a clinically grounded ICU Assessment & Plan while
preserving concrete, evidence-supported clinical concepts.

Rules:
- Do not make the note shorter by deleting supported clinical concepts.
- Preserve specific diagnoses, medications/treatments, procedures/devices,
  lab/physiology abnormalities, microbiology/infection findings, and care-state
  concepts when they are supported by today's EHR or an unresolved prior A&P.
- If a medication, procedure, infusion, device, or culture is documented but
  its indication is unclear, keep the concept and write "clarify indication" or
  "continue/adjust per team guidance" rather than deleting it.
- Downgrade stale or contradicted concepts by status: resolved/off/held/
  discontinued/historical background. Do not keep them active.
- Do not replace specific concepts with generic labels. Use "sustained VT on
  amiodarone/quinidine", "neutropenic fever on vancomycin/cefepime", "AKI on
  CRRT", "PE/DVT on heparin", etc. rather than "arrhythmia", "infection", or
  "renal dysfunction" when the chart is specific.
- Use active_ap_problems as the main sections, but also use
  candidate_problem_pool and carry_forward_major_headings to recover supported
  concepts that should not be lost.
- Before finalizing, silently check that no supported concept from the scaffold
  or today's EHR was deleted solely for concision.

{scaffold_str}

{AP_INSTRUCTION_2}
"""


def build_transition_judge_prompt(case_inputs: dict, scaffold: dict, candidate_ap: str, args: argparse.Namespace) -> str:
    if args.prompt_version == "v2_judge_cui_guard":
        return build_transition_judge_prompt_cui_guard(case_inputs, scaffold, candidate_ap)

    previous_note = case_inputs["previous_latest_note"] or "(No previous A&P note is available.)"
    scaffold_str = json.dumps(scaffold, ensure_ascii=False, indent=2)
    return f"""Judge whether the candidate ICU Assessment & Plan is a safe and evidence-supported day-to-day update.

Use only:
- the latest previous A&P note,
- today's raw EHR,
- the candidate A&P,
- the candidate memory-gated scaffold.

Do not use the gold current-day A&P. This judge is part of generation-time
quality control, not final evaluation.

Check:
1. Which candidate changes from the previous A&P are supported by today's EHR?
2. Which new diagnoses, severity changes, treatment plans, or discontinued
   problems are unsupported?
3. Which carried-forward major problems from the previous A&P were forgotten?
4. Which scaffold items should be downgraded from active_ap_problems to
   watchlist/supportive_care?
5. Which evidence-supported items should be added, but only if they satisfy the
   general promotion gate?

Return only valid JSON:
{{
  "supported_changes": [
    {{
      "change": "",
      "evidence": [],
      "target_scaffold_update": ""
    }}
  ],
  "unsupported_changes": [
    {{
      "change": "",
      "reason": "",
      "suggested_revision": ""
    }}
  ],
  "forgotten_carried_problems": [
    {{
      "problem": "",
      "previous_evidence": "",
      "suggested_revision": ""
    }}
  ],
  "missing_updates": [
    {{
      "problem_or_event": "",
      "today_evidence": "",
      "suggested_revision": ""
    }}
  ],
  "scaffold_revision_suggestions": [
    {{
      "target": "active_ap_problems|watchlist|supportive_care|resolved_problems|uncertainties",
      "action": "add|remove|downgrade|merge|revise",
      "item": "",
      "reason": ""
    }}
  ],
  "summary": {{
    "unsupported_change_count": 0,
    "missing_update_count": 0,
    "forgotten_carried_problem_count": 0,
    "overall_decision": "accept|revise_minor|revise_major"
  }}
}}

Latest previous A&P:
{previous_note}

Today's raw EHR:
{case_inputs["current_ehr"]}

Candidate scaffold:
{scaffold_str}

Candidate A&P:
{candidate_ap}
"""


def build_transition_judge_prompt_cui_guard(case_inputs: dict, scaffold: dict, candidate_ap: str) -> str:
    previous_note = case_inputs["previous_latest_note"] or "(No previous A&P note is available.)"
    scaffold_str = json.dumps(scaffold, ensure_ascii=False, indent=2)
    return f"""Judge the candidate ICU Assessment & Plan with a coverage-preserving clinical concept guard.

Use only:
- the latest previous A&P note,
- today's raw EHR,
- the candidate A&P,
- the candidate memory-gated scaffold.

Do not use the gold current-day A&P.

Goal:
Improve evidence grounding without reducing supported clinical concept coverage.
Your job is not to make the note shorter. Your job is to preserve all
evidence-supported concepts while fixing unsupported or stale claims.

Definitions:
- Supported concept: a specific diagnosis/problem, medication/treatment,
  procedure/device, lab/physiology abnormality, microbiology/infection finding,
  infusion, nutrition route, respiratory/hemodynamic/renal replacement state,
  goals-of-care or disposition state that appears in today's EHR or remains an
  unresolved prior major A&P issue.
- Stale concept: contradicted by today's EHR or only historical with no current
  relevance.
- Unclear-indication concept: documented medication/procedure/device/culture
  where the chart shows exposure but not the reason. Preserve it with "clarify
  indication" instead of deleting it.

Check:
1. Which candidate concepts are supported and must be preserved?
2. Which supported concepts from today's EHR/scaffold are missing and must be
   added?
3. Which candidate concepts are stale, contradicted, or unsupported and should
   be removed or downgraded?
4. Which specific concepts were replaced by generic wording and should be made
   concrete again?

Return only valid JSON:
{{
  "supported_concepts_to_preserve": [
    {{
      "concept": "",
      "category": "diagnosis_problem|medication_treatment|procedure_device|lab_physiology|infection_microbiology|care_state|goals_disposition",
      "status": "active|improving|worsening|resolved|administered_or_ordered|held_or_discontinued|historical_background|unclear",
      "evidence": [],
      "must_appear_in_revision": true
    }}
  ],
  "missing_high_confidence_concepts_to_add": [
    {{
      "concept": "",
      "category": "diagnosis_problem|medication_treatment|procedure_device|lab_physiology|infection_microbiology|care_state|goals_disposition",
      "today_evidence": [],
      "target_section": "",
      "required_wording": ""
    }}
  ],
  "unsupported_or_stale_concepts_to_remove_or_downgrade": [
    {{
      "concept": "",
      "reason": "",
      "today_evidence_or_conflict": "",
      "required_handling": "remove|brief_resolved_update|historical_background|uncertain_clarify_indication"
    }}
  ],
  "generic_to_specific_rewrites": [
    {{
      "generic_phrase": "",
      "specific_replacement": "",
      "evidence": ""
    }}
  ],
  "scaffold_revision_suggestions": [
    {{
      "target": "active_ap_problems|candidate_problem_pool|watchlist|supportive_care|resolved_problems|uncertainties",
      "action": "add|remove|downgrade|merge|revise|preserve",
      "item": "",
      "reason": ""
    }}
  ],
  "coverage_guard": {{
    "candidate_supported_concept_count_estimate": 0,
    "concepts_at_risk_of_being_lost": [],
    "overall_decision": "accept|revise_minor|revise_major"
  }}
}}

Keep the JSON concise:
- At most 12 supported_concepts_to_preserve.
- At most 10 missing_high_confidence_concepts_to_add.
- At most 10 unsupported_or_stale_concepts_to_remove_or_downgrade.
- At most 8 generic_to_specific_rewrites.
- At most 10 scaffold_revision_suggestions.
- Each evidence array may contain at most 2 short quotes.

Latest previous A&P:
{previous_note}

Today's raw EHR:
{case_inputs["current_ehr"]}

Candidate scaffold:
{scaffold_str}

Candidate A&P:
{candidate_ap}
"""


def build_revise_scaffold_prompt(scaffold: dict, judge: dict, case_inputs: dict, args: argparse.Namespace) -> str:
    if args.prompt_version == "v2_judge_cui_guard":
        return build_revise_scaffold_prompt_cui_guard(scaffold, judge, case_inputs)

    scaffold_str = json.dumps(scaffold, ensure_ascii=False, indent=2)
    judge_str = json.dumps(judge, ensure_ascii=False, indent=2)
    schema_hint = "V2" if args.prompt_version == "v2" else "V1"
    return f"""Revise the memory-gated scaffold using the generation-time judge feedback.

Keep the same {schema_hint} scaffold schema and return only valid JSON. Do not
write the final A&P. Do not use gold current-day A&P.

Revision rules:
- Remove or downgrade unsupported active problems.
- Preserve unresolved carried-forward major problems unless today's raw EHR
  clearly resolves them.
- Add missing evidence-supported updates only when today's raw EHR supports
  them.
- Put isolated labs, single medication administrations, prophylaxis, nutrition,
  access, routine monitoring, generic risk, and low-confidence items into
  watchlist/supportive_care unless they satisfy the general promotion gate.
- Keep active_ap_problems at most 6.
- For V2, preserve carry_forward_major_headings and candidate_problem_pool as
  recall support, but only active_ap_problems with appropriate section_role
  should drive main A&P sections.

Original scaffold:
{scaffold_str}

Judge feedback:
{judge_str}

Today's raw EHR:
{case_inputs["current_ehr"]}
"""


def build_revise_scaffold_prompt_cui_guard(scaffold: dict, judge: dict, case_inputs: dict) -> str:
    scaffold_str = json.dumps(scaffold, ensure_ascii=False, indent=2)
    judge_str = json.dumps(judge, ensure_ascii=False, indent=2)
    return f"""Revise the V2 memory-gated scaffold using the coverage-preserving judge feedback.

Return only valid JSON in the same V2 scaffold schema. Do not write the final
A&P. Do not use gold current-day A&P.

Coverage-preserving revision rules:
- Preserve every item in supported_concepts_to_preserve unless today's raw EHR
  clearly contradicts it.
- Add every item in missing_high_confidence_concepts_to_add when supported by
  today's raw EHR.
- Remove or downgrade only concepts listed in
  unsupported_or_stale_concepts_to_remove_or_downgrade, and follow the required
  handling. If required_handling is uncertain_clarify_indication, keep the
  concept in watchlist/supportive_care or a related active problem with explicit
  uncertainty.
- Do not delete documented medication/procedure/device/infusion/culture concepts
  solely because the indication is unclear; preserve them with a clarify plan.
- Do not replace specific concepts with generic headings. Apply
  generic_to_specific_rewrites.
- Put important supported concepts that should not be standalone sections into
  candidate_problem_pool, watchlist, or supportive_care so the final generator
  can still mention them.
- Keep active_ap_problems at most 8. Prefer merging related concepts over
  deleting supported concepts.

Original scaffold:
{scaffold_str}

Coverage-preserving judge feedback:
{judge_str}

Today's raw EHR:
{case_inputs["current_ehr"]}
"""


def generate_text(prompt: str, args: argparse.Namespace) -> str:
    return call_deepseek(
        prompt=prompt,
        model=args.model,
        api_url=args.api_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
        api_key_env=args.api_key_env,
    )


def count_rejected_by_type(scaffold: dict, evidence_type: str) -> int:
    if not isinstance(scaffold, dict):
        return 0
    rejected = scaffold.get("rejected_candidates", [])
    if not isinstance(rejected, list):
        return 0
    return sum(1 for item in rejected if isinstance(item, dict) and item.get("evidence_type") == evidence_type)


def count_prior_decisions(scaffold: dict, decision: str) -> int:
    if not isinstance(scaffold, dict):
        return 0
    states = scaffold.get("prior_problem_states", [])
    if not isinstance(states, list):
        return 0
    return sum(
        1
        for item in states
        if isinstance(item, dict) and item.get("carry_forward_decision") == decision
    )


def generate_case(args: argparse.Namespace, admission_id: str, day: int) -> dict:
    out_dir = Path(args.output_dir) / args.config_name
    scaffold_dir = Path(args.output_dir) / "scaffolds" / args.config_name
    judge_dir = Path(args.output_dir) / "generation_judges" / args.config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    case_file = out_dir / f"{admission_id}_day{day}.txt"
    scaffold_file = scaffold_dir / f"{admission_id}_day{day}.json"
    candidate_file = out_dir / f"{admission_id}_day{day}.candidate.txt"
    judge_file = judge_dir / f"{admission_id}_day{day}.json"
    revised_scaffold_file = scaffold_dir / f"{admission_id}_day{day}.revised.json"

    case_inputs = read_case_inputs(args, admission_id, day)
    scaffold = load_or_run_json(scaffold_file, lambda: call_json(build_scaffold_prompt(case_inputs, args), args))

    if case_file.exists():
        generated = case_file.read_text(encoding="utf-8")
    else:
        if args.use_judge_revise:
            if candidate_file.exists():
                candidate = candidate_file.read_text(encoding="utf-8")
            else:
                candidate = generate_text(build_generation_prompt(case_inputs, scaffold, args), args)
                candidate_file.write_text(candidate, encoding="utf-8")
            judge = load_or_run_json(
                judge_file,
                lambda: call_json(
                    build_transition_judge_prompt(case_inputs, scaffold, candidate, args),
                    args,
                    max_tokens=args.judge_max_tokens,
                ),
            )
            revised_scaffold = load_or_run_json(
                revised_scaffold_file,
                lambda: call_json(
                    build_revise_scaffold_prompt(scaffold, judge, case_inputs, args),
                    args,
                    max_tokens=args.revise_max_tokens,
                ),
            )
            generated = generate_text(build_generation_prompt(case_inputs, revised_scaffold, args), args)
            scaffold = revised_scaffold
        else:
            generated = generate_text(build_generation_prompt(case_inputs, scaffold, args), args)
        case_file.write_text(generated, encoding="utf-8")

    active_count = len(scaffold.get("active_ap_problems", [])) if isinstance(scaffold, dict) else 0
    watchlist_count = len(scaffold.get("watchlist", [])) if isinstance(scaffold, dict) else 0
    supportive_count = len(scaffold.get("supportive_care", [])) if isinstance(scaffold, dict) else 0
    candidate_count = len(scaffold.get("candidate_problem_pool", [])) if isinstance(scaffold, dict) else 0
    carry_count = len(scaffold.get("carry_forward_major_headings", [])) if isinstance(scaffold, dict) else 0
    must_cover_count = len(scaffold.get("must_cover_concepts", [])) if isinstance(scaffold, dict) else 0
    active_state_audit_count = len(scaffold.get("active_state_audit", [])) if isinstance(scaffold, dict) else 0
    unsupported_or_stale_count = (
        len(scaffold.get("unsupported_or_stale_concepts", [])) if isinstance(scaffold, dict) else 0
    )
    evidence_event_count = len(scaffold.get("evidence_events", [])) if isinstance(scaffold, dict) else 0
    prior_problem_state_count = len(scaffold.get("prior_problem_states", [])) if isinstance(scaffold, dict) else 0
    must_carry_forward_count = len(scaffold.get("must_carry_forward_sections", [])) if isinstance(scaffold, dict) else 0
    rejected_count = len(scaffold.get("rejected_candidates", [])) if isinstance(scaffold, dict) else 0
    contradiction_count = len(scaffold.get("contradiction_audit", [])) if isinstance(scaffold, dict) else 0
    disposition_and_goals_present = (
        int(any(scaffold.get("disposition_and_goals", {}).values()))
        if isinstance(scaffold, dict) and isinstance(scaffold.get("disposition_and_goals"), dict)
        else 0
    )
    low_confidence_active_count = (
        sum(
            1
            for item in scaffold.get("active_ap_problems", [])
            if isinstance(item, dict) and item.get("confidence") == "low"
        )
        if isinstance(scaffold, dict)
        else 0
    )
    return {
        "config": args.config_name,
        "admission_id": admission_id,
        "day": day,
        "baseline_run_name": args.baseline_run_name,
        "baseline_setting": args.baseline_setting,
        "baseline_method": args.baseline_method,
        "memory_source": args.memory_source,
        "method": (
            f"memory_gated_scaffold_with_judge_revise_{args.prompt_version}"
            if args.use_judge_revise
            else f"memory_gated_scaffold_no_judge_{args.prompt_version}"
        ),
        "prompt_version": args.prompt_version,
        "generation_time_judge_revise": bool(args.use_judge_revise),
        "baseline_rouge_l": rouge_l_f1(case_inputs["baseline_note"], case_inputs["gold_note"]),
        "augmented_rouge_l": rouge_l_f1(generated, case_inputs["gold_note"]),
        "rouge_delta": rouge_l_f1(generated, case_inputs["gold_note"])
        - rouge_l_f1(case_inputs["baseline_note"], case_inputs["gold_note"]),
        "baseline_words": len(case_inputs["baseline_note"].split()),
        "augmented_words": len(generated.split()),
        "gold_words": len(case_inputs["gold_note"].split()),
        "active_ap_problem_count": active_count,
        "carry_forward_heading_count": carry_count,
        "candidate_problem_count": candidate_count,
        "must_cover_concept_count": must_cover_count,
        "active_state_audit_count": active_state_audit_count,
        "unsupported_or_stale_concept_count": unsupported_or_stale_count,
        "watchlist_count": watchlist_count,
        "supportive_care_count": supportive_count,
        "evidence_event_count": evidence_event_count,
        "prior_problem_state_count": prior_problem_state_count,
        "must_carry_forward_section_count": must_carry_forward_count,
        "disposition_and_goals_present": disposition_and_goals_present,
        "rejected_candidate_count": rejected_count,
        "contradiction_count": contradiction_count,
        "low_confidence_active_problem_count": low_confidence_active_count,
        "med_admin_only_rejected_count": count_rejected_by_type(scaffold, "med_admin_only"),
        "isolated_lab_rejected_count": count_rejected_by_type(scaffold, "isolated_lab_only"),
        "previous_claim_revised_count": count_prior_decisions(scaffold, "revise"),
        "previous_claim_dropped_count": count_prior_decisions(scaffold, "drop"),
        "scaffold_path": str(scaffold_file),
        "revised_scaffold_path": str(revised_scaffold_file) if args.use_judge_revise else "",
        "generation_judge_path": str(judge_file) if args.use_judge_revise else "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AP with memory-gated scaffold augmentation.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--problem-state-dir", default="outputs/ap_problem_state_fair")
    parser.add_argument("--output-dir", default="outputs/ap_memory_gated_scaffold")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--cases-file", default="")
    parser.add_argument("--baseline-run-name", required=True)
    parser.add_argument("--baseline-setting", choices=["gt", "gen"], required=True)
    parser.add_argument("--baseline-method", choices=["method1", "method2", "method-1"], required=True)
    parser.add_argument("--memory-source", choices=["gold", "baseline_method", "none"], required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--scaffold-max-tokens", type=int, default=3200)
    parser.add_argument("--judge-max-tokens", type=int, default=3200)
    parser.add_argument("--revise-max-tokens", type=int, default=4200)
    parser.add_argument(
        "--prompt-version",
        choices=["v1", "v2", "v2_cui_recall", "v2_judge_cui_guard", "v3"],
        default="v1",
    )
    parser.add_argument("--use-judge-revise", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel patient-day API workers. Keep 1 for deterministic sequential runs.",
    )
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--parse-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_values = args.cases
    if args.cases_file:
        case_values = [
            line.strip()
            for line in Path(args.cases_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    cases = parse_cases(case_values)
    rows = []

    if args.workers <= 1:
        for admission_id, day in cases:
            print(f"Generating {args.config_name}: {admission_id} day {day}")
            rows.append(generate_case(args, admission_id, day))
    else:
        print(f"Generating {args.config_name} with {args.workers} workers over {len(cases)} cases")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_case = {
                executor.submit(generate_case, args, admission_id, day): (admission_id, day)
                for admission_id, day in cases
            }
            for future in as_completed(future_to_case):
                admission_id, day = future_to_case[future]
                try:
                    row = future.result()
                except Exception as exc:
                    print(f"Failed {args.config_name}: {admission_id} day {day}: {exc}")
                    raise
                rows.append(row)
                print(f"Completed {args.config_name}: {admission_id} day {day}")

        case_order = {case: idx for idx, case in enumerate(cases)}
        rows.sort(key=lambda row: case_order[(str(row["admission_id"]), int(row["day"]))])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.config_name}_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
