"""
AP-Memory no-training experiment.

This runner compares four A&P generation settings on the same day-level AP
samples used by ProblemFlow-AP:

  direct              Today's EHR evidence only.
  history_ap          Previous day's A&P plus today's EHR evidence.
  ap_memory_no_judge  Patient memory JSON plus previous A&P and today's EHR.
  ap_memory           Same as above, with a judge/reviser loop that checks
                      whether day-to-day A&P changes are supported by today's
                      evidence.

The experiment is intentionally no-training: an external structured memory is
updated across days, while the LLM acts as a state transition function. A mock
mode is included for smoke tests and plumbing checks; use --llm deepseek for the
actual no-training LLM experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from problemflow_ap import (  # noqa: E402
    DEFAULT_API_URL,
    build_samples,
    call_deepseek,
    detect_problem_ids,
    evidence_agent,
    f1_sets,
    normalize_text,
    read_jsonl,
    rouge_l_f1,
    verifier_agent,
    write_jsonl,
)


DEFAULT_OUTDIR = Path("experiments/problemflow_ap/outputs_ap_memory")
METHODS = [
    "direct",
    "history_ap",
    "ap_memory_no_judge",
    "ap_memory",
    "ap_memory_v2_no_judge",
    "ap_memory_v2",
]
MEMORY_SCHEMA = {
    "global_status": {
        "overall_trajectory": "unknown",
        "current_severity": "unknown",
        "one_sentence_summary": "",
    },
    "active_problems": [],
    "resolved_problems": [],
    "key_events_today": [],
    "interventions_today": [],
    "treatment_response": [],
    "risks": [],
    "uncertainties": [],
    "judge_feedback": [],
}
MEMORY_SCHEMA_V2 = {
    "global_status": {
        "overall_trajectory": "unknown",
        "current_severity": "unknown",
        "one_sentence_summary": "",
    },
    "active_ap_problems": [],
    "watchlist": [],
    "supportive_care": [],
    "resolved_problems": [],
    "key_events_today": [],
    "treatment_response": [],
    "uncertainties": [],
    "judge_feedback": [],
}

MOCK_SUPPORTIVE_PROBLEM_IDS = {"fen_nutrition", "prophylaxis_access", "heme"}


def parse_json_response(text: str) -> Any:
    cleaned = str(text or "").strip()
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


def dump_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def evidence_blob(sample: dict, max_items: int) -> str:
    events = sample.get("ehr_events_before_cutoff", [])[:max_items]
    lines = []
    for item in events:
        rel_time = normalize_text(item.get("rel_time", ""))
        text = normalize_text(item.get("text", ""))
        if not text:
            continue
        prefix = f"[{rel_time}] " if rel_time else ""
        lines.append(f"- {prefix}{text}")
    return "\n".join(lines)


def compact_evidence_for_memory(sample: dict, max_items: int) -> list[dict]:
    evidence = evidence_agent(sample)[:max_items]
    return [
        {
            "evidence_id": item.get("evidence_id", ""),
            "text": item.get("text", ""),
            "rel_time": item.get("rel_time", ""),
            "problem_ids": item.get("problem_ids", []),
            "trend": item.get("trend", "unknown"),
        }
        for item in evidence
    ]


def call_text(prompt: str, args: argparse.Namespace, max_tokens: int | None = None, system: str | None = None) -> str:
    return call_deepseek(
        prompt=prompt,
        model=args.model,
        api_url=args.api_url,
        temperature=args.temperature,
        max_tokens=max_tokens or args.max_tokens,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
        system=system or "You are an experienced ICU attending. Be concise, evidence-grounded, and clinically precise.",
    )


def call_json(prompt: str, args: argparse.Namespace, max_tokens: int | None = None, system: str | None = None) -> Any:
    return parse_json_response(call_text(prompt, args, max_tokens=max_tokens, system=system))


def prompt_direct(sample: dict, args: argparse.Namespace) -> str:
    prompt = f"""Write today's ICU Assessment & Plan using only today's available pre-A&P information.

Do not use the gold A&P. Do not invent diagnoses, treatments, or outcomes not
supported by the provided evidence. If information is sparse, write a concise
problem-oriented A&P with explicit uncertainty.

Current note context before A&P:
{sample.get("current_note_context", "")}

Today's EHR evidence:
{evidence_blob(sample, args.max_evidence_items)}

Output format:
Assessment:
...

Plan:
...
"""
    return call_text(prompt, args, max_tokens=args.generation_max_tokens)


def prompt_history_ap(sample: dict, previous_ap: str, args: argparse.Namespace) -> str:
    prompt = f"""Write today's ICU Assessment & Plan from the previous day's A&P and today's new EHR evidence.

Rules:
- Treat the previous A&P as historical context, not as today's truth.
- Carry forward unresolved active problems unless today's evidence supports resolution.
- Any day-to-day change must be supported by today's evidence.
- Do not invent diagnoses, treatments, or outcomes.

Previous day's A&P:
{previous_ap or "None available."}

Current note context before A&P:
{sample.get("current_note_context", "")}

Today's EHR evidence:
{evidence_blob(sample, args.max_evidence_items)}

Output format:
Assessment:
...

Plan:
...
"""
    return call_text(prompt, args, max_tokens=args.generation_max_tokens)


def prompt_memory_update(sample: dict, previous_ap: str, memory: dict, args: argparse.Namespace) -> dict:
    prompt = f"""Update this patient's structured progression memory.

You are given yesterday's A&P, the previous patient memory JSON, and today's
new EHR evidence. Perform an incremental state transition:
M_t = Update(M_(t-1), AP_(t-1), X_t).

Rules:
1. Preserve unresolved clinically relevant history.
2. Add or change only what is supported by today's evidence or the prior memory.
3. Move problems to resolved only with evidence of resolution or completed plan.
4. Mark uncertainty explicitly instead of guessing.
5. Keep JSON compact and valid.

Return only valid JSON with this schema:
{dump_json(MEMORY_SCHEMA)}

Previous A&P:
{previous_ap or "None available."}

Previous memory JSON:
{dump_json(memory)}

Today's compact evidence:
{dump_json(compact_evidence_for_memory(sample, args.max_memory_evidence_items))}
"""
    result = call_json(prompt, args, max_tokens=args.memory_max_tokens)
    return normalize_memory(result)


def prompt_memory_update_v2(sample: dict, previous_ap: str, memory: dict, args: argparse.Namespace) -> dict:
    prompt = f"""Update this patient's structured progression memory using a general A&P problem-promotion gate.

You are given yesterday's A&P, the previous patient memory JSON, and today's
new EHR evidence. Perform an incremental state transition:
M_t = Update(M_(t-1), AP_(t-1), X_t).

Do not optimize for a single disease or case. Apply these general rules:
1. active_ap_problems are only problems that should receive their own A&P section today:
   - a prior A&P problem that remains unresolved, OR
   - a new/changed diagnosis, syndrome, severity, treatment decision, disposition blocker, or major complication supported by today's evidence.
2. watchlist stores abnormalities, risks, uncertain diagnoses, and trends that should be remembered but should not necessarily become A&P sections.
3. supportive_care stores routine ICU care, nutrition, access, prophylaxis, sedation, and monitoring unless they are a primary active problem today.
4. Do not promote an isolated lab abnormality, single medication administration, prophylaxis item, or generic risk into active_ap_problems unless it changes today's assessment or plan.
5. Prefer the granularity of the prior A&P and gold-style ICU notes: concise major problems, not an exhaustive checklist.
6. Keep at most 6 active_ap_problems. Put lower-priority carried history into watchlist/supportive_care.
7. Move problems to resolved only with evidence of resolution or completed treatment.
8. Return compact valid JSON only.

Return only valid JSON with this schema:
{dump_json(MEMORY_SCHEMA_V2)}

Previous A&P:
{previous_ap or "None available."}

Previous memory JSON:
{dump_json(memory)}

Today's compact evidence:
{dump_json(compact_evidence_for_memory(sample, args.max_memory_evidence_items))}
"""
    result = call_json(prompt, args, max_tokens=args.memory_max_tokens)
    return normalize_memory_v2(result)


def prompt_generate_from_memory(sample: dict, memory: dict, args: argparse.Namespace) -> str:
    prompt = f"""Write today's ICU Assessment & Plan from this evidence-grounded progression memory.

Rules:
- Use the memory as the patient state.
- Use today's EHR only to support today's changes and plans.
- Preserve carried-forward active problems when today's evidence is sparse.
- Do not invent unsupported treatments, diagnoses, or outcomes.
- Be concise and problem-oriented.

Progression memory JSON:
{dump_json(memory)}

Today's EHR evidence:
{evidence_blob(sample, args.max_evidence_items)}

Output format:
Assessment:
...

Plan:
...
"""
    return call_text(prompt, args, max_tokens=args.generation_max_tokens)


def prompt_generate_from_memory_v2(sample: dict, memory: dict, args: argparse.Namespace) -> str:
    prompt = f"""Write today's ICU Assessment & Plan from this V2 progression memory.

Use the memory as a patient state, but apply the same problem-promotion gate:
- Write A&P sections only for active_ap_problems.
- Do not turn watchlist or supportive_care items into separate sections unless today's evidence shows they became a major active problem.
- Routine prophylaxis, nutrition, access, and generic monitoring should be compactly folded into ICU Care when needed.
- Match the granularity of a real ICU A&P: concise major problems and plans, not an exhaustive safety checklist.
- Preserve unresolved prior A&P problems, but do not expand every historical risk into today's A&P.
- Use today's EHR only to support today's changes and plans.
- Do not invent unsupported treatments, diagnoses, or outcomes.

V2 progression memory JSON:
{dump_json(memory)}

Today's EHR evidence:
{evidence_blob(sample, args.max_evidence_items)}

Output format:
Assessment:
...

Plan:
...
"""
    return call_text(prompt, args, max_tokens=args.generation_max_tokens)


def prompt_judge(previous_ap: str, candidate_ap: str, sample: dict, previous_memory: dict, args: argparse.Namespace) -> dict:
    prompt = f"""Judge whether the day-to-day A&P changes are supported by today's evidence.

Compare the previous A&P and today's candidate A&P. Identify:
- supported_changes: changes that are explained by today's EHR evidence
- unsupported_changes: new, removed, resolved, or changed claims without support
- missing_updates: important today's evidence missing from the candidate A&P
- forgotten_active_problems: prior active problems that disappeared without evidence
- memory_revision_suggestions: concrete edits for the progression memory

Return only valid JSON:
{{
  "supported_changes": [
    {{"change": "", "evidence": [], "target_memory_update": ""}}
  ],
  "unsupported_changes": [
    {{"change": "", "reason": "", "suggested_revision": ""}}
  ],
  "missing_updates": [
    {{"evidence": "", "should_update": ""}}
  ],
  "forgotten_active_problems": [],
  "memory_revision_suggestions": [],
  "summary": {{
    "supported_change_count": 0,
    "unsupported_change_count": 0,
    "missing_update_count": 0,
    "forgotten_active_problem_count": 0
  }}
}}

Previous A&P:
{previous_ap or "None available."}

Previous memory JSON:
{dump_json(previous_memory)}

Today's EHR evidence:
{evidence_blob(sample, args.max_evidence_items)}

Today's candidate A&P:
{candidate_ap}
"""
    response = call_text(prompt, args, max_tokens=args.judge_max_tokens)
    try:
        result = parse_json_response(response)
    except (json.JSONDecodeError, ValueError):
        result = mock_judge(previous_ap, candidate_ap, sample, previous_memory)
        result["raw_judge_response"] = response
        result["parse_error"] = "judge_json_parse_failed"
    if not isinstance(result, dict):
        result = {"raw_judge": result}
    return result


def prompt_revise_memory(candidate_memory: dict, judge: dict, args: argparse.Namespace) -> dict:
    prompt = f"""Revise the candidate progression memory using the judge feedback.

Rules:
- Remove or downgrade unsupported changes.
- Add missing evidence-supported updates.
- Preserve important active problems if their resolution was unsupported.
- Keep the same compact JSON schema.
- Return only valid JSON.

Candidate memory:
{dump_json(candidate_memory)}

Judge feedback:
{dump_json(judge)}
"""
    response = call_text(prompt, args, max_tokens=args.memory_max_tokens)
    try:
        result = parse_json_response(response)
    except (json.JSONDecodeError, ValueError):
        fallback = normalize_memory(candidate_memory)
        fallback["judge_feedback"] = [judge]
        fallback["memory_revision_parse_error"] = "revision_json_parse_failed"
        return fallback
    return normalize_memory(result)


def prompt_revise_memory_v2(candidate_memory: dict, judge: dict, args: argparse.Namespace) -> dict:
    prompt = f"""Revise the candidate V2 progression memory using the judge feedback.

Use a general problem-promotion gate; do not overfit to any single case:
- Remove or downgrade unsupported active_ap_problems.
- Add missing evidence-supported updates, but place them in watchlist/supportive_care unless they meet active_ap_problems criteria.
- Preserve unresolved prior A&P problems, but keep low-priority history as watchlist when it should not receive its own section today.
- Routine ICU care, prophylaxis, nutrition, access, single medication administrations, and isolated lab abnormalities should not become active_ap_problems unless they change today's assessment or plan.
- Keep at most 6 active_ap_problems.
- Return only valid JSON with the same V2 schema.

Candidate V2 memory:
{dump_json(candidate_memory)}

Judge feedback:
{dump_json(judge)}
"""
    response = call_text(prompt, args, max_tokens=args.memory_max_tokens)
    try:
        result = parse_json_response(response)
    except (json.JSONDecodeError, ValueError):
        fallback = normalize_memory_v2(candidate_memory)
        fallback["judge_feedback"] = [judge]
        fallback["memory_revision_parse_error"] = "revision_json_parse_failed"
        return fallback
    return normalize_memory_v2(result)


def normalize_memory(memory: Any) -> dict:
    if not isinstance(memory, dict):
        memory = {}
    normalized = json.loads(json.dumps(MEMORY_SCHEMA))
    for key, value in memory.items():
        normalized[key] = value
    for key in [
        "active_problems",
        "resolved_problems",
        "key_events_today",
        "interventions_today",
        "treatment_response",
        "risks",
        "uncertainties",
        "judge_feedback",
    ]:
        if not isinstance(normalized.get(key), list):
            normalized[key] = []
    if not isinstance(normalized.get("global_status"), dict):
        normalized["global_status"] = dict(MEMORY_SCHEMA["global_status"])
    return normalized


def normalize_memory_v2(memory: Any) -> dict:
    if not isinstance(memory, dict):
        memory = {}
    normalized = json.loads(json.dumps(MEMORY_SCHEMA_V2))
    for key, value in memory.items():
        normalized[key] = value
    for key in [
        "active_ap_problems",
        "watchlist",
        "supportive_care",
        "resolved_problems",
        "key_events_today",
        "treatment_response",
        "uncertainties",
        "judge_feedback",
    ]:
        if not isinstance(normalized.get(key), list):
            normalized[key] = []
    if not isinstance(normalized.get("global_status"), dict):
        normalized["global_status"] = dict(MEMORY_SCHEMA_V2["global_status"])
    return normalized


def mock_direct(sample: dict) -> str:
    evidence = evidence_agent(sample)
    problem_ids = sorted({pid for item in evidence for pid in item.get("problem_ids", [])})
    return mock_ap_from_problem_ids(problem_ids, evidence)


def mock_history_ap(sample: dict, previous_ap: str) -> str:
    evidence = evidence_agent(sample)
    problem_ids = set(detect_problem_ids(previous_ap))
    problem_ids.update(pid for item in evidence for pid in item.get("problem_ids", []))
    return mock_ap_from_problem_ids(sorted(problem_ids), evidence)


def mock_memory_update(sample: dict, previous_ap: str, memory: dict) -> dict:
    evidence = evidence_agent(sample)
    problem_ids = set(detect_problem_ids(previous_ap))
    for problem in memory.get("active_problems", []):
        if isinstance(problem, dict):
            problem_ids.update(detect_problem_ids(problem.get("problem", "")))
    problem_ids.update(pid for item in evidence for pid in item.get("problem_ids", []))
    updated = normalize_memory(memory)
    updated["active_problems"] = [
        {
            "problem": pid,
            "status": "active",
            "assessment": f"{pid} remains relevant to today's A&P.",
            "today_evidence": [
                item["text"][:180]
                for item in evidence
                if pid in item.get("problem_ids", [])
            ][:2],
            "plan": ["continue evidence-guided monitoring and treatment"],
            "confidence": "medium",
        }
        for pid in sorted(problem_ids)
    ]
    updated["key_events_today"] = [item["text"][:180] for item in evidence[:5]]
    updated["global_status"] = {
        "overall_trajectory": "unknown",
        "current_severity": "unknown",
        "one_sentence_summary": "Mock progression memory updated from previous A&P and today's evidence.",
    }
    return updated


def mock_memory_update_v2(sample: dict, previous_ap: str, memory: dict) -> dict:
    evidence = evidence_agent(sample)
    problem_ids = set(detect_problem_ids(previous_ap))
    for problem in memory.get("active_ap_problems", []):
        if isinstance(problem, dict):
            problem_ids.update(detect_problem_ids(problem.get("problem", "") or problem.get("name", "")))
    problem_ids.update(pid for item in evidence for pid in item.get("problem_ids", []))
    active_ids = sorted(pid for pid in problem_ids if pid not in MOCK_SUPPORTIVE_PROBLEM_IDS)[:6]
    supportive_ids = sorted(pid for pid in problem_ids if pid in MOCK_SUPPORTIVE_PROBLEM_IDS)
    updated = normalize_memory_v2(memory)
    updated["active_ap_problems"] = [
        {
            "problem": pid,
            "status": "active",
            "assessment": f"{pid} is a major active A&P problem or carried-forward unresolved problem.",
            "today_evidence": [
                item["text"][:180]
                for item in evidence
                if pid in item.get("problem_ids", [])
            ][:2],
            "plan": ["continue evidence-guided management"],
            "confidence": "medium",
        }
        for pid in active_ids
    ]
    updated["supportive_care"] = [{"item": pid, "rationale": "routine/supportive ICU care or lower-priority abnormality"} for pid in supportive_ids]
    updated["watchlist"] = []
    updated["key_events_today"] = [item["text"][:180] for item in evidence[:5]]
    updated["global_status"] = {
        "overall_trajectory": "unknown",
        "current_severity": "unknown",
        "one_sentence_summary": "Mock V2 memory updated with major A&P problems separated from watchlist/supportive care.",
    }
    return updated


def mock_ap_from_memory(memory: dict, sample: dict) -> str:
    problem_ids = []
    for problem in memory.get("active_problems", []):
        if isinstance(problem, dict):
            problem_ids.extend(detect_problem_ids(problem.get("problem", "")))
        elif isinstance(problem, str):
            problem_ids.extend(detect_problem_ids(problem))
    return mock_ap_from_problem_ids(sorted(set(problem_ids)), evidence_agent(sample))


def mock_ap_from_memory_v2(memory: dict, sample: dict) -> str:
    problem_ids = []
    for problem in memory.get("active_ap_problems", []):
        if isinstance(problem, dict):
            problem_ids.extend(detect_problem_ids(problem.get("problem", "") or problem.get("name", "")))
        elif isinstance(problem, str):
            problem_ids.extend(detect_problem_ids(problem))
    return mock_ap_from_problem_ids(sorted(set(problem_ids)), evidence_agent(sample))


def mock_ap_from_problem_ids(problem_ids: list[str], evidence: list[dict]) -> str:
    if not problem_ids:
        return "Assessment:\nNo specific active problem could be confidently inferred.\n\nPlan:\nContinue close monitoring and reassess with new data."
    lines = ["Assessment:"]
    for pid in problem_ids:
        supporting = [item["text"] for item in evidence if pid in item.get("problem_ids", [])]
        summary = supporting[0][:160] if supporting else "carried forward from prior A&P"
        lines.append(f"- {pid}: {summary}")
    lines.append("\nPlan:")
    for pid in problem_ids:
        lines.append(f"- {pid}: continue evidence-guided management and monitoring.")
    return "\n".join(lines)


def mock_judge(previous_ap: str, candidate_ap: str, sample: dict, previous_memory: dict) -> dict:
    previous_probs = set(detect_problem_ids(previous_ap))
    candidate_probs = set(detect_problem_ids(candidate_ap))
    evidence_probs = {pid for item in evidence_agent(sample) for pid in item.get("problem_ids", [])}
    unsupported_new = sorted((candidate_probs - previous_probs) - evidence_probs)
    forgotten = sorted(previous_probs - candidate_probs)
    summary = {
        "supported_change_count": len((candidate_probs - previous_probs) & evidence_probs),
        "unsupported_change_count": len(unsupported_new),
        "missing_update_count": len(evidence_probs - candidate_probs),
        "forgotten_active_problem_count": len(forgotten),
    }
    return {
        "supported_changes": [],
        "unsupported_changes": [{"change": pid, "reason": "not seen in today evidence", "suggested_revision": "downgrade or remove"} for pid in unsupported_new],
        "missing_updates": [{"evidence": pid, "should_update": "consider in active problems"} for pid in sorted(evidence_probs - candidate_probs)],
        "forgotten_active_problems": forgotten,
        "memory_revision_suggestions": [],
        "summary": summary,
    }


def revise_memory_with_mock_judge(memory: dict, judge: dict) -> dict:
    revised = normalize_memory(memory)
    unsupported = {
        item.get("change")
        for item in judge.get("unsupported_changes", [])
        if isinstance(item, dict) and item.get("change")
    }
    if unsupported:
        revised["active_problems"] = [
            problem
            for problem in revised.get("active_problems", [])
            if not (isinstance(problem, dict) and problem.get("problem") in unsupported)
        ]
    revised["judge_feedback"] = [judge]
    return revised


def revise_memory_v2_with_mock_judge(memory: dict, judge: dict) -> dict:
    revised = normalize_memory_v2(memory)
    unsupported = {
        item.get("change")
        for item in judge.get("unsupported_changes", [])
        if isinstance(item, dict) and item.get("change")
    }
    if unsupported:
        kept = []
        downgraded = revised.get("watchlist", [])
        for problem in revised.get("active_ap_problems", []):
            name = problem.get("problem") if isinstance(problem, dict) else str(problem)
            if name in unsupported:
                downgraded.append(problem)
            else:
                kept.append(problem)
        revised["active_ap_problems"] = kept
        revised["watchlist"] = downgraded
    revised["judge_feedback"] = [judge]
    return revised


def generate(args: argparse.Namespace) -> Path:
    samples = read_jsonl(Path(args.samples))
    out_dir = Path(args.outdir)
    generation_dir = out_dir / "generations"
    memory_dir = out_dir / "memories"
    judge_dir = out_dir / "judges"
    generation_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    judge_dir.mkdir(parents=True, exist_ok=True)

    sample_order = [s for s in sorted(samples, key=lambda row: (row["hadm_id"], int(row["day"])))]
    if not args.allow_leakage_warnings:
        sample_order = [s for s in sample_order if s.get("leakage_check_pass", False)]

    methods = METHODS if args.method == "all" else [args.method]
    for method in methods:
        outputs = []
        use_v2_memory = method.startswith("ap_memory_v2")
        memories: dict[str, dict] = defaultdict(lambda: normalize_memory_v2({}) if use_v2_memory else normalize_memory({}))
        previous_gold_ap_by_hadm: dict[str, str] = {}
        previous_generated_ap_by_hadm: dict[str, str] = {}
        previous_memory_by_hadm: dict[str, dict] = {}
        for sample in sample_order:
            if args.limit and len(outputs) >= args.limit:
                break
            hadm_id = str(sample["hadm_id"])
            day = int(sample["day"])
            previous_ap = previous_generated_ap_by_hadm.get(hadm_id, "") if args.autoregressive_history else previous_gold_ap_by_hadm.get(hadm_id, "")
            previous_gold_ap = previous_gold_ap_by_hadm.get(hadm_id, "")
            memory_before = normalize_memory_v2(memories[hadm_id]) if use_v2_memory else normalize_memory(memories[hadm_id])
            judge = {}
            candidate_memory = {}
            revised_memory = {}

            if method == "direct":
                generated_ap = prompt_direct(sample, args) if args.llm == "deepseek" else mock_direct(sample)
            elif method == "history_ap":
                generated_ap = prompt_history_ap(sample, previous_ap, args) if args.llm == "deepseek" else mock_history_ap(sample, previous_ap)
            elif method in {"ap_memory_no_judge", "ap_memory", "ap_memory_v2_no_judge", "ap_memory_v2"}:
                use_judge = method in {"ap_memory", "ap_memory_v2"}
                if args.llm == "deepseek":
                    if use_v2_memory:
                        candidate_memory = prompt_memory_update_v2(sample, previous_ap, memory_before, args)
                        candidate_ap = prompt_generate_from_memory_v2(sample, candidate_memory, args)
                    else:
                        candidate_memory = prompt_memory_update(sample, previous_ap, memory_before, args)
                        candidate_ap = prompt_generate_from_memory(sample, candidate_memory, args)
                    if use_judge:
                        judge = prompt_judge(previous_ap, candidate_ap, sample, memory_before, args)
                        if use_v2_memory:
                            revised_memory = prompt_revise_memory_v2(candidate_memory, judge, args)
                            generated_ap = prompt_generate_from_memory_v2(sample, revised_memory, args)
                        else:
                            revised_memory = prompt_revise_memory(candidate_memory, judge, args)
                            generated_ap = prompt_generate_from_memory(sample, revised_memory, args)
                    else:
                        revised_memory = candidate_memory
                        generated_ap = candidate_ap
                else:
                    if use_v2_memory:
                        candidate_memory = mock_memory_update_v2(sample, previous_ap, memory_before)
                        candidate_ap = mock_ap_from_memory_v2(candidate_memory, sample)
                    else:
                        candidate_memory = mock_memory_update(sample, previous_ap, memory_before)
                        candidate_ap = mock_ap_from_memory(candidate_memory, sample)
                    if use_judge:
                        judge = mock_judge(previous_ap, candidate_ap, sample, memory_before)
                        if use_v2_memory:
                            revised_memory = revise_memory_v2_with_mock_judge(candidate_memory, judge)
                            generated_ap = mock_ap_from_memory_v2(revised_memory, sample)
                        else:
                            revised_memory = revise_memory_with_mock_judge(candidate_memory, judge)
                            generated_ap = mock_ap_from_memory(revised_memory, sample)
                    else:
                        judge = {}
                        revised_memory = candidate_memory
                        generated_ap = candidate_ap
                memories[hadm_id] = normalize_memory_v2(revised_memory) if use_v2_memory else normalize_memory(revised_memory)
            else:
                raise ValueError(method)

            evidence = evidence_agent(sample)
            verification = verifier_agent(generated_ap, evidence)
            row = {
                "sample_id": sample["sample_id"],
                "hadm_id": hadm_id,
                "day": day,
                "method": method,
                "generated_ap": generated_ap,
                "gold_ap": sample["gold_ap"],
                "previous_ap_used": previous_ap,
                "previous_gold_ap": previous_gold_ap,
                "candidate_memory": candidate_memory,
                "revised_memory": revised_memory,
                "judge": judge,
                "detected_problems": sorted(detect_problem_ids(generated_ap)),
                "gold_problems": sorted(detect_problem_ids(sample["gold_ap"])),
                "previous_gold_problems": sorted(detect_problem_ids(previous_gold_ap)),
                "evidence_problems": sorted({pid for item in evidence for pid in item.get("problem_ids", [])}),
                "verification_summary": verification["summary"],
            }
            outputs.append(row)
            previous_gold_ap_by_hadm[hadm_id] = sample["gold_ap"]
            previous_generated_ap_by_hadm[hadm_id] = generated_ap
            previous_memory_by_hadm[hadm_id] = memory_before
            if method.startswith("ap_memory") and args.sleep_between_cases > 0:
                time.sleep(args.sleep_between_cases)

        write_jsonl(generation_dir / f"{method}.jsonl", outputs)
        if method.startswith("ap_memory"):
            memory_rows = [{"hadm_id": hadm_id, "memory": memory} for hadm_id, memory in sorted(memories.items())]
            write_jsonl(memory_dir / f"{method}_latest_memory.jsonl", memory_rows)
            judge_rows = [
                {
                    "sample_id": row["sample_id"],
                    "hadm_id": row["hadm_id"],
                    "day": row["day"],
                    "judge": row.get("judge", {}),
                }
                for row in outputs
                if row.get("judge")
            ]
            write_jsonl(judge_dir / f"{method}_judges.jsonl", judge_rows)
        print(f"Generated {len(outputs)} rows for {method} -> {generation_dir / (method + '.jsonl')}")
    return generation_dir


def safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


def judge_summary(row: dict) -> dict:
    judge = row.get("judge") or {}
    summary = judge.get("summary") if isinstance(judge, dict) else {}
    return summary if isinstance(summary, dict) else {}


def evaluate(args: argparse.Namespace) -> Path:
    generation_dir = Path(args.generation_dir)
    out_path = Path(args.output)
    rows = []
    for gen_path in sorted(generation_dir.glob("*.jsonl")):
        for row in read_jsonl(gen_path):
            pred = normalize_text(row.get("generated_ap", ""))
            gold = normalize_text(row.get("gold_ap", ""))
            pred_probs = set(row.get("detected_problems") or detect_problem_ids(pred))
            gold_probs = set(row.get("gold_problems") or detect_problem_ids(gold))
            previous_gold_probs = set(row.get("previous_gold_problems") or [])
            evidence_probs = set(row.get("evidence_problems") or [])
            carried_gold_probs = previous_gold_probs & gold_probs
            unsupported_pred_probs = pred_probs - gold_probs
            unsupported_change_probs = (pred_probs - previous_gold_probs) - evidence_probs
            forgotten_carried_probs = carried_gold_probs - pred_probs
            p, r, f1 = f1_sets(pred_probs, gold_probs)
            verification = row.get("verification_summary", {})
            summary = judge_summary(row)
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
                    "grounded_claim_rate": verification.get("grounded_claim_rate", 0.0),
                    "unsupported_claim_rate": verification.get("unsupported_claim_rate", 0.0),
                    "unsupported_problem_rate": safe_rate(len(unsupported_pred_probs), len(pred_probs)),
                    "unsupported_change_rate": safe_rate(len(unsupported_change_probs), len(pred_probs - previous_gold_probs)),
                    "forgotten_carried_problem_rate": safe_rate(len(forgotten_carried_probs), len(carried_gold_probs)),
                    "pred_problem_count": len(pred_probs),
                    "gold_problem_count": len(gold_probs),
                    "carried_gold_problem_count": len(carried_gold_probs),
                    "judge_unsupported_change_count": summary.get("unsupported_change_count", math.nan),
                    "judge_missing_update_count": summary.get("missing_update_count", math.nan),
                    "judge_forgotten_active_problem_count": summary.get("forgotten_active_problem_count", math.nan),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["method"]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary_rows = summarize(rows)
    summary_fieldnames = sorted({key for row in summary_rows for key in row.keys()}) if summary_rows else ["method"]
    if "method" in summary_fieldnames:
        summary_fieldnames = ["method", *[key for key in summary_fieldnames if key != "method"]]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote per-sample metrics -> {out_path}")
    print(f"Wrote summary metrics -> {summary_path}")
    print_summary(summary_rows)
    return out_path


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    metric_keys = [
        "rouge_l_f1",
        "problem_f1",
        "problem_precision",
        "problem_recall",
        "grounded_claim_rate",
        "unsupported_claim_rate",
        "unsupported_problem_rate",
        "unsupported_change_rate",
        "forgotten_carried_problem_rate",
    ]
    summary = []
    baseline = grouped.get("direct", [])
    baseline_by_sample = {row["sample_id"]: row for row in baseline}
    for method in METHODS:
        method_rows = grouped.get(method, [])
        if not method_rows:
            continue
        out = {"method": method, "n": len(method_rows)}
        for metric in metric_keys:
            values = [float(row[metric]) for row in method_rows if row.get(metric) == row.get(metric)]
            out[f"{metric}_mean"] = sum(values) / len(values) if values else 0.0
            out[f"{metric}_std"] = (
                math.sqrt(sum((value - out[f"{metric}_mean"]) ** 2 for value in values) / len(values))
                if values
                else 0.0
            )
            paired_deltas = [
                float(row[metric]) - float(baseline_by_sample[row["sample_id"]][metric])
                for row in method_rows
                if row["sample_id"] in baseline_by_sample
                and row.get(metric) == row.get(metric)
                and baseline_by_sample[row["sample_id"]].get(metric) == baseline_by_sample[row["sample_id"]].get(metric)
            ]
            if method != "direct" and paired_deltas:
                out[f"{metric}_delta_vs_direct"] = sum(paired_deltas) / len(paired_deltas)
        summary.append(out)
    return summary


def print_summary(summary_rows: list[dict]) -> None:
    print("\nAP-Memory summary")
    for row in summary_rows:
        line = (
            f"{row['method']}: n={row['n']} "
            f"rouge={row.get('rouge_l_f1_mean', 0)*100:.2f} "
            f"problem_f1={row.get('problem_f1_mean', 0)*100:.2f} "
            f"unsupported_change={row.get('unsupported_change_rate_mean', 0)*100:.2f} "
            f"forgotten={row.get('forgotten_carried_problem_rate_mean', 0)*100:.2f}"
        )
        delta = row.get("problem_f1_delta_vs_direct")
        if isinstance(delta, (int, float)) and delta == delta:
            line += f" problem_f1_delta={delta*100:+.2f}"
        print(line)


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
    gen_args = argparse.Namespace(**vars(args))
    gen_args.samples = str(samples_path)
    generate(gen_args)
    eval_args = argparse.Namespace(
        generation_dir=str(outdir / "generations"),
        output=str(outdir / "metrics" / "ap_memory_metrics.csv"),
    )
    evaluate(eval_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AP-Memory no-training experiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--inputdir", default="data/AP/input")
    build.add_argument("--golddir", default="data/AP/gold")
    build.add_argument("--output", default=str(DEFAULT_OUTDIR / "data" / "ap_samples.jsonl"))
    build.add_argument("--failed-log", default=str(DEFAULT_OUTDIR / "logs" / "failed_ap_extraction.jsonl"))
    build.add_argument("--include-first-note", action="store_true")
    build.add_argument("--keep-full-note", action="store_true")

    gen = subparsers.add_parser("generate")
    add_generation_args(gen)
    gen.add_argument("--samples", default=str(DEFAULT_OUTDIR / "data" / "ap_samples.jsonl"))

    ev = subparsers.add_parser("evaluate")
    ev.add_argument("--generation-dir", default=str(DEFAULT_OUTDIR / "generations"))
    ev.add_argument("--output", default=str(DEFAULT_OUTDIR / "metrics" / "ap_memory_metrics.csv"))

    run = subparsers.add_parser("run-all")
    run.add_argument("--inputdir", default="data/AP/input")
    run.add_argument("--golddir", default="data/AP/gold")
    run.add_argument("--include-first-note", action="store_true")
    add_generation_args(run)

    return parser.parse_args()


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all")
    parser.add_argument("--allow-leakage-warnings", action="store_true")
    parser.add_argument("--autoregressive-history", action="store_true", help="Use the previous generated A&P instead of previous gold A&P as history.")
    parser.add_argument("--llm", choices=["mock", "deepseek"], default="mock")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--generation-max-tokens", type=int, default=1600)
    parser.add_argument("--memory-max-tokens", type=int, default=2600)
    parser.add_argument("--judge-max-tokens", type=int, default=2200)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--sleep-seconds", type=float, default=3.0)
    parser.add_argument("--sleep-between-cases", type=float, default=0.0)
    parser.add_argument("--max-evidence-items", type=int, default=35)
    parser.add_argument("--max-memory-evidence-items", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)


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
