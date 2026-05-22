from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_ANNOTATIONS = Path(r"C:\Users\dsw54\Downloads\verifier_truth_annotations_30cases.md")
DEFAULT_JUDGE_DIR = Path(
    "outputs/ap_memory_gated_scaffold_ap100/generation_judges/ap100full_generated_method2_gen_v2_judge_revise"
)
DEFAULT_OUTDIR = Path("outputs/oracle_claim_verifier_qwen653/manual_verifier_truth_usable")


MANUAL_REWRITES: dict[str, list[dict[str, str]]] = {
    "105351_day13": [
        {
            "match": "Course complicated by acute kidney injury requiring CRRT",
            "corrected": "Course complicated by persistent renal failure requiring intermittent HD/dialysis, not CRRT/CVVHD.",
            "reason": "Current gold A&P says renal failure with HD today; CRRT/CVVHD is not the supported modality.",
        },
        {
            "match": "Hypoxic Respiratory Failure / ARDS with Failed Extubation",
            "corrected": "Hypoxic respiratory failure on mechanical ventilation with failed CPAP/weaning attempt.",
            "reason": "Current-day respiratory note documents CPAP wean with tachypnea/increased work of breathing and return to AC; failed extubation is the wrong wording.",
        },
        {
            "match": "Today",
            "claim_contains_all": ["failed extubation attempt", "tachypnea", "increased work of breathing"],
            "corrected": "Today’s course notable for failed CPAP/weaning attempt with tachypnea and increased work of breathing.",
            "reason": "The failed weaning physiology is supported, but it should not be called a failed extubation attempt.",
        },
        {
            "match": "Failed CPAP trial due to tachypnea and increased WOB",
            "move_to_keep": "true",
            "reason": "Current-day respiratory note states the patient was weaned to CPAP but became tachypneic with marked increased work of breathing and was returned to AC.",
        },
        {
            "match": "Continue AC ventilation; PEEP weaned to 5",
            "corrected": "Continue AC ventilation after failed CPAP/weaning attempt; PEEP was adjusted during the day and reached 5 cmH2O.",
            "reason": "PEEP changes and return to AC are supported, but the exact oxygenation target should be written cautiously.",
        },
    ],
    "128729_day13": [
        {
            "match": "76-year-old male with RA",
            "move_to_keep": "true",
            "reason": "Current gold A&P explicitly describes a 76-year-old with RA and underlying UIP; RA therapies are on hold.",
        },
        {
            "match": "Currently intubated on pressure support ventilation",
            "corrected": "The patient is no longer intubated in the current-day truth; he is on nasal cannula with persistent oxygen requirement.",
            "reason": "Current-day gold A&P and flowsheet document nasal cannula, SpO2 97%, and home hospice planning.",
        },
        {
            "match": "Ventilator: CPAP/PSV",
            "corrected": "Remove ventilator settings; current respiratory support is nasal cannula with significant oxygen requirement.",
            "reason": "Current-day gold A&P does not support ongoing invasive ventilation.",
        },
        {
            "match": "Daily SBT assessment",
            "corrected": "Replace extubation/SBT planning with home hospice planning and chest tube management.",
            "reason": "Current-day gold A&P states home with chest tube and discharge home to hospice.",
        },
        {
            "match": "Remain in ICU; reassess for extubation readiness",
            "corrected": "Disposition should be discharge home to hospice with chest tube, DNR/DNI.",
            "reason": "Current gold A&P states D/C home to hospice and code status DNR/DNI.",
        },
    ],
    "121846_day34": [
        {
            "match": "New hypokalemia (K 2.9) requiring repletion",
            "move_to_keep": "true",
            "reason": "Original generation judge and current-day evidence identify K 2.9 with potassium repletion as a real missing/update item.",
        },
        {
            "match": "Failed SBT today",
            "move_to_keep": "true",
            "reason": "Original verifier treats failed SBT parameters as a forgotten carried problem; this should not be deleted if carried forward from prior truth.",
        },
        {
            "match": "Continue mechanical ventilation on PSV 10/5",
            "corrected": "Continue mechanical ventilation and daily SBT/RSBI; avoid unsupported exact settings if not present in current-day evidence.",
            "reason": "Ventilator dependence and SBT planning are supported, but exact settings should be used only when evidenced.",
        },
    ],
}


MANUAL_MISSING: dict[str, list[dict[str, str]]] = {
    "127713_day19": [
        {
            "claim_text_to_add": "Respiratory failure is improving after mucus plugging improved with pulmonary toilet; no respiratory events for the last 48 hours.",
            "target_section": "Respiratory",
            "evidence": ["Current-day A&P: no respiratory events for last 48 hours; improved mucus plugging with pulmonary toilet."],
            "reason": "Replaces the incorrect acute GI bleed trajectory with the true current-day respiratory trajectory.",
        },
        {
            "claim_text_to_add": "Stable for transfer to the floor while continuing pulmonary toilet.",
            "target_section": "Disposition",
            "evidence": ["Current-day A&P: stable for transfer to floor, continue pulmonary toilet; disposition transfer to floor."],
            "reason": "Restores the actual disposition context.",
        },
        {
            "claim_text_to_add": "Neurologic status is stable and close to baseline after recent PEA arrest.",
            "target_section": "Neurologic",
            "evidence": ["Current-day A&P: stable neuro status, close to baseline; mental status unchanged."],
            "reason": "Restores the actual neurologic trajectory.",
        },
        {
            "claim_text_to_add": "ESRD: HD was done yesterday; continue renal plan rather than holding dialysis for GI bleeding.",
            "target_section": "Renal",
            "evidence": ["Current-day A&P: ESRD - HD yesterday."],
            "reason": "Corrects the unsupported dialysis-hold-for-bleeding statement.",
        },
        {
            "claim_text_to_add": "Goals of care: arrange/continue family meeting and communication with PCP; code status remains full code.",
            "target_section": "Goals of care",
            "evidence": ["Current-day A&P: attempting to arrange family meeting; communication with PCP; code status full code."],
            "reason": "Restores important communication and goals-of-care context.",
        },
    ],
    "128729_day13": [
        {
            "claim_text_to_add": "Current respiratory support is nasal cannula with persistent oxygen requirement, not invasive ventilation.",
            "target_section": "Respiratory",
            "evidence": ["Current-day truth A&P documents nasal cannula, SpO2 97%, and significant oxygen requirement."],
            "reason": "Corrects the V2 trajectory away from intubation/SBT planning.",
        },
        {
            "claim_text_to_add": "Left pneumothorax/chest tube remains central; plan is home with chest tube after failed pneumostat.",
            "target_section": "Pneumothorax / Chest tube",
            "evidence": ["Current-day truth A&P: did well with chest tube, failed pneumostat, home with chest tube."],
            "reason": "This is the key current-day procedural/disposition issue.",
        },
        {
            "claim_text_to_add": "Family/goals of care: discharge home with hospice today; code status DNR/DNI.",
            "target_section": "Disposition / Goals of care",
            "evidence": ["Current-day truth A&P: home with hospice today; Disposition: D/C home to hospice; Code status: DNR/DNI."],
            "reason": "V2 misses the most important disposition and goals-of-care outcome.",
        },
    ],
    "121846_day34": [
        {
            "claim_text_to_add": "Add hypokalemia K 2.9 with aggressive potassium repletion and recheck plan.",
            "target_section": "Fluid / Electrolytes",
            "evidence": ["Original verifier missing update: potassium 2.90 mEq/L with multiple potassium chloride administrations."],
            "reason": "V2/pseudo revise under-covered a real current-day metabolic issue.",
        },
        {
            "claim_text_to_add": "Add metabolic alkalosis on ABG (pH 7.48, calculated total CO2/HCO3 37) and monitor with diuresis.",
            "target_section": "Fluid / Electrolytes",
            "evidence": ["Original verifier missing update: pH 7.48, pCO2 48, calculated total CO2 37."],
            "reason": "Qwen specifically penalized missing hypochloremic/metabolic alkalosis.",
        },
        {
            "claim_text_to_add": "Update anemia to current Hgb 9.1 g/dL / Hct 27.3% and monitor transfusion threshold.",
            "target_section": "Hematology",
            "evidence": ["Original verifier missing update: Hgb 9.1 g/dL, Hct 27.3%."],
            "reason": "Avoids stale hematology values.",
        },
    ],
    "110458_day10": [
        {
            "claim_text_to_add": "Respiratory status: remains intubated/mechanically ventilated; include secretions/sputum changes, including yellow/thick and later blood-tinged plug if clinically relevant.",
            "target_section": "Respiratory",
            "evidence": ["Original verifier missing updates cite secretions changing to yellow/thick and blood-tinged plug."],
            "reason": "Restores clinically relevant respiratory trajectory that pseudo truth did not add.",
        },
        {
            "claim_text_to_add": "Hypotension should be attributed to sepsis/hypovolemia when supported, rather than unsupported ETT malposition/reactive airway framing.",
            "target_section": "Hemodynamics",
            "evidence": ["Qwen qualitative review flagged missed hypotension attributed to sepsis and hypovolemia."],
            "reason": "Addresses the main case-level miss against V2 judge-revise.",
        },
    ],
    "174752_day51": [
        {
            "claim_text_to_add": "Add hypernatremia Na 150-153 with free-water correction plan.",
            "target_section": "Fluid / Electrolytes",
            "evidence": ["Original verifier missing update: hypernatremia Na 150-153."],
            "reason": "Qwen penalized pseudo revision for missing hypernatremia.",
        },
        {
            "claim_text_to_add": "Add metabolic alkalosis (pH 7.54, HCO3 32) and monitor in relation to ventilation/diuresis.",
            "target_section": "Fluid / Electrolytes",
            "evidence": ["Original verifier missing update: metabolic alkalosis pH 7.54, HCO3 32."],
            "reason": "Important current-day physiology absent from pseudo revision.",
        },
        {
            "claim_text_to_add": "Disposition planning should mention LTAC/rehab screening if supported.",
            "target_section": "Disposition",
            "evidence": ["Original verifier missing update: disposition to LTAC/rehab screening this week."],
            "reason": "Restores disposition context.",
        },
    ],
    "199046_day10": [
        {
            "claim_text_to_add": "Add septic arthritis as a key active problem when supported by the supplied case evidence.",
            "target_section": "Infectious / Musculoskeletal",
            "evidence": ["Qwen qualitative review flagged missed septic arthritis."],
            "reason": "This was a major reason pseudo-oracle revision underperformed.",
        },
        {
            "claim_text_to_add": "Remove unsupported PMH/problem list items such as cirrhosis, colon cancer, and IgA nephropathy unless directly supported.",
            "target_section": "Assessment",
            "evidence": ["Qwen qualitative review flagged those as unsupported in the oracle-revised note."],
            "reason": "Diagnosis/PMH hallucinations persisted after pseudo verifier.",
        },
    ],
    "191230_day26": [
        {
            "claim_text_to_add": "Mention PEG placement/tolerance without complications and PEG site stable.",
            "target_section": "FEN / Procedures",
            "evidence": ["Current-day truth A&P: tolerated PEG without complications; PEG site looks fine."],
            "reason": "Important current-day procedure outcome.",
        },
        {
            "claim_text_to_add": "Mention ARDS with tracheostomy, recent VAP due to Pseudomonas, and lowering PSV support as tolerated.",
            "target_section": "Respiratory / Infectious",
            "evidence": ["Current-day truth A&P: ARDS, s/p trach, Pseudomonas nosocomial pneumonia, RSBI 109, lower PSV as tolerated."],
            "reason": "Key respiratory/infectious trajectory.",
        },
        {
            "claim_text_to_add": "Mention completed linezolid for VRE bacteremia and completed ganciclovir for CMV viremia with repeat CMV viral load negative.",
            "target_section": "Infectious disease",
            "evidence": ["Current-day truth A&P: completed linezolid 14d; completed ganciclovir; repeat CMV viral load negative."],
            "reason": "Avoids missing key infectious disease context.",
        },
    ],
}


MANUAL_SUPPORTED_TO_FIX: dict[str, list[dict[str, str]]] = {
    "127713_day19": [
        {
            "contains": "acute upper gi bleeding",
            "action": "REWRITE",
            "rewrite": "74-year-old male with ESRD and recent PEA arrest, with improving respiratory failure after mucus plugging and stable neurologic status.",
            "reason": "Current-day A&P does not contain acute GI bleeding; the active trajectory is improving respiratory failure, ESRD, stable neurologic status, and floor transfer.",
        },
        {
            "contains": "hemorrhagic shock",
            "action": "DELETE",
            "rewrite": "",
            "reason": "No hemorrhagic shock is present in current-day A&P; hemodynamics are stable enough for floor transfer.",
        },
        {
            "contains": "maintain npo",
            "action": "DELETE",
            "rewrite": "",
            "reason": "Current-day A&P documents tube feeds, not NPO for EGD.",
        },
        {
            "contains": "pantoprazole 80 mg iv bid",
            "action": "REWRITE",
            "rewrite": "Continue PPI for stress ulcer prophylaxis.",
            "reason": "PPI is supported, but high-dose GI bleed dosing is not supported.",
        },
        {
            "contains": "hold heparin and aspirin",
            "action": "DELETE",
            "rewrite": "",
            "reason": "Current-day A&P lists SQ unfractionated heparin for DVT prophylaxis.",
        },
        {
            "contains": "serial hct q6h",
            "action": "DELETE",
            "rewrite": "",
            "reason": "Serial Hct monitoring for active GI bleed is not supported by current-day A&P.",
        },
        {
            "contains": "given gi bleed",
            "action": "REWRITE",
            "rewrite": "Continue pulmonary toilet and monitor respiratory status.",
            "reason": "Aspiration/GI-bleed framing is unsupported; respiratory plan should focus on pulmonary toilet.",
        },
        {
            "contains": "hold dialysis today due to active bleeding",
            "action": "REWRITE",
            "rewrite": "ESRD: HD was done yesterday; continue renal plan.",
            "reason": "Current-day A&P says HD yesterday, not dialysis held for bleeding.",
        },
        {
            "contains": "hold glargine today given npo",
            "action": "REWRITE",
            "rewrite": "Continue regular insulin sliding scale with tube feeds.",
            "reason": "Current-day A&P documents tube feeds and regular insulin sliding scale.",
        },
        {
            "contains": "ongoing bleeding",
            "action": "DELETE",
            "rewrite": "",
            "reason": "No active/ongoing bleeding is in the current-day A&P.",
        },
        {
            "contains": "bleeding resolved",
            "action": "DELETE",
            "rewrite": "",
            "reason": "DVT prophylaxis is SQ UF heparin in current-day A&P; bleeding-resolved condition is unsupported.",
        },
        {
            "contains": "update family on gi bleed",
            "action": "REWRITE",
            "rewrite": "Continue goals-of-care communication and family meeting planning.",
            "reason": "Family communication is supported, but GI bleed/transfusion/EGD details are not.",
        },
    ]
}


def extract_json_blocks(text: str) -> list[dict]:
    return [json.loads(block) for block in re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)]


def matches_rule(claim: str, rule: dict[str, str]) -> bool:
    low = claim.lower()
    if rule.get("match", "").lower() not in low:
        return False
    required = rule.get("claim_contains_all")
    if required:
        return all(part.lower() in low for part in required)
    return True


def make_missing_from_original(case_id: str, judge_dir: Path) -> tuple[list[dict], list[dict]]:
    path = judge_dir / f"{case_id}.json"
    if not path.exists():
        return [], []
    judge = json.loads(path.read_text(encoding="utf-8"))
    missing = []
    for item in judge.get("missing_updates", []) or []:
        problem = str(item.get("problem_or_event", "")).strip()
        revision = str(item.get("suggested_revision", "")).strip()
        evidence = str(item.get("today_evidence", "")).strip()
        if not problem or problem.lower().startswith("no "):
            continue
        if revision.lower() in {"none needed", "none"}:
            continue
        missing.append(
            {
                "claim_text_to_add": revision if revision else problem,
                "target_section": problem,
                "evidence": [evidence] if evidence else [],
                "reason": "Original V2 judge-revise verifier identified this as a missing current-day update.",
            }
        )
    carried = []
    for item in judge.get("forgotten_carried_problems", []) or []:
        problem = str(item.get("problem", "")).strip()
        revision = str(item.get("suggested_revision", "")).strip()
        previous = str(item.get("previous_evidence", "")).strip()
        if not problem or problem.lower().startswith("no "):
            continue
        if revision.lower() in {"none needed", "none"}:
            continue
        carried.append(
            {
                "problem": problem,
                "claim_text_to_add": revision if revision else problem,
                "prior_gold_evidence": previous,
                "current_day_evidence_or_current_gold_support": "Supported by original verifier's carried-forward problem audit.",
                "reason": "Original V2 judge-revise verifier identified this as a carried-forward problem that should not be dropped.",
            }
        )
    return missing, carried


def dedupe(items: list[dict], text_key: str) -> list[dict]:
    seen = set()
    out = []
    for item in items:
        text = re.sub(r"\s+", " ", str(item.get(text_key, ""))).strip().lower()
        if not text or text in seen:
            continue
        toks = {tok for tok in re.findall(r"[a-z0-9]+", text) if len(tok) > 2}
        duplicate = False
        for prev in out:
            prev_text = re.sub(r"\s+", " ", str(prev.get(text_key, ""))).strip().lower()
            prev_toks = {tok for tok in re.findall(r"[a-z0-9]+", prev_text) if len(tok) > 2}
            if toks and prev_toks and len(toks & prev_toks) / len(toks | prev_toks) >= 0.45:
                duplicate = True
                break
        if duplicate:
            continue
        seen.add(text)
        out.append(item)
    return out


def apply_manual_rewrites(obj: dict) -> list[dict]:
    case_id = obj["case_id"]
    vt = obj["verifier_truth"]
    unsupported = vt.get("unsupported_claims_to_remove_or_rewrite", [])
    supported = vt.get("supported_claims_to_keep", [])
    corrections = []
    remaining = []
    for item in unsupported:
        claim = str(item.get("claim_text", ""))
        applied = False
        for rule in MANUAL_REWRITES.get(case_id, []):
            if not matches_rule(claim, rule):
                continue
            if rule.get("move_to_keep") == "true":
                kept = {
                    "claim_text": claim,
                    "support_label": "SUPPORTED",
                    "evidence": item.get("evidence", []),
                    "reason": rule["reason"],
                }
                supported.append(kept)
                corrections.append(
                    {
                        "pseudo_claim_id": "",
                        "pseudo_label_was": f"{item.get('support_label')} / {item.get('action')}",
                        "correct_label_should_be": "SUPPORTED / KEEP",
                        "claim_text": claim,
                        "reason": rule["reason"],
                    }
                )
                applied = True
                break
            item["action"] = "REWRITE"
            item["support_label"] = "PARTIALLY_SUPPORTED"
            item["corrected_claim_or_rewrite"] = rule["corrected"]
            item["reason"] = rule["reason"]
            corrections.append(
                {
                    "pseudo_claim_id": "",
                    "pseudo_label_was": "UNSUPPORTED / DELETE",
                    "correct_label_should_be": "PARTIALLY_SUPPORTED / REWRITE",
                    "claim_text": claim,
                    "reason": rule["reason"],
                }
            )
            remaining.append(item)
            applied = True
            break
        if not applied:
            remaining.append(item)
    vt["unsupported_claims_to_remove_or_rewrite"] = remaining
    vt["supported_claims_to_keep"] = supported
    vt["pseudo_verifier_corrections"] = vt.get("pseudo_verifier_corrections", []) + corrections
    extra_rules = MANUAL_SUPPORTED_TO_FIX.get(case_id, [])
    if extra_rules:
        new_supported = []
        for item in vt["supported_claims_to_keep"]:
            claim = str(item.get("claim_text", ""))
            low = claim.lower()
            matched = None
            for rule in extra_rules:
                if rule["contains"] in low:
                    matched = rule
                    break
            if not matched:
                new_supported.append(item)
                continue
            fix = {
                "claim_text": claim,
                "support_label": "UNSUPPORTED" if matched["action"] == "DELETE" else "PARTIALLY_SUPPORTED",
                "action": matched["action"],
                "corrected_claim_or_rewrite": matched["rewrite"],
                "evidence": item.get("evidence", []),
                "reason": matched["reason"],
            }
            vt["unsupported_claims_to_remove_or_rewrite"].append(fix)
            corr = {
                "pseudo_claim_id": "",
                "pseudo_label_was": f"{item.get('support_label')} / KEEP",
                "correct_label_should_be": f"{fix['support_label']} / {fix['action']}",
                "claim_text": claim,
                "reason": matched["reason"],
            }
            corrections.append(corr)
            vt["pseudo_verifier_corrections"].append(corr)
        vt["supported_claims_to_keep"] = new_supported
    return corrections


def render_markdown(objs: list[dict], audit: list[dict]) -> str:
    lines = [
        "# Curated Claim-Level Verifier Truth Annotations",
        "",
        "This file is a manually curated usable version of the 30-case verifier truth annotations.",
        "It corrects obvious pseudo-verifier label errors and replaces noisy heading-based missing items with clinically actionable missing/carried-forward verifier instructions.",
        "",
        "## Index",
    ]
    for obj in objs:
        vt = obj["verifier_truth"]
        lines.append(
            f"- {obj['case_id']}: {len(vt.get('unsupported_claims_to_remove_or_rewrite', []))} fix, "
            f"{len(vt.get('supported_claims_to_keep', []))} keep, "
            f"{len(vt.get('missing_supported_claims_to_add', []))} add, "
            f"{len(vt.get('carried_forward_problems_to_restore', []))} restore, "
            f"{len(vt.get('pseudo_verifier_corrections', []))} label corrections"
        )
    lines.extend(["", "## Cases", ""])
    for obj in objs:
        lines.append(f"### {obj['case_id']}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(obj, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    lines.append("## Curation Audit")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(audit, ensure_ascii=False, indent=2))
    lines.append("```")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--judge-dir", type=Path, default=DEFAULT_JUDGE_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    objs = extract_json_blocks(args.annotations.read_text(encoding="utf-8"))
    audit = []
    for obj in objs:
        case_id = obj["case_id"]
        vt = obj["verifier_truth"]
        old_add = len(vt.get("missing_supported_claims_to_add", []))
        old_restore = len(vt.get("carried_forward_problems_to_restore", []))
        old_fix = len(vt.get("unsupported_claims_to_remove_or_rewrite", []))
        old_keep = len(vt.get("supported_claims_to_keep", []))

        original_missing, original_carried = make_missing_from_original(case_id, args.judge_dir)
        manual_missing = MANUAL_MISSING.get(case_id, [])
        vt["missing_supported_claims_to_add"] = dedupe([*manual_missing, *original_missing], "claim_text_to_add")
        vt["carried_forward_problems_to_restore"] = dedupe(original_carried, "claim_text_to_add")
        corrections = apply_manual_rewrites(obj)

        vt["case_level_summary"]["major_missing_items"] = [
            item["claim_text_to_add"] for item in vt.get("missing_supported_claims_to_add", [])
        ] + [item["claim_text_to_add"] for item in vt.get("carried_forward_problems_to_restore", [])]
        vt["case_level_summary"][
            "recommended_reviser_instruction"
        ] = "Use the fix labels to remove or rewrite unsupported details, then add the missing/current-day and carried-forward items listed here. Do not leave empty sections."

        audit.append(
            {
                "case_id": case_id,
                "old_fix": old_fix,
                "new_fix": len(vt.get("unsupported_claims_to_remove_or_rewrite", [])),
                "old_keep": old_keep,
                "new_keep": len(vt.get("supported_claims_to_keep", [])),
                "old_missing_add": old_add,
                "new_missing_add": len(vt.get("missing_supported_claims_to_add", [])),
                "old_restore": old_restore,
                "new_restore": len(vt.get("carried_forward_problems_to_restore", [])),
                "label_corrections": len(corrections),
            }
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.outdir / "verifier_truth_annotations_curated_usable.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    md_path = args.outdir / "verifier_truth_annotations_curated_usable.md"
    md_path.write_text(render_markdown(objs, audit), encoding="utf-8", newline="\n")
    audit_path = args.outdir / "curation_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(f"Wrote {md_path}")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {audit_path}")
    print(json.dumps({"cases": len(objs), "total_label_corrections": sum(x["label_corrections"] for x in audit)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
