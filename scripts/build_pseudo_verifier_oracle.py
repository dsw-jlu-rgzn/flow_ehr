from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SELECTED = Path("outputs/oracle_claim_verifier_qwen653/selected_cases_with_failure_seed.json")
DEFAULT_TRUTH = Path("outputs/oracle_claim_verifier_qwen653/truth_replacement/verifier_truth.jsonl")
DEFAULT_V2_DIR = Path("outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2")
DEFAULT_OUTDIR = Path("outputs/oracle_claim_verifier_qwen653/pseudo_verifier_oracle")


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "continue",
    "current",
    "for",
    "from",
    "given",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "patient",
    "per",
    "plan",
    "reassess",
    "the",
    "to",
    "today",
    "with",
}

CLAIM_TYPE_PATTERNS = [
    ("numeric_lab_vital", re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg/dl|g/dl|mEq/L|mmHg|cmH2O|%|units?|mcg|mg|ml|L/min|F|C)\b", re.I)),
    ("medication", re.compile(r"\b(vancomycin|cefepime|ceftriaxone|ceftazidime|ciprofloxacin|azithro|linezolid|ganciclovir|metoprolol|amio|amiodarone|warfarin|heparin|insulin|glargine|humalog|fentanyl|midazolam|versed|phenylephrine|norepinephrine|vasopressin|lactulose|pantoprazole|ppi|lasix|furosemide)\b", re.I)),
    ("procedure_device", re.compile(r"\b(crrt|cvvh|cvvhd|dialysis|hd\b|intubat|extubat|trach|peg|ett|ventilat|catheter|line|sbt|cpap|peep|fio2|cvvh|transfusion|prbc)\b", re.I)),
    ("trajectory", re.compile(r"\b(improv|worsen|decreas|increas|wean|failed|stable|resolved|off|transition|drop|rising|falling|tolerat)\b", re.I)),
    ("disposition_context", re.compile(r"\b(code status|full code|dnr|family|goals|communication|disposition|transfer|floor|rehab|discharge)\b", re.I)),
    ("plan_action", re.compile(r"\b(continue|monitor|hold|start|stop|obtain|await|adjust|replete|notify|consider|target|maintain|wean|reassess|update)\b", re.I)),
]

HIGH_RISK_GROUPS = {
    "crrt": [r"\bcrrt\b", r"\bcvvh\b", r"\bcvvhd\b", r"continuous venovenous"],
    "hd": [r"\bhd\b", r"\bhemodialysis\b", r"\bdialysis\b"],
    "failed_extubation": [r"failed extubat"],
    "extubation": [r"\bextubat"],
    "trach": [r"\btrach", r"tracheostomy"],
    "pressor": [r"\bpressor", r"norepinephrine", r"phenylephrine", r"vasopressin", r"levophed"],
    "abg": [r"\babg\b", r"\bpco2\b", r"\bpo2\b", r"\bph\b"],
    "peep": [r"\bpeep\b"],
    "fio2": [r"\bfio2\b"],
    "transfusion": [r"transfus", r"\bprbc\b"],
    "anticoagulation": [r"warfarin", r"coumadin", r"heparin", r"anticoag"],
    "code_status": [r"code status", r"full code", r"\bdnr\b"],
}

MEDICATIONS = [
    "vancomycin",
    "cefepime",
    "ceftriaxone",
    "ceftazidime",
    "ciprofloxacin",
    "azithromycin",
    "linezolid",
    "ganciclovir",
    "metoprolol",
    "amiodarone",
    "warfarin",
    "heparin",
    "insulin",
    "glargine",
    "humalog",
    "fentanyl",
    "midazolam",
    "versed",
    "phenylephrine",
    "norepinephrine",
    "vasopressin",
    "lactulose",
    "pantoprazole",
    "furosemide",
    "lasix",
]

MED_ALIASES = {
    "warfarin": ["warfarin", "coumadin"],
    "furosemide": ["furosemide", "lasix"],
    "lasix": ["lasix", "furosemide"],
    "midazolam": ["midazolam", "versed"],
    "versed": ["versed", "midazolam"],
    "pantoprazole": ["pantoprazole", "ppi", "protonix"],
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text: str) -> str:
    text = text.replace("鈥檚", "'s").replace("掳", " deg ").replace("鈮?", ">=")
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z][a-z0-9/-]{2,}", text.lower())
        if t not in STOPWORDS and not t.isdigit()
    }


def numbers(text: str) -> list[str]:
    out = []
    for n in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", text):
        if len(n) >= 3 and n.startswith("219"):
            continue
        out.append(n)
    return out


def approx_num_present(num: str, evidence: str) -> bool:
    try:
        value = float(num)
    except ValueError:
        return num in evidence
    for candidate in numbers(evidence):
        try:
            cand = float(candidate)
        except ValueError:
            continue
        tol = max(0.11, abs(value) * 0.03)
        if abs(value - cand) <= tol:
            return True
    return False


def claim_type(claim: str) -> str:
    for label, pattern in CLAIM_TYPE_PATTERNS:
        if pattern.search(claim):
            return label
    if len(claim.split()) <= 7:
        return "diagnosis"
    return "diagnosis"


def split_claims(text: str) -> list[dict]:
    claims = []
    section = "assessment"
    current_sentence = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^\*{0,2}assessment", line, re.I):
            section = "assessment"
            continue
        if re.match(r"^\*{0,2}plan", line, re.I):
            section = "plan"
            continue
        heading = re.match(r"^\d+\.\s+\*{0,2}(.+?)\*{0,2}:?\s*$", line)
        if heading:
            section = clean_text(heading.group(1)).lower().replace(" ", "_")[:80]
            claims.append({"claim_text": clean_text(heading.group(1)), "original_sentence": clean_text(line), "section": section})
            continue
        bullet = re.match(r"^[-*]\s+(.+)$", line)
        if bullet:
            for piece in split_sentence_into_claims(clean_text(bullet.group(1))):
                claims.append({"claim_text": piece, "original_sentence": clean_text(line), "section": section})
            continue
        current_sentence += " " + line
        if re.search(r"[.!?]$", line):
            for piece in split_sentence_into_claims(clean_text(current_sentence)):
                claims.append({"claim_text": piece, "original_sentence": clean_text(current_sentence), "section": section})
            current_sentence = ""
    if current_sentence.strip():
        for piece in split_sentence_into_claims(clean_text(current_sentence)):
            claims.append({"claim_text": piece, "original_sentence": clean_text(current_sentence), "section": section})
    filtered = []
    seen = set()
    for claim in claims:
        text_norm = claim["claim_text"].strip(" .;:")
        if len(text_norm) < 8:
            continue
        key = (claim["section"], text_norm.lower())
        if key in seen:
            continue
        seen.add(key)
        claim["claim_text"] = text_norm + ("." if not re.search(r"[.!?]$", text_norm) else "")
        filtered.append(claim)
    return filtered


def split_sentence_into_claims(sentence: str) -> list[str]:
    if len(sentence.split()) <= 22:
        return [sentence]
    pieces = re.split(r";\s+|\s+-\s+|\.\s+(?=[A-Z])", sentence)
    out = []
    for piece in pieces:
        piece = piece.strip(" .;:")
        if not piece:
            continue
        subpieces = re.split(r",\s+(?=(?:and\s+)?(?:new|with|and|but|while|currently|PEEP|FiO2|BUN|Cr|Creatinine|Hemoglobin|Glucose)\b)", piece)
        out.extend(p.strip(" .;:") for p in subpieces if len(p.strip()) >= 8)
    return out or [sentence]


def evidence_corpus(truth: dict) -> list[dict[str, str]]:
    snippets = [
        {"source": "gold_ap_oracle_reference", "text": truth["llm_like_oracle_output"]},
    ]
    for ev in truth.get("current_day_evidence_compact", []):
        snippets.append({"source": ev.get("evidence_id", "current_day_evidence"), "text": ev.get("text", "")})
    return snippets


def retrieve_evidence(claim: str, snippets: list[dict[str, str]], limit: int = 5) -> tuple[str, list[str]]:
    claim_tokens = tokens(claim)
    scored = []
    for snippet in snippets:
        stokens = tokens(snippet["text"])
        overlap = len(claim_tokens & stokens)
        if overlap:
            scored.append((overlap / math.sqrt(max(len(stokens), 1)), overlap, snippet))
    scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
    chosen = [s for _, _, s in scored[:limit]]
    if not chosen:
        chosen = snippets[:2]
    evidence_text = "\n".join(f"{item['source']}: {clean_text(item['text'])[:900]}" for item in chosen)
    return evidence_text, [item["source"] for item in chosen]


def group_present(group: str, text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in HIGH_RISK_GROUPS[group])


def high_risk_groups(claim: str) -> list[str]:
    return [group for group in HIGH_RISK_GROUPS if group_present(group, claim)]


def medication_mentions(claim: str) -> list[str]:
    low = claim.lower()
    return [med for med in MEDICATIONS if re.search(rf"\b{re.escape(med)}\b", low)]


def medication_supported(med: str, text: str) -> bool:
    low = text.lower()
    aliases = MED_ALIASES.get(med, [med])
    return any(re.search(rf"\b{re.escape(alias)}\b", low) for alias in aliases)


def support_decision(
    claim: str,
    ctype: str,
    evidence_text: str,
    full_evidence: str,
    oracle_reference: str,
) -> tuple[str, str, str, str, str]:
    claim_low = claim.lower()
    oracle_low = oracle_reference.lower()
    full_low = full_evidence.lower()
    claim_tokens = tokens(claim)
    ev_tokens = tokens(full_evidence)
    overlap = len(claim_tokens & ev_tokens) / max(len(claim_tokens), 1)
    nums = numbers(claim)
    missing_nums = [num for num in nums if not approx_num_present(num, full_evidence)]
    meds = medication_mentions(claim)
    missing_meds = [med for med in meds if not medication_supported(med, full_evidence)]
    risky = high_risk_groups(claim)
    missing_risk = []
    for group in risky:
        support_text = full_evidence if group in {"anticoagulation", "transfusion", "code_status"} else oracle_reference
        if not group_present(group, support_text):
            missing_risk.append(group)

    if "failed extubat" in claim_low and not re.search(r"failed extubat", full_low):
        return (
            "UNSUPPORTED",
            "FIX",
            "DELETE",
            "",
            "High-risk failed-extubation claim is not explicitly documented in oracle/reference evidence.",
        )
    if re.search(r"\b(crrt|cvvhd|cvvh)\b", claim_low) and not re.search(
        r"\b(crrt|cvvhd|cvvh|continuous venovenous)\b", oracle_low
    ):
        rewrite = ""
        if re.search(r"\b(hd|dialysis|hemodialysis)\b", oracle_low):
            rewrite = "The patient has renal failure and is receiving dialysis/HD as documented."
        return (
            "PARTIALLY_SUPPORTED" if rewrite else "UNSUPPORTED",
            "FIX",
            "REWRITE" if rewrite else "DELETE",
            rewrite,
            "Renal failure may be supported, but CRRT/CVVHD specifically is not documented in the oracle/reference evidence.",
        )
    if re.search(r"\b(phenylephrine|norepinephrine|vasopressin|pressor)", claim_low) and re.search(
        r"pressors?\s+off", oracle_low
    ):
        return (
            "CONTRADICTED",
            "FIX",
            "REWRITE",
            "Pressors are off currently.",
            "Oracle/reference evidence states pressors are off, contradicting ongoing vasopressor support.",
        )
    if re.search(r"failed\s+cpap|failed\s+sbt|failed\s+wean", claim_low) and not re.search(
        r"failed\s+(cpap|sbt|wean)", oracle_low
    ):
        return (
            "UNSUPPORTED",
            "FIX",
            "DELETE",
            "",
            "Failed CPAP/SBT/weaning claim is high-risk and is not explicitly documented in oracle/reference evidence.",
        )
    if missing_risk:
        return (
            "UNSUPPORTED",
            "FIX",
            "DELETE",
            "",
            "High-risk claim lacks explicit matching evidence for: " + ", ".join(missing_risk),
        )
    if ctype == "numeric_lab_vital" and missing_nums:
        return (
            "UNSUPPORTED",
            "FIX",
            "REWRITE" if overlap >= 0.35 else "DELETE",
            "",
            "Numeric claim contains values not found in evidence: " + ", ".join(missing_nums),
        )
    if ctype == "medication" and missing_meds:
        return (
            "UNSUPPORTED",
            "FIX",
            "DELETE",
            "",
            "Medication claim includes medication(s) not explicitly found in evidence: " + ", ".join(missing_meds),
        )
    if overlap >= 0.58:
        return ("SUPPORTED", "KEEP", "KEEP", "", f"Claim terms are well covered by retrieved evidence (token overlap {overlap:.2f}).")
    if overlap >= 0.36:
        return (
            "PARTIALLY_SUPPORTED",
            "KEEP",
            "KEEP",
            "",
            f"Core concept appears supported, but wording may be more specific than evidence (token overlap {overlap:.2f}).",
        )
    if ctype in {"plan_action", "disposition_context"} and overlap >= 0.28:
        return (
            "PARTIALLY_SUPPORTED",
            "KEEP",
            "KEEP",
            "",
            f"Plan/context claim has partial supporting evidence (token overlap {overlap:.2f}).",
        )
    return (
        "UNSUPPORTED",
        "FIX",
        "DELETE",
        "",
        f"Insufficient lexical/clinical support in retrieved evidence (token overlap {overlap:.2f}).",
    )


def build(args: argparse.Namespace) -> None:
    selected = json.loads(args.selected.read_text(encoding="utf-8"))
    truth_by_case = {row["case_id"]: row for row in read_jsonl(args.truth)}
    args.outdir.mkdir(parents=True, exist_ok=True)

    raw_claims = []
    evidence_rows = []
    labeled_rows = []
    case_summaries = []

    for item in selected:
        case_id = item["case_id"]
        v2_path = args.v2_dir / f"{case_id}.txt"
        v2_text = v2_path.read_text(encoding="utf-8", errors="replace")
        truth = truth_by_case[case_id]
        snippets = evidence_corpus(truth)
        full_evidence = "\n".join(item["text"] for item in snippets)
        claims = split_claims(v2_text)

        counts = Counter()
        for idx, claim in enumerate(claims, start=1):
            ctype = claim_type(claim["claim_text"])
            claim_id = f"{case_id}_v2_c{idx:03d}"
            raw = {
                "case_id": case_id,
                "source_output": "memory_gated_scaffold_v2",
                "claim_id": claim_id,
                "claim_text": claim["claim_text"],
                "claim_type": ctype,
                "original_sentence": claim["original_sentence"],
                "section": claim["section"],
            }
            raw_claims.append(raw)
            evidence_text, evidence_sources = retrieve_evidence(claim["claim_text"], snippets)
            evidence_row = {
                **raw,
                "evidence_text": evidence_text,
                "evidence_sources": evidence_sources,
                "evidence_time_window": "oracle_reference_ap_plus_current_day_evidence_for_pseudo_labeling",
            }
            evidence_rows.append(evidence_row)
            support_label, binary_label, action, rewrite, reason = support_decision(
                claim["claim_text"], ctype, evidence_text, full_evidence, truth["llm_like_oracle_output"]
            )
            labeled = {
                **evidence_row,
                "support_label": support_label,
                "binary_label": binary_label,
                "action": action,
                "oracle_rewrite": rewrite,
                "reason": reason,
                "pseudo_oracle_label": True,
                "manual_review_status": "pseudo_checked_by_rules",
            }
            labeled_rows.append(labeled)
            counts[(support_label, binary_label, action)] += 1
            counts[("claim_type", ctype)] += 1
        case_summaries.append(
            {
                "case_id": case_id,
                "num_claims": len(claims),
                "keep_claims": sum(1 for row in labeled_rows if row["case_id"] == case_id and row["binary_label"] == "KEEP"),
                "fix_claims": sum(1 for row in labeled_rows if row["case_id"] == case_id and row["binary_label"] == "FIX"),
                "delete_claims": sum(1 for row in labeled_rows if row["case_id"] == case_id and row["action"] == "DELETE"),
                "rewrite_claims": sum(1 for row in labeled_rows if row["case_id"] == case_id and row["action"] == "REWRITE"),
                "claim_type_counts": {
                    key[1]: value for key, value in counts.items() if isinstance(key, tuple) and len(key) == 2 and key[0] == "claim_type"
                },
                "selection_failure_modes": item.get("failure_modes", []),
            }
        )

    write_jsonl(args.outdir / "claims_raw.jsonl", raw_claims)
    write_jsonl(args.outdir / "claims_with_evidence.jsonl", evidence_rows)
    write_jsonl(args.outdir / "claims_oracle_labeled_pseudo.jsonl", labeled_rows)
    write_jsonl(args.outdir / "case_claim_summary.jsonl", case_summaries)
    review_dir = args.outdir / "per_case_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    rows_by_case = defaultdict(list)
    for row in labeled_rows:
        rows_by_case[row["case_id"]].append(row)
    for summary in case_summaries:
        cid = summary["case_id"]
        rows = rows_by_case[cid]
        lines = [
            f"# {cid} Pseudo Verifier Review",
            "",
            f"- claims: {summary['num_claims']}",
            f"- KEEP: {summary['keep_claims']}",
            f"- FIX: {summary['fix_claims']}",
            f"- DELETE: {summary['delete_claims']}",
            f"- REWRITE: {summary['rewrite_claims']}",
            "",
            "## FIX Claims",
            "",
        ]
        for row in rows:
            if row["binary_label"] != "FIX":
                continue
            lines.extend(
                [
                    f"### {row['claim_id']} | {row['claim_type']} | {row['support_label']} | {row['action']}",
                    "",
                    f"Claim: {row['claim_text']}",
                    "",
                    f"Reason: {row['reason']}",
                    "",
                ]
            )
            if row["oracle_rewrite"]:
                lines.extend([f"Rewrite: {row['oracle_rewrite']}", ""])
        lines.extend(["## KEEP High-Risk Or Specific Claims", ""])
        for row in rows:
            if row["binary_label"] == "KEEP" and (high_risk_groups(row["claim_text"]) or row["claim_type"] in {"numeric_lab_vital", "medication", "procedure_device"}):
                lines.extend(
                    [
                        f"- `{row['claim_id']}` {row['claim_type']} {row['support_label']}: {row['claim_text']}",
                    ]
                )
        (review_dir / f"{cid}.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    with (args.outdir / "claim_level_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["case_id", "num_claims", "keep_claims", "fix_claims", "delete_claims", "rewrite_claims"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in case_summaries:
            writer.writerow({key: row[key] for key in fieldnames})

    audit_path = args.outdir / "high_risk_claim_audit.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "case_id",
            "claim_id",
            "claim_type",
            "support_label",
            "binary_label",
            "action",
            "risk_groups",
            "claim_text",
            "reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in labeled_rows:
            risks = high_risk_groups(row["claim_text"])
            if not risks and row["claim_type"] not in {"numeric_lab_vital", "medication", "procedure_device"}:
                continue
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "claim_id": row["claim_id"],
                    "claim_type": row["claim_type"],
                    "support_label": row["support_label"],
                    "binary_label": row["binary_label"],
                    "action": row["action"],
                    "risk_groups": "|".join(risks),
                    "claim_text": row["claim_text"],
                    "reason": row["reason"],
                }
            )

    global_counts = Counter((row["support_label"], row["binary_label"], row["action"]) for row in labeled_rows)
    type_counts = Counter(row["claim_type"] for row in labeled_rows)
    type_fix = Counter(row["claim_type"] for row in labeled_rows if row["binary_label"] == "FIX")
    report = [
        "# Pseudo Verifier Oracle Report",
        "",
        f"- cases: {len(selected)}",
        f"- claims: {len(labeled_rows)}",
        f"- KEEP: {sum(1 for row in labeled_rows if row['binary_label'] == 'KEEP')}",
        f"- FIX: {sum(1 for row in labeled_rows if row['binary_label'] == 'FIX')}",
        f"- per-case review dir: {review_dir.as_posix()}",
        f"- high-risk audit csv: {audit_path.as_posix()}",
        "",
        "## Label Counts",
        "",
    ]
    for key, value in sorted(global_counts.items()):
        report.append(f"- {key}: {value}")
    report += ["", "## Fix Rate By Claim Type", ""]
    for ctype, total in sorted(type_counts.items()):
        report.append(f"- {ctype}: {type_fix[ctype]}/{total} ({type_fix[ctype] / total:.1%})")
    report += [
        "",
        "## Caveat",
        "",
        "These labels are pseudo-oracle labels for upper-bound plumbing. They are strict for high-risk clinical claims and use oracle/reference A&P plus current-day evidence for labeling. They should be spot-checked before being used as final human annotation.",
    ]
    (args.outdir / "summary_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {len(labeled_rows)} pseudo verifier labels for {len(selected)} cases -> {args.outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pseudo-oracle claim-level verifier labels for selected AP100 cases.")
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
