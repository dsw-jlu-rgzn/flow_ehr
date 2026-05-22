from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


DEFAULT_SELECTED = Path("outputs/oracle_claim_verifier_qwen653/selected_cases_with_failure_seed.json")
DEFAULT_OUTDIR = Path("outputs/oracle_claim_verifier_qwen653/truth_replacement")
DEFAULT_DATA_ROOT = Path("data_ap100_ap/AP")
DEFAULT_V2_DIR = Path("outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2")
DEFAULT_JUDGE_REVISE_DIR = Path(
    "outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2_judge_revise"
)
DEFAULT_BASELINE_ROOT = Path("data_ap100_ap/AP/generated/DG/deepseek_api_full_gen/gen/method2")


HIGH_RISK_PATTERNS = {
    "renal_replacement": re.compile(r"\b(crrt|cvvh|cvvhd|dialysis|hd\b|renal replacement)\b", re.I),
    "ventilation": re.compile(r"\b(intubat|extubat|trach|ventilat|sbt|rsbi|peep|fio2|abg|ett)\b", re.I),
    "vasopressor": re.compile(r"\b(pressor|norepinephrine|phenylephrine|vasopressin|levophed|shock)\b", re.I),
    "bleeding_transfusion": re.compile(r"\b(bleed|hemorrhage|hematoma|hgb|hemoglobin|transfus|warfarin|inr)\b", re.I),
    "infection_antibiotic": re.compile(r"\b(fever|sepsis|pneumonia|vap|culture|antibiotic|vanco|vancomycin|cef|cipro|azithro)\b", re.I),
    "disposition_goals": re.compile(r"\b(code status|full code|dnr|family|goals|disposition|transfer|rehab|trach discussion)\b", re.I),
    "nutrition": re.compile(r"\b(nutrition|tube feed|enteral|tpn|diet|npo|feeding)\b", re.I),
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def day_key(day: int | str) -> str:
    return str(int(day))


def load_gold_ap(data_root: Path, hadm_id: str, day: int) -> str:
    path = data_root / "gold" / f"gt_{hadm_id}.csv"
    for row in read_csv_rows(path):
        if day_key(row["DAY"]) == day_key(day):
            return row["TEXT"].strip()
    raise KeyError(f"Missing gold A&P for {hadm_id}_day{day}: {path}")


def load_current_day_evidence(data_root: Path, hadm_id: str, day: int) -> list[dict[str, str]]:
    path = data_root / "input" / f"input_{hadm_id}.csv"
    rows = []
    if not path.exists():
        return rows
    for idx, row in enumerate(read_csv_rows(path)):
        if day_key(row.get("DAY", 0)) != day_key(day):
            continue
        rows.append(
            {
                "evidence_id": f"{hadm_id}_day{day}_raw{idx:04d}",
                "time": row.get("TIME", ""),
                "rel_time": row.get("REL_TIME", ""),
                "is_note": row.get("IS_NOTE", ""),
                "text": row.get("TEXT", "").strip(),
            }
        )
    return rows


def load_baseline_ap(baseline_root: Path, hadm_id: str, day: int) -> str:
    path = baseline_root / f"genpns_{hadm_id}.csv"
    if not path.exists():
        return ""
    for row in read_csv_rows(path):
        if day_key(row["DAY"]) == day_key(day):
            return row["TEXT"].strip()
    return ""


def split_sentences(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    pieces = re.split(r"(?<=[.!?])\s+|(?:\n\s*[-*]\s*)", clean)
    return [p.strip(" -") for p in pieces if len(p.strip(" -")) >= 20]


def high_risk_summary(text: str, limit: int = 24) -> dict[str, list[str]]:
    sentences = split_sentences(text)
    summary: dict[str, list[str]] = {}
    for label, pattern in HIGH_RISK_PATTERNS.items():
        hits = [s for s in sentences if pattern.search(s)]
        if hits:
            summary[label] = hits[:limit]
    return summary


def compact_evidence(evidence: list[dict[str, str]], limit: int = 80) -> list[dict[str, str]]:
    note_rows = [e for e in evidence if e.get("is_note") == "1"]
    non_note_rows = [e for e in evidence if e.get("is_note") != "1"]
    prioritized = note_rows + non_note_rows
    return prioritized[:limit]


def extract_llm_like_ap(gold_note: str) -> str:
    matches = list(re.finditer(r"(?im)^\s*assessment\s+and\s+plan\.?\s*$", gold_note))
    if matches:
        start = matches[-1].start()
        extracted = gold_note[start:].strip()
    else:
        fallback = re.search(r"(?im)^\s*(?:\d+\s*(?:yo|year old)|\[\*\*Age over 90 \*\*\]).*$", gold_note)
        extracted = gold_note[fallback.start() :].strip() if fallback else gold_note.strip()
        extracted = "Assessment and Plan\n" + extracted
    extracted = re.sub(r"(?im)^\s*Total time spent:.*$", "", extracted).strip()
    extracted = re.sub(r"(?im)^\s*Patient is critically ill\s*$", "", extracted).strip()
    return extracted


def leakage_terms(text: str) -> list[str]:
    terms = [
        "oracle",
        "truth",
        "gold_ap",
        "gold a&p",
        "replacement_source",
        "selection",
        "qwen",
        "verifier",
        "failure_modes",
        "current_day_evidence",
    ]
    low = text.lower()
    return [term for term in terms if term in low]


def markdown_case(record: dict) -> str:
    def block(title: str, text: str) -> str:
        return f"\n## {title}\n\n```text\n{text.strip()}\n```\n"

    lines = [
        f"# {record['case_id']} Verifier Truth Packet",
        "",
        "## Selection",
        "",
        f"- Qwen source: full 653 V2 detail, with judge-revise detail when available",
        f"- Failure modes: {', '.join(record['selection']['failure_modes'])}",
        f"- Case types: {', '.join(record['selection']['case_type'])}",
        f"- Reason: {record['selection']['reason_for_selection']}",
        "",
        "## Oracle Truth Use",
        "",
        "Use `gold_ap` below as the verifier upper-bound replacement output for this case.",
        "This file is a manual audit packet; the machine-readable row is in `oracle_gold_generation.jsonl`.",
    ]
    lines.append(block("Gold A&P / Oracle Replacement", record["gold_ap"]))
    lines.append(block("Baseline DeepSeek AP", record.get("baseline_ap", "")))
    lines.append(block("V2 AP", record.get("v2_ap", "")))
    if record.get("judge_revise_ap"):
        lines.append(block("V2 Judge-Revise AP", record["judge_revise_ap"]))
    lines.append("## High-Risk Truth Extracted From Gold A&P\n")
    lines.append("```json")
    lines.append(json.dumps(record["gold_high_risk_truth"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("\n## Current-Day Evidence Snippets\n")
    for ev in record["current_day_evidence_compact"]:
        text = re.sub(r"\s+", " ", ev["text"]).strip()
        lines.append(f"- `{ev['evidence_id']}` `{ev['rel_time']}` {text[:900]}")
    return "\n".join(lines).strip() + "\n"


def build(args: argparse.Namespace) -> None:
    selected = json.loads(args.selected.read_text(encoding="utf-8"))
    args.outdir.mkdir(parents=True, exist_ok=True)
    gold_txt_dir = args.outdir / "gold_txt"
    gold_csv_dir = args.outdir / "gold_csv"
    generation_method2_dir = args.outdir / "generation_run" / "method2"
    prediction_only_dir = args.outdir / "prediction_only"
    prediction_txt_dir = prediction_only_dir / "txt"
    prediction_generation_method2_dir = prediction_only_dir / "generation_run" / "method2"
    packets_dir = args.outdir / "manual_case_truth"

    jsonl_rows = []
    prediction_rows = []
    truth_rows = []
    rows_by_hadm: dict[str, list[dict[str, str]]] = {}
    prediction_rows_by_hadm: dict[str, list[dict[str, str]]] = {}
    audit_rows = []

    for item in selected:
        hadm_id = item["admission_id"]
        day = int(item["day"])
        case_id = item["case_id"]
        gold_ap = load_gold_ap(args.data_root, hadm_id, day)
        llm_like_ap = extract_llm_like_ap(gold_ap)
        current_day_evidence = load_current_day_evidence(args.data_root, hadm_id, day)
        baseline_ap = load_baseline_ap(args.baseline_root, hadm_id, day)
        v2_ap = read_text(args.v2_dir / f"{case_id}.txt").strip()
        judge_revise_ap = read_text(args.judge_revise_dir / f"{case_id}.txt").strip()

        row = {
            "sample_id": case_id,
            "case_id": case_id,
            "hadm_id": hadm_id,
            "day": day,
            "generated_ap": gold_ap,
            "gold_ap": gold_ap,
            "replacement_source": "gold_ap_oracle_truth",
            "selection": item,
            "current_day_evidence": current_day_evidence,
        }
        jsonl_rows.append(row)
        prediction_rows.append(
            {
                "sample_id": case_id,
                "case_id": case_id,
                "hadm_id": hadm_id,
                "day": day,
                "generated_ap": llm_like_ap,
            }
        )

        truth = {
            "case_id": case_id,
            "hadm_id": hadm_id,
            "day": day,
            "truth_output_path": str((gold_txt_dir / f"{case_id}.txt").as_posix()),
            "gold_ap": gold_ap,
            "llm_like_oracle_output": llm_like_ap,
            "gold_high_risk_truth": high_risk_summary(gold_ap),
            "current_day_evidence_compact": compact_evidence(current_day_evidence),
            "selection": item,
            "baseline_ap": baseline_ap,
            "v2_ap": v2_ap,
            "judge_revise_ap": judge_revise_ap,
            "notes": [
                "This is the oracle replacement output for upper-bound verification.",
                "It is intentionally not a DeepSeek generation.",
                "Use it to replace generated_ap when validating whether the evaluation stack can recognize a near-perfect output.",
            ],
        }
        truth_rows.append(truth)

        write_text(gold_txt_dir / f"{case_id}.txt", gold_ap + "\n")
        write_text(prediction_txt_dir / f"{case_id}.txt", llm_like_ap + "\n")
        write_text(packets_dir / f"{case_id}.md", markdown_case(truth))
        rows_by_hadm.setdefault(hadm_id, []).append({"DAY": str(day), "TEXT": gold_ap})
        prediction_rows_by_hadm.setdefault(hadm_id, []).append({"DAY": str(day), "TEXT": llm_like_ap})
        leaks = leakage_terms(llm_like_ap)
        audit_rows.append(
            {
                "case_id": case_id,
                "hadm_id": hadm_id,
                "day": day,
                "status": "PASS" if not leaks and llm_like_ap.lower().startswith("assessment") else "CHECK",
                "leakage_terms": "|".join(leaks),
                "starts_with_assessment": str(llm_like_ap.lower().startswith("assessment")),
                "full_gold_chars": len(gold_ap),
                "prediction_chars": len(llm_like_ap),
                "prediction_words": len(llm_like_ap.split()),
                "starts_with": llm_like_ap[:100].replace("\n", " "),
            }
        )

    gold_csv_dir.mkdir(parents=True, exist_ok=True)
    generation_method2_dir.mkdir(parents=True, exist_ok=True)
    for hadm_id, rows in rows_by_hadm.items():
        rows.sort(key=lambda r: int(r["DAY"]))
        for out_csv in [
            gold_csv_dir / f"genpns_{hadm_id}.csv",
            generation_method2_dir / f"genpns_{hadm_id}.csv",
        ]:
            with out_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["DAY", "TEXT"])
                writer.writeheader()
                writer.writerows(rows)

    prediction_generation_method2_dir.mkdir(parents=True, exist_ok=True)
    for hadm_id, rows in prediction_rows_by_hadm.items():
        rows.sort(key=lambda r: int(r["DAY"]))
        with (prediction_generation_method2_dir / f"genpns_{hadm_id}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["DAY", "TEXT"])
            writer.writeheader()
            writer.writerows(rows)

    with (args.outdir / "oracle_gold_generation.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in jsonl_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    prediction_only_dir.mkdir(parents=True, exist_ok=True)
    with (prediction_only_dir / "oracle_prediction_only.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in prediction_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (prediction_only_dir / "leakage_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    with (args.outdir / "verifier_truth.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in truth_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "selected_cases": str(args.selected.as_posix()),
        "n_cases": len(selected),
        "oracle_gold_generation_jsonl": str((args.outdir / "oracle_gold_generation.jsonl").as_posix()),
        "verifier_truth_jsonl": str((args.outdir / "verifier_truth.jsonl").as_posix()),
        "gold_txt_dir": str(gold_txt_dir.as_posix()),
        "gold_csv_dir": str(gold_csv_dir.as_posix()),
        "generation_run_dir": str((args.outdir / "generation_run").as_posix()),
        "generation_method2_dir": str(generation_method2_dir.as_posix()),
        "prediction_only_jsonl": str((prediction_only_dir / "oracle_prediction_only.jsonl").as_posix()),
        "prediction_only_txt_dir": str(prediction_txt_dir.as_posix()),
        "prediction_only_generation_run_dir": str((prediction_only_dir / "generation_run").as_posix()),
        "prediction_only_leakage_audit_csv": str((prediction_only_dir / "leakage_audit.csv").as_posix()),
        "manual_case_truth_dir": str(packets_dir.as_posix()),
        "intended_use": "Replace DeepSeek generated_ap with gold_ap oracle truth for verifier/evaluation upper-bound validation.",
    }
    write_text(args.outdir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote oracle truth replacement for {len(selected)} cases -> {args.outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-case oracle truth packets and gold replacement outputs for AP verifier validation."
    )
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2_DIR)
    parser.add_argument("--judge-revise-dir", type=Path, default=DEFAULT_JUDGE_REVISE_DIR)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
