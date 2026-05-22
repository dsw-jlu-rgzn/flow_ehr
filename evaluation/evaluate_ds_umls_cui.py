"""Exact-match UMLS CUI-F1 evaluation for DS sections.

This is a lightweight fallback when QuickUMLS is unavailable. It builds a
section-specific CUI matcher by scanning MRSTY and MRCONSO from a raw UMLS
release archive and matching exact normalized UMLS strings against n-grams from
generated/gold sections.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path

from evaluate_ds_light import SECTIONS, extract_generated_sections, extract_gold_sections


SEMANTIC_TYPES = {
    "Diagnosis": [
        "T047",
        "T061",
        "T023",
        "T170",
        "T201",
        "T048",
        "T074",
        "T121",
        "T031",
        "T060",
        "T056",
        "T034",
        "T029",
        "T046",
        "T191",
        "T190",
        "T184",
        "T033",
        "T037",
        "T041",
        "T109",
    ],
    "Hospital Course": [
        "T121",
        "T170",
        "T033",
        "T109",
        "T058",
        "T047",
        "T031",
        "T061",
        "T023",
        "T046",
        "T201",
        "T074",
        "T184",
        "T116",
        "T123",
        "T059",
        "T060",
        "T195",
        "T041",
        "T029",
    ],
    "Discharge Instructions": [
        "T121",
        "T033",
        "T170",
        "T184",
        "T109",
        "T047",
        "T061",
        "T023",
        "T058",
        "T074",
        "T031",
        "T201",
        "T046",
        "T041",
        "T195",
        "T040",
        "T116",
        "T037",
        "T039",
        "T123",
    ],
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[\*\*.*?\*\*\]", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def ngrams(text: str, max_n: int) -> set[str]:
    toks = tokens(text)
    out: set[str] = set()
    for i in range(len(toks)):
        for n in range(1, max_n + 1):
            if i + n > len(toks):
                break
            gram_toks = toks[i : i + n]
            if n == 1 and (gram_toks[0] in STOPWORDS or len(gram_toks[0]) < 4):
                continue
            if gram_toks[0] in STOPWORDS or gram_toks[-1] in STOPWORDS:
                continue
            out.add(" ".join(gram_toks))
    return out


def admission_id_from_generated(path: Path) -> str:
    match = re.match(r"48h_all_abs_(\d+)\.txt$", path.name)
    return match.group(1) if match else ""


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def find_umls_archive(umls_dir: Path) -> Path:
    candidates = sorted(umls_dir.glob("*-1-meta.nlm"))
    if not candidates:
        candidates = sorted(umls_dir.glob("*.nlm"))
    if not candidates:
        raise FileNotFoundError(f"No .nlm UMLS archive found under {umls_dir}")
    return candidates[0]


def list_archive_members(archive: Path) -> list[str]:
    result = subprocess.run(
        ["tar", "-tf", str(archive)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8", errors="replace").splitlines()


def extract_member_bytes(archive: Path, member_name: str) -> bytes:
    result = subprocess.run(
        ["tar", "-xOf", str(archive), member_name],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def read_member_bytes(archive: Path, member_name_pattern: str) -> bytes:
    for member_name in list_archive_members(archive):
        if re.search(member_name_pattern, member_name):
            data = extract_member_bytes(archive, member_name)
            return gzip.decompress(data) if member_name.endswith(".gz") else data
    raise FileNotFoundError(member_name_pattern)


def load_cui_semtypes(umls_archive: Path) -> dict[str, set[str]]:
    cui_to_sty: dict[str, set[str]] = defaultdict(set)
    data = read_member_bytes(umls_archive, r"/MRSTY\.RRF\.gz$")
    for line in data.decode("utf-8", errors="replace").splitlines():
        parts = line.split("|")
        if len(parts) >= 2:
            cui_to_sty[parts[0]].add(parts[1])
    return cui_to_sty


def iter_mrconso_lines(umls_archive: Path):
    members = [
        name
        for name in list_archive_members(umls_archive)
        if re.search(r"/MRCONSO\.RRF\.[a-z]{2}\.gz$", name)
    ]
    for member_name in sorted(members):
        proc = subprocess.Popen(
            ["tar", "-xOf", str(umls_archive), member_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdout is not None
        with gzip.GzipFile(fileobj=proc.stdout) as gz:
            for raw in gz:
                yield raw.decode("utf-8", errors="replace")
        proc.stdout.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(f"tar failed for {member_name}: {stderr}")


def build_text_records(gen_dir: Path, gt_dir: Path) -> list[dict[str, str]]:
    records = []
    for gen_file in sorted(gen_dir.glob("48h_all_abs_*.txt")):
        admission_id = admission_id_from_generated(gen_file)
        gold_file = gt_dir / f"gtsummary_{admission_id}.txt"
        if not gold_file.exists():
            continue
        gen_sections = extract_generated_sections(read_text(gen_file))
        gold_sections = extract_gold_sections(read_text(gold_file))
        for section in SECTIONS:
            records.append(
                {
                    "admission_id": admission_id,
                    "section": section,
                    "side": "gen",
                    "text": gen_sections[section],
                }
            )
            records.append(
                {
                    "admission_id": admission_id,
                    "section": section,
                    "side": "gold",
                    "text": gold_sections[section],
                }
            )
    return records


def build_candidate_terms(records: list[dict[str, str]], max_ngram: int) -> set[str]:
    terms: set[str] = set()
    for record in records:
        terms.update(ngrams(record["text"], max_ngram))
    return terms


def build_term_cui_index(
    umls_archive: Path,
    candidate_terms: set[str],
    cui_to_sty: dict[str, set[str]],
    allowed_sty_union: set[str],
) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for line_no, line in enumerate(iter_mrconso_lines(umls_archive), start=1):
        parts = line.rstrip("\n").split("|")
        if len(parts) < 15:
            continue
        cui = parts[0]
        lat = parts[1]
        suppress = parts[16] if len(parts) > 16 else ""
        if lat != "ENG" or suppress == "O":
            continue
        if not (cui_to_sty.get(cui, set()) & allowed_sty_union):
            continue
        term = normalize(parts[14])
        if term in candidate_terms:
            index[term].add(cui)
    return index


def extract_cuis(text: str, section: str, term_to_cuis: dict[str, set[str]], max_ngram: int) -> set[str]:
    cuis: set[str] = set()
    allowed = set(SEMANTIC_TYPES[section])
    for term in ngrams(text, max_ngram):
        for cui in term_to_cuis.get(term, set()):
            cuis.add(cui)
    return cuis


def score_sets(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact-match UMLS CUI-F1 for DS sections.")
    parser.add_argument("--gen-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--umls-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--max-ngram", type=int, default=6)
    args = parser.parse_args()

    gen_dir = Path(args.gen_dir)
    gt_dir = Path(args.gt_dir)
    umls_archive = find_umls_archive(Path(args.umls_dir))

    records = build_text_records(gen_dir, gt_dir)
    candidate_terms = build_candidate_terms(records, args.max_ngram)
    cui_to_sty = load_cui_semtypes(umls_archive)
    allowed_union = set().union(*[set(v) for v in SEMANTIC_TYPES.values()])
    term_to_cuis = build_term_cui_index(umls_archive, candidate_terms, cui_to_sty, allowed_union)

    by_key: dict[tuple[str, str, str], str] = {}
    for record in records:
        by_key[(record["admission_id"], record["section"], record["side"])] = record["text"]

    rows = []
    for admission_id in sorted({r["admission_id"] for r in records}):
        for section in SECTIONS:
            gen_text = by_key.get((admission_id, section, "gen"), "")
            gold_text = by_key.get((admission_id, section, "gold"), "")
            gen_cuis = extract_cuis(gen_text, section, term_to_cuis, args.max_ngram)
            gold_cuis = extract_cuis(gold_text, section, term_to_cuis, args.max_ngram)
            precision, recall, f1 = score_sets(gen_cuis, gold_cuis)
            rows.append(
                {
                    "admission_id": admission_id,
                    "section": section,
                    "precision": precision,
                    "recall": recall,
                    "cui_f1": f1,
                    "gen_cuis": len(gen_cuis),
                    "gold_cuis": len(gold_cuis),
                    "overlap_cuis": len(gen_cuis & gold_cuis),
                }
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"UMLS archive: {umls_archive}")
    print(f"Candidate terms: {len(candidate_terms)}")
    print(f"Matched UMLS terms: {len(term_to_cuis)}")
    print(f"Detail CSV: {output_csv}")
    for section in SECTIONS:
        vals = [r["cui_f1"] for r in rows if r["section"] == section]
        mean = sum(vals) / len(vals) if vals else 0.0
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if vals else 0.0
        print(f"{section}: {mean * 100:.2f} +/- {std * 100:.2f} (n={len(vals)})")


if __name__ == "__main__":
    main()
