import argparse
import csv
import gzip
import json
import math
import re
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path


SEMANTIC_TYPES = {
    "T170", "T033", "T074", "T047", "T201", "T046", "T184",
    "T031", "T029", "T023", "T041", "T048", "T061",
}

STOP_NORMS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "he", "her",
    "his", "in", "is", "it", "no", "not", "of", "on", "or", "she", "the",
    "to", "was", "were", "with",
}


def normalize_term(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_norm(text):
    return re.findall(r"[a-z0-9]+", normalize_term(text))


def iter_ngrams(tokens, max_ngram):
    n = len(tokens)
    for start in range(n):
        for end in range(start + 1, min(n, start + max_ngram) + 1):
            yield " ".join(tokens[start:end])


def valid_norm(norm):
    if not norm or norm in STOP_NORMS:
        return False
    compact = norm.replace(" ", "")
    if len(compact) < 3:
        return False
    if compact.isdigit():
        return False
    return True


def read_csv_day_text(path, day):
    if not path.exists():
        return ""
    target = str(day)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("DAY", "")).strip() == target:
                return row.get("TEXT", "") or ""
    return ""


def read_txt(path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def load_cases(summary_csv, require_methods):
    cases = []
    seen = set()
    with Path(summary_csv).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            admission_id = str(row["admission_id"]).strip()
            day = int(row["day"])
            key = (admission_id, day)
            if key in seen:
                continue
            seen.add(key)
            cases.append({"admission_id": admission_id, "day": day})
    return cases


def load_cases_file(cases_file):
    cases = []
    for line in Path(cases_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        admission_id, day = line.split(":", 1)
        cases.append({"admission_id": admission_id.strip(), "day": int(day)})
    return cases


def collect_documents(cases, method_dirs, base_dir, gold_dir, require_all=True):
    docs = []
    skipped = []
    for case in cases:
        admission_id = case["admission_id"]
        day = case["day"]
        texts = {
            "gold": read_csv_day_text(Path(gold_dir) / f"gt_{admission_id}.csv", day),
            "base": read_csv_day_text(Path(base_dir) / f"genpns_{admission_id}.csv", day),
        }
        for method, method_dir in method_dirs.items():
            texts[method] = read_txt(Path(method_dir) / f"{admission_id}_day{day}.txt")
        missing = [name for name, text in texts.items() if not text.strip()]
        if missing and require_all:
            skipped.append({"admission_id": admission_id, "day": day, "missing": ",".join(missing)})
            continue
        docs.append({"admission_id": admission_id, "day": day, "texts": texts})
    return docs, skipped


def candidate_ngrams_from_docs(docs, max_ngram):
    candidates = set()
    for doc in docs:
        for text in doc["texts"].values():
            tokens = tokenize_norm(text)
            for ng in iter_ngrams(tokens, max_ngram):
                if valid_norm(ng):
                    candidates.add(ng)
    return candidates


def umls_zip_parts(umls_dir):
    umls_dir = Path(umls_dir)
    return sorted(umls_dir.glob("*-meta.nlm"))


def open_gzip_member(zip_path, member_name):
    zf = zipfile.ZipFile(zip_path)
    raw = zf.open(member_name)
    return zf, gzip.open(raw, mode="rt", encoding="utf-8", errors="replace", newline="")


def load_allowed_cuis(umls_dir, semantic_types):
    allowed = set()
    for zip_path in umls_zip_parts(umls_dir):
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if member.endswith("MRSTY.RRF.gz"):
                    with gzip.open(zf.open(member), mode="rt", encoding="utf-8", errors="replace", newline="") as f:
                        for line in f:
                            parts = line.rstrip("\n").split("|")
                            if len(parts) > 1 and parts[1] in semantic_types:
                                allowed.add(parts[0])
                    return allowed
    raise FileNotFoundError(f"Could not find MRSTY.RRF.gz under {umls_dir}")


def build_candidate_term_map(umls_dir, candidate_ngrams, allowed_cuis, max_terms=None):
    term_to_cuis = defaultdict(set)
    scanned = 0
    kept = 0
    conso_members = []
    for zip_path in umls_zip_parts(umls_dir):
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if re.search(r"MRCONSO\.RRF\.[a-z]{2}\.gz$", member) or member.endswith("MRCONSO.RRF.gz"):
                    conso_members.append((zip_path, member))

    if not conso_members:
        raise FileNotFoundError(f"Could not find MRCONSO.RRF.*.gz under {umls_dir}")

    for zip_path, member in conso_members:
        zf, f = open_gzip_member(zip_path, member)
        with zf, f:
            for line in f:
                scanned += 1
                parts = line.rstrip("\n").split("|")
                if len(parts) < 17:
                    continue
                cui, lat, suppress, term = parts[0], parts[1], parts[16], parts[14]
                if lat != "ENG" or cui not in allowed_cuis or suppress == "O":
                    continue
                norm = normalize_term(term)
                if norm in candidate_ngrams and valid_norm(norm):
                    term_to_cuis[norm].add(cui)
                    kept += 1
                    if max_terms and kept >= max_terms:
                        return term_to_cuis, {"mrconso_rows_scanned": scanned, "term_rows_kept": kept}
    return term_to_cuis, {"mrconso_rows_scanned": scanned, "term_rows_kept": kept}


def extract_cuis(text, term_to_cuis, max_ngram):
    cuis = set()
    tokens = tokenize_norm(text)
    for ng in iter_ngrams(tokens, max_ngram):
        if ng in term_to_cuis:
            cuis.update(term_to_cuis[ng])
    return cuis


def prf(pred, gold):
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return precision, recall, f1, tp


def mean_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def paired_summary(details, method_a, method_b):
    deltas = [float(row[f"{method_a}_f1"]) - float(row[f"{method_b}_f1"]) for row in details]
    mean_delta, std_delta = mean_std(deltas)
    wins = sum(1 for x in deltas if x > 1e-12)
    losses = sum(1 for x in deltas if x < -1e-12)
    ties = len(deltas) - wins - losses
    return {
        "comparison": f"{method_a}_minus_{method_b}",
        "mean_delta_f1": mean_delta,
        "std_delta_f1": std_delta,
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def parse_method_dirs(items):
    method_dirs = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--method_dirs entries must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name or name in {"gold", "base"}:
            raise ValueError(f"Invalid method name for --method_dirs: {name!r}")
        method_dirs[name] = path.strip()
    return method_dirs


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate AP100 patient-day outputs with UMLS CUI-F1.")
    parser.add_argument("--umls_dir", required=True)
    parser.add_argument("--summary_csv", default="outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2_summary.csv")
    parser.add_argument("--cases_file", default="", help="Optional case list with one admission_id:day per line.")
    parser.add_argument("--base_dir", default="data_ap100_ap/AP/generated/DG/deepseek_api_full_gen/gen/method2")
    parser.add_argument("--v2_dir", default="outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2")
    parser.add_argument("--v2_judge_dir", default="outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2_judge_revise")
    parser.add_argument(
        "--method_dirs",
        nargs="*",
        default=[],
        help="Optional extra generated AP directories as name=path. Names become metric prefixes.",
    )
    parser.add_argument("--gold_dir", default="data_ap100_ap/AP/gold")
    parser.add_argument("--out_dir", default="outputs/ap_memory_gated_scaffold_ap100/umls_eval")
    parser.add_argument("--max_ngram", type=int, default=8)
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    method_dirs = {
        "v2": args.v2_dir,
        "v2_judge": args.v2_judge_dir,
        **parse_method_dirs(args.method_dirs),
    }
    methods = ["base", *method_dirs.keys()]

    if args.cases_file:
        cases = load_cases_file(args.cases_file)
    else:
        cases = load_cases(args.summary_csv, require_methods=methods)
    docs, skipped = collect_documents(
        cases,
        method_dirs,
        args.base_dir,
        args.gold_dir,
        require_all=not args.allow_missing,
    )
    candidates = candidate_ngrams_from_docs(docs, args.max_ngram)
    allowed_cuis = load_allowed_cuis(args.umls_dir, SEMANTIC_TYPES)
    term_to_cuis, index_stats = build_candidate_term_map(args.umls_dir, candidates, allowed_cuis)

    details = []
    method_values = {m: {"precision": [], "recall": [], "f1": [], "pred_cuis": [], "gold_cuis": []}
                     for m in methods}
    for doc in docs:
        gold_cuis = extract_cuis(doc["texts"]["gold"], term_to_cuis, args.max_ngram)
        row = {
            "admission_id": doc["admission_id"],
            "day": doc["day"],
            "gold_cuis": len(gold_cuis),
        }
        for method in methods:
            pred_cuis = extract_cuis(doc["texts"][method], term_to_cuis, args.max_ngram)
            precision, recall, f1, tp = prf(pred_cuis, gold_cuis)
            row.update({
                f"{method}_pred_cuis": len(pred_cuis),
                f"{method}_tp": tp,
                f"{method}_precision": precision,
                f"{method}_recall": recall,
                f"{method}_f1": f1,
            })
            method_values[method]["precision"].append(precision)
            method_values[method]["recall"].append(recall)
            method_values[method]["f1"].append(f1)
            method_values[method]["pred_cuis"].append(len(pred_cuis))
            method_values[method]["gold_cuis"].append(len(gold_cuis))
        details.append(row)

    summary = []
    for method in methods:
        item = {"method": method, "n": len(docs)}
        for metric in ["precision", "recall", "f1", "pred_cuis", "gold_cuis"]:
            mean, std = mean_std(method_values[method][metric])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        summary.append(item)

    paired = []
    for method in methods:
        if method != "base":
            paired.append(paired_summary(details, method, "base"))
    for i, method_a in enumerate(methods):
        if method_a == "base":
            continue
        for method_b in methods[i + 1:]:
            if method_b != "base":
                paired.append(paired_summary(details, method_b, method_a))

    out_dir = Path(args.out_dir)
    detail_fields = list(details[0].keys()) if details else ["admission_id", "day"]
    write_csv(out_dir / "ap100_umls_cui_f1_detail.csv", details, detail_fields)
    write_csv(out_dir / "ap100_umls_cui_f1_summary.csv", summary, list(summary[0].keys()) if summary else ["method"])
    write_csv(out_dir / "ap100_umls_cui_f1_paired.csv", paired, list(paired[0].keys()))
    if skipped:
        write_csv(out_dir / "ap100_umls_cui_f1_skipped.csv", skipped, ["admission_id", "day", "missing"])

    metadata = {
        "umls_dir": str(Path(args.umls_dir).resolve()),
        "n_cases_from_summary": len(cases),
        "n_evaluated": len(docs),
        "n_skipped": len(skipped),
        "max_ngram": args.max_ngram,
        "semantic_types": sorted(SEMANTIC_TYPES),
        "methods": methods,
        "method_dirs": method_dirs,
        "candidate_ngrams": len(candidates),
        "allowed_cuis": len(allowed_cuis),
        "matched_terms": len(term_to_cuis),
        **index_stats,
    }
    (out_dir / "ap100_umls_cui_f1_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "paired": paired, "metadata": metadata}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
