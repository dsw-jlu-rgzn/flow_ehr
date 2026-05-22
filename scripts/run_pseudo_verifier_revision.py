from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_SELECTED = Path("outputs/oracle_claim_verifier_qwen653/selected_cases_with_failure_seed.json")
DEFAULT_LABELS = Path(
    "outputs/oracle_claim_verifier_qwen653/pseudo_verifier_oracle/claims_oracle_labeled_pseudo.jsonl"
)
DEFAULT_V2_DIR = Path("outputs/ap_memory_gated_scaffold_ap100/ap100full_generated_method2_gen_v2")
DEFAULT_OUTDIR = Path("outputs/oracle_claim_verifier_qwen653/pseudo_verifier_revision_smoke5")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean(text: str) -> str:
    text = text.replace("鈥檚", "'s").replace("掳", " deg ").replace("鈮?", ">=")
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_match(text: str) -> str:
    text = clean(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_line_sentences(line: str) -> list[str]:
    prefix = ""
    body = line
    match = re.match(r"^(\s*[-*]\s+)(.+)$", line)
    if match:
        prefix = match.group(1)
        body = match.group(2)
    pieces = re.split(r"(?<=[.!?])\s+|;\s+", body)
    out = []
    for piece in pieces:
        piece = piece.strip()
        if piece:
            out.append(prefix + piece if prefix else piece)
    return out or [line]


def strip_markup(text: str) -> str:
    text = re.sub(r"^\s*[-*]\s+", "", text)
    text = re.sub(r"^\s*\d+\.\s*", "", text)
    text = text.replace("**", "").strip(" :")
    return text


def labels_for_text(text: str, labels: list[dict]) -> list[dict]:
    ntext = normalize_for_match(text)
    matches = []
    for label in labels:
        claim = normalize_for_match(label["claim_text"])
        sent = normalize_for_match(label["original_sentence"])
        if claim and claim == ntext:
            matches.append(label)
        elif claim and len(claim.split()) >= 4 and claim in ntext:
            matches.append(label)
        elif sent and sent == ntext:
            matches.append(label)
    return matches


def revise_text(original: str, labels: list[dict]) -> tuple[str, dict]:
    deleted_claim_ids = []
    rewritten_claim_ids = []
    downgraded_claim_ids = []
    kept_claim_ids = [row["claim_id"] for row in labels if row["binary_label"] == "KEEP"]

    labels_by_claim_norm = {normalize_for_match(row["claim_text"]): row for row in labels}
    labels_by_sentence_norm = defaultdict(list)
    for row in labels:
        labels_by_sentence_norm[normalize_for_match(row["original_sentence"])].append(row)

    revised_lines: list[str] = []
    inserted_rewrites: set[str] = set()
    for raw_line in original.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            revised_lines.append(line)
            continue

        stripped = strip_markup(line)
        stripped_norm = normalize_for_match(stripped)
        heading_label = labels_by_claim_norm.get(stripped_norm)
        if heading_label and heading_label["binary_label"] == "FIX":
            if heading_label["action"] == "REWRITE" and heading_label.get("oracle_rewrite"):
                rewrite = heading_label["oracle_rewrite"].strip()
                if rewrite not in inserted_rewrites:
                    revised_lines.append(re.sub(re.escape(stripped), rewrite, line, count=1))
                    inserted_rewrites.add(rewrite)
                    rewritten_claim_ids.append(heading_label["claim_id"])
            else:
                deleted_claim_ids.append(heading_label["claim_id"])
            continue

        chunks = split_line_sentences(line)
        new_chunks: list[str] = []
        for chunk in chunks:
            chunk_norm = normalize_for_match(strip_markup(chunk))
            chunk_labels = labels_by_sentence_norm.get(chunk_norm) or labels_for_text(strip_markup(chunk), labels)
            if not chunk_labels:
                new_chunks.append(chunk)
                continue
            fix_labels = [row for row in chunk_labels if row["binary_label"] == "FIX"]
            keep_labels = [row for row in chunk_labels if row["binary_label"] == "KEEP"]
            rewrite_labels = [row for row in fix_labels if row["action"] == "REWRITE" and row.get("oracle_rewrite")]
            delete_labels = [row for row in fix_labels if row["action"] == "DELETE" or not row.get("oracle_rewrite")]

            if rewrite_labels:
                for row in rewrite_labels:
                    rewritten_claim_ids.append(row["claim_id"])
                    # Preserve the surrounding sentence when possible; replace only the
                    # unsupported claim span. If exact text is not present, add one
                    # rewrite once and drop the unsafe sentence.
                    rewrite = row["oracle_rewrite"].strip()
                    if not rewrite or rewrite in inserted_rewrites:
                        continue
                    if row["claim_text"].rstrip(".") in chunk:
                        updated = chunk.replace(row["claim_text"].rstrip("."), rewrite, 1)
                        new_chunks.append(updated)
                    elif row["claim_text"] in chunk:
                        updated = chunk.replace(row["claim_text"], rewrite, 1)
                        new_chunks.append(updated)
                    else:
                        indent = re.match(r"^(\s*[-*]\s*)", chunk).group(1) if re.match(r"^(\s*[-*]\s*)", chunk) else ""
                        new_chunks.append(indent + rewrite)
                    inserted_rewrites.add(rewrite)
                for row in delete_labels:
                    deleted_claim_ids.append(row["claim_id"])
                continue

            if fix_labels and not keep_labels:
                for row in fix_labels:
                    deleted_claim_ids.append(row["claim_id"])
                continue

            # Mixed keep/delete without a safe rewrite: remove only exact unsafe claim spans.
            updated = chunk
            for row in delete_labels:
                deleted_claim_ids.append(row["claim_id"])
                for unsafe in [row["claim_text"].rstrip("."), row["claim_text"], row["original_sentence"]]:
                    unsafe = unsafe.strip()
                    if unsafe and unsafe in updated:
                        updated = updated.replace(unsafe, "").strip(" ;,-")
            if updated and normalize_for_match(updated):
                new_chunks.append(updated)

        if new_chunks:
            if len(new_chunks) == 1:
                revised_lines.append(new_chunks[0])
            else:
                revised_lines.extend(new_chunks)

    revised = "\n".join(revised_lines)
    revised = re.sub(r"\n{3,}", "\n\n", revised).strip() + "\n"
    stats = {
        "deleted_claim_ids": sorted(set(deleted_claim_ids)),
        "rewritten_claim_ids": sorted(set(rewritten_claim_ids)),
        "downgraded_claim_ids": sorted(set(downgraded_claim_ids)),
        "kept_claim_ids": sorted(set(kept_claim_ids)),
    }
    return revised, stats


def build_case_review(case_id: str, labels: list[dict], stats: dict, original: str, revised: str) -> str:
    lines = [
        f"# {case_id} Pseudo Verify + Minimal Reviser",
        "",
        f"- claims: {len(labels)}",
        f"- KEEP: {sum(1 for row in labels if row['binary_label'] == 'KEEP')}",
        f"- FIX: {sum(1 for row in labels if row['binary_label'] == 'FIX')}",
        f"- deleted: {len(stats['deleted_claim_ids'])}",
        f"- rewritten: {len(stats['rewritten_claim_ids'])}",
        "",
        "## Rewritten Claims",
        "",
    ]
    label_by_id = {row["claim_id"]: row for row in labels}
    for claim_id in stats["rewritten_claim_ids"]:
        row = label_by_id[claim_id]
        lines.extend(
            [
                f"- `{claim_id}` {row['claim_type']} {row['support_label']}: {row['claim_text']}",
                f"  rewrite: {row.get('oracle_rewrite', '')}",
            ]
        )
    lines.extend(["", "## Deleted Claims", ""])
    for claim_id in stats["deleted_claim_ids"][:80]:
        row = label_by_id[claim_id]
        lines.append(f"- `{claim_id}` {row['claim_type']} {row['support_label']}: {row['claim_text']}")
    lines.extend(
        [
            "",
            "## Original V2",
            "",
            "```text",
            original.strip(),
            "```",
            "",
            "## Revised Output",
            "",
            "```text",
            revised.strip(),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pseudo verifier labels through a deterministic minimal reviser.")
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2_DIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    selected = json.loads(args.selected.read_text(encoding="utf-8"))[: args.limit]
    labels_by_case: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(args.labels):
        labels_by_case[row["case_id"]].append(row)

    output_rows = []
    verify_rows = []
    review_dir = args.outdir / "case_reviews"
    revised_dir = args.outdir / "revised_txt"
    review_dir.mkdir(parents=True, exist_ok=True)
    revised_dir.mkdir(parents=True, exist_ok=True)

    for item in selected:
        case_id = item["case_id"]
        original = (args.v2_dir / f"{case_id}.txt").read_text(encoding="utf-8", errors="replace")
        labels = labels_by_case[case_id]
        verify_rows.extend(labels)
        revised, stats = revise_text(original, labels)
        (revised_dir / f"{case_id}.txt").write_text(revised, encoding="utf-8", newline="\n")
        (review_dir / f"{case_id}.md").write_text(
            build_case_review(case_id, labels, stats, original, revised),
            encoding="utf-8",
            newline="\n",
        )
        output_rows.append(
            {
                "case_id": case_id,
                "source_output": "memory_gated_scaffold_v2",
                "revised_method": "v2_pseudo_oracle_claim_verifier_minimal_reviser",
                "original_output": original,
                "oracle_revised_output": revised,
                **stats,
            }
        )

    write_jsonl(args.outdir / "oracle_revised_outputs_smoke5.jsonl", output_rows)
    write_jsonl(args.outdir / "verify_labels_smoke5.jsonl", verify_rows)
    summary = {
        "n_cases": len(output_rows),
        "case_ids": [row["case_id"] for row in output_rows],
        "revised_txt_dir": str(revised_dir.as_posix()),
        "case_review_dir": str(review_dir.as_posix()),
        "output_jsonl": str((args.outdir / "oracle_revised_outputs_smoke5.jsonl").as_posix()),
        "verify_labels_jsonl": str((args.outdir / "verify_labels_smoke5.jsonl").as_posix()),
    }
    (args.outdir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote smoke revision for {len(output_rows)} cases -> {args.outdir}")


if __name__ == "__main__":
    main()
