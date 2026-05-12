"""
Fair paired comparison for AP generation outputs.

The script compares two or more generated AP directories against the same gold
directory. It uses the intersection of admission IDs for every method so the
reported deltas are paired and fair.

Expected generated layout:
  <gen_dir>/method-1/genpns_<hadm_id>.csv
  <gen_dir>/method1/genpns_<hadm_id>.csv
  <gen_dir>/method2/genpns_<hadm_id>.csv

Example:
  python evaluation/compare_ap_generation.py \
    --gt-dir data/AP/gold \
    --run baseline=data/AP/generated/EE/mistral/gt_v2 \
    --run embed_prev=data/AP/generated/EE/mistral/embedding_previous_note \
    --output-csv outputs/ap_compare.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METHODS = ["method-1", "method1", "method2"]


@dataclass(frozen=True)
class RunSpec:
    name: str
    path: Path


def parse_run(value: str) -> RunSpec:
    if "=" not in value:
        path = Path(value)
        return RunSpec(path.name, path)
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("--run name cannot be empty")
    return RunSpec(name, Path(path))


def admission_id_from_generated(path: Path) -> str:
    stem = path.stem
    if not stem.startswith("genpns_"):
        return ""
    return stem.replace("genpns_", "", 1)


def read_generated_text(path: Path) -> str:
    df = pd.read_csv(path)
    if "TEXT" not in df.columns:
        return ""
    return " ".join(df["TEXT"].fillna("").astype(str).tolist())


def read_gold_text(path: Path) -> str:
    df = pd.read_csv(path)
    if "TEXT" not in df.columns:
        return ""
    return " ".join(df["TEXT"].fillna("").astype(str).tolist())


def tokenize(text: str) -> list[str]:
    return str(text).lower().split()


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_len(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_sapbert(device: str = ""):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    resolved_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(resolved_device)
    model.eval()

    @torch.no_grad()
    def embed(text: str):
        encoded = tokenizer(
            str(text),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(resolved_device)
        outputs = model(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled, dim=-1)

    def score(pred: str, gold: str) -> float:
        pred_emb = embed(pred)
        gold_emb = embed(gold)
        return float((pred_emb * gold_emb).sum(dim=-1).item())

    return score


def collect_ids(run: RunSpec, method: str) -> set[str]:
    method_dir = run.path / method
    if not method_dir.is_dir():
        return set()
    return {
        admission_id_from_generated(path)
        for path in method_dir.glob("genpns_*.csv")
        if admission_id_from_generated(path)
    }


def evaluate_run_method(
    run: RunSpec,
    method: str,
    ids: list[str],
    gt_dir: Path,
    sapbert_score=None,
) -> list[dict]:
    rows = []
    for admission_id in ids:
        gen_file = run.path / method / f"genpns_{admission_id}.csv"
        gold_file = gt_dir / f"gt_{admission_id}.csv"
        pred_text = read_generated_text(gen_file)
        gold_text = read_gold_text(gold_file)
        row = {
            "run": run.name,
            "method": method,
            "admission_id": admission_id,
            "rouge_l_f1": rouge_l_f1(pred_text, gold_text),
            "pred_words": len(tokenize(pred_text)),
            "gold_words": len(tokenize(gold_text)),
        }
        if sapbert_score is not None:
            row["sapbert_cosine"] = sapbert_score(pred_text, gold_text)
        rows.append(row)
    return rows


def summarize(df: pd.DataFrame, baseline_name: str) -> pd.DataFrame:
    metric_cols = ["rouge_l_f1"]
    if "sapbert_cosine" in df.columns:
        metric_cols.append("sapbert_cosine")

    summary_rows = []
    for method, method_df in df.groupby("method", sort=False):
        baseline_df = method_df[method_df["run"] == baseline_name]
        for run, run_df in method_df.groupby("run", sort=False):
            row = {
                "method": method,
                "run": run,
                "n": int(len(run_df)),
                "pred_words_mean": float(run_df["pred_words"].mean()),
                "gold_words_mean": float(run_df["gold_words"].mean()),
            }
            for metric in metric_cols:
                row[f"{metric}_mean"] = float(run_df[metric].mean())
                row[f"{metric}_std"] = float(run_df[metric].std(ddof=0))
                if run != baseline_name and not baseline_df.empty:
                    paired = run_df.merge(
                        baseline_df[["admission_id", metric]],
                        on="admission_id",
                        suffixes=("", "_baseline"),
                    )
                    row[f"{metric}_delta_vs_{baseline_name}"] = float(
                        (paired[metric] - paired[f"{metric}_baseline"]).mean()
                    )
            summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def print_summary(summary_df: pd.DataFrame, baseline_name: str) -> None:
    for method, method_df in summary_df.groupby("method", sort=False):
        print(f"\n=== {method} ===")
        for _, row in method_df.iterrows():
            line = (
                f"{row['run']}: n={int(row['n'])}, "
                f"ROUGE-L={row['rouge_l_f1_mean'] * 100:.2f} +/- {row['rouge_l_f1_std'] * 100:.2f}, "
                f"pred_words={row['pred_words_mean']:.1f}"
            )
            delta_col = f"rouge_l_f1_delta_vs_{baseline_name}"
            if delta_col in row and pd.notna(row[delta_col]):
                line += f", delta={row[delta_col] * 100:+.2f}"
            if "sapbert_cosine_mean" in row and pd.notna(row["sapbert_cosine_mean"]):
                line += f", SapBERT-cos={row['sapbert_cosine_mean'] * 100:.2f}"
            print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AP generation runs fairly on paired IDs.")
    parser.add_argument("--gt-dir", default="data/AP/gold")
    parser.add_argument("--run", type=parse_run, action="append", required=True, help="name=generated_dir")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--baseline", default="", help="Run name to use for paired deltas. Defaults to first --run.")
    parser.add_argument("--output-csv", default="outputs/ap_generation_comparison.csv")
    parser.add_argument("--summary-csv", default="outputs/ap_generation_comparison_summary.csv")
    parser.add_argument("--sapbert", action="store_true", help="Also compute SapBERT cosine. Downloads/loads SapBERT.")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_dir = Path(args.gt_dir)
    runs = args.run
    baseline_name = args.baseline or runs[0].name

    sapbert_score = load_sapbert(args.device) if args.sapbert else None

    rows = []
    for method in args.methods:
        common_ids = None
        for run in runs:
            ids = collect_ids(run, method)
            common_ids = ids if common_ids is None else common_ids.intersection(ids)

        common_ids = common_ids or set()
        common_ids = {
            admission_id
            for admission_id in common_ids
            if (gt_dir / f"gt_{admission_id}.csv").is_file()
        }
        sorted_ids = sorted(common_ids)
        if not sorted_ids:
            print(f"Warning: no paired IDs for method {method}")
            continue

        print(f"Method {method}: evaluating paired IDs {', '.join(sorted_ids)}")
        for run in runs:
            rows.extend(evaluate_run_method(run, method, sorted_ids, gt_dir, sapbert_score))

    if not rows:
        raise SystemExit("No comparable generated files found.")

    detail_df = pd.DataFrame(rows)
    summary_df = summarize(detail_df, baseline_name)

    output_csv = Path(args.output_csv)
    summary_csv = Path(args.summary_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    summary_df.to_csv(summary_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    print_summary(summary_df, baseline_name)
    print(f"\nDetail CSV: {output_csv}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
