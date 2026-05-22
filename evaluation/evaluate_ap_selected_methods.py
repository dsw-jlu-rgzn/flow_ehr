"""Evaluate selected AP patient-days across method directories.

Supports base CSV outputs plus arbitrary per-case TXT directories. Intended for
small smoke tests where methods are not all in the original AP generated CSV
layout.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import pandas as pd


def tokenize(text: str) -> list[str]:
    return str(text or "").lower().split()


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b, start=1):
            curr.append(prev[j - 1] + 1 if x == y else max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(pred: str, gold: str) -> float:
    p = tokenize(pred)
    g = tokenize(gold)
    if not p or not g:
        return 0.0
    lcs = lcs_len(p, g)
    precision = lcs / len(p)
    recall = lcs / len(g)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def read_csv_day(path: Path, day: int) -> str:
    if not path.exists():
        return ""
    df = pd.read_csv(path)
    rows = df[df["DAY"].astype(int).eq(day)]
    return "\n".join(rows["TEXT"].fillna("").astype(str).tolist())


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def load_cases(path: Path) -> list[tuple[str, int]]:
    cases = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        admission_id, day = line.split(":", 1)
        cases.append((admission_id.strip(), int(day)))
    return cases


def parse_txt_methods(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        name, path = item.split("=", 1)
        out[name] = Path(path)
    return out


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-file", required=True)
    parser.add_argument("--gold-dir", default="data_ap100_ap/AP/gold")
    parser.add_argument("--base-dir", default="data_ap100_ap/AP/generated/DG/deepseek_api_full_gen/gen/method2")
    parser.add_argument("--txt-method", action="append", default=[], help="name=directory with <admission>_day<day>.txt")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cases = load_cases(Path(args.cases_file))
    methods = {"base": None, **parse_txt_methods(args.txt_method)}
    rows = []
    for admission_id, day in cases:
        gold = read_csv_day(Path(args.gold_dir) / f"gt_{admission_id}.csv", day)
        for method, method_dir in methods.items():
            if method == "base":
                pred = read_csv_day(Path(args.base_dir) / f"genpns_{admission_id}.csv", day)
            else:
                pred = read_txt(method_dir / f"{admission_id}_day{day}.txt")
            rows.append(
                {
                    "admission_id": admission_id,
                    "day": day,
                    "method": method,
                    "rouge_l_f1": rouge_l_f1(pred, gold),
                    "pred_words": len(tokenize(pred)),
                    "gold_words": len(tokenize(gold)),
                    "missing_pred": int(not bool(pred.strip())),
                }
            )

    detail = pd.DataFrame(rows)
    summary_rows = []
    for method, df in detail.groupby("method", sort=False):
        item = {
            "method": method,
            "n": int(len(df)),
            "rouge_l_f1_mean": mean(df["rouge_l_f1"].tolist()),
            "pred_words_mean": mean(df["pred_words"].tolist()),
            "missing_pred": int(df["missing_pred"].sum()),
        }
        if method != "base":
            paired = df.merge(
                detail[detail["method"].eq("base")][["admission_id", "day", "rouge_l_f1"]],
                on=["admission_id", "day"],
                suffixes=("", "_base"),
            )
            item["rouge_l_delta_vs_base"] = mean((paired["rouge_l_f1"] - paired["rouge_l_f1_base"]).tolist())
        summary_rows.append(item)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / "ap_selected_auto_metrics_detail.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(summary_rows).to_csv(out_dir / "ap_selected_auto_metrics_summary.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
