"""
No-training DeepSeek API prefilter for AP generation.

This is an API-based alternative to embedding_prefilter_no_train.py for
environments where loading a local HuggingFace embedding model is impractical.
It asks DeepSeek to select the most clinically relevant same-day EHR snippets
for writing an ICU Assessment & Plan, then exports a condensed AP input folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd

from deepseek_api_generation import DEFAULT_API_URL, call_deepseek
from flow_prefilter_mvp import build_trend_snippets, normalize_text, row_to_snippet


def make_query_context(day_df: pd.DataFrame, previous_note: str, query_mode: str) -> str:
    note_df = day_df[day_df["IS_NOTE"] == 1]
    data_df = day_df[day_df["IS_NOTE"] == 0]

    if query_mode == "oracle_gt" and not note_df.empty:
        return "Current-day target note:\n" + normalize_text(note_df.iloc[-1]["TEXT"])
    if query_mode == "previous_note" and previous_note:
        return "Previous progress note:\n" + previous_note

    snippets = [row_to_snippet(row) for _, row in data_df.iterrows()]
    snippets.extend(build_trend_snippets(day_df))
    return "Current-day EHR context:\n" + "\n".join(snippets)


def parse_selected_indices(text: str, max_index: int) -> list[int]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", cleaned, flags=re.DOTALL)
        if not match:
            payload = None
        else:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None

    if isinstance(payload, dict):
        candidates = payload.get("selected_indices", payload.get("indices", []))
    elif isinstance(payload, list):
        candidates = payload
    else:
        candidates = re.findall(r"\d+", cleaned)

    selected = []
    seen = set()
    for value in candidates:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < max_index and idx not in seen:
            selected.append(idx)
            seen.add(idx)
    return selected


def build_selection_prompt(
    query_context: str,
    snippets: list[str],
    top_k: int,
    query_mode: str,
) -> str:
    numbered = "\n".join(f"[{idx}] {snippet}" for idx, snippet in enumerate(snippets))
    return f"""You are selecting EHR evidence for an ICU daily Assessment & Plan note.

Query mode: {query_mode}

Selection goal:
- Keep the {top_k} snippets most useful for writing the current day's Assessment & Plan.
- Prioritize active problems, abnormal or changing labs/vitals, respiratory status, medications, procedures, imaging impressions, cultures, and care decisions.
- Avoid redundant snippets unless they show a meaningful change.
- Include trend snippets when they summarize important changes.

Context signal:
{query_context}

Candidate snippets:
{numbered}

Return only valid JSON in this exact shape:
{{"selected_indices":[0,1,2]}}
"""


def select_day_rows(
    day_df: pd.DataFrame,
    previous_note: str,
    query_mode: str,
    top_k: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, str, dict]:
    data_df = day_df[day_df["IS_NOTE"] == 0].copy()
    note_df = day_df[day_df["IS_NOTE"] == 1].copy()
    if data_df.empty:
        next_note = previous_note
        if not note_df.empty:
            next_note = normalize_text(note_df.iloc[-1]["TEXT"])
        return day_df, next_note, {"api_used": False, "selected": 0}

    raw_snippets = [row_to_snippet(row) for _, row in data_df.iterrows()]
    trend_snippets = build_trend_snippets(day_df)
    all_snippets = raw_snippets + trend_snippets

    if len(all_snippets) <= top_k:
        selected_indices = list(range(len(all_snippets)))
        api_used = False
    else:
        prompt = build_selection_prompt(
            query_context=make_query_context(day_df, previous_note, query_mode),
            snippets=all_snippets,
            top_k=top_k,
            query_mode=query_mode,
        )
        response = call_deepseek(
            prompt=prompt,
            model=args.model,
            api_url=args.api_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
        )
        selected_indices = parse_selected_indices(response, max_index=len(all_snippets))
        api_used = True

        if not selected_indices:
            selected_indices = list(range(min(top_k, len(all_snippets))))
        selected_indices = selected_indices[:top_k]

    selected_raw_indices = []
    synthetic_rows = []
    for idx in selected_indices:
        if idx < len(raw_snippets):
            selected_raw_indices.append(idx)
        else:
            template = data_df.iloc[-1].copy()
            template["TEXT"] = all_snippets[idx]
            template["IS_NOTE"] = 0
            template["DEEPSEEK_PREFILTER_RANK"] = len(selected_raw_indices) + len(synthetic_rows)
            synthetic_rows.append(template)

    selected_data = data_df.iloc[sorted(set(selected_raw_indices))].copy()
    if not selected_data.empty:
        rank_by_idx = {idx: rank for rank, idx in enumerate(selected_indices) if idx < len(raw_snippets)}
        selected_data["DEEPSEEK_PREFILTER_RANK"] = [
            rank_by_idx.get(idx, len(rank_by_idx)) for idx in sorted(set(selected_raw_indices))
        ]

    frames = [frame for frame in [selected_data, pd.DataFrame(synthetic_rows), note_df] if not frame.empty]
    output = pd.concat(frames, ignore_index=True) if frames else day_df
    sort_cols = [col for col in ["DAY", "TIME", "REL_TIME", "IS_NOTE"] if col in output.columns]
    if sort_cols:
        output = output.sort_values(sort_cols, kind="stable")

    next_note = previous_note
    if not note_df.empty:
        next_note = normalize_text(note_df.iloc[-1]["TEXT"])

    return output, next_note, {
        "api_used": api_used,
        "candidate_count": len(all_snippets),
        "trend_count": len(trend_snippets),
        "selected": len(selected_indices),
        "selected_raw": len(set(selected_raw_indices)),
        "selected_trend": len(synthetic_rows),
    }


def export_inputs(args: argparse.Namespace) -> None:
    input_path = Path(args.inputdir)
    output_path = Path(args.outputdir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for csv_path in sorted(input_path.glob("input_*.csv")):
        df = pd.read_csv(csv_path)
        required = {"DAY", "TIME", "REL_TIME", "TEXT", "IS_NOTE"}
        missing = required.difference(df.columns)
        if missing:
            print(f"Skipping {csv_path.name}: missing columns {sorted(missing)}")
            continue

        previous_note = ""
        output_days = []
        for day, day_df in df.groupby("DAY", sort=True):
            before = int((day_df["IS_NOTE"] == 0).sum())
            filtered_day, previous_note, stats = select_day_rows(
                day_df=day_df,
                previous_note=previous_note,
                query_mode=args.query_mode,
                top_k=args.top_k,
                args=args,
            )
            after = int((filtered_day["IS_NOTE"] == 0).sum())
            output_days.append(filtered_day)
            summary_rows.append(
                {
                    "file": csv_path.name,
                    "day": day,
                    "query_mode": args.query_mode,
                    "non_note_before": before,
                    "non_note_after": after,
                    **stats,
                }
            )
            print(f"{csv_path.name} day={day}: {before} -> {after} non-note rows")

        if output_days:
            output_file = output_path / csv_path.name
            pd.concat(output_days, ignore_index=True).to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)
            print(f"Wrote {output_file}")

    summary_file = output_path / "_deepseek_prefilter_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_file, index=False)
    print(f"Wrote {summary_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-training DeepSeek API AP prefilter.")
    parser.add_argument("--inputdir", default="data/AP/input")
    parser.add_argument("--outputdir", default="data/AP/input_deepseek_topk")
    parser.add_argument("--query-mode", choices=["day_context", "previous_note", "oracle_gt"], default="day_context")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    export_inputs(parse_args())


if __name__ == "__main__":
    main()
