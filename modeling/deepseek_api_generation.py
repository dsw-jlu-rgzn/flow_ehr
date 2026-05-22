"""
Generate AP/DS outputs with the DeepSeek chat API.

This keeps the original task data and output layouts, but replaces local
HuggingFace inference with an OpenAI-compatible API call.

Required environment variable:
  DEEPSEEK_API_KEY by default. Pass api_key_env to call_deepseek for other
  OpenAI-compatible providers.

Examples:
  python modeling/deepseek_api_generation.py ap --limit 2
  python modeling/deepseek_api_generation.py ds --limit 3
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"


def call_deepseek(
    prompt: str,
    model: str,
    api_url: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    sleep_seconds: float,
    api_key_env: str = "DEEPSEEK_API_KEY",
) -> str:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set.")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an experienced ICU clinician. Write concise, clinically grounded documentation.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if model.lower().startswith("qwen/"):
        payload["enable_thinking"] = False
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"].strip()
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            ssl.SSLError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)
    raise RuntimeError(f"DeepSeek API request failed after {retries} attempts: {last_error}")


def df2chron_str(df: pd.DataFrame) -> str:
    rel_time = df["REL_TIME"].fillna("").astype(str).tolist() if "REL_TIME" in df.columns else [""] * len(df)
    text = df["TEXT"].fillna("").astype(str).tolist()
    return "\n".join(f"{t}\t{x}" for t, x in zip(rel_time, text))


AP_INSTRUCTION_1 = """
You are an experienced ICU clinician tasked with reviewing the following EHR data and generating concise Assessment and Plan sections of a clinical progress note. Use professional and medically appropriate language.

EHR Data:
"""

AP_INSTRUCTION_2 = """

Write the next physician progress note content for the current day.

Assessment:
Briefly describe the active problems for the day, why the patient is admitted, and relevant comorbidities.

Plan:
Organize the plan into subsections for each active problem. Include proposed or ongoing interventions, medications, and care strategies.
"""


def generate_ap(args: argparse.Namespace) -> None:
    input_dir = Path(args.inputdir)
    output_root = Path(args.outputdir) / "AP" / "generated" / "DG" / args.run_name / args.setting
    output_root.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("input_*.csv"))
    if args.limit:
        files = files[: args.limit]

    for method in args.methods:
        method_dir = output_root / ("method-1" if method == -1 else f"method{method}")
        method_dir.mkdir(parents=True, exist_ok=True)

        for csv_path in files:
            df = pd.read_csv(csv_path)
            admission_id = csv_path.stem.split("_")[-1]
            output_path = method_dir / f"genpns_{admission_id}.csv"
            if output_path.exists():
                existing_df = pd.read_csv(output_path)
                days = existing_df["DAY"].astype(int).tolist() if "DAY" in existing_df.columns else []
                gen_pns = existing_df["TEXT"].fillna("").astype(str).tolist() if "TEXT" in existing_df.columns else []
            else:
                days = []
                gen_pns = []
            generated_days = set(days)
            day_groups = {day: group for day, group in df.groupby("DAY", sort=True)}

            first_day = None
            for day, day_df in day_groups.items():
                if len(day_df[day_df["IS_NOTE"] == 1]) != 0:
                    first_day = day
                    break

            previous_pn = ""
            for day, day_df in day_groups.items():
                if len(day_df[day_df["IS_NOTE"] == 1]) == 0:
                    continue

                ehr_str = df2chron_str(day_df[day_df["IS_NOTE"] == 0])
                if previous_pn:
                    ehr_str += "\n\nPrevious progress note context:\n" + previous_pn

                if day != first_day:
                    if day in generated_days:
                        text = gen_pns[days.index(day)]
                        print(f"AP {admission_id} day={day} method={method}: skipped existing")
                    else:
                        prompt = AP_INSTRUCTION_1 + ehr_str + AP_INSTRUCTION_2
                        text = call_deepseek(
                            prompt,
                            model=args.model,
                            api_url=args.api_url,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            retries=args.retries,
                            sleep_seconds=args.sleep_seconds,
                        )
                        days.append(day)
                        gen_pns.append(text)
                        generated_days.add(day)
                        pd.DataFrame(zip(days, gen_pns), columns=["DAY", "TEXT"]).to_csv(
                            output_path,
                            index=False,
                        )
                        print(f"AP {admission_id} day={day} method={method}: generated {len(text.split())} words")

                if args.setting == "gt":
                    next_prev = day_df[day_df["IS_NOTE"] == 1].iloc[-1]["TEXT"]
                else:
                    next_prev = day_df[day_df["IS_NOTE"] == 1].iloc[-1]["TEXT"] if day == first_day else gen_pns[-1]

                if method == 1:
                    previous_pn = str(next_prev)
                elif method == 2:
                    previous_pn += str(next_prev) + "\n"
            pd.DataFrame(zip(days, gen_pns), columns=["DAY", "TEXT"]).to_csv(output_path, index=False)


DS_ROLE = """
Role: You are a clinician in the ICU responsible for generating patient discharge documentation.

Hospital Data from the last available discharge window:

"""

DS_INSTRUCTION = """

Part 1: Diagnosis
Provide a diagnosis that summarizes the patient's primary medical condition(s) identified during their hospital stay.

---

Part 2: Hospital Course Summary
Summarize the patient's hospital course over the available discharge window, including key treatments, response, significant events, and progress.

---

Part 3: Discharge Instructions
Write personalized discharge instructions. Include medication guidance, activity/lifestyle recommendations, follow-up, and precautions when supported by the data.
"""


def generate_ds(args: argparse.Namespace) -> None:
    input_dir = Path(args.inputdir)
    output_dir = Path(args.outputdir) / "DS" / "generated" / "DG" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.csv"))
    if args.limit:
        files = files[: args.limit]

    for csv_path in files:
        admission_id = csv_path.stem.split("_")[-1]
        output_path = output_dir / f"48h_all_abs_{admission_id}.txt"
        if output_path.exists():
            print(f"DS {admission_id}: skipped existing")
            continue
        df = pd.read_csv(csv_path)
        chronology_str = df2chron_str(df)
        prompt = DS_ROLE + chronology_str + DS_INSTRUCTION
        text = call_deepseek(
            prompt,
            model=args.model,
            api_url=args.api_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
        )
        output_path.write_text(text, encoding="utf-8")
        print(f"DS {admission_id}: generated {len(text.split())} words")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--outputdir", default="data")
    parser.add_argument("--run-name", default="deepseek_api")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AP/DS with DeepSeek API.")
    subparsers = parser.add_subparsers(dest="task", required=True)

    ap = subparsers.add_parser("ap")
    add_common_args(ap)
    ap.add_argument("--inputdir", default="data/AP/input")
    ap.add_argument("--setting", choices=["gt", "gen"], default="gt")
    ap.add_argument("--methods", type=int, nargs="+", default=[-1, 1, 2])

    ds = subparsers.add_parser("ds")
    add_common_args(ds)
    ds.add_argument("--inputdir", default="data/DS/input")
    ds.set_defaults(max_tokens=2200)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "ap":
        generate_ap(args)
    elif args.task == "ds":
        generate_ds(args)
    else:
        raise ValueError(args.task)


if __name__ == "__main__":
    main()
