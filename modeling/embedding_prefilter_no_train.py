"""
No-training embedding prefilter for AP generation.

This is the smallest closed loop before training a flow model. It uses an
existing language model as a frozen embedding encoder, ranks same-day EHR rows
plus deterministic trend snippets, and exports a condensed AP input directory.

Query modes:
  previous_note: realistic mode. Use the previous note as the retrieval target.
  day_context: no-label mode. Use the whole same-day EHR context as the target.
  oracle_gt: upper-bound/debug mode. Use the same-day ground-truth note.

Example:
  python modeling/embedding_prefilter_no_train.py \
    --inputdir data/AP/input \
    --outputdir data/AP/input_embedding_topk \
    --model mistral \
    --query-mode previous_note \
    --top-k 40
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from flow_prefilter_mvp import build_trend_snippets, cosine_scores, normalize_text, row_to_snippet


if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


AVAILABLE_MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "llama2": "meta-llama/Llama-2-13b-chat-hf",
}


class FrozenLLMEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str = "",
        load_in_8bit: bool = True,
        max_length: int = 512,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"trust_remote_code": True}
        if load_in_8bit and torch.cuda.is_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if "device_map" not in kwargs:
            self.model.to(self.device)
        self.model.eval()
        self.input_device = getattr(self.model, "device", self.device)

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 4) -> np.ndarray:
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
            outputs = self.model(**encoded, output_hidden_states=True, use_cache=False)
            hidden = outputs.hidden_states[-1]
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = F.normalize(pooled, dim=-1)
            embeddings.append(pooled.detach().cpu().numpy().astype(np.float32))

            del encoded, outputs, hidden, mask, pooled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return np.vstack(embeddings)


def make_query_text(day_df: pd.DataFrame, previous_note: str, mode: str) -> str:
    data_df = day_df[day_df["IS_NOTE"] == 0]
    note_df = day_df[day_df["IS_NOTE"] == 1]

    if mode == "oracle_gt" and not note_df.empty:
        return normalize_text(note_df.iloc[-1]["TEXT"])

    if mode == "previous_note" and previous_note:
        return previous_note

    snippets = [row_to_snippet(row) for _, row in data_df.iterrows()]
    snippets.extend(build_trend_snippets(day_df))
    return "\n".join(snippets)


def select_day_rows(
    day_df: pd.DataFrame,
    embedder: FrozenLLMEmbedder,
    previous_note: str,
    query_mode: str,
    top_k: int,
    batch_size: int,
) -> tuple[pd.DataFrame, str]:
    data_df = day_df[day_df["IS_NOTE"] == 0].copy()
    note_df = day_df[day_df["IS_NOTE"] == 1].copy()
    if data_df.empty:
        next_note = previous_note
        if not note_df.empty:
            next_note = normalize_text(note_df.iloc[-1]["TEXT"])
        return day_df, next_note

    raw_snippets = [row_to_snippet(row) for _, row in data_df.iterrows()]
    trend_snippets = build_trend_snippets(day_df)
    all_snippets = raw_snippets + trend_snippets
    query_text = make_query_text(day_df, previous_note, query_mode)

    embeddings = embedder.encode([query_text] + all_snippets, batch_size=batch_size)
    query_embedding = embeddings[0]
    candidate_embeddings = embeddings[1:]
    scores = cosine_scores(candidate_embeddings, query_embedding)

    selected_raw_indices = set()
    synthetic_rows = []
    for idx in np.argsort(-scores):
        if len(selected_raw_indices) + len(synthetic_rows) >= top_k:
            break
        if idx < len(raw_snippets):
            selected_raw_indices.add(idx)
        else:
            template = data_df.iloc[-1].copy()
            template["TEXT"] = all_snippets[idx]
            template["IS_NOTE"] = 0
            template["EMBED_SCORE"] = float(scores[idx])
            synthetic_rows.append(template)

    selected_data = data_df.iloc[sorted(selected_raw_indices)].copy()
    if not selected_data.empty:
        selected_data["EMBED_SCORE"] = [float(scores[idx]) for idx in sorted(selected_raw_indices)]

    frames = [frame for frame in [selected_data, pd.DataFrame(synthetic_rows), note_df] if not frame.empty]
    output = pd.concat(frames, ignore_index=True) if frames else day_df
    sort_cols = [col for col in ["DAY", "TIME", "REL_TIME", "IS_NOTE"] if col in output.columns]
    if sort_cols:
        output = output.sort_values(sort_cols, kind="stable")

    next_note = previous_note
    if not note_df.empty:
        next_note = normalize_text(note_df.iloc[-1]["TEXT"])
    return output, next_note


def export_inputs(args) -> None:
    model_name = AVAILABLE_MODELS.get(args.model, args.model)
    embedder = FrozenLLMEmbedder(
        model_name=model_name,
        device=args.device,
        load_in_8bit=not args.no_8bit,
        max_length=args.max_length,
    )

    input_path = Path(args.inputdir)
    output_path = Path(args.outputdir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for csv_path in sorted(input_path.glob("*.csv")):
        df = pd.read_csv(csv_path)
        required = {"DAY", "TEXT", "IS_NOTE"}
        missing = required.difference(df.columns)
        if missing:
            print(f"Skipping {csv_path.name}: missing columns {sorted(missing)}")
            continue

        previous_note = ""
        output_days = []
        for day, day_df in df.groupby("DAY", sort=True):
            before = int((day_df["IS_NOTE"] == 0).sum())
            filtered_day, previous_note = select_day_rows(
                day_df=day_df,
                embedder=embedder,
                previous_note=previous_note,
                query_mode=args.query_mode,
                top_k=args.top_k,
                batch_size=args.batch_size,
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
                }
            )

        if not output_days:
            continue
        output_file = output_path / csv_path.name
        pd.concat(output_days, ignore_index=True).to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Wrote {output_file}")

    summary_file = output_path / "_embedding_prefilter_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_file, index=False)
    print(f"Wrote {summary_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-training LLM embedding AP prefilter.")
    parser.add_argument("--inputdir", default="data/AP/input")
    parser.add_argument("--outputdir", default="data/AP/input_embedding_topk")
    parser.add_argument("--model", default="mistral", help="Model key or HuggingFace model name/path.")
    parser.add_argument("--query-mode", choices=["previous_note", "day_context", "oracle_gt"], default="previous_note")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="")
    parser.add_argument("--no-8bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    export_inputs(parse_args())


if __name__ == "__main__":
    main()
