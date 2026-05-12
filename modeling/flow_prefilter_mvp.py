"""
Minimal supervised flow-prefilter loop for AP generation.

This script trains a small model before the LLM. The model predicts the semantic
direction of the daily ground-truth progress note from that day's EHR context.
At inference/export time, the predicted note embedding is used to rank raw EHR
rows plus simple trend snippets. The top-k snippets are exported as a condensed
AP input directory that can be passed to event_ap_fix_v2.py.

Example:
  python modeling/flow_prefilter_mvp.py train --inputdir data/AP/input --checkpoint checkpoints/flow_prefilter_mvp.pt
  python modeling/flow_prefilter_mvp.py export --inputdir data/AP/input --outputdir data/AP/input_flow_topk --checkpoint checkpoints/flow_prefilter_mvp.pt
  python modeling/flow_prefilter_mvp.py all --inputdir data/AP/input --outputdir data/AP/input_flow_topk
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_/\-]+|\d+(?:\.\d+)?")
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

TREND_SPECS = {
    "glucose": {
        "keywords": ["glucose"],
        "unit": "",
        "min_abs_delta": 30.0,
        "min_pct_delta": 0.15,
    },
    "creatinine": {
        "keywords": ["creatinine", "creat"],
        "unit": "",
        "min_abs_delta": 0.3,
        "min_pct_delta": 0.2,
    },
    "wbc": {
        "keywords": ["wbc", "white blood cell"],
        "unit": "",
        "min_abs_delta": 3.0,
        "min_pct_delta": 0.2,
    },
    "potassium": {
        "keywords": ["potassium", " k "],
        "unit": "",
        "min_abs_delta": 0.5,
        "min_pct_delta": 0.12,
    },
    "sodium": {
        "keywords": ["sodium", " na "],
        "unit": "",
        "min_abs_delta": 4.0,
        "min_pct_delta": 0.03,
    },
    "lactate": {
        "keywords": ["lactate"],
        "unit": "",
        "min_abs_delta": 1.0,
        "min_pct_delta": 0.25,
    },
    "hemoglobin": {
        "keywords": ["hemoglobin", "hgb"],
        "unit": "",
        "min_abs_delta": 1.0,
        "min_pct_delta": 0.12,
    },
    "heart rate": {
        "keywords": ["heart rate", "hr "],
        "unit": "bpm",
        "min_abs_delta": 20.0,
        "min_pct_delta": 0.2,
    },
    "temperature": {
        "keywords": ["temperature", "temp"],
        "unit": "",
        "min_abs_delta": 1.0,
        "min_pct_delta": 0.02,
    },
    "oxygen saturation": {
        "keywords": ["spo2", "o2 sat", "oxygen saturation"],
        "unit": "%",
        "min_abs_delta": 4.0,
        "min_pct_delta": 0.05,
    },
}


def stable_hash(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16)


class HashingTextEncoder:
    """Dependency-light frozen encoder for the first closed-loop experiment."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in TOKEN_RE.findall(str(text).lower()):
            idx = stable_hash(token) % self.dim
            sign = 1.0 if stable_hash("sign:" + token) % 2 == 0 else -1.0
            vector[idx] += sign

            # Lightweight character n-grams make misspellings/abbreviations less brittle.
            if not token[0].isdigit() and len(token) >= 5:
                for start in range(len(token) - 2):
                    gram = token[start : start + 3]
                    gidx = stable_hash("g:" + gram) % self.dim
                    vector[gidx] += 0.25

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return np.vstack([self.encode_one(text) for text in texts]).astype(np.float32)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def row_to_snippet(row: pd.Series) -> str:
    time_value = row.get("REL_TIME", row.get("TIME", ""))
    return f"{time_value} | {normalize_text(row.get('TEXT', ''))}"


def parse_numeric_value(text: str) -> float | None:
    values = []
    for match in NUMBER_RE.findall(str(text)):
        try:
            value = float(match)
        except ValueError:
            continue
        if -1000.0 <= value <= 10000.0:
            values.append(value)
    if not values:
        return None
    return values[0]


def contains_keyword(text: str, keywords: list[str]) -> bool:
    padded = f" {text.lower()} "
    return any(keyword in padded for keyword in keywords)


def build_trend_snippets(day_df: pd.DataFrame) -> list[str]:
    """Create deterministic trend snippets so the ranker can select temporal changes."""

    trend_snippets = []
    data_df = day_df[day_df.get("IS_NOTE", 0) == 0].copy()
    if data_df.empty:
        return trend_snippets

    for metric, spec in TREND_SPECS.items():
        observations = []
        for _, row in data_df.iterrows():
            text = normalize_text(row.get("TEXT", ""))
            if not contains_keyword(text, spec["keywords"]):
                continue
            value = parse_numeric_value(text)
            if value is None:
                continue
            observations.append((row.get("REL_TIME", row.get("TIME", "")), value))

        if len(observations) < 2:
            continue

        first_time, first_value = observations[0]
        last_time, last_value = observations[-1]
        delta = last_value - first_value
        pct_delta = abs(delta) / max(abs(first_value), 1.0)

        if abs(delta) < spec["min_abs_delta"] and pct_delta < spec["min_pct_delta"]:
            direction = "stable"
        elif delta > 0:
            direction = "rising"
        else:
            direction = "falling"

        unit = spec["unit"]
        trend_snippets.append(
            "[Trend] "
            f"{metric} {direction}: {first_value:g}{unit} -> {last_value:g}{unit} "
            f"(delta {delta:+g}) from {first_time} to {last_time}."
        )

    return trend_snippets


@dataclass
class DaySample:
    file_name: str
    admission_id: str
    day: float
    ehr_text: str
    note_text: str


def load_day_samples(inputdir: str) -> list[DaySample]:
    samples = []
    for csv_path in sorted(Path(inputdir).glob("*.csv")):
        df = pd.read_csv(csv_path)
        required = {"DAY", "TEXT", "IS_NOTE"}
        missing = required.difference(df.columns)
        if missing:
            print(f"Skipping {csv_path.name}: missing columns {sorted(missing)}")
            continue

        admission_id = csv_path.stem.split("_")[-1]
        for day, day_df in df.groupby("DAY", sort=True):
            data_df = day_df[day_df["IS_NOTE"] == 0]
            note_df = day_df[day_df["IS_NOTE"] == 1]
            if data_df.empty or note_df.empty:
                continue

            snippets = [row_to_snippet(row) for _, row in data_df.iterrows()]
            snippets.extend(build_trend_snippets(day_df))
            note_text = normalize_text(note_df.iloc[-1]["TEXT"])
            if not note_text:
                continue

            samples.append(
                DaySample(
                    file_name=csv_path.name,
                    admission_id=admission_id,
                    day=float(day),
                    ehr_text="\n".join(snippets),
                    note_text=note_text,
                )
            )
    return samples


class EmbeddingPairDataset(Dataset):
    def __init__(self, samples: list[DaySample], encoder: HashingTextEncoder):
        if not samples:
            raise ValueError("No supervised day samples found. AP input needs DAY, IS_NOTE, and note rows.")
        self.samples = samples
        self.x = torch.from_numpy(encoder.encode(sample.ehr_text for sample in samples))
        self.y = torch.from_numpy(encoder.encode(sample.note_text for sample in samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class MLPProjector(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class ConditionalFlowMatcher(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2 + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def time_features(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        return torch.cat(
            [
                t,
                t * t,
                torch.sin(2 * math.pi * t),
                torch.cos(2 * math.pi * t),
            ],
            dim=-1,
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, condition, self.time_features(t)], dim=-1))

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, steps: int = 16, seed: int = 13) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        z = torch.randn_like(condition)
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full((condition.shape[0],), step / steps, device=condition.device)
            z = z + dt * self.forward(z, t, condition)
        return F.normalize(z, dim=-1)


def train_model(args) -> None:
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder = HashingTextEncoder(dim=args.embed_dim)
    samples = load_day_samples(args.inputdir)
    dataset = EmbeddingPairDataset(samples, encoder)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.model_type == "mlp":
        model = MLPProjector(args.embed_dim, args.hidden_dim).to(device)
    else:
        model = ConditionalFlowMatcher(args.embed_dim, args.hidden_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_cos = 0.0
        for x, y in dataloader:
            x = x.to(device)
            y = F.normalize(y.to(device), dim=-1)
            optimizer.zero_grad(set_to_none=True)

            if args.model_type == "mlp":
                pred = model(x)
                loss = F.mse_loss(pred, y) + (1.0 - F.cosine_similarity(pred, y, dim=-1)).mean()
            else:
                noise = torch.randn_like(y)
                t = torch.rand(y.shape[0], device=device)
                z_t = (1.0 - t[:, None]) * noise + t[:, None] * y
                target_velocity = y - noise
                pred_velocity = model(z_t, t, x)
                loss = F.mse_loss(pred_velocity, target_velocity)

                if epoch % max(args.eval_every, 1) == 0:
                    pred = model.sample(x, steps=args.sample_steps)
                else:
                    pred = F.normalize(z_t + (1.0 - t[:, None]) * pred_velocity, dim=-1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                total_loss += float(loss.item()) * x.shape[0]
                total_cos += float(F.cosine_similarity(pred, y, dim=-1).sum().item())

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} loss={total_loss / len(dataset):.4f} "
                f"train_cos={total_cos / len(dataset):.4f}"
            )

    checkpoint = {
        "model_type": args.model_type,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "state_dict": model.state_dict(),
        "num_samples": len(dataset),
        "trend_specs": TREND_SPECS,
    }
    torch.save(checkpoint, args.checkpoint)

    meta_path = Path(args.checkpoint).with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "inputdir": args.inputdir,
                "model_type": args.model_type,
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
                "num_samples": len(dataset),
            },
            indent=2,
        )
    )
    print(f"Saved checkpoint: {args.checkpoint}")


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint["model_type"]
    embed_dim = int(checkpoint["embed_dim"])
    hidden_dim = int(checkpoint.get("hidden_dim", 512))
    if model_type == "mlp":
        model = MLPProjector(embed_dim, hidden_dim).to(device)
    else:
        model = ConditionalFlowMatcher(embed_dim, hidden_dim).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, model_type, embed_dim


def predict_note_embedding(
    model: nn.Module,
    model_type: str,
    condition: np.ndarray,
    device: torch.device,
    sample_steps: int,
) -> np.ndarray:
    x = torch.from_numpy(condition[None, :]).float().to(device)
    with torch.no_grad():
        if model_type == "mlp":
            pred = model(x)
        else:
            pred = model.sample(x, steps=sample_steps)
    return pred.cpu().numpy()[0]


def cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    matrix_norm = np.linalg.norm(matrix, axis=1)
    denom = np.maximum(matrix_norm * max(query_norm, 1e-8), 1e-8)
    return (matrix @ query) / denom


def select_day_rows(
    day_df: pd.DataFrame,
    encoder: HashingTextEncoder,
    model: nn.Module,
    model_type: str,
    device: torch.device,
    top_k: int,
    sample_steps: int,
) -> pd.DataFrame:
    data_df = day_df[day_df["IS_NOTE"] == 0].copy()
    note_df = day_df[day_df["IS_NOTE"] == 1].copy()
    if data_df.empty:
        return day_df

    raw_snippets = [row_to_snippet(row) for _, row in data_df.iterrows()]
    trend_snippets = build_trend_snippets(day_df)
    all_snippets = raw_snippets + trend_snippets
    all_embeddings = encoder.encode(all_snippets)
    day_embedding = encoder.encode(["\n".join(all_snippets)])[0]
    predicted_note_embedding = predict_note_embedding(model, model_type, day_embedding, device, sample_steps)
    scores = cosine_scores(all_embeddings, predicted_note_embedding)

    selected_raw_indices = set()
    synthetic_rows = []
    ranked_indices = np.argsort(-scores)
    for idx in ranked_indices:
        if len(selected_raw_indices) + len(synthetic_rows) >= top_k:
            break
        if idx < len(raw_snippets):
            selected_raw_indices.add(idx)
        else:
            template = data_df.iloc[-1].copy()
            template["TEXT"] = all_snippets[idx]
            template["IS_NOTE"] = 0
            template["FLOW_SCORE"] = float(scores[idx])
            if "TIME" in template:
                template["TIME"] = data_df.iloc[-1].get("TIME", "")
            if "REL_TIME" in template:
                template["REL_TIME"] = data_df.iloc[-1].get("REL_TIME", "")
            synthetic_rows.append(template)

    selected_data = data_df.iloc[sorted(selected_raw_indices)].copy()
    if not selected_data.empty:
        selected_data["FLOW_SCORE"] = [float(scores[idx]) for idx in sorted(selected_raw_indices)]

    frames = [frame for frame in [selected_data, pd.DataFrame(synthetic_rows), note_df] if not frame.empty]
    if not frames:
        return day_df

    output = pd.concat(frames, ignore_index=True)
    sort_cols = [col for col in ["DAY", "TIME", "REL_TIME", "IS_NOTE"] if col in output.columns]
    if sort_cols:
        output = output.sort_values(sort_cols, kind="stable")
    return output


def export_prefiltered_inputs(args) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, model_type, embed_dim = load_model(args.checkpoint, device)
    encoder = HashingTextEncoder(dim=embed_dim)

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

        output_days = []
        for day, day_df in df.groupby("DAY", sort=True):
            before = int((day_df["IS_NOTE"] == 0).sum())
            filtered_day = select_day_rows(
                day_df,
                encoder,
                model,
                model_type,
                device,
                top_k=args.top_k,
                sample_steps=args.sample_steps,
            )
            after = int((filtered_day["IS_NOTE"] == 0).sum())
            output_days.append(filtered_day)
            summary_rows.append(
                {
                    "file": csv_path.name,
                    "day": day,
                    "non_note_before": before,
                    "non_note_after": after,
                }
            )

        if not output_days:
            print(f"Skipping {csv_path.name}: no days could be exported")
            continue

        output_df = pd.concat(output_days, ignore_index=True)
        output_file = output_path / csv_path.name
        output_df.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Wrote {output_file}")

    summary_file = output_path / "_flow_prefilter_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_file, index=False)
    print(f"Wrote {summary_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/export a minimal AP flow prefilter.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--inputdir", default="data/AP/input")
        subparser.add_argument("--checkpoint", default="checkpoints/flow_prefilter_mvp.pt")
        subparser.add_argument("--device", default="")
        subparser.add_argument("--embed-dim", type=int, default=512)
        subparser.add_argument("--hidden-dim", type=int, default=512)
        subparser.add_argument("--sample-steps", type=int, default=16)

    train = subparsers.add_parser("train", help="Train the small prefilter model.")
    add_common(train)
    train.add_argument("--model-type", choices=["flow", "mlp"], default="flow")
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--eval-every", type=int, default=20)

    export = subparsers.add_parser("export", help="Export top-k filtered AP inputs.")
    add_common(export)
    export.add_argument("--outputdir", default="data/AP/input_flow_topk")
    export.add_argument("--top-k", type=int, default=40)

    all_cmd = subparsers.add_parser("all", help="Train and export in one command.")
    add_common(all_cmd)
    all_cmd.add_argument("--model-type", choices=["flow", "mlp"], default="flow")
    all_cmd.add_argument("--epochs", type=int, default=200)
    all_cmd.add_argument("--batch-size", type=int, default=8)
    all_cmd.add_argument("--lr", type=float, default=1e-3)
    all_cmd.add_argument("--weight-decay", type=float, default=1e-4)
    all_cmd.add_argument("--eval-every", type=int, default=20)
    all_cmd.add_argument("--outputdir", default="data/AP/input_flow_topk")
    all_cmd.add_argument("--top-k", type=int, default=40)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train_model(args)
    elif args.command == "export":
        export_prefiltered_inputs(args)
    elif args.command == "all":
        train_model(args)
        export_prefiltered_inputs(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
