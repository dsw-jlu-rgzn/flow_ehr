from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ChatSftDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_seq_len: int):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, messages: list[dict[str, str]]) -> dict[str, list[int]]:
        prompt_messages = messages[:-1]
        full_messages = messages
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if self.tokenizer.eos_token_id is not None and (
            not full_ids or full_ids[-1] != self.tokenizer.eos_token_id
        ):
            full_ids.append(self.tokenizer.eos_token_id)
        assistant_ids = full_ids[len(prompt_ids) :]

        if len(full_ids) > self.max_seq_len:
            min_prompt_tokens = min(512, max(64, self.max_seq_len // 4))
            prompt_budget = min(len(prompt_ids), min_prompt_tokens)
            assistant_budget = self.max_seq_len - prompt_budget
            if len(assistant_ids) > assistant_budget:
                assistant_ids = assistant_ids[:assistant_budget]
            else:
                prompt_budget = min(len(prompt_ids), self.max_seq_len - len(assistant_ids))
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
            full_ids = prompt_ids + assistant_ids
            prompt_len = len(prompt_ids)
        else:
            prompt_len = len(prompt_ids)

        labels = [-100] * min(prompt_len, len(full_ids)) + full_ids[min(prompt_len, len(full_ids)) :]
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self._encode(self.rows[idx]["messages"])


@dataclass
class DataCollatorForCausalSft:
    tokenizer: Any

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(x["input_ids"]) for x in features)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_target_modules(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT for direct AP generation.")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train-file", type=Path, default=Path("outputs/ap_direct_sft_dataset/train.jsonl"))
    parser.add_argument("--val-file", type=Path, default=Path("outputs/ap_direct_sft_dataset/val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ap_direct_sft_lora/qwen2_5_7b"))
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--qlora-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--merge-and-save", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if args.qlora_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.qlora_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=parse_target_modules(args.target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ChatSftDataset(args.train_file, tokenizer, args.max_seq_len)
    if args.max_train_samples:
        train_dataset.rows = train_dataset.rows[: args.max_train_samples]
    eval_dataset = ChatSftDataset(args.val_file, tokenizer, args.max_seq_len) if args.val_file.exists() else None
    if eval_dataset is not None and args.max_eval_samples:
        eval_dataset.rows = eval_dataset.rows[: args.max_eval_samples]

    training_kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps if eval_dataset is not None else None,
        "save_strategy": "steps",
        "bf16": args.precision == "bf16",
        "fp16": args.precision == "fp16",
        "optim": "paged_adamw_8bit" if args.qlora_4bit else "adamw_torch",
        "lr_scheduler_type": "cosine",
        "report_to": "none",
        "seed": args.seed,
        "remove_unused_columns": False,
    }
    signature = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
    training_kwargs[strategy_key] = "steps" if eval_dataset is not None else "no"
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForCausalSft(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    if args.merge_and_save:
        merged_dir = args.output_dir.with_name(args.output_dir.name + "_merged")
        merged = model.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        print(f"Merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()
