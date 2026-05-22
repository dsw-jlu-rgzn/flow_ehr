"""Run AP/DS LLM-as-judge evaluation with a Hugging Face evaluator model.

This wrapper keeps generation and evaluation separable while allowing both to
use open Hugging Face models. It reuses the existing AP/DS judge scripts and
replaces their OpenAI-compatible call with modeling.hf_generation.call_huggingface.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling.hf_generation import DEFAULT_HF_ROUTER_URL, call_huggingface


TASK_TO_MODULE = {
    "ap": "evaluation.judge_augmented_ap",
    "ds": "evaluation.judge_ds_pairwise_llm",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("task", choices=sorted(TASK_TO_MODULE))
    parser.add_argument("--hf-backend", choices=["router", "local", "mock"], default="router")
    parser.add_argument("--hf-api-url", default=DEFAULT_HF_ROUTER_URL)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--eval-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--hf-system-prompt",
        default=(
            "You are a careful clinical documentation evaluator. Return valid JSON only "
            "when the user requests JSON."
        ),
    )
    parser.add_argument("--help-hf", action="store_true")
    if "--help-hf" in sys.argv:
        parser.print_help()
        raise SystemExit(0)
    args, remaining = parser.parse_known_args()
    if args.help_hf:
        parser.print_help()
        raise SystemExit(0)
    return args, remaining


def main() -> None:
    hf_args, judge_args = parse_args()
    module = importlib.import_module(TASK_TO_MODULE[hf_args.task])

    def hf_eval_call(
        prompt: str,
        model: str,
        api_url: str,
        temperature: float,
        max_tokens: int,
        retries: int,
        sleep_seconds: float,
        api_key_env: str = "HF_TOKEN",
    ) -> str:
        return call_huggingface(
            prompt=prompt,
            model=model,
            backend=hf_args.hf_backend,
            api_url=hf_args.hf_api_url,
            hf_token_env=hf_args.hf_token_env,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            sleep_seconds=sleep_seconds,
            system_prompt=hf_args.hf_system_prompt,
        )

    module.call_deepseek = hf_eval_call
    forwarded = ["--model", hf_args.eval_model, "--api-url", hf_args.hf_api_url, "--api-key-env", hf_args.hf_token_env]
    sys.argv = [sys.argv[0]] + forwarded + judge_args
    module.main()


if __name__ == "__main__":
    main()
