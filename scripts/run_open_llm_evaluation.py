"""Run AP/DS LLM-as-judge evaluation with an OpenAI-compatible open API.

This wrapper keeps evaluation independent from the Hugging Face generation
backend. It forwards all task-specific arguments to the existing judge scripts
while setting open-evaluator defaults.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TASK_TO_MODULE = {
    "ap": "evaluation.judge_augmented_ap",
    "ds": "evaluation.judge_ds_pairwise_llm",
}


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("task", choices=sorted(TASK_TO_MODULE))
    parser.add_argument("--eval-model", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--eval-api-url", default="https://api.siliconflow.cn/v1/chat/completions")
    parser.add_argument("--eval-api-key-env", default="OPEN_LLM_EVAL_API_KEY")
    parser.add_argument("--help-wrapper", action="store_true")
    if "--help-wrapper" in sys.argv:
        parser.print_help()
        raise SystemExit(0)
    wrapper_args, task_args = parser.parse_known_args()
    if wrapper_args.help_wrapper:
        parser.print_help()
        raise SystemExit(0)

    forwarded = [
        "--model",
        wrapper_args.eval_model,
        "--api-url",
        wrapper_args.eval_api_url,
        "--api-key-env",
        wrapper_args.eval_api_key_env,
    ] + task_args

    module = importlib.import_module(TASK_TO_MODULE[wrapper_args.task])
    sys.argv = [sys.argv[0]] + forwarded
    module.main()


if __name__ == "__main__":
    main()
