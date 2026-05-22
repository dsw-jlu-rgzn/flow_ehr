"""Run the current AP scaffold/judge-revise experiment with Hugging Face.

This is a thin adapter around modeling/ap_memory_gated_scaffold_generation.py.
All original AP experiment arguments are still accepted after the HF adapter
arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELING_DIR = ROOT / "modeling"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_DIR))

from modeling.hf_generation import DEFAULT_HF_ROUTER_URL, call_huggingface


def parse_adapter_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--hf-backend", choices=["router", "local", "mock"], default="router")
    parser.add_argument("--hf-api-url", default=DEFAULT_HF_ROUTER_URL)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument(
        "--hf-system-prompt",
        default="You are an experienced ICU clinician. Write concise, clinically grounded documentation.",
    )
    parser.add_argument(
        "--help-hf",
        action="store_true",
        help="Show HF adapter help. Use --help after adapter args for the underlying AP script help.",
    )
    args, remaining = parser.parse_known_args()
    if args.help_hf:
        parser.print_help()
        raise SystemExit(0)
    return args, remaining


def main() -> None:
    hf_args, ap_args = parse_adapter_args()

    import ap_memory_gated_scaffold_generation as ap_module

    def hf_call(
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

    ap_module.call_deepseek = hf_call
    sys.argv = [sys.argv[0]] + ap_args
    ap_module.main()


if __name__ == "__main__":
    main()
