"""Run DS experiment stages with Hugging Face generation.

The script wraps existing DS stage scripts and monkey-patches their model call
to Hugging Face Router, local transformers, or mock mode. Run stages in order:
minimal -> variants -> dx2 -> dx3.
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


STAGE_TO_MODULE = {
    "minimal": "scripts.run_ds_minimal_closed_loop",
    "variants": "scripts.run_ds_v2_variants_10",
    "dx2": "scripts.run_ds_ours2_v4_dx2",
    "dx3": "scripts.run_ds_ours2_v4_dx3_agent",
}


def parse_adapter_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("stage", choices=sorted(STAGE_TO_MODULE))
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
        help="Show HF adapter help. Use --help after stage/adapter args for the wrapped DS script help.",
    )
    if "--help-hf" in sys.argv:
        parser.print_help()
        raise SystemExit(0)
    args, remaining = parser.parse_known_args()
    if args.help_hf:
        parser.print_help()
        raise SystemExit(0)
    return args, remaining


def main() -> None:
    hf_args, stage_args = parse_adapter_args()
    module = importlib.import_module(STAGE_TO_MODULE[hf_args.stage])

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

    if hasattr(module, "call_deepseek"):
        module.call_deepseek = hf_call
    sys.argv = [sys.argv[0]] + stage_args
    module.main()


if __name__ == "__main__":
    main()
