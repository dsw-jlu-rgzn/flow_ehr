"""Hugging Face generation helpers for AP/DS experiments.

The helpers intentionally keep the same simple string-in/string-out surface as
the existing OpenAI-compatible generation code. Tokens are read from environment
variables and should not be written into scripts or configs.
"""

from __future__ import annotations

import http.client
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Any


DEFAULT_HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_CLINICAL_SYSTEM_PROMPT = (
    "You are an experienced ICU clinician. Write concise, clinically grounded documentation."
)


def call_huggingface(
    prompt: str,
    model: str,
    backend: str = "router",
    api_url: str = DEFAULT_HF_ROUTER_URL,
    hf_token_env: str = "HF_TOKEN",
    temperature: float = 0.0,
    max_tokens: int = 1800,
    retries: int = 3,
    sleep_seconds: float = 2.0,
    system_prompt: str = DEFAULT_CLINICAL_SYSTEM_PROMPT,
) -> str:
    """Generate text with Hugging Face Router, local transformers, or mock mode."""

    if backend == "mock":
        return mock_generation(prompt=prompt, model=model)
    if backend == "router":
        return call_hf_router(
            prompt=prompt,
            model=model,
            api_url=api_url,
            hf_token_env=hf_token_env,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            sleep_seconds=sleep_seconds,
            system_prompt=system_prompt,
        )
    if backend == "local":
        return call_local_transformers(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_new_tokens=max_tokens,
            system_prompt=system_prompt,
        )
    raise ValueError(f"Unsupported Hugging Face backend: {backend}")


def mock_generation(prompt: str, model: str) -> str:
    """Return shape-compatible mock outputs for plumbing checks."""

    lower = prompt.lower()
    if "compare two generated icu assessment & plan notes" in lower:
        return json.dumps(
            {
                "baseline": {
                    "active_problem_coverage": 3,
                    "trajectory_capture": 3,
                    "plan_specificity": 3,
                    "evidence_grounding": 3,
                    "disposition_context": 3,
                    "unsupported_problem_count": 1,
                    "missed_key_problem_count": 1,
                    "brief_rationale": "mock",
                },
                "augmented": {
                    "active_problem_coverage": 3,
                    "trajectory_capture": 3,
                    "plan_specificity": 3,
                    "evidence_grounding": 3,
                    "disposition_context": 3,
                    "unsupported_problem_count": 1,
                    "missed_key_problem_count": 1,
                    "brief_rationale": "mock",
                },
                "winner": "tie",
            },
            ensure_ascii=False,
        )
    if "compare two generated discharge summaries" in lower:
        admission_id = "mock"
        marker = '"admission_id": "'
        if marker in prompt:
            admission_id = prompt.split(marker, 1)[1].split('"', 1)[0]
        scores = {
            "diagnosis_coverage": 3,
            "hospital_course_completeness": 3,
            "temporal_order_correctness": 3,
            "discharge_plan_correctness": 3,
            "evidence_grounding": 3,
            "unsupported_claim_count": 1,
            "missed_major_event_count": 1,
            "overall_quality": 3,
        }
        return json.dumps(
            {
                "admission_id": admission_id,
                "scores": {"A": scores, "B": scores},
                "winner": "tie",
                "rationale": "mock",
            },
            ensure_ascii=False,
        )
    if "role-classification json" in lower or "diagnosis_candidates" in lower:
        return json.dumps(
            {
                "diagnosis_candidates": [
                    {
                        "candidate": "mock diagnosis",
                        "role": "principal_discharge_diagnosis",
                        "include_in_diagnosis": True,
                        "reason": "mock",
                        "final_phrase": "Mock diagnosis",
                    }
                ]
            },
            ensure_ascii=False,
        )
    if "return only valid json" in lower or "return one compact valid json object" in lower:
        return json.dumps(
            {
                "global_status": {
                    "overall_trajectory": "unclear",
                    "current_severity": "unclear",
                    "one_sentence_summary": "Mock state for plumbing validation.",
                },
                "active_ap_problems": [],
                "watchlist": [],
                "supportive_care": [],
                "resolved_problems": [],
                "uncertainties": [],
                "promotion_gate_notes": [],
                "admission_reason": [],
                "principal_diagnoses": [],
                "secondary_diagnoses": [],
                "major_procedures": [],
                "hospital_course_timeline": [],
                "complications": [],
                "treatments": [],
                "unresolved_problems_at_discharge": [],
                "discharge_medications": [],
                "follow_up": [],
                "must_not_add": [],
                "unsupported_claims": [],
                "missed_major_events": [],
                "must_add": [],
                "must_remove": [],
            },
            ensure_ascii=False,
        )
    if "return only bullet lines" in lower:
        return "- Mock diagnosis"
    if "write exactly three sections" in lower:
        return (
            "## 1. Diagnosis:\n"
            "- Mock diagnosis\n\n"
            "## 2. Hospital Course Summary:\n"
            f"Mock hospital course generated by {model} mock backend.\n\n"
            "## 3. Discharge Instructions:\n"
            "Mock discharge instructions for plumbing validation.\n"
        )
    return (
        "Assessment:\n"
        f"Mock AP generated by {model} mock backend for plumbing validation.\n\n"
        "Plan:\n"
        "- Continue evidence-grounded clinical monitoring.\n"
    )


def call_hf_router(
    prompt: str,
    model: str,
    api_url: str,
    hf_token_env: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    sleep_seconds: float,
    system_prompt: str,
) -> str:
    token = os.environ.get(hf_token_env)
    if not token:
        raise RuntimeError(f"{hf_token_env} is not set.")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
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
    raise RuntimeError(f"Hugging Face Router request failed after {retries} attempts: {last_error}")


@lru_cache(maxsize=2)
def _load_local_model(model: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Local Hugging Face inference requires torch and transformers. "
            "Use --hf-backend router if local dependencies or GPU memory are unavailable."
        ) from exc

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
        except Exception:
            pass
    tokenizer = AutoTokenizer.from_pretrained(model, token=token, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if token:
        model_kwargs["token"] = token
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16
        if os.environ.get("HF_LOCAL_8BIT", "0") == "1":
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs.pop("torch_dtype", None)
            except Exception:
                pass
    loaded = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
    if not torch.cuda.is_available():
        loaded = loaded.to("cpu")
    loaded.eval()
    return tokenizer, loaded


def call_local_transformers(
    prompt: str,
    model: str,
    temperature: float,
    max_new_tokens: int,
    system_prompt: str,
) -> str:
    import torch

    tokenizer, loaded = _load_local_model(model)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    else:
        text = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        encoded = tokenizer(text, return_tensors="pt").input_ids
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids

    device = next(loaded.parameters()).device
    encoded = encoded.to(device)
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": max(temperature, 1e-5),
        "pad_token_id": tokenizer.eos_token_id,
    }
    with torch.no_grad():
        output = loaded.generate(encoded, **generate_kwargs)
    new_tokens = output[0, encoded.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
