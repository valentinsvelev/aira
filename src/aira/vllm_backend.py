"""Local-only vLLM adapter for optional ScholarQA synthesis checkpoints.

A vLLM process ordinarily serves one checkpoint. Use separate addresses for the
DR-Tulu and AstaBrief processes. An explicit SCHOLAR_MODEL_BASE_URL is retained
as a legacy override, primarily for existing local setups.
"""
from __future__ import annotations

import os
from ipaddress import ip_address
from urllib.parse import urlsplit

import httpx

MODEL_URL_ENV = {
    "rl-research/DR-Tulu-8B": "SCHOLAR_VLLM_DR_TULU_URL",
    "allenai/AstaBrief_8B": "SCHOLAR_VLLM_ASTA_URL",
}
MODEL_DEFAULT_URL = {
    "rl-research/DR-Tulu-8B": "http://host.docker.internal:8001/v1",
    "allenai/AstaBrief_8B": "http://host.docker.internal:8002/v1",
}
LOCAL_NAMES = {"localhost", "host.docker.internal", "vllm-dr", "vllm-asta", "vllm", "sglang"}


def model_url(model_id: str) -> str:
    if model_id not in MODEL_URL_ENV:
        raise ValueError("Unknown optional ScholarQA model")
    base = (os.getenv(MODEL_URL_ENV[model_id]) or os.getenv("SCHOLAR_MODEL_BASE_URL")
            or MODEL_DEFAULT_URL[model_id]).strip().rstrip("/")
    parts = urlsplit(base)
    host = parts.hostname or ""
    if parts.scheme != "http" or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("vLLM endpoint must be an uncredentialed local HTTP URL")
    if parts.path not in ("", "/v1"):
        raise ValueError("vLLM endpoint should end in /v1")
    if host not in LOCAL_NAMES:
        try:
            ip = ip_address(host)
            if not (ip.is_private or ip.is_loopback):
                raise ValueError("vLLM must remain on a private network")
        except ValueError as exc:
            raise ValueError("vLLM endpoint must use a local/private host") from exc
    return (base if parts.path == "/v1" else base + "/v1")


def _headers() -> dict[str, str]:
    token = os.getenv("SCHOLAR_VLLM_API_KEY", "").strip() or os.getenv("SCHOLAR_MODEL_API_KEY", "").strip()
    return {"Authorization": "Bearer " + token} if token else {}


def model_status(model_id: str) -> dict:
    """Explicit user-initiated readiness probe (no periodic tracing/polling)."""
    base = model_url(model_id)
    try:
        with httpx.Client(timeout=4.0) as client:
            response = client.get(base + "/models", headers=_headers())
            response.raise_for_status()
            ids = [item.get("id", "") for item in response.json().get("data", [])]
        return {"model": model_id, "ready": model_id in ids, "served_models": ids,
                "detail": "Model available" if model_id in ids else "Server is running a different model"}
    except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
        return {"model": model_id, "ready": False, "served_models": [],
                "detail": f"Cannot reach local vLLM server ({type(exc).__name__})"}


def generate(model_id: str, system_prompt: str, user_prompt: str, generation=None) -> str:
    base = model_url(model_id)
    payload = {
        "model": model_id,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        "temperature": (0.7 if model_id == "allenai/AstaBrief_8B" and
                        (generation is None or generation.temperature == 0.0)
                        else generation.temperature if generation else 0.7),
        "top_p": (0.95 if model_id == "allenai/AstaBrief_8B" and
                  (generation is None or generation.top_p == 0.9)
                  else generation.top_p if generation else 0.95),
        "stream": False,
    }
    if generation and generation.num_predict != -1:
        payload["max_tokens"] = generation.num_predict
    elif model_id == "allenai/AstaBrief_8B":
        payload["max_tokens"] = 4096
    try:
        with httpx.Client(timeout=240) as client:
            response = client.post(base + "/chat/completions", json=payload, headers=_headers())
            response.raise_for_status()
            body = response.json()["choices"][0]["message"]
            content = body.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Model returned no answer content")
            return content.strip()
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Local vLLM model {model_id} failed at {base}: {exc}") from exc
