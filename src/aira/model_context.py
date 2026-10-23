"""Resolve Ollama's advertised model context size when auto context is selected."""
import logging
from functools import lru_cache

_LOG = logging.getLogger(__name__)

@lru_cache(maxsize=24)
def ollama_model_max_context(base_url: str, model: str) -> int:
    """Read model_info's architecture-specific context_length via /api/show.

    A safe operational fallback is used if Ollama is unavailable; Ollama can
    advertise enormous contexts requiring more RAM than the local machine has.
    """
    try:
        import httpx
        response = httpx.post(f"{base_url.rstrip('/')}/api/show", json={"model": model}, timeout=5.0)
        response.raise_for_status()
        payload = response.json()
        info = payload.get("model_info") or {}
        lengths = [int(v) for k, v in info.items()
                   if k.endswith(".context_length") and isinstance(v, int) and v >= 2048]
        if lengths:
            return max(lengths)
    except Exception as exc:
        _LOG.warning("Could not read Ollama model context for %s: %s", model, type(exc).__name__)
    return 4096
