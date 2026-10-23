"""Second-stage local reranker transport. No model downloads inside the API container.

The separate inference service runs native on the host (MPS / CUDA / CPU).
HTTP failures propagate: NEVER silently return unre-ranked results when enabled.
"""
from __future__ import annotations

import math
import os

import httpx

RERANK_MODELS = ("jinaai/jina-reranker-v3.5", "Alibaba-NLP/gte-reranker-modernbert-base")


def rerank_hits(query: str, hits: list[dict], model: str, top_k: int) -> list[dict]:
    """Preserve IDs/source metadata; preserve first-stage scores for inspectability."""
    if not hits or model == "none":
        return hits[:top_k]
    if model not in RERANK_MODELS:
        raise ValueError("Unknown reranker")
    # The API runs in Docker on macOS; override for API running directly on host.
    base_url = os.getenv("RERANKER_BASE_URL", "http://host.docker.internal:8010").rstrip("/")
    try:
        response = httpx.post(f"{base_url}/rerank", json={
            "model": model, "query": query, "documents": [hit["text"] for hit in hits],
        }, timeout=httpx.Timeout(float(os.getenv("RERANKER_TIMEOUT_SECONDS", "300")), connect=4.0))
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(
            f"Reranker '{model}' unavailable at {base_url}. "
            "Start the optional local reranker service or choose 'None'. "
            f"Details: {exc}"
        ) from exc

    ordered = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(ordered, list) or len(ordered) != len(hits):
        raise RuntimeError("Reranker returned an incomplete candidate ranking")
    seen: set[int] = set()
    output: list[dict] = []
    for result in ordered:
        try:
            index = result["index"]
            score = float(result["relevance_score"])
            if type(index) is not int or not 0 <= index < len(hits) or index in seen or not math.isfinite(score):
                raise ValueError("Invalid index or score")
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Reranker returned invalid index or score") from exc
        seen.add(index)
        item = {**hits[index], "first_stage_score": hits[index].get("score"),
                "first_stage_rank": index + 1,
                "score": score, "reranker": model}
        output.append(item)
    # Trust only scores/indices, never texts or IDs returned by external inference.
    output.sort(key=lambda hit: -hit["score"])  # stable ties preserve first-stage order
    return output[:top_k]
