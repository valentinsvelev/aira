"""Opt-in native inference server for second-stage retrieval ranking.

Start on localhost only. The calling API container uses RERANKER_BASE_URL.
The model is downloaded on first request, not on startup.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

JINA = "jinaai/jina-reranker-v3.5"
GTE = "Alibaba-NLP/gte-reranker-modernbert-base"
app = FastAPI(title="AIRA Reranking Service", docs_url=None, redoc_url=None)


class RerankRequest(BaseModel):
    model: Literal["jinaai/jina-reranker-v3.5", "Alibaba-NLP/gte-reranker-modernbert-base"]
    query: str = Field(min_length=1, max_length=5000)
    documents: list[str] = Field(min_length=1, max_length=100)


def inference_device() -> str:
    requested = os.getenv("RERANKER_DEVICE", "auto").lower()
    if requested != "auto":
        if requested not in {"cpu", "mps", "cuda"}:
            raise ValueError("RERANKER_DEVICE must be auto, cpu, mps or cuda")
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)  # avoid retaining both large models in unified memory
def load_model(name: str):
    device = inference_device()
    if name == GTE:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install reranking dependencies: pip install -e '.[reranking]'") from exc
        model = CrossEncoder(GTE, device=device, max_length=int(os.getenv("RERANKER_MAX_LENGTH", "2048")))
        return model
    if name == JINA:
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("Install transformers >= 4.51 for Jina v3.5") from exc
        # The authors' custom model.rerank implements true listwise inference.
        # Remote model code is explicitly enabled ONLY for this pinned, trusted ID.
        model = AutoModel.from_pretrained(
            JINA, revision=os.getenv("RERANKER_JINA_REVISION") or None,
            trust_remote_code=True, dtype="auto")
        model = model.to(device)
        model.eval()
        return model
    raise ValueError("Unknown reranker")


def score_documents(query: str, documents: list[str], model_name: str) -> list[dict]:
    model = load_model(model_name)
    if model_name == GTE:
        import numpy as np
        scores = np.asarray(model.predict([(query, doc) for doc in documents]), dtype=float).reshape(-1)
        ranking = [{"index": i, "relevance_score": float(score)} for i, score in enumerate(scores)]
    else:
        # The listwise API returns index, document, relevance_score and sorts by relevance.
        # We only keep indices + scores; raw paper text never comes back to callers.
        raw = model.rerank(query, documents)
        ranking = [{"index": item["index"], "relevance_score": float(item["relevance_score"])}
                   for item in raw]
    if (len(ranking) != len(documents) or
            {item["index"] for item in ranking} != set(range(len(documents))) or
            any(not math.isfinite(item["relevance_score"]) for item in ranking)):
        raise RuntimeError("Invalid model output (missing or non-finite scores)")
    return sorted(ranking, key=lambda item: -item["relevance_score"])


@app.get("/health")
def health():
    # Deliberately no model loading or Phoenix instrumentation.
    return {"status": "ready", "loaded_models": load_model.cache_info().currsize}


@app.post("/rerank")
def rerank(body: RerankRequest):
    try:
        return {"results": score_documents(body.query, body.documents, body.model)}
    except Exception as exc:
        # Local-only service. Avoid returning untrusted paper content on errors.
        raise HTTPException(status_code=503, detail=f"Reranker model failed: {type(exc).__name__}: {exc}") from exc
