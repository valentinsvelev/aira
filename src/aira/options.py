"""Bounded, request-scoped retrieval and generation parameters."""
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class RetrievalOptions(BaseModel):
    mode: Literal["sparse", "dense", "hybrid"] = "sparse"
    embedding_model: Literal["sentence-transformers/all-MiniLM-L6-v2", "Alibaba-NLP/gte-large-en-v1.5"] = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = Field(default=5, ge=1, le=12)
    bm25_candidates: int = Field(default=20, ge=1, le=100)
    dense_candidates: int = Field(default=20, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=200)
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    # Optional second-stage ranking, applied AFTER sparse/dense/hybrid candidate retrieval.
    reranker: Literal["none", "jinaai/jina-reranker-v3.5", "Alibaba-NLP/gte-reranker-modernbert-base"] = "none"
    rerank_candidates: int = Field(default=20, ge=1, le=100)

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_legacy_mode(cls, value):
        # Previously saved profiles and API clients may still use bm25.
        return "sparse" if value == "bm25" else value

    @model_validator(mode="after")
    def enough_candidates(self):
        if self.reranker != "none" and self.rerank_candidates < self.top_k:
            raise ValueError("rerank_candidates must be >= top_k when reranking is enabled")
        if self.mode == "sparse" and self.bm25_candidates < self.top_k:
            raise ValueError("bm25_candidates must be >= top_k")
        if self.mode == "dense" and self.dense_candidates < self.top_k:
            raise ValueError("dense_candidates must be >= top_k")
        return self


class GenerationOptions(BaseModel):
    model: Literal["llama3.2:3b", "qwen3.5:4b", "granite4.1:3b", "openbmb/minicpm-v4.7:1b"] | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=200)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    num_predict: int = Field(default=-1, description="-1 leaves output length to the model/Ollama")

    @field_validator("num_predict")
    @classmethod
    def valid_output_limit(cls, value: int) -> int:
        if value != -1 and not 32 <= value <= 32768:
            raise ValueError("num_predict must be -1 (model-decided) or 32..32768")
        return value
    num_ctx: int = Field(default=0, ge=0, le=1048576, description="0: use the model-advertised maximum context length (Ollama /api/show)")

    @field_validator("num_ctx")
    @classmethod
    def valid_context_window(cls, value: int) -> int:
        if value != 0 and value < 2048:
            raise ValueError("num_ctx must be 0 (model maximum) or at least 2048")
        return value
    repeat_penalty: float = Field(default=1.1, ge=0.8, le=2.0)
    seed: int | None = Field(default=None, ge=0, le=2147483647)
