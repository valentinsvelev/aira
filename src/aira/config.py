"""Configuration, with local-first defaults."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///./.data/aira.db"
    papers_dir: Path = Path("papers")
    llm_provider: str = "ollama"  # ollama | openai
    ollama_model: str = "qwen3.5:4b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    faithfulness_mode: str = "semantic"  # semantic | extractive | tev1
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.78
    retrieval_mode: str = "sparse"
    tev1_model: str = "tev1:0.8b"
    tev1_supported_probability: float = 0.75
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    tavily_api_key: str | None = None
    web_search_provider: str = "ddgs"  # ddgs (free) | tavily | disabled
    web_search_max_results: int = 4
    openalex_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.getenv("DATABASE_URL", cls.database_url),
            papers_dir=Path(os.getenv("PAPERS_DIR", "papers")),
            llm_provider=os.getenv("LLM_PROVIDER", "ollama").lower(),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            faithfulness_mode=os.getenv("FAITHFULNESS_MODE", "semantic").lower(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.78")),
            retrieval_mode=("sparse" if os.getenv("RETRIEVAL_MODE", "sparse").lower() == "bm25" else os.getenv("RETRIEVAL_MODE", "sparse").lower()),
            tev1_model=os.getenv("TEV1_MODEL", "tev1:0.8b"),
            tev1_supported_probability=float(os.getenv("TEV1_SUPPORTED_PROBABILITY", "0.75")),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
            web_search_provider=os.getenv("WEB_SEARCH_PROVIDER", "ddgs").lower(),
            web_search_max_results=int(os.getenv("WEB_SEARCH_MAX_RESULTS", "4")),
            openalex_api_key=os.getenv("OPENALEX_API_KEY") or None,
        )


def make_context_manager(session_factory, settings: Settings):
    """Construct configured verifier lazily: normal ingestion needs no model download."""
    from aira.context import ContextManager
    if settings.faithfulness_mode == "extractive":
        return ContextManager(session_factory)
    if settings.faithfulness_mode == "semantic":
        from aira.semantic import SemanticChecker
        return ContextManager(session_factory, checker=SemanticChecker(
            settings.embedding_model, settings.similarity_threshold))
    if settings.faithfulness_mode == "tev1":
        from aira.tev1_checker import Tev1Checker
        return ContextManager(session_factory, checker=Tev1Checker(
            settings.ollama_base_url, settings.tev1_model, settings.tev1_supported_probability))
    raise ValueError("FAITHFULNESS_MODE must be extractive, semantic, or tev1")
