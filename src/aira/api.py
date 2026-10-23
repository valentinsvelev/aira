"""Local-only development API. No authentication; do not expose to the internet."""
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select, text

from aira import __version__
from aira.academic import AcademicSearchError, import_approved_pdf, scholar_url, search_academic
from aira.agent import answer_with_local_context, run_agent
from aira.agent_workflows import (
    WorkflowDefinition,
    execute_workflow,
    get_workflow,
    get_workflow_run,
    list_workflow_runs,
    save_workflow,
)
from aira.chat_titles import fallback_title, update_auto_title
from aira.config import Settings, make_context_manager
from aira.context import ContextManager
from aira.conversations import (
    create_conversation,
    delete_conversation,
    get_conversation,
    get_messages,
    list_conversations,
    recent_history,
    rename_conversation,
    save_exchange,
)
from aira.db import Document, DocumentVersion, EvidenceChunk, make_session_factory
from aira.ingestion import Ingestor
from aira.literature import (
    delete_review,
    discover,
    get_review,
    make_survey,
    new_review,
    review_list,
    set_paper_status,
)
from aira.memory import read_conversation_turns, search_conversation
from aira.observability import (
    configure_tracing,
    conversation_scope,
    emit_test_trace,
    trace_span,
    tracing_status,
)
from aira.options import GenerationOptions, RetrievalOptions
from aira.profiles import (
    ALLOWED_TOOLS,
    DEFAULT_TOOLS,
    create_profile,
    delete_profile,
    get_profile,
    list_profiles,
    save_default_profile,
    update_profile,
)
from aira.progress import get_progress, track_request
from aira.rag_systems import create_system, delete_system, get_system, list_systems, update_system
from aira.retrieval import LocalRetriever
from aira.scholarqa import answer_scholar_question
from aira.semantic import SemanticChecker
from aira.sources import evidence_detail
from aira.tev1_checker import Tev1Checker
from aira.vector_store import GTE_LARGE, MINILM, EmbeddingIndex
from aira.vllm_backend import MODEL_URL_ENV, model_status
from aira.websearch import WebSearchError, search_web


class RetrievalEvaluationRequest(BaseModel):
    dataset: Literal["smoke", "scifact"] = "smoke"
    retrieval: RetrievalOptions = Field(default_factory=RetrievalOptions)
    count: int | None = Field(default=10, ge=1, le=300)
    seed: int = Field(default=42, ge=0, le=2147483647)


class BenchmarkExample(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    answer: str = Field(min_length=1, max_length=2000)
    context: str = Field(default="", max_length=15000)


class BenchmarkRunRequest(BaseModel):
    dataset: Literal["qasper", "druid", "custom"]
    count: int = Field(default=3, ge=1, le=10)
    offset: int = Field(default=0, ge=0, le=5000)
    examples: list[BenchmarkExample] = Field(default_factory=list, max_length=10)
    save_results: bool = True  # local immutable artifacts for optional DuckDB analytics
    random_sample: bool = False  # legacy offset callers remain supported
    sampling_seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


class BenchmarkJobRequest(BaseModel):
    dataset: Literal["qasper", "druid", "custom"]
    examples: list[BenchmarkExample] = Field(default_factory=list, max_length=20000)
    save_results: bool = True


class AgentProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    system_instructions: str = Field(default="", max_length=6000)
    enabled_tools: list[str] = Field(default_factory=lambda: list(DEFAULT_TOOLS), max_length=6)
    generation: GenerationOptions | None = None

    @field_validator("enabled_tools")
    @classmethod
    def validate_tools(cls, tools: list[str]) -> list[str]:
        if len(set(tools)) != len(tools) or not set(tools).issubset(ALLOWED_TOOLS):
            raise ValueError("Unknown or repeated tool; allowed: " + ", ".join(sorted(ALLOWED_TOOLS)))
        if "local_search" not in tools:
            raise ValueError("local_search is required for a research assistant")
        return tools


class AgentProfileUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    system_instructions: str | None = Field(default=None, max_length=6000)
    enabled_tools: list[str] | None = Field(default=None, max_length=6)
    generation: GenerationOptions | None = None

    @field_validator("enabled_tools")
    @classmethod
    def validate_tools(cls, tools: list[str] | None) -> list[str] | None:
        return AgentProfileCreate.validate_tools(tools) if tools is not None else None


class RAGSystemBody(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    system_instructions: str = Field(default="", max_length=6000)
    retrieval: RetrievalOptions = Field(default_factory=RetrievalOptions)
    generation: GenerationOptions = Field(default_factory=GenerationOptions)


class ConversationCreate(BaseModel):
    title: str = Field(default="New chat", min_length=1, max_length=120)


class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class IndexRequest(BaseModel):
    embedding_model: str = MINILM


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=12)
    retrieval: RetrievalOptions | None = None


class ChatRequest(BaseModel):
    progress_id: str | None = Field(default=None, pattern=r"^[a-fA-F0-9-]{36}$")
    question: str = Field(min_length=1, max_length=5000)
    session_id: str = Field(default="default", min_length=1, max_length=100)
    agent_profile_id: str = Field(default="default", min_length=1, max_length=36)
    rag_system_id: str = Field(default="default", min_length=1, max_length=36)
    retrieval: RetrievalOptions | None = None
    generation: GenerationOptions = Field(default_factory=GenerationOptions)
    use_web: bool = False  # Only search the web when explicitly requested
    use_academic: bool = False  # Explicit opt-in for arXiv and OpenAlex API queries
    web_fetch_pages: int = Field(default=2, ge=0, le=3)  # Read actual pages, not only snippets
    memory_route: str = Field(default="auto", pattern="^(auto|conversation|research|both|neither)$")


class AcademicSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)
    sources: list[str] = Field(default_factory=lambda: ["arxiv", "openalex"], min_length=1)
    year_from: int | None = Field(default=None, ge=1900, le=2100)


class ScholarAnswerRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=8, ge=1, le=12)
    sources: list[Literal["arxiv", "openalex", "acl", "acm"]] = Field(
        default_factory=lambda: ["arxiv", "openalex"], min_length=1, max_length=4)
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    context_questions: list[str] = Field(default_factory=list, max_length=3)
    evidence_mode: Literal["full_text", "abstracts_only"] = "full_text"
    max_full_text_papers: int = Field(default=4, ge=1, le=6)
    model_id: Literal["llama3.2:3b", "qwen3.5:4b", "granite4.1:3b", "rl-research/DR-Tulu-8B", "allenai/AstaBrief_8B"] | None = None
    generation: GenerationOptions = Field(default_factory=GenerationOptions)


class LiteratureCreateRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    year_from: int | None = Field(default=None, ge=1900, le=2100)


class LiteratureDiscoverRequest(BaseModel):
    query: str | None = Field(default=None, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)
    sources: list[str] = Field(default_factory=lambda: ["arxiv", "openalex"], min_length=1)


class LiteratureStatusRequest(BaseModel):
    status: str


class LiteratureSurveyRequest(BaseModel):
    generation: GenerationOptions = Field(default_factory=GenerationOptions)


class ScholarLinkRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class PaperImportRequest(BaseModel):
    url: str = Field(min_length=10, max_length=1000)


class NoteRequest(BaseModel):
    verification_mode: str | None = None
    claim: str = Field(min_length=1, max_length=4000)
    quote: str = Field(min_length=1, max_length=6000)
    evidence_id: str = Field(min_length=1, max_length=64)
    replace_note_id: str | None = None
    expected_revision: int | None = Field(default=None, ge=0)


class DeleteNoteRequest(BaseModel):
    expected_revision: int | None = Field(default=None, ge=0)


class CompareRequest(BaseModel):
    verification_mode: str | None = None
    original: str = Field(min_length=1, max_length=2000)
    edited: str = Field(min_length=1, max_length=2000)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_tracing()
    factory = make_session_factory(settings.database_url)
    ingestor = Ingestor(factory, settings.papers_dir)
    retriever = LocalRetriever(factory, settings.embedding_model)
    indexer = EmbeddingIndex(factory, settings.embedding_model)
    context = make_context_manager(factory, settings)
    app = FastAPI(title="AIRA", version=__version__,
                  description="Local retrieval-augmented AI research assistant")

    def checked_embedding_model(model: str) -> str:
        if model not in (MINILM, GTE_LARGE):
            raise HTTPException(status_code=422, detail="Unsupported embedding model")
        return model

    def selected_checker(mode: str | None):
        mode = mode or settings.faithfulness_mode
        if mode == "semantic":
            return SemanticChecker(settings.embedding_model, settings.similarity_threshold)
        if mode == "tev1":
            return Tev1Checker(settings.ollama_base_url, settings.tev1_model,
                                settings.tev1_supported_probability)
        if mode == "extractive":
            from aira.context import check_extractively_supported
            return check_extractively_supported
        raise HTTPException(status_code=422, detail="Unknown verification mode")

    @app.get("/tracing/status")
    def tracing_diagnostics():
        return tracing_status()

    @app.post("/tracing/test")
    def trace_diagnostic():
        return emit_test_trace()

    @app.get("/sessions")
    def phoenix_sessions(limit: int = 30):
        from aira.trace_reader import recent_sessions
        return recent_sessions(limit)

    @app.get("/traces")
    def traces(limit: int = 10, session_id: str | None = None):
        if not 1 <= limit <= 20:
            raise HTTPException(status_code=422, detail="limit must be 1..20")
        from aira.trace_reader import recent_traces
        try:
            return recent_traces(limit, session_id=session_id) if session_id else recent_traces(limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Offline, relevance-judged retrieval evaluation. No LLM answer generation,
    # and never writes benchmark corpus documents into the user's library.
    @app.get("/retrieval-evals/datasets")
    def retrieval_eval_datasets():
        from aira.retrieval_evaluation import dataset_status
        return {"datasets": dataset_status(Path(os.getenv("AIRA_DATA_DIR", ".data")))}

    @app.get("/retrieval-evals/runs")
    def retrieval_eval_runs():
        from aira.retrieval_evaluation import list_runs
        return {"runs": list_runs(Path(os.getenv("AIRA_DATA_DIR", ".data")))}

    @app.get("/retrieval-evals/runs/{run_id}")
    def retrieval_eval_run(run_id: str):
        from aira.retrieval_evaluation import read_run
        result = read_run(Path(os.getenv("AIRA_DATA_DIR", ".data")), run_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Retrieval run not found")
        return result

    @app.post("/retrieval-evals/jobs", status_code=202)
    def start_retrieval_eval(body: RetrievalEvaluationRequest):
        from aira.retrieval_evaluation import start_job
        try:
            job_id = start_job(Path(os.getenv("AIRA_DATA_DIR", ".data")),
                               body.dataset, body.retrieval, body.count, body.seed)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"job_id": job_id, "status": "queued"}

    @app.get("/retrieval-evals/jobs/{job_id}")
    def retrieval_eval_job_status(job_id: str):
        from aira.retrieval_evaluation import job_status
        job = job_status(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found (API may have restarted)")
        return job

    @app.delete("/retrieval-evals/jobs/{job_id}")
    def cancel_retrieval_eval(job_id: str):
        from aira.retrieval_evaluation import cancel_job
        if not cancel_job(job_id):
            raise HTTPException(status_code=404, detail="Job not found or already finished")
        return {"status": "cancellation_requested"}

    @app.post("/benchmarks/run")
    def run_benchmark(body: BenchmarkRunRequest):
        """Explicitly initiated, bounded experiment; not a chat or Phoenix trace."""
        from aira.benchmarks import cases_for_benchmark, dataset_rows, run_cases
        if body.dataset == "custom":
            if not body.examples:
                raise HTTPException(status_code=422, detail="Upload JSON, JSONL or CSV with question and answer columns")
            cases = [{"id": str(i+1), "question": item.question, "context": item.context,
                      "gold": item.answer} for i, item in enumerate(body.examples[:body.count])]
            provenance = "Local browser upload (only selected rows sent to API)"
        else:
            try:
                if body.random_sample:
                    from aira.benchmarks import random_dataset_cases
                    cases, provenance, sample_seed = random_dataset_cases(
                        body.dataset, body.count, body.sampling_seed)
                else:
                    rows, provenance = dataset_rows(body.dataset, body.offset, max(body.count, 5))
                    cases = cases_for_benchmark(body.dataset, rows, body.count)
                    sample_seed = None
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not cases:
            raise HTTPException(status_code=422, detail="No usable labeled cases found in this sample")
        try:
            result = run_cases(body.dataset, cases, settings, retriever=retriever)
        except Exception as exc:
            # Do not leak model URLs/API tokens in detailed provider exceptions.
            raise HTTPException(status_code=503, detail=(
                "Model evaluation failed. Check that your configured LLM and, for "
                f"custom retrieval, your index are available ({type(exc).__name__}).")) from exc
        response = {"dataset": body.dataset, "source": provenance, **result}
        if body.random_sample and body.dataset != "custom":
            response["sampling_seed"] = sample_seed
        if body.save_results:
            from aira.experiment_store import save_experiment
            try:
                root = Path(os.getenv("AIRA_DATA_DIR", ".data"))
                response.update(save_experiment(root, dataset=body.dataset, provenance=provenance,
                    sample_offset=body.offset, model=(settings.ollama_model if settings.llm_provider == "ollama"
                    else settings.openai_model), result=result, configuration={
                        "llm_provider": settings.llm_provider,
                        "retrieval_mode": settings.retrieval_mode,
                        "embedding_model": settings.embedding_model,
                        "generation_temperature": 0,
                        "generation_num_predict": 256,
                        "sampling": "random" if body.random_sample else "sequential",
                        "sampling_seed": sample_seed if body.dataset != "custom" else body.sampling_seed,
                    }))
            except (OSError, ValueError, TypeError) as exc:
                # Scoring succeeded; surface persistence failures rather than pretending it was saved.
                response["save_error"] = f"Experiment completed but could not be saved ({type(exc).__name__})"
        return response

    @app.post("/benchmarks/jobs", status_code=202)
    def start_full_benchmark(body: BenchmarkJobRequest):
        from aira.benchmark_jobs import run_all_job
        if body.dataset == "custom" and not body.examples:
            raise HTTPException(status_code=422, detail="A local labeled dataset is required")
        try:
            job_id = run_all_job(dataset=body.dataset,
                                 examples=[item.model_dump() for item in body.examples],
                                 settings=settings, retriever=retriever, save_results=body.save_results)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"job_id": job_id, "status": "queued"}

    @app.get("/benchmarks/jobs/{job_id}")
    def full_benchmark_status(job_id: str):
        from aira.benchmark_jobs import get_job
        result = get_job(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Benchmark run not found or API restarted")
        return result

    @app.delete("/benchmarks/jobs/{job_id}")
    def stop_full_benchmark(job_id: str):
        from aira.benchmark_jobs import cancel_job
        if not cancel_job(job_id):
            raise HTTPException(status_code=404, detail="Benchmark run not found")
        return {"status": "cancellation_requested"}

    @app.get("/rag-systems")
    def available_rag_systems():
        return {"systems": list_systems(factory)}

    @app.post("/rag-systems", status_code=201)
    def add_rag_system(body: RAGSystemBody):
        if not body.name.strip():
            raise HTTPException(status_code=422, detail="RAG name cannot be blank")
        return create_system(factory, body.model_dump())

    @app.patch("/rag-systems/{profile_id}")
    def change_rag_system(profile_id: str, body: RAGSystemBody):
        found = update_system(factory, profile_id, body.model_dump())
        if not found: raise HTTPException(status_code=404, detail="Unknown RAG system")
        return found

    @app.delete("/rag-systems/{profile_id}", status_code=204)
    def remove_rag_system(profile_id: str):
        if profile_id == "default": raise HTTPException(status_code=403, detail="Built-in RAG cannot be deleted")
        if not delete_system(factory, profile_id): raise HTTPException(status_code=404, detail="Unknown RAG system")

    @app.get("/agents")
    def available_agents():
        return {"agents": [default_agent(), *list_profiles(factory)]}

    def default_agent():
        saved = get_profile(factory, "default")
        return {**saved, "builtin": True} if saved else {
            "id": "default", "name": "Research assistant",
            "description": "Built-in evidence-grounded researcher", "system_instructions": "",
            "enabled_tools": list(DEFAULT_TOOLS), "generation": None, "builtin": True}

    @app.post("/agents", status_code=201)
    def add_agent(body: AgentProfileCreate):
        if not body.name.strip():
            raise HTTPException(status_code=422, detail="Agent name cannot be blank")
        return create_profile(factory, body.name, body.description, body.system_instructions,
                              body.enabled_tools, (body.generation or GenerationOptions()).model_dump())

    @app.get("/agents/{profile_id}")
    def agent_detail(profile_id: str):
        found = default_agent() if profile_id == "default" else get_profile(factory, profile_id)
        if found is None:
            raise HTTPException(status_code=404, detail="Unknown agent profile")
        return found

    @app.patch("/agents/{profile_id}")
    def change_agent(profile_id: str, body: AgentProfileUpdate):
        if body.enabled_tools is not None:
            saved_graph = get_workflow(factory, profile_id)
            if saved_graph and saved_graph.get("enabled"):
                active_graph_tools = {node["type"] for node in saved_graph["nodes"] if node["type"] in {"local_search", "web_search", "academic_search"}}
                if not active_graph_tools.issubset(set(body.enabled_tools)):
                    raise HTTPException(status_code=422, detail="Disable or edit the executable workflow before removing its tool permissions")
        if profile_id == "default":
            if body.name is not None and not body.name.strip():
                raise HTTPException(status_code=422, detail="Agent name cannot be blank")
            return save_default_profile(factory, **body.model_dump(exclude_unset=True))
        if body.name is not None and not body.name.strip():
            raise HTTPException(status_code=422, detail="Agent name cannot be blank")
        updated = update_profile(factory, profile_id, **body.model_dump(exclude_unset=True))
        if updated is None:
            raise HTTPException(status_code=404, detail="Unknown agent profile")
        return updated

    @app.get("/agents/{profile_id}/workflow")
    def agent_workflow_detail(profile_id: str):
        if profile_id != "default" and get_profile(factory, profile_id) is None:
            raise HTTPException(status_code=404, detail="Unknown agent profile")
        return {"workflow": get_workflow(factory, profile_id)}

    @app.get("/agents/{profile_id}/workflow/runs")
    def agent_workflow_runs(profile_id: str):
        if profile_id != "default" and get_profile(factory, profile_id) is None:
            raise HTTPException(status_code=404, detail="Unknown agent profile")
        return {"runs": list_workflow_runs(factory, profile_id)}

    @app.get("/agents/{profile_id}/workflow/runs/{turn_id}/phoenix")
    def agent_workflow_phoenix(profile_id: str, turn_id: int):
        if profile_id != "default" and get_profile(factory, profile_id) is None:
            raise HTTPException(status_code=404, detail="Unknown agent profile")
        saved = get_workflow_run(factory, profile_id, turn_id)
        if saved is None:
            raise HTTPException(status_code=404, detail="Unknown workflow run")
        from aira.trace_reader import workflow_spans_for_trace
        return workflow_spans_for_trace(saved.get("trace_id") or "")

    @app.put("/agents/{profile_id}/workflow")
    def change_agent_workflow(profile_id: str, body: WorkflowDefinition):
        agent = default_agent() if profile_id == "default" else get_profile(factory, profile_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Unknown agent profile")
        try:
            return {"workflow": save_workflow(factory, profile_id, body, agent["enabled_tools"])}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/agents/{profile_id}", status_code=204)
    def remove_agent(profile_id: str):
        if profile_id == "default":
            raise HTTPException(status_code=403, detail="Built-in profile cannot be deleted")
        if not delete_profile(factory, profile_id):
            raise HTTPException(status_code=404, detail="Unknown agent profile")

    @app.get("/conversations")
    def conversations(limit: int = 50, offset: int = 0):
        if not 1 <= limit <= 100 or not 0 <= offset <= 100000:
            raise HTTPException(status_code=422, detail="Invalid pagination")
        return list_conversations(factory, limit=limit, offset=offset)

    @app.post("/conversations", status_code=201)
    def new_conversation(body: ConversationCreate):
        return create_conversation(factory, body.title)

    @app.get("/conversations/{conversation_id}")
    def conversation_detail(conversation_id: str):
        found = get_conversation(factory, conversation_id)
        if found is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return found

    @app.patch("/conversations/{conversation_id}")
    def update_conversation(conversation_id: str, body: ConversationRename):
        if not body.title.strip():
            raise HTTPException(status_code=422, detail="Title cannot be blank")
        found = rename_conversation(factory, conversation_id, body.title)
        if found is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return found

    @app.delete("/conversations/{conversation_id}", status_code=204)
    def remove_conversation(conversation_id: str):
        if not delete_conversation(factory, conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")
        # Context notes and provenance events remain in their existing audit tables.

    @app.get("/conversations/{conversation_id}/messages")
    def conversation_messages(conversation_id: str, limit: int = 50, before: int | None = None):
        if not 1 <= limit <= 100 or (before is not None and before < 1):
            raise HTTPException(status_code=422, detail="Invalid pagination")
        result = get_messages(factory, conversation_id, limit=limit, before=before)
        if result is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return result

    @app.get("/health")
    def health():
        configured = (settings.llm_provider == "ollama"
                      or (settings.llm_provider == "openai" and bool(settings.openai_api_key)))
        return {"status": "ok", "llm_enabled": configured,
                "llm_provider": settings.llm_provider, "local_model": settings.ollama_model,
                "model_name": settings.ollama_model if settings.llm_provider == "ollama" else settings.openai_model,
                "llm_reachable": "not_checked", "faithfulness_mode": settings.faithfulness_mode,
                "web_enabled": settings.web_search_provider != "disabled" and (settings.web_search_provider != "tavily" or bool(settings.tavily_api_key)),
                "web_provider": settings.web_search_provider,
                "retrieval": "sparse_dense_hybrid_pgvector",
                "vector_index": indexer.stats()}

    @app.get("/ready")
    def readiness():
        """Database-only readiness check; never calls inference or emits a test trace.

        This is intentionally different from the lightweight /health liveness endpoint.
        Do not return credentials, database URLs, or raw driver exceptions.
        """
        try:
            engine = factory.kw["bind"]
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                if engine.dialect.name == "postgresql":
                    vector = connection.scalar(text(
                        "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                    ))
                    if not vector:
                        raise RuntimeError("Required vector extension is not installed")
        except Exception as exc:
            # Log the exception class only; no database credentials or query results.
            import logging
            logging.getLogger(__name__).warning("Readiness failed: %s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Database not ready") from None
        return {"status": "ready", "database": "ok",
                "vector_extension": "ok" if engine.dialect.name == "postgresql" else "not_applicable"}

    @app.post("/ingest/sync")
    def ingest(index_vectors: bool = True):
        result = ingestor.sync()
        if index_vectors and factory.kw['bind'].dialect.name == 'postgresql':
            try:
                result['vector_index'] = indexer.sync()
            except Exception as exc:
                # Relational ingestion is committed; retain a visible, retryable error.
                result['vector_index_error'] = f"{type(exc).__name__}: {exc}"
        return result

    @app.post('/retrieval/reindex')
    def reindex(body: IndexRequest | None = None):
        try:
            return EmbeddingIndex(factory, checked_embedding_model(
                body.embedding_model if body else settings.embedding_model)).sync()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Vector indexing failed: {type(exc).__name__}") from exc

    @app.get('/retrieval/status')
    def retrieval_status(embedding_model: str = MINILM):
        return EmbeddingIndex(factory, checked_embedding_model(embedding_model)).stats()

    @app.get("/documents")
    def list_documents():
        """Return corpus inventory; inactive versions remain in the database."""
        documents = []
        with factory() as db:
            for document in db.scalars(select(Document).order_by(Document.path)).all():
                version = db.scalar(select(DocumentVersion).where(
                    DocumentVersion.document_id == document.id,
                    DocumentVersion.active.is_(True),
                ))
                count = (db.scalar(select(func.count(EvidenceChunk.id)).where(
                    EvidenceChunk.document_version_id == version.id)) if version else 0)
                documents.append({"path": document.path, "active": document.active,
                                  "version": version.version if version else None,
                                  "chunks": count or 0})
        return {"documents": documents}

    @app.post("/documents/upload")
    async def upload_document(file: UploadFile = File(...)):
        """Local-only controlled upload: save atomically inside PAPERS_DIR, then ingest."""
        name = file.filename or ""
        if (not name or name in {".", ".."} or name.startswith(".")
                or Path(name).name != name or "\\" in name
                or Path(name).suffix.lower() not in {".pdf", ".txt", ".md"}):
            raise HTTPException(status_code=400, detail="Use a PDF, TXT or MD filename without directory components")
        root = settings.papers_dir.resolve()
        root.mkdir(parents=True, exist_ok=True)
        destination = root / name
        if destination.is_symlink():
            raise HTTPException(status_code=400, detail="Refusing to replace a symlink")
        temporary = root / f".{uuid4().hex}.uploadtmp"
        total = 0
        try:
            with temporary.open("xb") as stream:
                while True:
                    block = await file.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > 20 * 1024 * 1024:
                        raise HTTPException(status_code=413, detail="File exceeds 20 MiB")
                    stream.write(block)
            if not total:
                raise HTTPException(status_code=400, detail="Empty upload")
            os.replace(temporary, destination)
            result = ingestor.ingest_file(destination)
            if result["status"] == "failed":
                raise HTTPException(status_code=422, detail=result["error"])
            if factory.kw['bind'].dialect.name == 'postgresql':
                try:
                    result['vector_index'] = indexer.sync()
                except Exception as exc:
                    result['vector_index_error'] = f"{type(exc).__name__}: {exc}"
            return result
        finally:
            await file.close()
            temporary.unlink(missing_ok=True)


    @app.post("/retrieval/search")
    def local_search(body: SearchRequest):
        options = body.retrieval or RetrievalOptions(top_k=body.top_k)
        try:
            return {"results": retriever.search(body.query, options=options),
                    "parameters": options.model_dump()}
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/web/search")
    def web_search(body: SearchRequest):
        try:
            return {"results": search_web(factory, body.query,
                provider=settings.web_search_provider, api_key=settings.tavily_api_key,
                max_results=min(body.top_k, settings.web_search_max_results))}
        except (WebSearchError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/academic/search")
    def academic_search(body: AcademicSearchRequest):
        try:
            return search_academic(factory, body.query, sources=body.sources,
                                   top_k=body.top_k, year_from=body.year_from,
                                   openalex_api_key=settings.openalex_api_key)
        except (AcademicSearchError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/scholar/model-status")
    def scholar_model_status():
        """Only requested from the settings button; no background polling."""
        return {"models": [model_status(model) for model in MODEL_URL_ENV]}

    @app.post("/scholar/answer")
    def scholar_answer(body: ScholarAnswerRequest):
        try:
            with trace_span("research.scholar_answer", model=body.model_id or body.generation.model or settings.ollama_model):
                return answer_scholar_question(
                    factory, body.question, settings, generation=body.generation,
                    sources=body.sources, top_k=body.top_k, year_from=body.year_from,
                    context_questions=body.context_questions, model_id=body.model_id,
                    evidence_mode=body.evidence_mode, max_full_text_papers=body.max_full_text_papers,
                )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502,
                                detail=f"Paper synthesis failed: {type(exc).__name__}: {str(exc)[:180]}") from exc

    @app.get("/literature/reviews")
    def literature_reviews():
        return {"reviews": review_list(factory)}

    @app.post("/literature/reviews", status_code=201)
    def literature_create(body: LiteratureCreateRequest):
        return new_review(factory, body.topic, body.year_from)

    @app.get("/literature/reviews/{review_id}")
    def literature_detail(review_id: str):
        result = get_review(factory, review_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Review not found")
        return result

    @app.delete("/literature/reviews/{review_id}", status_code=204)
    def literature_delete(review_id: str):
        if not delete_review(factory, review_id):
            raise HTTPException(status_code=404, detail="Review not found")

    @app.post("/literature/reviews/{review_id}/discover")
    def literature_discover(review_id: str, body: LiteratureDiscoverRequest):
        try:
            with trace_span("literature.discover", source_count=len(body.sources)):
                return discover(factory, review_id, body.query, settings, top_k=body.top_k, sources=body.sources)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Review not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.patch("/literature/reviews/{review_id}/papers/{paper_id}")
    def literature_set_status(review_id: str, paper_id: str, body: LiteratureStatusRequest):
        try:
            return set_paper_status(factory, review_id, paper_id, body.status)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Review paper not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/literature/reviews/{review_id}/papers/{paper_id}/import")
    def literature_import(review_id: str, paper_id: str):
        review = get_review(factory, review_id)
        if review is None:
            raise HTTPException(status_code=404, detail="Review not found")
        paper = next((p for p in review["papers"] if p["id"] == paper_id), None)
        if paper is None:
            raise HTTPException(status_code=404, detail="Review paper not found")
        if paper["status"] != "included":
            raise HTTPException(status_code=409, detail="Mark the paper included before importing")
        if not paper.get("pdf_url"):
            raise HTTPException(status_code=422, detail="No PDF URL available")
        try:
            result = import_approved_pdf(factory, settings.papers_dir, paper["pdf_url"])
            if result.get("status") == "failed":
                raise HTTPException(status_code=422, detail=result.get("error", "PDF processing failed"))
            set_paper_status(factory, review_id, paper_id, "included", imported=True)
            if factory.kw["bind"].dialect.name == "postgresql":
                try:
                    result["vector_index"] = indexer.sync()
                except Exception as exc:
                    result["vector_index_error"] = type(exc).__name__
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail="PDF provider unavailable") from exc

    @app.post("/literature/reviews/{review_id}/survey")
    def literature_survey(review_id: str, body: LiteratureSurveyRequest):
        try:
            with trace_span("literature.survey", model=body.generation.model or settings.ollama_model):
                return make_survey(factory, review_id, settings, body.generation)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Review not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Survey generation failed: {type(exc).__name__}") from exc

    @app.post("/academic/scholar-url")
    def academic_scholar(body: ScholarLinkRequest):
        return {"url": scholar_url(body.query),
                "mode": "human_browser", "automatic_scholar_scraping": False}

    @app.post("/academic/import")
    def import_paper(body: PaperImportRequest):
        # Human-triggered action. Strict URL allowlist in academic.py blocks SSRF.
        try:
            result = import_approved_pdf(factory, settings.papers_dir, body.url)
            if result.get("status") == "failed":
                raise HTTPException(status_code=422, detail=result.get("error", "PDF parsing failed"))
            if factory.kw["bind"].dialect.name == "postgresql":
                try:
                    result["vector_index"] = indexer.sync()
                except Exception as exc:
                    result["vector_index_error"] = f"{type(exc).__name__}: {exc}"
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"PDF download failed: {type(exc).__name__}") from exc

    @app.post("/context/compare")
    def compare_text(body: CompareRequest):
        checker = selected_checker(body.verification_mode)
        if isinstance(checker, Tev1Checker):
            verdict = checker.check(body.edited, body.original, body.original)
            return {"verification_method": "tev1_decision", "accepted": verdict.accepted,
                    "supported_probability": verdict.similarity, "reason": verdict.reason,
                    "warning": "TEV1 is experimental and does not prove entailment"}
        if not isinstance(checker, SemanticChecker):
            from aira.context import check_extractively_supported
            verdict = check_extractively_supported(body.edited, body.original, body.original)
            return {"verification_method": "extractive", "accepted": verdict.accepted,
                    "reason": verdict.reason, "warning": "Exact textual support only"}
        try:
            score = checker.compare(body.original, body.edited)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"cosine_similarity": round(score, 6),
                "threshold": settings.similarity_threshold,
                "above_threshold": score >= settings.similarity_threshold,
                "verification_method": "semantic_cosine",
                "warning": "Similarity is not factual entailment"}

    @app.get("/evidence/{evidence_id}")
    def get_evidence(evidence_id: str):
        detail = evidence_detail(factory, evidence_id)
        if not detail:
            raise HTTPException(status_code=404, detail="Unknown evidence ID")
        return detail

    @app.get("/context/{session_id}")
    def get_context(session_id: str):
        return context.notes(session_id)

    @app.get("/context/{session_id}/events")
    def get_context_events(session_id: str):
        return {"events": context.events(session_id)}

    @app.post("/context/{session_id}/notes")
    def propose_note(session_id: str, body: NoteRequest):
        verifier = (ContextManager(factory, checker=selected_checker(body.verification_mode))
                    if body.verification_mode else context)
        result = verifier.propose(session_id, body.claim, body.quote, body.evidence_id,
                                  body.replace_note_id, body.expected_revision)
        if not result["accepted"]:
            raise HTTPException(status_code=422, detail=result)
        return result

    @app.delete("/context/{session_id}/notes/{note_id}")
    def delete_note(session_id: str, note_id: str, body: DeleteNoteRequest | None = None):
        result = context.remove(session_id, note_id, body.expected_revision if body else None)
        if not result["accepted"]:
            raise HTTPException(status_code=409, detail=result)
        return result

    @app.get("/conversations/{conversation_id}/memory/search")
    def search_chat_memory(conversation_id: str, q: str, limit: int = 5):
        if not get_conversation(factory, conversation_id):
            raise HTTPException(status_code=404, detail="Unknown conversation")
        if not 1 <= limit <= 12 or not q.strip() or len(q) > 1000:
            raise HTTPException(status_code=422, detail="Invalid memory search")
        return {"results": search_conversation(factory, conversation_id, q, top_k=limit)}

    @app.get("/conversations/{conversation_id}/memory/turns")
    def read_chat_memory(conversation_id: str, ids: str):
        if not get_conversation(factory, conversation_id):
            raise HTTPException(status_code=404, detail="Unknown conversation")
        try:
            parts = ids.split(",")
            if len(parts) > 12 or any(not p.isdigit() for p in parts):
                raise ValueError
            turn_ids = [int(p) for p in parts]
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid turn identifiers") from None
        return {"results": read_conversation_turns(factory, conversation_id, turn_ids)}

    def execute_chat(body: ChatRequest, *, agent_mode: bool,
                     background_tasks: BackgroundTasks | None = None) -> dict:
        if settings.llm_provider == "openai" and not settings.openai_api_key:
            raise HTTPException(status_code=503, detail="Set OPENAI_API_KEY")
        options = body.retrieval or RetrievalOptions(
            mode=settings.retrieval_mode if factory.kw['bind'].dialect.name == 'postgresql' else 'sparse')
        rag_system = get_system(factory, body.rag_system_id) if not agent_mode else None
        if not agent_mode and rag_system is None:
            raise HTTPException(status_code=404, detail="Unknown RAG system")
        if not agent_mode and body.rag_system_id != "default":
            # Saved RAG presets determine both retrieval and generation.
            options = RetrievalOptions.model_validate(rag_system["retrieval"])
        # Legacy CLI session IDs still work without persisted chat history.
        saved_conversation = get_conversation(factory, body.session_id)
        is_saved_chat = saved_conversation is not None
        needs_title = bool(saved_conversation and saved_conversation["title"] == "New chat")
        history = recent_history(factory, body.session_id) if is_saved_chat else []
        profile = get_profile(factory, body.agent_profile_id)
        if body.agent_profile_id != "default" and profile is None:
            raise HTTPException(status_code=404, detail="Unknown agent profile")
        if profile:
            allowed = set(profile["enabled_tools"])
            if body.use_web and "web_search" not in allowed:
                raise HTTPException(status_code=422, detail="Selected agent does not allow web search")
            if body.use_academic and "academic_search" not in allowed:
                raise HTTPException(status_code=422, detail="Selected agent does not allow academic search")
        # Custom agents own their generation parameters. Request-level overrides
        # belong exclusively to the built-in/default profile.
        generation = (GenerationOptions.model_validate(profile["generation"] or {})
                      if profile and body.agent_profile_id != "default" else
                      GenerationOptions.model_validate(profile["generation"])
                      if profile and profile.get("generation") and "generation" not in body.model_fields_set
                      else body.generation)
        if not agent_mode and body.rag_system_id != "default":
            generation = GenerationOptions.model_validate(rag_system["generation"])
        try:
            with track_request(body.progress_id), conversation_scope(body.session_id), trace_span("research.chat", mode="agent" if agent_mode else "rag",
                            model=generation.model or settings.ollama_model,
                            **{"session.id": body.session_id,
                               "agent.profile_id": body.agent_profile_id,
                               "retrieval.strategy": options.mode,
                               "retrieval.top_k": options.top_k,
                               "generation.temperature": generation.temperature,
                               "generation.top_p": generation.top_p,
                               "generation.num_ctx": generation.num_ctx,
                               "research.web_enabled": body.use_web,
                               "research.academic_enabled": body.use_academic}) as research_span:
                # The trace ID is assigned by OTel and can be looked up directly
                # in Phoenix. Do not generate a pretend ID if tracing is disabled.
                run_trace_id = (f"{research_span.get_span_context().trace_id:032x}"
                                if research_span is not None and research_span.is_recording()
                                else None)
                if research_span is not None:
                    import json

                    from aira.observability import capture_content
                    research_span.set_attribute("metadata", json.dumps({
                        "agent_profile_id": body.agent_profile_id,
                        "retrieval_strategy": options.mode,
                        "retrieval_top_k": options.top_k,
                        "requested_memory_route": body.memory_route,
                        "web_enabled": body.use_web,
                        "academic_enabled": body.use_academic,
                    }, sort_keys=True))
                    if capture_content():
                        research_span.set_attribute("input.value", body.question[:1800])
                workflow = get_workflow(factory, body.agent_profile_id) if agent_mode else None
                if agent_mode and workflow and workflow.get("enabled"):
                    result = execute_workflow(workflow, question=body.question, session_id=body.session_id,
                        settings=settings, factory=factory, retriever=retriever, retrieval=options,
                        generation=generation, profile=profile or default_agent(),
                        use_web=body.use_web, use_academic=body.use_academic, history=history)
                elif agent_mode:
                    result = run_agent(body.question, body.session_id, settings, factory, retriever,
                                   context, options, generation, use_web=body.use_web,
                                   use_academic=body.use_academic,
                                   web_fetch_pages=body.web_fetch_pages, history=history,
                    **({"profile": {"system_instructions": rag_system["system_instructions"], "enabled_tools": list(DEFAULT_TOOLS)}} if rag_system and rag_system.get("system_instructions") else {}),
                                   **({"memory_route": body.memory_route} if body.memory_route != "auto" else {}),
                                   **({"profile": profile} if profile else {}))
                else:
                    result = answer_with_local_context(body.question, body.session_id, settings,
                    retriever, context, options, generation, use_web=body.use_web,
                    session_factory=factory, use_academic=body.use_academic,
                    web_fetch_pages=body.web_fetch_pages, history=history,
                    **({"profile": {"system_instructions": rag_system["system_instructions"], "enabled_tools": list(DEFAULT_TOOLS)}} if rag_system and rag_system.get("system_instructions") else {}),
                                   **({"memory_route": body.memory_route} if body.memory_route != "auto" else {}),
                                   **({"profile": profile} if profile else {}))
                if research_span is not None:
                    research_span.set_attribute("research.memory_route", result.get("memory_route", "unknown"))
                    research_span.set_attribute("research.source_count", len(result.get("sources") or []))
                    research_span.set_attribute("research.conversation_hits", len(result.get("conversation_sources") or []))
                    research_span.set_attribute("research.web_pages_read", result.get("web_pages_fetched", 0))
                    if capture_content():
                        research_span.set_attribute("output.value", str(result.get("answer", ""))[:1800])
            result["trace_id"] = run_trace_id
            result["agent_profile_id"] = body.agent_profile_id
            result["agent_profile_name"] = profile["name"] if profile else "Research assistant"
            if is_saved_chat:
                save_exchange(factory, body.session_id, body.question, result,
                              "chat" if agent_mode else "answer",
                              parameters={"retrieval": options.model_dump(),
                                          "generation": generation.model_dump(),
                                          "use_web": body.use_web,
                                          "use_academic": body.use_academic,
                                          "web_fetch_pages": body.web_fetch_pages,
                                          "memory_route": result.get("memory_route", body.memory_route),
                                          "agent_profile_id": body.agent_profile_id, "rag_system_id": body.rag_system_id})
                if needs_title and os.getenv("CHAT_TITLES_ENABLED", "true").lower() == "true":
                    provisional = fallback_title(body.question)
                    kwargs = {"model": os.getenv("OLLAMA_TITLE_MODEL", "qwen3.5:0.8b"),
                              "base_url": settings.ollama_base_url}
                    if background_tasks is not None:
                        # The answer is returned before this small optional local model call.
                        background_tasks.add_task(update_auto_title, factory, body.session_id,
                                                  provisional, body.question, result["answer"], **kwargs)
                    else:
                        update_auto_title(factory, body.session_id, provisional,
                                          body.question, result["answer"], **kwargs)
            return result
        except WebSearchError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            if str(exc).startswith("Reranker"):
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            raise HTTPException(status_code=502, detail=f"Research runtime error: {type(exc).__name__}") from exc
        except Exception as exc:
            label = "Agent" if agent_mode else "Local LLM"
            raise HTTPException(status_code=502, detail=f"{label} error: {type(exc).__name__}") from exc

    @app.get("/chat/progress/{progress_id}")
    async def chat_progress(progress_id: str):
        try:
            return get_progress(progress_id)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid progress identifier") from None

    @app.post("/answer")
    def answer(body: ChatRequest, background_tasks: BackgroundTasks):
        return execute_chat(body, agent_mode=False, background_tasks=background_tasks)

    @app.post("/chat")
    def chat(body: ChatRequest, background_tasks: BackgroundTasks):
        return execute_chat(body, agent_mode=True, background_tasks=background_tasks)

    return app


app = create_app()
