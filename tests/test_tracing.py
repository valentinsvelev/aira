"""OpenTelemetry tracing: opt-in export, content privacy, sessions and span metadata."""
from pathlib import Path

from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from aira import observability
from aira.api import create_app
from aira.config import Settings
from aira.trace_reader import normalize_trace


def _tracer(monkeypatch):
    from aira import observability
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(observability, "_TRACER", provider.get_tracer("test-evidence-lab"))
    return exporter


def test_opentelemetry_parent_child_and_private_attributes(monkeypatch):
    """Manual LangChain spans should belong to the active agent trace."""
    import importlib.util
    import sys
    from types import ModuleType

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from aira import observability

    # CI installs LangChain; this minimal test environment may not.
    if importlib.util.find_spec("langchain_core") is None:
        package = ModuleType("langchain_core")
        package.__path__ = []
        module = ModuleType("langchain_core.callbacks")
        module.BaseCallbackHandler = type("BaseCallbackHandler", (), {})
        monkeypatch.setitem(sys.modules, "langchain_core", package)
        monkeypatch.setitem(sys.modules, "langchain_core.callbacks", module)

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(observability, "_TRACER", provider.get_tracer("test"))
    handler = observability.callbacks()[0]
    with observability.trace_span("research.chat", mode="rag") as root:
        handler.on_chain_start({}, {}, run_id="chain")
        handler.on_tool_start({"name": "search_the_web"}, "SECRET_QUERY", run_id="tool", parent_run_id="chain")
        handler.on_tool_end("SECRET_TOOL_OUTPUT", run_id="tool")
        handler.on_chain_end({}, run_id="chain")

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["agent.chain"].parent.span_id == root.context.span_id
    assert spans["tool.search_the_web"].parent.span_id == spans["agent.chain"].context.span_id
    assert spans["tool.search_the_web"].attributes["openinference.span.kind"] == "TOOL"
    assert spans["tool.search_the_web"].status.status_code.name == "OK"
    assert "SECRET_QUERY" not in str(spans)
    assert "SECRET_TOOL_OUTPUT" not in str(spans)


def test_chat_response_has_real_trace_id_and_durable_metadata(tmp_path: Path, monkeypatch):
    from aira import api as api_module
    from aira.api import create_app
    from aira.config import Settings
    monkeypatch.setenv("OTEL_CAPTURE_CONTENT", "false")
    exporter = _tracer(monkeypatch)

    def fake_answer(question, session_id, settings, retriever, context, *args, **kwargs):
        from aira.observability import annotate_retrieval, trace_span
        with trace_span("retrieval.search", strategy="bm25") as child:
            annotate_retrieval(child, [{"id": "paper-1", "score": .87,
                                        "text": "PRIVATE PAPER PASSAGE"}])
        with trace_span("llm.answer", **{"llm.model_name": "qwen3.5:4b"}):
            pass
        return {"answer": "SECRET ANSWER", "model": "qwen3.5:4b", "sources": [],
                "memory_route": "research", "web_pages_fetched": 0}

    monkeypatch.setattr(api_module, "answer_with_local_context", fake_answer)
    monkeypatch.setenv("CHAT_TITLES_ENABLED", "false")
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'test.db'}",
                        papers_dir=tmp_path, llm_provider="ollama")
    with TestClient(create_app(settings)) as client:
        created = client.post("/conversations", json={"title": "Existing saved chat"}).json()
        response = client.post("/answer", json={"question": "PRIVATE QUESTION",
                                                 "session_id": created["id"]})
        assert response.status_code == 200, response.text
        data = response.json()
        assert len(data["trace_id"]) == 32
        turns = client.get(f"/conversations/{created['id']}/messages").json()["messages"]
        assert turns[-1]["metadata"]["trace_id"] == data["trace_id"]

    spans = {s.name: s for s in exporter.get_finished_spans()}
    root = spans["research.chat"]
    assert f"{root.context.trace_id:032x}" == data["trace_id"]
    assert root.status.status_code.name == "OK"
    assert spans["retrieval.search"].parent.span_id == root.context.span_id
    assert spans["llm.answer"].parent.span_id == root.context.span_id
    assert root.attributes["research.memory_route"] == "research"
    assert root.attributes["research.source_count"] == 0
    assert "PRIVATE QUESTION" not in str(root.attributes)
    assert "SECRET ANSWER" not in str(root.attributes)
    assert "PRIVATE PAPER PASSAGE" not in str(spans["retrieval.search"].attributes)


def test_content_capture_is_explicit(tmp_path: Path, monkeypatch):
    from aira import api as api_module
    from aira.api import create_app
    from aira.config import Settings
    monkeypatch.setenv("OTEL_CAPTURE_CONTENT", "true")
    exporter = _tracer(monkeypatch)

    monkeypatch.setattr(api_module, "answer_with_local_context", lambda *args, **kwargs: {
        "answer": "Some final answer", "sources": [], "memory_route": "neither"
    })
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'test.db'}",
                        papers_dir=tmp_path, llm_provider="ollama")
    with TestClient(create_app(settings)) as client:
        resp = client.post("/answer", json={"question": "What is AI?"})
        assert resp.status_code == 200
    root = next(s for s in exporter.get_finished_spans() if s.name == "research.chat")
    assert root.attributes["input.value"] == "What is AI?"
    assert root.attributes["output.value"] == "Some final answer"


def test_exporter_drops_health_and_fastapi_inner_spans():
    class Span:
        def __init__(self, name, attrs=None):
            self.name = name
            self.attributes = attrs or {}
    for name in ("GET /health", "fastapi.endpoint", "fastapi.dependencies", "GET /tracing/status"):
        assert observability._is_noise_span(Span(name))
    assert observability._is_noise_span(Span("GET", {"http.route": "/health"}))
    assert observability._is_noise_span(Span("GET /health", {"url.path": "/health"}))
    assert observability._is_noise_span(Span("POST /chat"))  # HTTP wrapper only; research.chat children remain
    assert not observability._is_noise_span(Span("research.chat", {"session.id": "chat-a"}))
    assert not observability._is_noise_span(Span("llm.answer", {"session.id": "chat-a"}))


def test_session_id_attaches_to_each_research_run(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(observability, "_TRACER", provider.get_tracer("session-test"))
    for _ in range(2):
        with observability.conversation_scope("saved-chat-123"):
            with observability.trace_span("research.chat"):
                with observability.trace_span("llm.answer"):
                    pass
    spans = exporter.get_finished_spans()
    roots = [s for s in spans if s.name == "research.chat"]
    assert len(roots) == 2
    assert roots[0].context.trace_id != roots[1].context.trace_id
    assert all(span.attributes.get("session.id") == "saved-chat-123" for span in spans)


def test_active_http_parent_is_tagged_with_phoenix_session(monkeypatch):
    from opentelemetry import trace
    class Parent:
        def __init__(self):
            self.attrs = {}
        def is_recording(self): return True
        def set_attribute(self, key, value): self.attrs[key] = value
    parent = Parent()
    monkeypatch.setattr(trace, "get_current_span", lambda: parent)
    with observability.conversation_scope("my-chat"):
        assert parent.attrs["session.id"] == "my-chat"
        assert parent.attrs["gen_ai.conversation.id"] == "my-chat"


def test_tracing_endpoints_report_disabled_and_do_not_export_content(monkeypatch, tmp_path):
    from aira import observability
    monkeypatch.setenv("OTEL_TRACES_ENABLED", "false")
    monkeypatch.setattr(observability, "_TRACER", None)
    monkeypatch.setattr(observability, "_PROVIDER", None)
    client = TestClient(create_app(Settings(database_url=f"sqlite:///{tmp_path / 'tracing.db'}", papers_dir=tmp_path)))
    status = client.get("/tracing/status")
    assert status.status_code == 200 and not status.json()["enabled"]
    emitted = client.post("/tracing/test")
    assert emitted.status_code == 200 and emitted.json()["flushed"] is False


def test_trace_capture_explicit_opt_in(monkeypatch):
    trace = {"trace_id":"trace", "spans":[{"span_id":"root","name":"research.chat",
             "start_time":"2026-10-15T08:00:00Z","end_time":"2026-10-15T08:00:01Z",
             "attributes":{"input.value":"private question","output.value":"private answer"}}]}
    monkeypatch.setenv("OTEL_CAPTURE_CONTENT", "false")
    hidden = normalize_trace(trace)
    assert hidden["spans"][0]["input"] is None
    assert hidden["spans"][0]["output"] is None
    monkeypatch.setenv("OTEL_CAPTURE_CONTENT", "true")
    shown = normalize_trace(trace)
    assert shown["spans"][0]["input"] == "private question"
    assert shown["spans"][0]["output"] == "private answer"


def test_observability_missing_sdk_is_optional(monkeypatch):
    import builtins

    import aira.observability as observability
    monkeypatch.setenv("OTEL_TRACES_ENABLED", "true")
    monkeypatch.setattr(observability, "_TRACER", None)
    original_import = builtins.__import__
    def block_sdk(name, *args, **kwargs):
        if name.startswith("opentelemetry.sdk"):
            raise ModuleNotFoundError("No module named 'opentelemetry.sdk'")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", block_sdk)
    assert observability.configure_tracing() is None


def test_tracing_is_off_by_default_and_excludes_source_text(monkeypatch):
    from aira.observability import callbacks, configure_tracing, trace_span
    monkeypatch.setenv("OTEL_TRACES_ENABLED", "false")
    assert configure_tracing() is None
    with trace_span("retrieval.search", strategy="hybrid") as span:
        assert span is None
    assert callbacks()==[]


def test_exporter_excludes_all_http_polls_but_preserves_manual_chat():
    class Span:
        def __init__(self, name, attributes=None):
            self.name = name
            self.attributes = attributes or {}
    for path in ('/health', '/traces', '/sessions', '/literature/reviews', '/context/123', '/chat'):
        assert observability._is_noise_span(Span('GET ' + path))
        assert observability._is_noise_span(Span('POST ' + path))
    assert observability._is_noise_span(Span('HTTP GET', {'http.request.method': 'GET'}))
    assert observability._is_noise_span(Span('some-auto-instrumented-span'))
    for name in ('research.chat', 'retrieval.search', 'llm.answer', 'tool.read_web_page'):
        assert not observability._is_noise_span(Span(name, {'session.id': 'chat-1'}))
        assert observability._is_noise_span(Span(name))
    assert observability._is_noise_span(Span('literature.discover', {'session.id': 'chat-1'}))
