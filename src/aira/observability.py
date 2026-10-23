"""Local-first OpenTelemetry tracing with OpenInference-compatible attributes.

Operational metadata is collected by default. Prompt, source and tool content
are collected only when OTEL_CAPTURE_CONTENT=true (explicit local opt-in).
"""
import json
import logging
import os
import re
import socket
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import urlparse

_TRACER = None
_PROVIDER = None
_TRACE_ERROR = None
_SESSION_ID = ContextVar("evidence_lab_session", default=None)


def capture_content() -> bool:
    return os.getenv("OTEL_CAPTURE_CONTENT", "false").strip().lower() == "true"


def _short(value, limit=1800) -> str:
    """Keep explicitly opted-in payloads bounded, even for enormous documents."""
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        text = "<unrenderable>"
    return text[:limit]


def _kind(name: str) -> str:
    if name.startswith("tool."):
        return "TOOL"
    if name.startswith(("retrieval.", "web.", "academic.")):
        return "RETRIEVER"
    if name.startswith("llm."):
        return "LLM"
    if name.startswith(("research.", "literature.", "agent.")):
        return "AGENT"
    return "CHAIN"


def _is_noise_span(span) -> bool:
    """Export only *application research spans*, never HTTP/ASGI/Phoenix polling.

    Filtering at the exporter boundary also handles third-party FastAPI auto-
    instrumentation. An exclusion list for /health alone is insufficient:
    /traces, /sessions, /literature/reviews and /context are polled by the UI.
    """
    name = str(getattr(span, "name", "") or "").lower()
    attributes = getattr(span, "attributes", {}) or {}
    # Instrumented request spans are infrastructure even for POST /chat;
    # their manually instrumented research.chat children remain visible.
    if (name.startswith(("fastapi.", "asgi.", "http.", "httpx.", "requests.",
                         "starlette.", "uvicorn.", "diagnostics.")) or
            re.match(r"^(get|post|put|patch|delete|options|head)\s+[/ ]", name)):
        return True
    if any(key in attributes for key in ("http.route", "http.target", "http.request.method",
                                           "http.method", "url.path", "http.url")):
        return True
    # A trace must belong to an actual saved research conversation. Prevent
    # standalone library browsing, benchmark calls and literature maintenance
    # from growing the Phoenix DB. Each real chat span carries session.id.
    if not attributes.get("session.id"):
        return True
    # Auto-instrumented framework internals must never become research spans.
    return not name.startswith(("research.chat", "agent.", "llm.",
                                "retrieval.", "tool.", "memory.", "web.", "academic.", "workflow."))


@contextmanager
def conversation_scope(session_id: str):
    """Attach one stable Phoenix session.id to every turn of a saved chat."""
    session_id = str(session_id).strip()
    token = _SESSION_ID.set(session_id or None)
    try:
        if session_id:
            # FastAPI/ASGI auto-instrumentors may already have a root HTTP span.
            # Phoenix derives sessions from this root as well as child spans.
            try:
                from opentelemetry import trace
                parent = trace.get_current_span()
                if parent.is_recording():
                    parent.set_attribute("session.id", session_id)
                    parent.set_attribute("gen_ai.conversation.id", session_id)
            except (ImportError, AttributeError):
                pass
        yield
    finally:
        _SESSION_ID.reset(token)


def configure_tracing():
    """Initialize once. Missing optional packages must never stop FastAPI."""
    global _TRACER, _PROVIDER, _TRACE_ERROR
    if os.getenv("OTEL_TRACES_ENABLED", "false").lower() != "true":
        return None
    if _TRACER is not None:
        return _TRACER
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        project = os.getenv("PHOENIX_PROJECT_NAME", "aira")
        provider = TracerProvider(resource=Resource.create({
            "service.name": "aira",
            "openinference.project.name": project,
        }))
        class NoiseFilteringExporter(OTLPSpanExporter):
            def export(self, spans):
                from opentelemetry.sdk.trace.export import SpanExportResult
                filtered = [span for span in spans if not _is_noise_span(span)]
                if not filtered:
                    return SpanExportResult.SUCCESS
                return super().export(filtered)

        provider.add_span_processor(BatchSpanProcessor(NoiseFilteringExporter(
            endpoint=os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://phoenix:6006/v1/traces"),
            headers={"x-project-name": project}, timeout=5,
        )))
        trace.set_tracer_provider(provider)
        _PROVIDER = provider
        _TRACE_ERROR = None
        _TRACER = trace.get_tracer("aira")
        logging.getLogger(__name__).info("Tracing initialized: Phoenix project=%s", project)
        return _TRACER
    except (ImportError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        _TRACE_ERROR = f"{type(exc).__name__}: {exc}"
        logging.getLogger(__name__).warning("OpenTelemetry unavailable (%s); continuing without traces", exc)
        return None


@contextmanager
def trace_span(name: str, **attributes):
    from aira.progress import record_operation
    record_operation(name)
    if _TRACER is None:
        yield None
        return
    with _TRACER.start_as_current_span(name) as span:
        span.set_attribute("openinference.span.kind", _kind(name))
        if _SESSION_ID.get():
            span.set_attribute("session.id", _SESSION_ID.get())
        for key, value in attributes.items():
            if isinstance(value, (str, bool, int, float)) and value is not None:
                span.set_attribute(key, value)
        try:
            yield span
            # Mark successful application operations explicitly, rather than
            # leaving Phoenix's status as "Unset" (as for HTTP health probes).
            from opentelemetry.trace.status import Status, StatusCode
            if span.is_recording() and span.status.status_code == StatusCode.UNSET:
                span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            span.record_exception(exc)
            span.set_attribute("error.type", type(exc).__name__)
            from opentelemetry.trace.status import Status, StatusCode
            span.set_status(Status(StatusCode.ERROR))
            raise


def annotate_retrieval(span, hits: list[dict]):
    """Attach inspectable evidence IDs/ranks/scores without raw PDF text by default."""
    if span is None:
        return
    span.set_attribute("retrieval.returned_count", len(hits))
    for idx, hit in enumerate(hits[:12]):
        prefix = f"retrieval.documents.{idx}.document"
        if hit.get("id") is not None:
            span.set_attribute(f"{prefix}.id", str(hit["id"]))
        if isinstance(hit.get("score"), (float, int)):
            span.set_attribute(f"{prefix}.score", float(hit["score"]))
        if isinstance(hit.get("page"), int):
            span.set_attribute(f"{prefix}.metadata", json.dumps({"page": hit["page"]}))
        if capture_content():
            if hit.get("text"):
                span.set_attribute(f"{prefix}.content", _short(hit["text"]))
            if hit.get("path"):
                span.set_attribute(f"{prefix}.metadata", json.dumps({
                    "page": hit.get("page"), "path": hit["path"],
                    "version": hit.get("version"),
                }))


def annotate_llm_response(span, reply):
    """Record real provider token counts when available; never estimate them."""
    if span is None or reply is None:
        return
    usage = getattr(reply, "usage_metadata", None) or {}
    metadata = getattr(reply, "response_metadata", None) or {}
    if not isinstance(usage, dict):
        usage = {}
    if not isinstance(metadata, dict):
        metadata = {}
    fields = {
        "llm.token_count.prompt": usage.get("input_tokens", metadata.get("prompt_eval_count")),
        "llm.token_count.completion": usage.get("output_tokens", metadata.get("eval_count")),
        "llm.token_count.total": usage.get("total_tokens"),
    }
    for attr, value in fields.items():
        if isinstance(value, int) and value >= 0:
            span.set_attribute(attr, value)
    # Provider identification makes this an LLM span Phoenix can classify.
    span.set_attribute("llm.system", str(metadata.get("llm_provider") or "ollama"))
    name = metadata.get("model_name") or metadata.get("model")
    if isinstance(name, str):
        span.set_attribute("llm.model_name", name)
    reason = metadata.get("done_reason") or metadata.get("finish_reason")
    if isinstance(reason, str):
        span.set_attribute("llm.finish_reason", reason)
    if capture_content():
        content = getattr(reply, "content", None)
        if content is not None:
            span.set_attribute("output.value", _short(content))


def callbacks():
    """LangChain callbacks that preserve OTel parents and enrich Phoenix spans.

    Deliberately avoids automatic instrumentation that records prompts by default.
    """
    if _TRACER is None:
        return []
    from threading import Lock

    from langchain_core.callbacks import BaseCallbackHandler
    from opentelemetry import trace

    class TraceHandler(BaseCallbackHandler):
        def __init__(self):
            self.spans = {}
            self.lock = Lock()

        def _start(self, run_id, name: str, parent_run_id=None, attrs=None):
            from aira.progress import record_operation
            record_operation(name)
            with self.lock:
                parent = self.spans.get(str(parent_run_id)) if parent_run_id else None
                context = trace.set_span_in_context(parent) if parent is not None else None
                span = _TRACER.start_span(name, context=context)
                span.set_attribute("openinference.span.kind", _kind(name))
                if _SESSION_ID.get():
                    span.set_attribute("session.id", _SESSION_ID.get())
                for attr, value in (attrs or {}).items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(attr, value)
                self.spans[str(run_id)] = span
                return span

        def _get(self, run_id):
            with self.lock:
                return self.spans.get(str(run_id))

        def _end(self, run_id, error=None):
            with self.lock:
                span = self.spans.pop(str(run_id), None)
            if span is not None:
                if not error and span.is_recording():
                    from opentelemetry.trace.status import Status, StatusCode
                    span.set_status(Status(StatusCode.OK))
                if error:
                    span.set_attribute("error.type", type(error).__name__)
                    span.record_exception(error)
                    from opentelemetry.trace.status import Status, StatusCode
                    span.set_status(Status(StatusCode.ERROR))
                span.end()

        def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
            span = self._start(run_id, "agent.chain", parent_run_id)
            if capture_content() and inputs is not None:
                span.set_attribute("input.value", _short(inputs))

        def on_chain_end(self, outputs, *, run_id, **kwargs):
            span = self._get(run_id)
            if span and capture_content() and outputs is not None:
                span.set_attribute("output.value", _short(outputs))
            self._end(run_id)

        def on_chain_error(self, error, *, run_id, **kwargs):
            self._end(run_id, error)

        def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
            invocation = kwargs.get("invocation_params") or {}
            model = invocation.get("model") or invocation.get("model_name")
            attrs = {"llm.system": "ollama"}
            if isinstance(model, str):
                attrs["llm.model_name"] = model
            # Model configuration is not private research content. Persist
            # only recognized, scalar inference parameters (no prompts).
            safe_config = {key: invocation[key] for key in
                           ("temperature", "top_p", "num_ctx", "num_predict", "repeat_penalty", "seed")
                           if isinstance(invocation.get(key), (str, int, float, bool))}
            if safe_config:
                attrs["llm.invocation_parameters"] = json.dumps(safe_config, sort_keys=True)
            span = self._start(run_id, "llm.chat", parent_run_id, attrs)
            if capture_content():
                # OpenInference's numbered message format can be displayed by Phoenix.
                for idx, message in enumerate((messages[0] if messages else [])[:12]):
                    role = getattr(message, "type", None)
                    content = getattr(message, "content", None)
                    if role:
                        span.set_attribute(f"llm.input_messages.{idx}.message.role", "user" if role == "human" else "assistant" if role == "ai" else str(role))
                    if content is not None:
                        span.set_attribute(f"llm.input_messages.{idx}.message.content", _short(content))
                if messages and messages[0]:
                    span.set_attribute("input.value", _short([getattr(m, "content", "") for m in messages[0]]))

        def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
            span = self._start(run_id, "llm.generate", parent_run_id, {"llm.system": "ollama"})
            if capture_content():
                for idx, prompt in enumerate(prompts[:12]):
                    span.set_attribute(f"llm.prompts.{idx}.prompt.text", _short(prompt))
                if prompts:
                    span.set_attribute("input.value", _short(prompts[0]))

        def on_llm_end(self, response, *, run_id, **kwargs):
            span = self._get(run_id)
            if span is not None:
                generations = getattr(response, "generations", None) or []
                first = generations[0][0] if generations and generations[0] else None
                message = getattr(first, "message", None)
                if message is not None:
                    annotate_llm_response(span, message)
                    if capture_content():
                        span.set_attribute("llm.output_messages.0.message.role", "assistant")
                        span.set_attribute("llm.output_messages.0.message.content", _short(getattr(message, "content", "")))
                        span.set_attribute("output.value", _short(getattr(message, "content", "")))
                usage = (getattr(response, "llm_output", None) or {}).get("token_usage", {})
                for key, attr in (("prompt_tokens", "llm.token_count.prompt"),
                                  ("completion_tokens", "llm.token_count.completion"),
                                  ("total_tokens", "llm.token_count.total")):
                    if isinstance(usage.get(key), int):
                        span.set_attribute(attr, usage[key])
            self._end(run_id)

        def on_llm_error(self, error, *, run_id, **kwargs):
            self._end(run_id, error)

        def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
            name = str(serialized.get("name", "unknown"))[:80] if isinstance(serialized, dict) else "unknown"
            span = self._start(run_id, f"tool.{name}", parent_run_id, {"tool.name": name})
            if capture_content():
                span.set_attribute("input.value", _short(input_str))

        def on_tool_end(self, output, *, run_id, **kwargs):
            span = self._get(run_id)
            if span is not None and capture_content():
                span.set_attribute("output.value", _short(output))
            self._end(run_id)

        def on_tool_error(self, error, *, run_id, **kwargs):
            self._end(run_id, error)

    return [TraceHandler()]


def tracing_status() -> dict:
    enabled = os.getenv("OTEL_TRACES_ENABLED", "false").strip().lower() == "true"
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://phoenix:6006/v1/traces")
    hostname = urlparse(endpoint).hostname
    port = urlparse(endpoint).port or 6006
    reachable = False
    if enabled and hostname:
        try:
            with socket.create_connection((hostname, port), timeout=1.5):
                reachable = True
        except OSError:
            pass
    return {
        "enabled": enabled,
        "initialized": _TRACER is not None,
        "collector_reachable": reachable,
        "project": os.getenv("PHOENIX_PROJECT_NAME", "aira"),
        "capture_content": capture_content(),
        "error": _TRACE_ERROR,
        "explanation": "Collector reachability and flush do not prove Phoenix accepted the trace. Check the named project after sending a test trace.",
    }


def emit_test_trace() -> dict:
    configure_tracing()
    status = tracing_status()
    if not status["initialized"]:
        return {**status, "flushed": False}
    with trace_span("diagnostics.test_trace", source="manual_diagnostic") as span:
        trace_id = f"{span.get_span_context().trace_id:032x}" if span else None
    flushed = bool(_PROVIDER.force_flush(timeout_millis=5000)) if _PROVIDER is not None else False
    return {**status, "flushed": flushed, "trace_id": trace_id}
