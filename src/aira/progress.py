"""Best-effort, process-local research activity snapshots (not token streaming).

Only stage names, timestamps and counts are stored. Never prompts or results.
Development runs one API worker; callers must not assume progress survives restarts
or works with multiple independent workers.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from time import monotonic
from uuid import UUID

_REQUEST: ContextVar[str | None] = ContextVar("research_progress_id", default=None)
_LOCK = RLock()
_ITEMS: dict[str, dict] = {}
_STAGES = {
    "research.chat": "Preparing research request",
    "web.search": "Searching the web",
    "academic.search": "Searching academic sources",
    "retrieval.search": "Searching indexed papers",
    "memory.search": "Searching earlier conversation",
    "memory.conversation.search": "Searching earlier conversation",
    "agent.chain": "Running research agent",
    "llm.answer": "Generating the answer",
    "llm.chat": "Consulting the language model",
    "llm.generate": "Consulting the language model",
    "workflow.local_search": "Workflow · searching local papers",
    "workflow.academic_search": "Workflow · searching academic sources",
    "workflow.web_search": "Workflow · searching the web",
    "workflow.evidence_gate": "Workflow · checking whether evidence is sufficient",
    "workflow.synthesize": "Workflow · synthesizing evidence",
    "workflow.answer": "Workflow · completing the answer",
}


def _valid(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError):
        return None


def _update(key: str, stage: str, *, finished: bool = False, failed: bool = False) -> None:
    with _LOCK:
        now = monotonic()
        # Keep at most 128 short-lived snapshots across concurrent requests.
        for old_key in list(_ITEMS):
            if now - _ITEMS[old_key]["updated"] > 300:
                del _ITEMS[old_key]
        if len(_ITEMS) >= 128 and key not in _ITEMS:
            oldest = min(_ITEMS, key=lambda item: _ITEMS[item]["updated"])
            del _ITEMS[oldest]
        previous = _ITEMS.get(key)
        history = list(previous["history"]) if previous else []
        if not history or history[-1] != stage:
            history.append(stage)
        _ITEMS[key] = {**(previous or {}), "stage": stage, "history": history[-10:], "finished": finished,
                       "failed": failed, "updated": now}


@contextmanager
def track_request(identifier: str | None):
    key = _valid(identifier)
    token = _REQUEST.set(key)
    try:
        if key:
            _update(key, "Preparing research request")
        yield
        if key:
            _update(key, "Research request completed", finished=True)
    except Exception:
        if key:
            _update(key, "Request failed", finished=True, failed=True)
        raise
    finally:
        _REQUEST.reset(token)


def record_operation(name: str) -> None:
    key = _REQUEST.get()
    if key:
        label = _STAGES.get(name) or ("Using " + name.removeprefix("tool.").replace("_", " ")
                                      if name.startswith("tool.") else None)
        if label:
            _update(key, label)


def begin_workflow(graph: dict) -> None:
    """Publish only diagram topology, never prompts, search templates or passages.

    Live state is ephemeral and addressed by the unguessable request progress UUID.
    It shares the existing process-local progress store and its five-minute TTL.
    """
    key = _REQUEST.get()
    if not key:
        return
    safe = {
        "schema_version": graph.get("schema_version", 1),
        "nodes": [{"id": n["id"], "type": n["type"], "x": n.get("x", 0), "y": n.get("y", 0)}
                  for n in graph.get("nodes", [])[:45]],
        "edges": [{"source": e["source"], "target": e["target"], "condition": e.get("condition")}
                  for e in graph.get("edges", [])[:95]],
    }
    # Capture only the current request's trace identifier; never span attributes.
    trace_id = None
    try:
        from opentelemetry import trace
        context = trace.get_current_span().get_span_context()
        if context.is_valid:
            trace_id = f"{context.trace_id:032x}"
    except (ImportError, AttributeError):
        pass
    with _LOCK:
        item = _ITEMS.get(key)
        if item is not None:
            item["workflow"] = {"graph": safe, "events": [], "sequence": 0,
                                "trace_id": trace_id}
            item["updated"] = monotonic()


def record_workflow_event(node: str, status: str, *, stage: int = 0,
                          branch: str | None = None, sources: int = 0,
                          iteration: int | None = None, latency_ms: int | None = None,
                          span_id: str | None = None, reason: str | None = None) -> None:
    """Publish a bounded, content-free state transition for an executing node."""
    key = _REQUEST.get()
    if not key or status not in {"running", "completed", "skipped", "failed"}:
        return
    with _LOCK:
        item = _ITEMS.get(key)
        workflow = item.get("workflow") if item else None
        if not workflow or node not in {n["id"] for n in workflow["graph"]["nodes"]}:
            return
        workflow["sequence"] += 1
        event = {"sequence": workflow["sequence"], "node": node, "status": status,
                 "stage": int(stage), "sources": int(sources)}
        if branch in {"enough", "insufficient", "repeat", "done"}:
            event["branch"] = branch
        if iteration is not None:
            event["iteration"] = int(iteration)
        if latency_ms is not None:
            event["latency_ms"] = max(0, min(int(latency_ms), 3600000))
        # Span IDs must be opaque OTEL hex identifiers; never return other span data.
        import re
        if isinstance(span_id, str) and re.fullmatch(r"[0-9a-fA-F]{16}", span_id):
            event["span_id"] = span_id.lower()
        if isinstance(reason, str) and reason in {
                "Per-chat permission is off", "Disabled in Research Chat settings", "Evidence threshold met"}:
            event["reason"] = reason
        workflow["events"].append(event)
        workflow["events"] = workflow["events"][-360:]
        item["updated"] = monotonic()


def get_progress(identifier: str) -> dict:
    key = _valid(identifier)
    if key is None:
        raise ValueError("Invalid progress identifier")
    with _LOCK:
        item = _ITEMS.get(key)
        if item is None or monotonic() - item["updated"] > 300:
            return {"stage": "Waiting for research to start", "history": [], "finished": False, "failed": False}
        result = {field: item[field] for field in ("stage", "history", "finished", "failed")}
        if "workflow" in item:
            workflow = item["workflow"]
            result["workflow"] = {"graph": workflow["graph"], "events": list(workflow["events"]),
                                  "sequence": workflow["sequence"],
                                  "trace_id": workflow.get("trace_id")}
        return result
