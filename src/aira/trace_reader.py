"""Read-only local Phoenix trace adapter for Behind the Scenes.

Only safe operation metadata is returned; prompts, span attributes, outputs and
retrieved passages are *never* copied to the browser. The Phoenix URL is an
operator-supplied environment variable, never a user-provided address.
"""
import os
from datetime import UTC, datetime
from urllib.parse import quote

import httpx

from aira.observability import capture_content


def _timestamp_us(value: object) -> int:
    if not isinstance(value, str):
        return 0
    try:
        # datetime.fromisoformat accepts Phoenix's ISO8601 UTC timestamps.
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return int(parsed.timestamp() * 1_000_000)
    except (ValueError, OverflowError):
        return 0


def _span_kind(kind: object, name: str) -> str:
    label = str(kind or "").lower()
    operation = name.lower()
    if label == "llm" or operation.startswith("llm."):
        return "model"
    if label == "tool" or operation.startswith(("tool.", "web.", "academic.", "memory.")):
        return "tool"
    if label in {"retriever", "embedding"} or operation.startswith("retrieval."):
        return "retrieval"
    if label in {"agent", "chain"} or operation.startswith(("research.", "agent.", "literature.")):
        return "agent"
    return "operation"


def normalize_trace(trace: dict) -> dict:
    """Translate Phoenix's TraceData / SpanData to the existing UI format."""
    records = trace.get("spans") or []
    spans = []
    for record in records[:100]:
        # Phoenix returns span_id alongside a separate database id.
        span_id = str(record.get("span_id") or (record.get("context") or {}).get("span_id") or "")
        if not span_id:
            continue
        name = str(record.get("name") or "operation")[:100]
        start = _timestamp_us(record.get("start_time"))
        end = _timestamp_us(record.get("end_time"))
        # Whitelist operational metadata only. Phoenix may hold private prompts
        # or source passages: expose bounded input/output only with explicit opt-in.
        attrs = record.get("attributes") or {}
        if not isinstance(attrs, dict):
            attrs = {}
        def safe_str(key: str, limit: int = 100):
            val = attrs.get(key)  # noqa: B023 (used within this iteration)
            return str(val)[:limit] if isinstance(val, (str, int, float, bool)) else None
        data = {
            "id": span_id,
            "parent_id": str(record["parent_id"]) if record.get("parent_id") else None,
            "name": name,
            "kind": _span_kind(record.get("span_kind"), name),
            "start_us": start,
            "duration_us": max(0, end - start),
            "error": str(record.get("status_code") or "").upper() == "ERROR",
            "session_id": safe_str("session.id", 100) or safe_str("session_id", 100),
            "model": safe_str("llm.model_name"),
            "strategy": safe_str("retrieval.strategy"),
            "tool": safe_str("tool.name"),
            "evidence_count": next((attrs[k] for k in ("retrieval.returned_count", "search.results_count") if isinstance(attrs.get(k), int)), None),
            "input": None,
            "output": None,
        }
        if capture_content():
            # Never blindly return arbitrary span attributes to the web client.
            data["input"] = safe_str("input.value", 600)
            data["output"] = safe_str("output.value", 600)
        spans.append(data)
    start = _timestamp_us(trace.get("start_time"))
    end = _timestamp_us(trace.get("end_time"))
    if end <= start and spans:
        start = min(s["start_us"] for s in spans)
        end = max(s["start_us"] + s["duration_us"] for s in spans)
    research = next((s for s in spans if s["name"] == "research.chat"), None)
    return {
        "id": str(trace.get("trace_id") or ""),
        "session_id": (research.get("session_id") if research else None) or next((span.get("session_id") for span in spans if span.get("session_id")), None) or str(trace.get("session_id") or "") or None,
        "spans": spans,
        "duration_us": max(0, end - start),
    }


def _is_agent_trace(trace: dict) -> bool:
    """Show only user research runs, not health/diagnostic/admin requests."""
    return any(str(span.get("name") or "") in {"research.chat", "literature.survey"}
               for span in (trace.get("spans") or []))


def recent_sessions(limit: int = 30) -> dict:
    """Query real Phoenix sessions separately from the recent-trace window."""
    limit = max(1, min(int(limit), 100))
    endpoint = os.getenv("PHOENIX_BASE_URL", "http://phoenix:6006").rstrip("/")
    project = os.getenv("PHOENIX_PROJECT_NAME", "aira")
    try:
        with httpx.Client(timeout=8, follow_redirects=False) as client:
            response = client.get(f"{endpoint}/v1/projects/{quote(project, safe='')}/sessions", params={"limit": limit, "order": "desc"})
            if response.status_code == 404:
                return {"available": True, "sessions": [], "message": "No sessions recorded in Phoenix yet."}
            response.raise_for_status()
            payload = response.json()
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            raise ValueError("Unexpected Phoenix session response")
        sessions = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            session_id = str(row.get("session_id") or "")[:100]
            if not session_id:
                continue
            sessions.append({"session_id": session_id, "num_traces": row.get("num_traces") if isinstance(row.get("num_traces"), int) else len(row.get("traces") or []), "start_time": str(row.get("start_time") or "")[:64]})
        return {"available": True, "sessions": sessions, "message": None}
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        return {"available": False, "sessions": [], "message": "Phoenix sessions API is unavailable; recent research traces remain accessible."}


def recent_traces(limit: int = 10, session_id: str | None = None) -> dict:
    """Fetch a bounded list of actual traces using Phoenix's read-only REST API."""
    limit = max(1, min(int(limit), 20))
    if session_id is not None and (not 1 <= len(session_id) <= 100 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in session_id)):
        raise ValueError("Invalid session identifier")
    endpoint = os.getenv("PHOENIX_BASE_URL", "http://phoenix:6006").rstrip("/")
    project = os.getenv("PHOENIX_PROJECT_NAME", "aira")
    url = f"{endpoint}/v1/projects/{quote(project, safe='')}/traces"
    try:
        with httpx.Client(timeout=5, follow_redirects=False) as client:
            query = {"limit": min(100, max(40, limit * 6)), "include_spans": "true", "sort": "start_time", "order": "desc"}
            if session_id:
                query["session_identifier"] = session_id
            response = client.get(url, params=query)
            # Phoenix returns 404 if a project hasn't been created by its first span.
            if response.status_code == 404:
                return {"available": True, "traces": [],
                        "message": "No Phoenix project yet. Enable tracing and run a research query."}
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("Unexpected Phoenix response")
        research_traces = [t for t in payload["data"]
                           if isinstance(t, dict) and _is_agent_trace(t)]
        selected = research_traces[:limit]
        # The traces-list endpoint returns only summary spans. Their attributes
        # (LLM tokens, retrieval metadata, optional input/output) are absent.
        # Fetch full spans for each selected trace from Phoenix's spans API.
        # Preserve the summary when the Phoenix version cannot supply details.
        for trace in selected:
            # Always request detailed span records. Some Phoenix releases return
            # empty/partial attributes in trace summaries; their presence alone
            # does not mean the input, output and model metadata are included.
            trace_id = trace.get("trace_id")
            if not trace_id:
                continue
            try:
                with httpx.Client(timeout=8, follow_redirects=False) as span_client:
                    span_response = span_client.get(
                        f"{endpoint}/v1/projects/{quote(project, safe='')}/spans",
                        params={"trace_id": trace_id, "limit": 150, "sort": "start_time", "order": "asc"},
                    )
                    span_response.raise_for_status()
                    full = span_response.json()
                if isinstance(full, dict) and isinstance(full.get("data"), list) and full["data"]:
                    # Phoenix spans API nests span_id under context.
                    trace["spans"] = full["data"]
            except (httpx.HTTPError, ValueError, TypeError, AttributeError):
                pass  # Display the recorded summary rather than fabricate spans.
        normalized = [normalize_trace(t) for t in selected]
        if session_id:
            for item in normalized:
                item["session_id"] = item["session_id"] or session_id
        return {"available": True, "traces": normalized,
                "message": ("No research executions found. Send a NEW chat message with tracing enabled; "
                            "health checks and diagnostic traces are intentionally hidden."
                            if not research_traces else None)}
    except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
        # Do not leak service URLs or any HTTP response data to the browser.
        import logging
        logging.getLogger(__name__).warning("Phoenix trace lookup failed: %s", type(exc).__name__)
        return {"available": False, "traces": [],
                "message": "Phoenix is unavailable or its trace API cannot be queried. Start Phoenix and check the API logs."}


def workflow_spans_for_trace(trace_id: str) -> dict:
    """On-demand, read-only Phoenix lookup for a trusted saved chat trace ID.

    The caller resolves the trace ID from a persisted conversation turn. Return
    only an allowlisted set of operational fields, never model content.
    """
    import re
    if not re.fullmatch(r"[0-9a-fA-F]{32}", trace_id):
        return {"available": False, "spans": [], "message": "This run has no valid Phoenix trace ID."}
    endpoint = os.getenv("PHOENIX_BASE_URL", "http://phoenix:6006").rstrip("/")
    project = os.getenv("PHOENIX_PROJECT_NAME", "aira")
    try:
        with httpx.Client(timeout=8, follow_redirects=False) as client:
            response = client.get(f"{endpoint}/v1/projects/{quote(project, safe='')}/spans",
                                  params={"trace_id": trace_id, "limit": 100, "sort": "start_time", "order": "asc"})
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise ValueError("Unexpected Phoenix response")
        allowed = {"question", "local_search", "academic_search", "web_search", "evidence_gate", "loop", "synthesize", "answer"}
        spans = []
        for item in data["data"][:100]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            if not name.startswith("workflow.") or name.removeprefix("workflow.") not in allowed:
                continue
            span_id = str(item.get("span_id") or (item.get("context") or {}).get("span_id") or "")
            started = _timestamp_us(item.get("start_time"))
            ended = _timestamp_us(item.get("end_time"))
            # Schema-version-2 graphs may contain several nodes of the same kind. Prefer the
            # explicit node ID so playback highlights the correct instance.
            attrs = item.get("attributes") or {}
            actual_node = attrs.get("workflow.node") if isinstance(attrs, dict) else None
            node_id = actual_node if isinstance(actual_node, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,40}", actual_node) else name.removeprefix("workflow.")
            spans.append({"node": node_id, "span_id": span_id[:32],
                          "latency_ms": round(max(0, ended-started)/1000),
                          "status": "failed" if str(item.get("status_code") or "").upper() == "ERROR" else "completed"})
        return {"available": True, "spans": spans,
                "message": None if spans else "No workflow spans for this trace; confirm tracing is enabled for new runs."}
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        return {"available": False, "spans": [], "message": "Phoenix span details are currently unavailable."}
