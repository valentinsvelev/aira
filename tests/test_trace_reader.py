"""Reading traces and sessions back from Phoenix for the in-app views."""
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from aira import trace_reader
from aira.api import create_app
from aira.config import Settings
from aira.trace_reader import (
    _is_agent_trace,
    _span_kind,
    normalize_trace,
    recent_sessions,
    recent_traces,
)


def client(tmp_path: Path):
    return TestClient(create_app(Settings(database_url=f"sqlite:///{tmp_path / 'test.db'}",
                                         papers_dir=tmp_path, llm_provider="ollama")))


def test_phoenix_reader_queries_local_project_and_redacts(monkeypatch):
    monkeypatch.setenv("PHOENIX_BASE_URL", "http://phoenix:6006")
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "aira")
    calls = []

    def respond(request):
        calls.append(request)
        if request.url.path.endswith("/spans"):
            # Even a trace summary with an attributes dict may lack detailed
            # model and prompt fields, so the adapter always fetches spans.
            return httpx.Response(200, json={"data": []})
        assert request.url.path == "/v1/projects/aira/traces"
        assert request.url.params["include_spans"] == "true"
        assert request.url.params["limit"] == "40"
        return httpx.Response(200, json={"data": [{
            "trace_id": "trace-1", "start_time": "2026-10-14T12:00:00Z",
            "end_time": "2026-10-14T12:00:01Z", "spans": [{
                "span_id": "span-root", "name": "research.chat", "span_kind": "AGENT",
                "start_time": "2026-10-14T12:00:00Z",
                "end_time": "2026-10-14T12:00:01Z", "attributes": {"session.id": "chat-a", "secret": "DO_NOT_LEAK"},
            }, {
                "span_id": "span-1", "name": "llm.chat", "span_kind": "LLM", "parent_id": "span-root",
                "start_time": "2026-10-14T12:00:00Z",
                "end_time": "2026-10-14T12:00:01Z", "attributes": {"secret": "DO_NOT_LEAK"},
            }],
        }], "next_cursor": None})

    transport = httpx.MockTransport(respond)
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=transport, **kwargs))
    output = recent_traces(2)
    assert len(calls) == 2
    assert calls[1].url.path.endswith("/spans")
    assert output["available"] is True
    assert output["traces"][0]["session_id"] == "chat-a"
    assert output["traces"][0]["spans"][1]["kind"] == "model"
    assert "DO_NOT_LEAK" not in str(output)


def test_phoenix_project_not_created(monkeypatch):
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(
        transport=httpx.MockTransport(lambda req: httpx.Response(404)), **kwargs))
    result = recent_traces(2)
    assert result["available"] and not result["traces"]


def test_phoenix_unreachable(monkeypatch):
    original = httpx.Client
    def broken(req):
        raise httpx.ConnectError("Connection refused")
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(
        transport=httpx.MockTransport(broken), **kwargs))
    assert recent_traces()["available"] is False


def test_invalid_span_timestamp_doesnt_crash():
    out = normalize_trace({"trace_id": "id", "spans": [{
        "span_id": "abc", "name": "tool.read_web_page",
        "start_time": "nonsense", "end_time": "nonsense",
    }]})
    assert out["duration_us"] == 0
    assert out["spans"][0]["duration_us"] == 0


def test_in_app_trace_listing_hides_http_health_checks(monkeypatch):
    from aira.trace_reader import recent_traces
    payload = {"data": [
        {"trace_id": "http-only", "spans": [{"name": "GET /health", "span_id": "h"}]},
        {"trace_id": "actual-research", "spans": [{"name": "research.chat", "span_id": "r"},
                                                     {"name": "llm.answer", "span_id": "m"}]},
    ]}
    orig = httpx.Client
    def respond(request):
        assert request.url.params["limit"] in {"40", "150"}
        return httpx.Response(200, json=payload)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: orig(
        transport=httpx.MockTransport(respond), **kwargs))
    result = recent_traces(2)
    assert result["available"]
    assert len(result["traces"]) == 1
    assert result["traces"][0]["id"] == "actual-research"


def test_in_app_trace_listing_explains_no_agent_runs(monkeypatch):
    from aira.trace_reader import recent_traces
    orig = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: orig(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"data": [
            {"trace_id": "health", "spans": [{"name": "GET /health", "span_id": "x"}]}
        ]})), **kwargs))
    out = recent_traces(3)
    assert out["available"] and not out["traces"]
    assert "health" in out["message"]


def test_trace_reader_removes_diagnostics_and_preserves_safe_metadata(monkeypatch):
    monkeypatch.setenv("OTEL_CAPTURE_CONTENT", "false")
    trace = {"trace_id": "ab", "spans": [
        {"span_id": "root", "name": "research.chat", "attributes": {
            "session.id": "chat-xyz", "input.value": "SECRET", "output.value": "SECRET ANSWER"}},
        {"span_id": "r", "name": "retrieval.search", "attributes": {
            "retrieval.strategy": "hybrid", "retrieval.returned_count": 4,
            "retrieval.documents.0.document.content": "PRIVATE PDF"}},
    ]}
    assert _is_agent_trace(trace)
    assert not _is_agent_trace({"spans": [{"name": "diagnostics.test_trace"}]})
    assert not _is_agent_trace({"spans": [{"name": "GET /health"}]})
    view = normalize_trace(trace)
    assert view["session_id"] == "chat-xyz"
    assert view["spans"][1]["strategy"] == "hybrid"
    assert view["spans"][1]["evidence_count"] == 4
    assert "PRIVATE PDF" not in str(view)
    assert "SECRET" not in str(view)
    monkeypatch.setenv("OTEL_CAPTURE_CONTENT", "true")
    revealed = normalize_trace(trace)
    assert revealed["spans"][0]["input"] == "SECRET"
    assert revealed["spans"][0]["output"] == "SECRET ANSWER"
    assert "PRIVATE PDF" not in str(revealed)


def test_reader_returns_only_research_runs(monkeypatch):
    original = httpx.Client
    traces = [{"trace_id": "x", "spans": [{"span_id": "probe", "name": "diagnostics.test_trace"}]},
              {"trace_id": "y", "spans": [{"span_id": "root", "name": "research.chat"}]}]
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"data": traces})), **kwargs))
    output = recent_traces(4)
    assert [t["id"] for t in output["traces"]] == ["y"]


def test_phoenix_session_trace_details_use_correct_rest_parameters(monkeypatch):
    monkeypatch.setenv('OTEL_CAPTURE_CONTENT','true')
    requests=[]
    def respond(req):
        requests.append(req)
        if req.url.path.endswith('/traces'):
            assert req.url.params['session_identifier'] == 'chat-001'
            assert 'session_id' not in req.url.params
            return httpx.Response(200, json={'data':[{
                'trace_id':'a'*32, 'start_time':'2026-10-15T10:00:00Z',
                'end_time':'2026-10-15T10:00:03Z',
                'spans':[{'span_id':'root','name':'research.chat', 'span_kind':'AGENT',
                          'start_time':'2026-10-15T10:00:00Z', 'end_time':'2026-10-15T10:00:03Z'}]
            }]})
        assert req.url.path.endswith('/spans')
        assert req.url.params['trace_id'] == 'a'*32
        return httpx.Response(200, json={'data':[
            {'context':{'trace_id':'a'*32,'span_id':'root'}, 'name':'research.chat','span_kind':'AGENT',
             'start_time':'2026-10-15T10:00:00Z', 'end_time':'2026-10-15T10:00:03Z',
             'attributes':{'session.id':'chat-001','input.value':'What is QLoRA?',
                           'output.value':'QLoRA is a quantization approach.'}},
            {'context':{'trace_id':'a'*32,'span_id':'child'}, 'parent_id':'root',
             'name':'llm.chat', 'span_kind':'LLM', 'attributes':{'llm.model_name':'qwen3.5:4b'},
             'start_time':'2026-10-15T10:00:01Z', 'end_time':'2026-10-15T10:00:03Z'},
        ]})
    orig=httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: orig(transport=httpx.MockTransport(respond), **kwargs))
    response=trace_reader.recent_traces(3,session_id='chat-001')
    assert response['available'] and len(response['traces'])==1
    trace=response['traces'][0]
    assert trace['session_id']=='chat-001'
    assert trace['spans'][0]['input']=='What is QLoRA?'
    assert trace['spans'][0]['output']=='QLoRA is a quantization approach.'
    assert trace['spans'][1]['parent_id']=='root'
    assert trace['spans'][1]['model']=='qwen3.5:4b'
    assert len(requests)==2


def test_phoenix_sessions_only_return_safe_metadata(monkeypatch):
    def fake_get(self, url, **kwargs):
        assert url.endswith('/v1/projects/aira/sessions')
        return httpx.Response(200, request=httpx.Request('GET', url), json={"data": [
            {"session_id": "chat-1", "num_traces": 4, "start_time": "2026-10-15T08:00:00Z",
             "input": "private text should never be returned"},
        ]})
    monkeypatch.setattr(httpx.Client, 'get', fake_get)
    response = recent_sessions()
    assert response['available']
    assert response['sessions'] == [{"session_id": "chat-1", "num_traces": 4, "start_time": "2026-10-15T08:00:00Z"}]
    assert 'private text' not in str(response)


def test_phoenix_session_filter_restricts_query_and_rejects_invalid_input(monkeypatch):
    params = []
    def fake_get(self, url, **kwargs):
        params.append(kwargs['params'])
        return httpx.Response(200, request=httpx.Request('GET', url), json={"data": []})
    monkeypatch.setattr(httpx.Client, 'get', fake_get)
    assert recent_traces(3, session_id="abc-123")['available']
    assert params[0]['session_identifier'] == 'abc-123'
    with pytest.raises(ValueError):
        recent_traces(3, session_id='../unsafe')


def test_trace_reader_only_returns_safe_fields():
    payload = {"trace_id": "testtrace", "start_time": "2026-10-14T12:00:00Z",
        "end_time": "2026-10-14T12:00:02Z", "spans": [
        {"span_id": "a", "name": "research.chat", "start_time": "2026-10-14T12:00:00Z",
         "end_time": "2026-10-14T12:00:02Z",
         "attributes": {"llm.input_messages": "PRIVATE_PROMPT"}},
        {"span_id": "b", "parent_id": "a", "name": "tool.web_search",
         "span_kind": "TOOL", "status_code": "ERROR",
         "start_time": "2026-10-14T12:00:00.250Z",
         "end_time": "2026-10-14T12:00:00.750Z",
         "attributes": {"tool.output": "PRIVATE_PROMPT"}},
    ]}
    normalized = normalize_trace(payload)
    assert normalized["id"] == "testtrace"
    assert normalized["spans"][1]["parent_id"] == "a"
    assert normalized["spans"][1]["kind"] == "tool"
    assert normalized["spans"][1]["duration_us"] == 500_000
    assert normalized["duration_us"] == 2_000_000
    assert normalized["spans"][1]["error"] is True
    assert "PRIVATE_PROMPT" not in str(normalized)


def test_traces_are_safe_when_phoenix_is_off(tmp_path, monkeypatch):
    import aira.trace_reader as tracer
    monkeypatch.setattr(tracer, "recent_traces", lambda limit: {"available": False, "traces": []})
    c = client(tmp_path)
    assert c.get("/traces").json() == {"available": False, "traces": []}
    assert c.get("/traces?limit=30").status_code == 422


def test_web_and_academic_search_are_tools_not_document_retrieval():
    assert _span_kind(None, "web.search") == "tool"
    assert _span_kind(None, "academic.search") == "tool"
    assert _span_kind(None, "memory.conversation.search") == "tool"
    assert _span_kind(None, "retrieval.search") == "retrieval"
    assert _span_kind(None, "llm.chat") == "model"
