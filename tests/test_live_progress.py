"""Live, privacy-safe progress reporting for chat requests and workflow runs."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

from aira.agent_workflows import execute_workflow, starter_workflow
from aira.config import Settings
from aira.options import GenerationOptions, RetrievalOptions
from aira.progress import (
    begin_workflow,
    get_progress,
    record_operation,
    record_workflow_event,
    track_request,
)


def test_progress_tracks_operation_stages_without_user_content():
    request_id = str(uuid4())
    with track_request(request_id):
        record_operation("web.search")
        record_operation("academic.search")
        record_operation("llm.answer")
        snapshot = get_progress(request_id)
        assert snapshot["stage"] == "Generating the answer"
        assert "Searching the web" in snapshot["history"]
        assert snapshot["finished"] is False
    finished = get_progress(request_id)
    assert finished["finished"] is True
    assert finished["stage"] == "Research request completed"
    assert all("http" not in stage for stage in finished["history"])


def test_invalid_progress_identifier_does_not_store_anything():
    import pytest
    with pytest.raises(ValueError):
        get_progress("../../tracing/status")


def test_failed_request_marks_error():
    import pytest
    request_id = str(uuid4())
    with pytest.raises(RuntimeError), track_request(request_id):
        raise RuntimeError("Private prompt that must not be returned")
    progress = get_progress(request_id)
    assert progress["failed"] is True
    assert "Private prompt" not in str(progress)


def test_workflow_progress_does_not_expose_research_text():
    identifier = str(uuid4())
    secret = "DO-NOT-ECHO-RESEARCH-CONTENT"
    graph = {"nodes": [{"id": "q", "type": "question", "x": 1, "y": 2,
                        "query_template": secret}], "edges": []}
    with track_request(identifier):
        begin_workflow(graph)
        record_workflow_event("q", "running", stage=0)
        record_workflow_event("other", "completed", stage=0)  # unknown nodes rejected
        record_workflow_event("q", "completed", stage=0, branch="enough")
    result = get_progress(identifier)
    assert result["finished"] is True
    assert result["workflow"]["sequence"] == 2
    assert [event["status"] for event in result["workflow"]["events"]] == ["running", "completed"]
    assert secret not in repr(result)
    assert "query_template" not in repr(result)
    assert result["workflow"]["graph"]["nodes"][0] == {"id": "q", "type": "question", "x": 1, "y": 2}


def test_live_progress_reflects_running_search_and_completion(tmp_path, monkeypatch):
    import aira.agent as agent
    class Model:
        model = "offline"
        def invoke(self, messages, **kwargs):
            return SimpleNamespace(content="A summary [1].")
    monkeypatch.setattr(agent, "make_model", lambda *_a, **_k: Model())
    started, release = Event(), Event()
    class Retriever:
        def search(self, *, query, options):
            started.set()
            assert release.wait(4)
            return [{"id": "pdf-1", "path": "test.pdf", "text": "Relevant passage.", "page": 1}]
    identifier = str(uuid4())
    def run():
        with track_request(identifier):
            return execute_workflow(starter_workflow(), question="Private question",
                session_id="demo", settings=Settings(papers_dir=tmp_path), factory=None,
                retriever=Retriever(), retrieval=RetrievalOptions(), generation=GenerationOptions(),
                profile={"enabled_tools": ["local_search"]}, use_web=False, use_academic=False)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run)
        try:
            assert started.wait(4)
            progress = get_progress(identifier)
            events = progress["workflow"]["events"]
            assert any(event["node"] == "local_search" and event["status"] == "running" for event in events)
            assert not progress["finished"]
            assert "Private question" not in repr(progress)
        finally:
            release.set()
        assert future.result(timeout=5)["answer"] == "A summary [1]."
    finished = get_progress(identifier)
    assert finished["finished"]
    assert any(event["node"] == "answer" and event["status"] == "completed"
               for event in finished["workflow"]["events"])


def test_advanced_nested_loop_emits_repeated_visits(tmp_path, monkeypatch):
    from test_workflow_loops import nested

    import aira.agent as agent
    monkeypatch.setattr(agent, "make_model", lambda *_a, **_k:
                        SimpleNamespace(model="offline", invoke=lambda *args, **kw: SimpleNamespace(content="OK [1].")))
    count = 0
    class Retriever:
        def search(self, *, query, options):
            nonlocal count
            count += 1
            return [{"id": str(count), "path": "paper.pdf", "text": "Relevant passage"}]
    identifier = str(uuid4())
    with track_request(identifier):
        execute_workflow(nested(), question="Query", session_id="demo",
            settings=Settings(papers_dir=tmp_path), factory=None, retriever=Retriever(),
            retrieval=RetrievalOptions(), generation=GenerationOptions(),
            profile={"enabled_tools": ["local_search"]}, use_web=False, use_academic=False)
    events = get_progress(identifier)["workflow"]["events"]
    outer = [event for event in events if event["node"] == "outer" and event["status"] == "completed"]
    assert [event["branch"] for event in outer] == ["repeat", "repeat", "done"]
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))


def test_progress_eviction_stays_bounded():
    identifier = str(uuid4())
    with track_request(identifier):
        begin_workflow({"nodes": [{"id": "a", "type": "question"}], "edges": []})
        for i in range(410):
            record_workflow_event("a", "running" if i % 2 == 0 else "completed", stage=i)
    result = get_progress(identifier)["workflow"]
    assert result["sequence"] == 410
    assert len(result["events"]) == 360


def test_live_node_details_include_timing_and_valid_span_reference():
    request_id = str(uuid4())
    with track_request(request_id):
        begin_workflow({"schema_version": 2,
                        "nodes": [{"id": "search-1", "type": "local_search", "x": 3, "y": 5,
                                   "query_template": "SECRET-DOCUMENT-TEXT"}], "edges": []})
        record_workflow_event("search-1", "running", stage=2, iteration=1)
        record_workflow_event("search-1", "completed", stage=2, iteration=1,
                              latency_ms=137, sources=4, span_id="a" * 16)
        record_workflow_event("search-1", "running", stage=4, iteration=2)
        record_workflow_event("search-1", "completed", stage=4, iteration=2,
                              latency_ms=83, sources=2, span_id="b" * 16)
    result = get_progress(request_id)["workflow"]
    assert [event["iteration"] for event in result["events"]] == [1, 1, 2, 2]
    assert [event["latency_ms"] for event in result["events"] if event["status"] == "completed"] == [137, 83]
    assert [event["span_id"] for event in result["events"] if event["status"] == "completed"] == ["a" * 16, "b" * 16]
    assert "SECRET-DOCUMENT-TEXT" not in repr(result)


def test_progress_rejects_untrusted_details_and_unknown_nodes():
    request_id = str(uuid4())
    with track_request(request_id):
        begin_workflow({"nodes": [{"id": "gate", "type": "evidence_gate"}], "edges": []})
        record_workflow_event("fake", "completed", latency_ms=10, sources=10)
        record_workflow_event("gate", "completed", stage=2, latency_ms=-5,
                              sources=3, branch="insufficient", span_id="not-a-span-id",
                              reason="EXFILTRATE-DOCUMENT-SECRET")
    events = get_progress(request_id)["workflow"]["events"]
    assert len(events) == 1
    assert events[0]["latency_ms"] == 0
    assert events[0]["branch"] == "insufficient"
    assert "reason" not in events[0]
    assert "span_id" not in events[0]
