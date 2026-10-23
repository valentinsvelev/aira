"""Workflow evidence gates, parallel retrieval and run playback."""
from threading import Barrier
from time import sleep
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aira.agent_workflows import WorkflowDefinition, execute_workflow, starter_workflow
from aira.api import create_app
from aira.config import Settings
from aira.options import GenerationOptions, RetrievalOptions


def gated(*, threshold=1, parallel=True):
    graph = starter_workflow()
    graph["parallel"] = parallel
    graph["nodes"].extend([
        {"id": "evidence_gate", "type": "evidence_gate", "min_sources": threshold, "x": 550, "y": 150},
        {"id": "web_search", "type": "web_search", "x": 770, "y": 200},
    ])
    graph["edges"] = [
        {"source": "question", "target": "local_search"},
        {"source": "local_search", "target": "evidence_gate"},
        {"source": "evidence_gate", "target": "synthesize", "condition": "enough"},
        {"source": "evidence_gate", "target": "web_search", "condition": "insufficient"},
        {"source": "web_search", "target": "synthesize"},
        {"source": "synthesize", "target": "answer"},
    ]
    return graph


def setup_model(monkeypatch):
    import aira.agent as module
    class Model:
        model = "offline"
        def invoke(self, messages, **kwargs):
            return SimpleNamespace(content="Answer with evidence [1].")
    monkeypatch.setattr(module, "make_model", lambda *_a, **_k: Model())
    monkeypatch.setattr(module, "_diversified_web_search",lambda *_a,**_k:[{"id":"web-id","excerpt":"Web corroboration."}])
    monkeypatch.setattr(module, "_web_sources",lambda hits:[{"id":item["id"],"path":"web.pdf"} for item in hits])


def run(graph, monkeypatch, tmp_path, hits, *, use_web=True):
    setup_model(monkeypatch)
    class Retriever:
        def search(self, *, query, options):
            return hits
    return execute_workflow(graph,question="Evidence?",session_id="demo",
        settings=Settings(papers_dir=tmp_path),factory=None,retriever=Retriever(),
        retrieval=RetrievalOptions(),generation=GenerationOptions(),
        profile={"enabled_tools":["local_search", "web_search", "academic_search"]},
        use_web=use_web,use_academic=False)


def test_gate_accepts_valid_and_rejects_bypass_or_missing_conditions():
    graph = gated()
    assert WorkflowDefinition.model_validate(graph).parallel
    for invalid in (
        {**graph, "edges":[{**e, "condition":None} if e["source"]=="evidence_gate" else e for e in graph["edges"]]},
        {**graph, "edges":[*graph["edges"],{"source":"question","target":"synthesize"}]},
        {**graph, "edges":[e for e in graph["edges"] if e["target"]!="web_search"]},
    ):
        with pytest.raises(ValueError):
            WorkflowDefinition.model_validate(invalid)
    assert WorkflowDefinition.model_validate(starter_workflow()).parallel


def test_gate_enough_skips_fallback(tmp_path,monkeypatch):
    result = run(gated(),monkeypatch,tmp_path,[{"id":"paper","path":"a.pdf","text":"Actual local evidence.","page":2}])
    assert result["sources"][0]["id"]=="paper"
    assert not result["web_used"]
    assert next(s for s in result["workflow_steps"] if s["node"]=="web_search")["status"]=="skipped"
    assert next(s for s in result["workflow_steps"] if s["node"]=="evidence_gate")["branch"]=="enough"
    assert result["workflow_graph"]["nodes"][-1]["id"]=="web_search"


def test_gate_missing_runs_fallback_and_respects_permission(tmp_path,monkeypatch):
    result = run(gated(threshold=2),monkeypatch,tmp_path,[])
    assert result["web_used"]
    assert result["sources"][0]["id"]=="web-id"
    assert next(s for s in result["workflow_steps"] if s["node"]=="evidence_gate")["branch"]=="insufficient"
    disabled = run(gated(),monkeypatch,tmp_path,[],use_web=False)
    assert disabled["sources"] == []
    assert next(s for s in disabled["workflow_steps"] if s["node"]=="web_search")["reason"]=="Per-chat permission is off"


def test_parallel_stage_citations_are_deterministic(tmp_path,monkeypatch):
    setup_model(monkeypatch)
    import aira.agent as module
    barrier = Barrier(2)
    class Retriever:
        def search(self,*,query,options):
            barrier.wait(timeout=3)
            sleep(.035)
            return [{"id":"local","path":"first.pdf","text":"First evidence."}]
    def web(*a, **kw):
        barrier.wait(timeout=3)
        return [{"id":"online","excerpt":"Second evidence."}]
    monkeypatch.setattr(module,"_diversified_web_search",web)
    wf=starter_workflow()
    wf["nodes"].append({"id":"web_search","type":"web_search"})
    wf["edges"].extend([{"source":"question","target":"web_search"},{"source":"web_search","target":"synthesize"}])
    result=execute_workflow(wf,question="Q",session_id="s",settings=Settings(papers_dir=tmp_path),
       factory=None,retriever=Retriever(),retrieval=RetrievalOptions(),generation=GenerationOptions(),
       profile={"enabled_tools":["local_search","web_search"]},use_web=True,use_academic=False)
    assert [s["id"] for s in result["sources"]]==["local","online"]
    stage1=[step for step in result["workflow_steps"] if step["stage"]==1]
    assert {step["node"] for step in stage1}=={"local_search","web_search"}
    assert "[1] first.pdf" in result["answer"] or result["answer"] == "Answer with evidence [1]."


def test_chat_persists_graph_and_playback_is_profile_scoped(tmp_path,monkeypatch):
    from aira.retrieval import LocalRetriever
    setup_model(monkeypatch)
    monkeypatch.setattr(LocalRetriever,"search",lambda *a,**kw: [{"id":"doc","path":"doc.pdf","text":"evidence"}])
    client=TestClient(create_app(Settings(database_url=f"sqlite:///{tmp_path/'workflow.db'}",papers_dir=tmp_path)))
    first=client.post("/agents",json={"name":"Graph", "enabled_tools":["local_search","web_search"]}).json()
    second=client.post("/agents",json={"name":"Other", "enabled_tools":["local_search"]}).json()
    graph=gated()
    assert client.put(f"/agents/{first['id']}/workflow",json=graph).status_code==200
    conversation=client.post("/conversations",json={"title":"Playback"}).json()
    response=client.post("/chat",json={"question":"Test", "session_id":conversation["id"],"agent_profile_id":first["id"]})
    assert response.status_code==200,response.text
    listed=client.get(f"/agents/{first['id']}/workflow/runs")
    assert listed.status_code==200,listed.text
    runs=listed.json()["runs"]
    assert len(runs)==1 and runs[0]["workflow"]["nodes"][-1]["id"]=="web_search"
    assert runs[0]["steps"][2]["node"]=="evidence_gate"
    assert not client.get(f"/agents/{second['id']}/workflow/runs").json()["runs"]
    # Historical graph remains the one saved with the chat even after editing the profile.
    assert client.put(f"/agents/{first['id']}/workflow",json=starter_workflow()).status_code==200
    assert client.get(f"/agents/{first['id']}/workflow/runs").json()["runs"][0]["workflow"]["nodes"][-1]["id"]=="web_search"
    assert client.get("/agents/missing/workflow/runs").status_code==404


def test_phoenix_replay_fetch_is_whitelisted_and_profile_scoped(tmp_path, monkeypatch):
    import httpx

    from aira.observability import _is_noise_span
    from aira.retrieval import LocalRetriever
    from aira.trace_reader import workflow_spans_for_trace
    setup_model(monkeypatch)
    monkeypatch.setattr(LocalRetriever, "search", lambda *a,**kw:[{"id":"pdf-1","path":"a.pdf","text":"Relevant text."}])
    client = TestClient(create_app(Settings(database_url=f"sqlite:///{tmp_path/'tracing.db'}",papers_dir=tmp_path)))
    profile = client.post("/agents",json={"name":"Trace test", "enabled_tools":["local_search"]}).json()
    conversation=client.post("/conversations",json={"title":"Trace"}).json()
    assert client.put(f"/agents/{profile['id']}/workflow",json=starter_workflow()).status_code==200
    res=client.post("/chat",json={"question":"Citations", "session_id":conversation['id'],"agent_profile_id":profile['id']})
    assert res.status_code==200
    run_id=client.get(f"/agents/{profile['id']}/workflow/runs").json()["runs"][0]["id"]
    no_trace=client.get(f"/agents/{profile['id']}/workflow/runs/{run_id}/phoenix")
    assert no_trace.status_code==200 and not no_trace.json()["available"]
    assert client.get(f"/agents/default/workflow/runs/{run_id}/phoenix").status_code==404

    calls=[]
    def fake(req):
        calls.append(req)
        assert req.url.path.endswith("/spans")
        return httpx.Response(200,json={"data":[
            {"name":"workflow.local_search","span_id":"span123","start_time":"2026-10-16T12:00:00Z","end_time":"2026-10-16T12:00:00.050Z", "attributes":{"input.value":"PRIVATE QUERY"}},
            {"name":"llm.answer","span_id":"span-secret","attributes":{"output.value":"PRIVATE TEXT"}},
        ]})
    original=httpx.Client
    monkeypatch.setattr(httpx,"Client",lambda **kwargs:original(transport=httpx.MockTransport(fake),**kwargs))
    details=workflow_spans_for_trace("0123456789abcdef0123456789abcdef")
    assert len(calls)==1
    assert details["available"] and details["spans"][0]["latency_ms"]==50
    assert "PRIVATE" not in str(details)
    from types import SimpleNamespace
    assert not _is_noise_span(SimpleNamespace(name="workflow.local_search",attributes={"session.id":"a"}))
    assert _is_noise_span(SimpleNamespace(name="workflow.local_search",attributes={}))
