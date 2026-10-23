"""Executable agent workflows: persistence, permissions, validation and execution."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aira.agent_workflows import (
    WorkflowDefinition,
    execute_workflow,
    get_workflow,
    starter_workflow,
)
from aira.api import create_app
from aira.config import Settings
from aira.db import make_session_factory
from aira.options import GenerationOptions, RetrievalOptions


def test_graph_validation_denies_cycles_unapproved_edges_and_missing_connections():
    good = starter_workflow()
    assert WorkflowDefinition.model_validate(good).enabled
    bad = {**good, "edges": [*good["edges"], {"source": "answer", "target": "question"}]}
    with pytest.raises(ValueError, match="invalid"):
        WorkflowDefinition.model_validate(bad)
    with pytest.raises(ValueError, match="unique"):
        WorkflowDefinition.model_validate({**good, "nodes": [*good["nodes"], good["nodes"][0]]})
    broken = {**good, "edges": [edge for edge in good["edges"] if edge["target"] != "synthesize"]}
    with pytest.raises(ValueError, match="requires"):
        WorkflowDefinition.model_validate(broken)


def test_graph_save_and_restart_and_permissions(tmp_path):
    path = str(tmp_path / "db.sqlite")
    def client():
        return TestClient(create_app(Settings(database_url=f"sqlite:///{path}", papers_dir=tmp_path, llm_provider="ollama")))
    api = client()
    agent = api.post("/agents", json={"name":"Pipeline", "enabled_tools":["local_search"]}).json()
    agent_id = agent["id"]
    wf = starter_workflow()
    response = api.put(f"/agents/{agent_id}/workflow", json=wf)
    assert response.status_code == 200, response.text
    assert api.get(f"/agents/{agent_id}/workflow").json()["workflow"] == response.json()["workflow"]
    assert client().get(f"/agents/{agent_id}/workflow").json()["workflow"] == response.json()["workflow"]
    changed = {**wf, "nodes": [*wf["nodes"], {"id":"web_search","type":"web_search"}],
               "edges": [*wf["edges"], {"source":"question","target":"web_search"}, {"source":"web_search","target":"synthesize"}]}
    blocked = api.put(f"/agents/{agent_id}/workflow", json=changed)
    assert blocked.status_code == 422 and "permitted" in blocked.text
    assert api.put(f"/agents/{agent_id}/workflow", json={**wf,"nodes":[*wf["nodes"],{"id":"exec","type":"exec"}]}).status_code == 422
    assert api.put("/agents/missing/workflow", json=wf).status_code == 404
    assert api.delete(f"/agents/{agent_id}").status_code == 204
    assert get_workflow(make_session_factory(f"sqlite:///{path}"), agent_id) is None


def test_running_graph_preserves_source_citations_and_order(monkeypatch, tmp_path):
    import aira.agent as agent_module
    messages = []
    class Retriever:
        def search(self, *, query, options):
            assert options.mode == "sparse"
            return [{"id":"chunk1","path":"paper.pdf","page":3,"version":1,"score":.9,
                     "text":"The experiment improves recall for scientific retrieval."}]
    class Model:
        model = "fake-model"
        def invoke(self, data, **kwargs):
            messages.extend(data)
            return SimpleNamespace(content="Recall improves [1].")
    monkeypatch.setattr(agent_module,"make_model",lambda *a,**kw: Model())
    result = execute_workflow(starter_workflow(),question="What improves recall?",session_id="test",
        settings=Settings(papers_dir=tmp_path),factory=None,retriever=Retriever(),
        retrieval=RetrievalOptions(),generation=GenerationOptions(),
        profile={"enabled_tools":["local_search"],"system_instructions":"Be concise."},
        use_web=False,use_academic=False)
    assert result["answer"] == "Recall improves [1]."
    assert result["sources"][0]["id"] == "chunk1"
    assert [step["node"] for step in result["workflow_steps"]] == ["question","local_search","synthesize","answer"]
    assert "paper.pdf page=3" in messages[-1][1]
    assert "Be concise." in messages[0][1]


def test_workflow_used_by_chat_only_when_enabled(tmp_path,monkeypatch):
    import aira.api as api_module
    c = TestClient(create_app(Settings(database_url=f"sqlite:///{tmp_path/'execution.db'}", papers_dir=tmp_path, llm_provider="ollama")))
    agent = c.post("/agents",json={"name":"Pipeline","enabled_tools":["local_search"]}).json()
    conversation = c.post("/conversations",json={"title":"Graph example"}).json()
    c.put(f"/agents/{agent['id']}/workflow",json=starter_workflow())
    calls=[]
    def stub(*args, **kwargs):
        calls.append("graph")
        return {"answer":"Graph answer [1]", "sources":[],"model":"stub","web_used":False,"academic_used":False,"workflow_steps":[],"memory_route":"research"}
    monkeypatch.setattr(api_module, "execute_workflow", stub)
    response = c.post("/chat",json={"question":"How?","session_id":conversation['id'],"agent_profile_id":agent['id']})
    assert response.status_code == 200, response.text
    assert calls == ["graph"]
    assert c.get(f"/conversations/{conversation['id']}/messages").json()["messages"][-1]["content"] == "Graph answer [1]"
    assert c.put(f"/agents/{agent['id']}/workflow",json={**starter_workflow(),"enabled":False}).status_code == 200
    def old_agent(*args,**kwargs):
        calls.append("old")
        return {"answer":"Original agent answer", "sources":[],"model":"stub","web_used":False,"academic_used":False,"memory_route":"auto"}
    monkeypatch.setattr(api_module,"run_agent",old_agent)
    response = c.post("/chat",json={"question":"Why?","session_id":conversation['id'],"agent_profile_id":agent['id']})
    assert response.status_code == 200, response.text
    assert calls[-1] == "old"


def test_active_graph_cannot_have_its_tool_permissions_revoked(tmp_path):
    api = TestClient(create_app(Settings(database_url=f"sqlite:///{tmp_path/'permissions.db'}", papers_dir=tmp_path)))
    profile = api.post("/agents", json={"name": "Graph", "enabled_tools": ["local_search", "web_search"]}).json()
    definition = starter_workflow()
    definition["nodes"].append({"id": "web_search", "type": "web_search"})
    definition["edges"].extend([{"source": "question", "target": "web_search"}, {"source": "web_search", "target": "synthesize"}])
    assert api.put(f"/agents/{profile['id']}/workflow", json=definition).status_code == 200
    denied = api.patch(f"/agents/{profile['id']}", json={"enabled_tools": ["local_search"]})
    assert denied.status_code == 422 and "workflow" in denied.text
    definition["enabled"] = False
    assert api.put(f"/agents/{profile['id']}/workflow", json=definition).status_code == 200
    assert api.patch(f"/agents/{profile['id']}", json={"enabled_tools": ["local_search"]}).status_code == 200


def test_chat_api_runs_real_executor_with_stubbed_local_model(tmp_path, monkeypatch):
    """End-to-end API path without requiring an Ollama server or embedding model."""
    import aira.agent as model_adapter
    from aira.retrieval import LocalRetriever

    client = TestClient(create_app(Settings(database_url=f"sqlite:///{tmp_path/'e2e.db'}", papers_dir=tmp_path)))
    profile = client.post("/agents", json={"name": "Graph researcher", "enabled_tools": ["local_search"]}).json()
    assert client.put(f"/agents/{profile['id']}/workflow", json=starter_workflow()).status_code == 200
    conversation = client.post("/conversations", json={"title": "Evidence"}).json()

    class FakeModel:
        model = "qwen-stub"
        def invoke(self, messages, **kwargs):
            assert "UNTRUSTED RETRIEVED EVIDENCE" in messages[-1][1]
            assert "[1]" in messages[-1][1]
            return SimpleNamespace(content="The source reports stronger retrieval [1].")
    monkeypatch.setattr(model_adapter, "make_model", lambda *a,**kw: FakeModel())
    monkeypatch.setattr(LocalRetriever, "search", lambda *a,**kw: [
        {"id":"local-1", "path":"local.pdf", "version":1, "page":5, "score":0.8,
         "text":"Comparison on labeled queries reports stronger retrieval."}])
    response = client.post("/chat",json={"question":"How does performance change?",
        "session_id":conversation["id"], "agent_profile_id":profile["id"]})
    assert response.status_code == 200, response.text
    assert response.json()["mode"] == "visual_workflow"
    assert len(response.json()["workflow_steps"]) == 4
    saved = client.get(f"/conversations/{conversation['id']}/messages").json()["messages"][-1]
    assert saved["sources"][0]["id"] == "local-1"
    assert saved["metadata"]["workflow_steps"][1]["node"] == "local_search"
