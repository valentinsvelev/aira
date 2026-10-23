"""Research-agent profiles: CRUD, tool permissions, instructions and saved generation settings."""
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from aira.api import create_app
from aira.config import Settings


def make_client(tmp_path: Path):
    return TestClient(create_app(Settings(database_url=f"sqlite:///{tmp_path / 'agents.db'}",
                                          papers_dir=tmp_path, llm_provider="ollama")))


def client(tmp_path: Path):
    return TestClient(create_app(Settings(database_url=f"sqlite:///{tmp_path / 'test.db'}",
                                         papers_dir=tmp_path, llm_provider="ollama")))


def test_profiles_crud_and_validation(tmp_path):
    client = make_client(tmp_path)
    initial = client.get("/agents").json()["agents"]
    assert len(initial) == 1 and initial[0]["id"] == "default"
    body = {"name": "Literature reviewer", "description": "Critically review methods",
            "system_instructions": "List methodological limitations and uncertainties.",
            "enabled_tools": ["local_search", "academic_search"]}
    resp = client.post("/agents", json=body)
    assert resp.status_code == 201, resp.text
    identifier = resp.json()["id"]
    assert client.get("/agents").json()["agents"][1]["id"] == identifier
    updated = client.patch(f"/agents/{identifier}", json={"name": "Survey writer"})
    assert updated.status_code == 200 and updated.json()["name"] == "Survey writer"
    assert updated.json()["enabled_tools"] == ["local_search", "academic_search"]
    edited_default = client.patch("/agents/default", json={"name": "Custom assistant", "system_instructions": "Prefer concise answers."})
    assert edited_default.status_code == 200, edited_default.text
    assert edited_default.json()["name"] == "Custom assistant"
    assert client.get("/agents").json()["agents"][0]["id"] == "default"
    assert client.get("/agents").json()["agents"][0]["name"] == "Custom assistant"
    assert client.delete("/agents/default").status_code == 403
    assert client.post("/agents", json={**body, "enabled_tools": ["web_search"]}).status_code == 422
    assert client.post("/agents", json={**body, "enabled_tools": ["local_search", "shell_exec"]}).status_code == 422
    assert client.delete(f"/agents/{identifier}").status_code == 204
    assert client.get(f"/agents/{identifier}").status_code == 404


def test_profiles_reject_disabled_web_and_unknown_profile(tmp_path):
    client = make_client(tmp_path)
    profile = client.post("/agents", json={"name": "Offline", "enabled_tools": ["local_search"]}).json()
    assert client.post("/chat", json={"question": "Hi", "agent_profile_id": "missing"}).status_code == 404
    denied = client.post("/chat", json={"question": "Find info", "agent_profile_id": profile["id"], "use_web": True})
    assert denied.status_code == 422 and "web search" in denied.json()["detail"]
    denied = client.post("/answer", json={"question": "Find info", "agent_profile_id": profile["id"], "use_academic": True})
    assert denied.status_code == 422


def test_profile_instructions_and_selected_id_are_persisted(tmp_path, monkeypatch):
    import aira.api as api_module
    client = make_client(tmp_path)
    profile = client.post("/agents", json={"name": "Skeptic", "system_instructions": "Always state limitations.",
        "enabled_tools": ["local_search", "context_read"]}).json()
    chat = client.post("/conversations", json={"title": "Profile test"}).json()
    received = []
    def stub_answer(question, session_id, settings, retriever, context, options, generation,
                    *, use_web, session_factory, use_academic, web_fetch_pages, history, profile):
        received.append(profile)
        return {"answer": "Evidence is inconclusive.", "model": "dummy", "sources": []}
    monkeypatch.setattr(api_module, "answer_with_local_context", stub_answer)
    response = client.post("/answer", json={"question": "How reliable is this?", "session_id": chat["id"],
                                               "agent_profile_id": profile["id"]})
    assert response.status_code == 200, response.text
    assert received[0]["system_instructions"] == "Always state limitations."
    history = client.get(f"/conversations/{chat['id']}/messages").json()["messages"]
    assert history[-1]["metadata"]["agent_profile_id"] == profile["id"]
    # A second instance against the same database sees the profile.
    another_client = make_client(tmp_path)
    assert another_client.get(f"/agents/{profile['id']}").json()["name"] == "Skeptic"


def test_profile_specializes_model_prompt_without_exposing_disabled_notes(monkeypatch):
    from aira import agent
    from aira.config import Settings
    from aira.options import RetrievalOptions
    calls = []
    class FakeContext:
        def notes(self, session_id):
            return {"revision": 2, "notes": [{"claim": "SECRET_WORKING_NOTE",
                                                "evidence_id": "test1", "source_active": True}]}
    class FakeRetriever:
        def search(self, query, top_k=5, options=None):
            return [{"id": "test2", "text": "Paper evidence about quantization.",
                     "path": "paper.pdf", "page": 4, "version": 1, "score": 1.0}]
    class FakeModel:
        model = "test"
        def invoke(self, messages):
            calls.append(messages)
            return SimpleNamespace(content="Reported from the paper [1].")
    monkeypatch.setattr(agent, "make_model", lambda *args, **kwargs: FakeModel())
    result = agent.answer_with_local_context(
        "What is quantization?", "demo", Settings(), FakeRetriever(), FakeContext(),
        retrieval=RetrievalOptions(),
        profile={"name": "Skeptic", "system_instructions": "Explain experimental limitations.",
                 "enabled_tools": ["local_search"]},
    )
    assert result["answer"].startswith("Reported")
    assert "Explain experimental limitations" in calls[0][0][1]
    assert "Paper evidence about quantization" in calls[0][1][1]
    assert "SECRET_WORKING_NOTE" not in calls[0][1][1]


def test_builtin_agent_can_be_edited_and_controls_tools_across_restart(tmp_path):
    client = make_client(tmp_path)
    updated = client.patch('/agents/default', json={
        'name': 'My Researcher',
        'system_instructions': 'Separate facts from hypotheses.',
        'enabled_tools': ['local_search', 'context_read'],
    })
    assert updated.status_code == 200
    assert updated.json()['builtin'] is True
    assert updated.json()['id'] == 'default'
    fresh_client = make_client(tmp_path)
    assert fresh_client.get('/agents/default').json()['system_instructions'] == 'Separate facts from hypotheses.'
    agents = fresh_client.get('/agents').json()['agents']
    assert len([p for p in agents if p['id'] == 'default']) == 1
    assert agents[0]['name'] == 'My Researcher'
    denied = fresh_client.post('/chat', json={'question': 'Search for new work',
                                            'agent_profile_id': 'default', 'use_web': True})
    assert denied.status_code == 422
    assert fresh_client.delete('/agents/default').status_code == 403


def test_default_agent_generation_persists_and_can_be_overridden(tmp_path, monkeypatch):
    import aira.api as module
    client = TestClient(create_app(Settings(database_url=f"sqlite:///{tmp_path/'default.db'}",
                                           papers_dir=tmp_path, llm_provider="ollama")))
    configuration={"model":"qwen3.5:4b","temperature":0.45,"top_p":0.85,
                   "num_predict":512,"num_ctx":8192,"repeat_penalty":1.15,"seed":3}
    changed=client.patch('/agents/default', json={'generation':configuration})
    assert changed.status_code == 200, changed.text
    assert changed.json()['generation']==configuration
    assert client.get('/agents').json()['agents'][0]['generation']==configuration

    seen=[]
    def fake_answer(question, session_id, settings, retriever, context, options, generation,
                    *, use_web, session_factory, use_academic, web_fetch_pages, history, profile, **kwargs):
        seen.append(generation)
        return {'answer':'Example answer.', 'model':'dummy','sources':[]}
    monkeypatch.setattr(module, 'answer_with_local_context', fake_answer)
    a=client.post('/answer',json={'question':'Hi'})
    assert a.status_code==200, a.text
    assert seen[-1].num_ctx==8192 and seen[-1].temperature==0.45
    b=client.post('/answer',json={'question':'Hi again','generation':{'num_ctx':4096,'temperature':0.1}})
    assert b.status_code==200,b.text
    assert seen[-1].num_ctx==4096 and seen[-1].temperature==0.1


def test_custom_agent_generation_is_enforced_even_if_request_overrides_it(tmp_path, monkeypatch):
    import aira.api as module
    c = client(tmp_path)
    profile = c.post("/agents", json={"name": "Method reviewer", "enabled_tools": ["local_search"],
        "generation": {"model": "qwen3.5:4b", "temperature": .44, "num_predict": -1}}).json()
    assert profile["generation"]["temperature"] == .44
    received = []

    def fake_answer(question, session_id, settings, retriever, context, options, generation, **kwargs):
        received.append(generation)
        return {"answer": "Test response", "model": "fake", "sources": []}

    monkeypatch.setattr(module, "answer_with_local_context", fake_answer)
    response = c.post("/answer", json={"question": "Explain RAG", "agent_profile_id": profile["id"],
        "generation": {"model": "llama3.2:3b", "temperature": 0.99, "num_predict": 40}})
    assert response.status_code == 200, response.text
    assert received[-1].model == "qwen3.5:4b"
    assert received[-1].temperature == .44
    assert received[-1].num_predict == -1

    default = c.post("/answer", json={"question": "Explain RAG", "generation": {"temperature": .3}})
    assert default.status_code == 200
    assert received[-1].temperature == .3
    assert c.patch(f"/agents/{profile['id']}", json={"generation": {"temperature": .12}}).status_code == 200
    assert c.get(f"/agents/{profile['id']}").json()["generation"]["temperature"] == .12
    assert c.patch(f"/agents/{profile['id']}", json={"generation": {"temperature": 4}}).status_code == 422
