"""Web search: providers, evidence capture, opt-in use in answers and broader coverage queries."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aira import agent, websearch
from aira.api import create_app
from aira.config import Settings
from aira.context import ContextManager
from aira.db import make_session_factory
from aira.sources import evidence_detail


@pytest.fixture
def session_factory(tmp_path):
    return make_session_factory(f"sqlite:///{tmp_path / 'web.db'}")


def fake_hits():
    return [
        {"title": "LoRA overview", "href": "https://example.org/lora", "body": "Adapters can reduce trainable parameters."},
        {"title": "duplicate", "href": "https://example.org/lora", "body": "Should be skipped"},
        {"title": "bad", "href": "javascript:alert(1)", "body": "Not a URL"},
        {"title": "no text", "href": "https://example.org/blank", "body": ""},
        {"title": "QLoRA", "href": "https://example.org/qlora", "body": "QLoRA uses four-bit quantization."},
    ]


def test_ddgs_evidence_is_persisted_and_deduplicated(session_factory, monkeypatch):
    monkeypatch.setattr(websearch, "_ddgs_results", lambda query, limit: fake_hits())
    results = websearch.search_web(session_factory, "LoRA QLoRA", max_results=4)
    assert len(results) == 2
    assert all(item["source_kind"] == "web_snippet_unverified" for item in results)
    assert all(item["retrieved_at"] for item in results)
    one = evidence_detail(session_factory, results[0]["evidence_id"])
    assert one["text"] == "Adapters can reduce trainable parameters."
    assert one["source_url"] == "https://example.org/lora"
    assert one["source_kind"] == "web_snippet"
    assert one["source_active"] is True
    # Stable source identity across calls.
    assert websearch.search_web(session_factory, "LoRA QLoRA")[0]["evidence_id"] == results[0]["evidence_id"]


def test_web_provider_rejects_bad_configuration(session_factory):
    with pytest.raises(websearch.WebSearchError, match="disabled"):
        websearch.search_web(session_factory, "quantization", provider="disabled")
    with pytest.raises(websearch.WebSearchError, match="TAVILY_API_KEY"):
        websearch.search_web(session_factory, "quantization", provider="tavily")
    with pytest.raises(ValueError):
        websearch.search_web(session_factory, "", provider="ddgs")
    with pytest.raises(ValueError):
        websearch.search_web(session_factory, "quantization", max_results=9)


def test_web_api_status_and_search_route(tmp_path, monkeypatch):
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'api.db'}", papers_dir=tmp_path,
                        web_search_provider="ddgs", faithfulness_mode="extractive")
    monkeypatch.setattr(websearch, "_ddgs_results", lambda query, limit: fake_hits())
    client = TestClient(create_app(settings))
    assert client.get("/health").json()["web_provider"] == "ddgs"
    assert client.get("/health").json()["web_enabled"] is True
    response = client.post("/web/search", json={"query": "QLoRA", "top_k": 4})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert client.get(f"/evidence/{results[0]['evidence_id']}").json()["text"]


def test_direct_rag_web_is_opt_in_and_passed_as_untrusted_data(session_factory, monkeypatch):
    context = ContextManager(session_factory)
    class Retriever:
        def search(self, question, **kwargs):
            return []
    class FakeModel:
        model = "fake-local"
        def invoke(self, messages):
            self.seen = messages
            return SimpleNamespace(content="Answered from evidence")
    fake_model = FakeModel()
    fake_web = [{"title": "Source", "url": "https://example.com/paper", "excerpt": "NF4 uses 4 bits.",
                 "evidence_id": "a" * 64, "retrieved_at": "2026-10-13T00:00:00Z"}]
    calls = []
    def fake_search(*args, **kwargs):
        calls.append(1)
        return fake_web
    monkeypatch.setattr(agent, "make_model", lambda *args: fake_model)
    monkeypatch.setattr(agent, "_configured_web_search", fake_search)
    settings = Settings(faithfulness_mode="extractive")
    plain = agent.answer_with_local_context("What is NF4?", "demo", settings, Retriever(), context)
    assert not calls
    assert plain["web_used"] is False
    enriched = agent.answer_with_local_context("What is NF4?", "demo", settings, Retriever(),
                                               context, use_web=True, session_factory=session_factory)
    assert len(calls) == 1
    assert enriched["sources"][0]["id"] == "a" * 64
    assert enriched["sources"][0]["source_kind"] == "web_snippet_unverified"
    assert "UNVERIFIED WEB SEARCH SNIPPET" in fake_model.seen[1][1]
    assert "https://example.com/paper" in fake_model.seen[1][1]


def test_web_context_retained_with_large_local_evidence():
    snapshot = {"notes": []}
    local = [{"id": "paper", "path": "paper.pdf", "version": 1, "page": 3,
              "text": "ABC " * 1000} for _ in range(10)]
    web = [{"title": "A", "url": "https://example.org", "excerpt": "web important",
            "evidence_id": "abc", "retrieved_at": "today"}]
    composed = agent.build_research_context(snapshot, local, web_results=web)
    assert "web important" in composed
    assert len(composed) <= 8000


def test_agent_enables_web_tool_only_when_requested(session_factory, monkeypatch):
    agents = pytest.importorskip("langchain.agents")
    context = ContextManager(session_factory)
    tools_seen = []
    class Retriever:
        def search(self, question, **kwargs):
            return []
    class FakeRunnable:
        def invoke(self, payload, config=None):
            return {"messages": [SimpleNamespace(content="Found an answer") ]}
    def fake_create_agent(model, tools, system_prompt):
        tools_seen.append({tool.name for tool in tools})
        return FakeRunnable()
    monkeypatch.setattr(agents, "create_agent", fake_create_agent)
    monkeypatch.setattr(agent, "make_model", lambda *args: object())
    monkeypatch.setattr(agent, "_configured_web_search", lambda *args: [])
    settings = Settings(faithfulness_mode="extractive")
    agent.run_agent("compare methods", "demo", settings, session_factory, Retriever(), context)
    agent.run_agent("compare methods", "demo", settings, session_factory, Retriever(), context, use_web=True)
    assert "search_the_web" not in tools_seen[0]
    assert "search_the_web" in tools_seen[1]


def test_coverage_query_includes_linear_and_nonlinear_methods():
    query = agent._coverage_search_query("Which dimensionality reduction methods are most common?")
    assert "PCA" in query and "t-SNE" in query and "UMAP" in query
    assert agent._coverage_search_query("What is PCA?") is None


def test_coverage_search_uses_two_bounded_queries_and_preserves_provenance(monkeypatch):
    calls = []
    def fake_search(settings, db, query):
        calls.append(query)
        suffix = "base" if len(calls) == 1 else "expanded"
        return [{"url":f"https://{suffix}.example.org/pca", "title":suffix, "evidence_id":suffix}]
    monkeypatch.setattr(agent,"_configured_web_search",fake_search)
    hits = agent._diversified_web_search(SimpleNamespace(), None,
                                        "Which dimensionality reduction methods are most common?")
    assert len(calls) == 2
    assert [hit['evidence_id'] for hit in hits] == ["base", "expanded"]
    assert all("url" in hit for hit in hits)


def test_coverage_failure_preserves_first_successful_search(monkeypatch):
    calls = 0
    def fake_search(settings, db, query):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise agent.WebSearchError("Search temporarily unavailable")
        return [{"url":"https://example.org/one", "evidence_id":"one"}]
    monkeypatch.setattr(agent,"_configured_web_search",fake_search)
    assert len(agent._diversified_web_search(None,None,"List the most common retrieval methods")) == 1
    assert calls == 2


def test_result_merger_keeps_unique_urls():
    common={"url":"https://example.org/pca", "evidence_id":"one"}
    more={"url":"https://example.org/umap", "evidence_id":"two"}
    results=agent._interleave_web_sources([common],[common,more])
    assert [r['evidence_id'] for r in results] == ['one','two']


def test_domain_diversity_precedes_repeated_domain():
    primary=[{"url":"https://wikipedia.org/a","evidence_id":"a"},
             {"url":"https://wikipedia.org/b","evidence_id":"b"}]
    secondary=[{"url":"https://example.edu/overview","evidence_id":"c"}]
    merged=agent._interleave_web_sources(primary,secondary)
    assert [r["evidence_id"] for r in merged] == ["a","c","b"]
