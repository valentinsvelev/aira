"""Reading web pages safely: SSRF guards, size limits and evidence capture (network mocked)."""
import socket
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from aira import agent, webbrowse, websearch
from aira.context import ContextManager
from aira.db import make_session_factory
from aira.sources import evidence_detail


def resolve_public(host, port, type):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", port))]


def test_private_url_and_metadata_ip_denied():
    bad_urls = ["http://127.0.0.1/secrets", "http://169.254.169.254/latest/meta-data/",
                "http://10.0.1.5", "http://localhost:8000/docs", "file:///etc/passwd",
                "http://example.com:8080/internal", "https://user:pass@example.org/a"]
    for url in bad_urls:
        with pytest.raises(webbrowse.WebPageError):
            webbrowse.validate_public_web_url(url, resolver=resolve_public)
    webbrowse.validate_public_web_url("https://example.org/article", resolver=resolve_public)


def test_read_html_page_and_capture_evidence(tmp_path, monkeypatch):
    module = ModuleType("trafilatura")
    module.extract = lambda html, **kwargs: "Article text: " + "evidence " * 32
    monkeypatch.setitem(sys.modules, "trafilatura", module)

    def router(request):
        if request.url.path == "/redirect":
            return httpx.Response(302, headers={"location": "https://example.org/paper"})
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"},
                              text="<html><body>Some publication text</body></html>")

    client = httpx.Client(transport=httpx.MockTransport(router))
    factory = make_session_factory(f"sqlite:///{tmp_path / 'browse.db'}")
    page = webbrowse.read_search_result(
        factory, {"title": "Paper", "url": "https://example.org/redirect"},
        client=client, resolver=resolve_public)
    assert page["url"] == "https://example.org/paper"
    assert page["source_kind"] == "web_page_unverified"
    stored = evidence_detail(factory, page["evidence_id"])
    assert stored["source_kind"] == "web_page"
    assert stored["source_url"] == "https://example.org/paper"
    assert stored["text"] == page["excerpt"]
    assert "Captured webpage" in stored["note"]
    assert ContextManager(factory)._evidence_active is not None


def test_redirect_to_internal_ip_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "trafilatura", ModuleType("trafilatura"))
    http = httpx.Client(transport=httpx.MockTransport(lambda req:
        httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})))
    with pytest.raises(webbrowse.WebPageError, match="blocked"):
        webbrowse.fetch_page_text("https://example.org/a", client=http, resolver=resolve_public)


def test_oversized_body_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "trafilatura", ModuleType("trafilatura"))
    http = httpx.Client(transport=httpx.MockTransport(lambda req:
        httpx.Response(200, headers={"content-type": "text/html"},
                       content=b"x" * (webbrowse.MAX_BYTES + 1))))
    with pytest.raises(webbrowse.WebPageError, match="size"):
        webbrowse.fetch_page_text("https://example.org/oversize", client=http,
                                  resolver=resolve_public)


def test_failed_page_reads_reported_and_snippets_kept(tmp_path, monkeypatch):
    factory = make_session_factory(f"sqlite:///{tmp_path / 'search.db'}")
    monkeypatch.setattr(websearch, "_ddgs_results", lambda query, limit: [
        {"title": "A", "href": "https://example.org/a", "body": "Snippet exists"}])
    monkeypatch.setattr(webbrowse, "fetch_page_text", lambda *args, **kwargs:
                        (_ for _ in ()).throw(webbrowse.WebPageError("blocked by site")))
    snippets = websearch.search_web(factory, "test")
    pages, failures = webbrowse.read_top_results(factory, snippets)
    assert pages == []
    assert failures == [{"url": "https://example.org/a", "reason": "blocked by site"}]
    assert evidence_detail(factory, snippets[0]["evidence_id"])["source_kind"] == "web_snippet"


def test_answer_uses_fetched_page_text_not_only_snippet(tmp_path, monkeypatch):
    factory = make_session_factory(f"sqlite:///{tmp_path / 'answer.db'}")
    web_hit = {"title": "Site", "url": "https://example.org/page", "excerpt": "Search snippet",
               "evidence_id": "a"*64, "retrieved_at": "2026-10-13T00:00:00Z",
               "source_kind": "web_snippet_unverified"}
    page_hit = dict(web_hit, excerpt="FULL PAGE EVIDENCE: " + "long text " * 35,
                    evidence_id="b"*64, source_kind="web_page_unverified")
    monkeypatch.setattr(agent, "_configured_web_search", lambda *a: [web_hit])
    monkeypatch.setattr(agent, "read_top_results", lambda *a: ([page_hit], []))
    class FakeModel:
        model = "fake"
        def invoke(self, messages):
            assert "FULL PAGE EVIDENCE" in messages[1][1]
            assert "RETRIEVED WEB PAGE" in messages[1][1]
            return SimpleNamespace(content="answer [evidence:" + "b" * 64 + "]")
    monkeypatch.setattr(agent, "make_model", lambda *a: FakeModel())
    class Retriever:
        def search(self, *a, **kw):
            return []
    output = agent.answer_with_local_context("Research the subject", "demo", SimpleNamespace(
        llm_provider="ollama", ollama_model="test"), Retriever(), ContextManager(factory),
        use_web=True, session_factory=factory)
    assert output["web_pages_fetched"] == 1
    assert output["sources"][0]["source_kind"] == "web_page_unverified"
    assert len(output["sources"]) == 2


def test_agent_web_tools_opt_in_and_bounded(tmp_path, monkeypatch):
    agents = pytest.importorskip("langchain.agents")
    factory = make_session_factory(f"sqlite:///{tmp_path / 'agent.db'}")
    monkeypatch.setattr(agent, "_configured_web_search", lambda *a: [{
        "title": "Article", "url": "https://example.org/a", "excerpt": "Relevant snippet",
        "evidence_id": "a"*64, "retrieved_at": "today", "source_kind": "web_snippet_unverified"}])
    monkeypatch.setattr(agent, "read_top_results", lambda *a: ([], []))
    monkeypatch.setattr(agent, "read_search_result", lambda *a: {
        "title": "Article", "url": "https://example.org/a", "excerpt": "Actual page contents " * 20,
        "evidence_id": "b"*64, "retrieved_at": "today", "source_kind": "web_page_unverified"})
    recorded = {}
    class Runner:
        def invoke(self, payload, config=None):
            tools = recorded["tools"]
            forbidden = tools["read_web_page"].invoke({"url": "http://127.0.0.1/admin"})
            assert "URL must first appear" in forbidden
            allowed = tools["read_web_page"].invoke({"url": "https://example.org/a"})
            assert "Actual page contents" in allowed
            exhausted = tools["read_web_page"].invoke({"url": "https://example.org/a"})
            assert "budget exhausted" in exhausted
            return {"messages": [SimpleNamespace(content="Answer from page") ]}
    def fake_create_agent(model, tools, system_prompt):
        recorded["tools"] = {t.name: t for t in tools}
        return Runner()
    monkeypatch.setattr(agents, "create_agent", fake_create_agent)
    monkeypatch.setattr(agent, "make_model", lambda *a: object())
    class Retriever:
        def search(self, *a, **kw): return []
    output = agent.run_agent("What is new?", "demo", SimpleNamespace(
        llm_provider="ollama"), factory, Retriever(), ContextManager(factory),
        use_web=True, web_fetch_pages=1)
    assert output["web_pages_fetched"] == 1
    assert any(source["id"] == "b"*64 for source in output["sources"])
