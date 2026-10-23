"""Academic metadata search, Crossref enrichment and approved PDF import (all network mocked)."""
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from aira import academic, agent
from aira.academic import _crossref_results, _validate_query
from aira.api import create_app
from aira.config import Settings
from aira.context import ContextManager
from aira.db import make_session_factory
from aira.sources import evidence_detail


@pytest.fixture
def factory(tmp_path):
    return make_session_factory(f"sqlite:///{tmp_path / 'academic.db'}")


def make_paper(provider, *, title="LoRA: Low-Rank Adaptation", doi=None):
    return {
        "provider": provider, "provider_id": provider + "-1",
        "title": title, "authors": ["A. Scientist"], "year": 2021,
        "abstract": "Low-rank updates reduce trainable parameters.",
        "url": f"https://{provider}.example.org/paper1",
        "pdf_url": "https://arxiv.org/pdf/2106.09685" if provider == "arxiv" else None,
        "doi": doi, "arxiv_id": "2106.09685" if provider == "arxiv" else None,
    }


def test_scholar_link_human_only():
    assert academic.scholar_url("LoRA & QLoRA") == "https://scholar.google.com/scholar?q=LoRA+%26+QLoRA"
    with pytest.raises(ValueError):
        academic.scholar_url("  ")


def test_academic_metadata_is_normalized_deduplicated_and_captured(factory, monkeypatch):
    monkeypatch.setattr(academic, "_arxiv_results", lambda query, limit: [make_paper("arxiv")])
    monkeypatch.setattr(academic, "_openalex_results", lambda query, limit, year, key:
                        [make_paper("openalex", doi="10.1234/a")])
    first = academic.search_academic(factory, "low-rank adaptation", top_k=5)
    assert first["errors"] == {}
    assert len(first["results"]) == 1
    paper = first["results"][0]
    assert paper["found_in"] == ["arxiv", "openalex"]
    assert paper["doi"] == "10.1234/a"
    assert paper["pdf_url"] == "https://arxiv.org/pdf/2106.09685"
    evidence = evidence_detail(factory, paper["evidence_id"])
    assert evidence["source_kind"] == "academic_abstract"
    assert evidence["source_active"]
    assert "full paper was not fetched" in evidence["note"]
    second = academic.search_academic(factory, "low-rank adaptation")
    assert paper["evidence_id"] == second["results"][0]["evidence_id"]
    ctx = ContextManager(factory)
    accepted = ctx.propose("demo", "Low-rank updates reduce trainable parameters.", "Low-rank updates reduce trainable parameters.", paper["evidence_id"])
    assert accepted["accepted"]


def test_partial_provider_failure_is_reported(factory, monkeypatch):
    monkeypatch.setattr(academic, "_arxiv_results", lambda q, lim: [make_paper("arxiv")])
    def failure(*_):
        raise httpx.ConnectTimeout("search unavailable")
    monkeypatch.setattr(academic, "_openalex_results", failure)
    found = academic.search_academic(factory, "low rank")
    assert len(found["results"]) == 1
    assert "openalex" in found["errors"]


def test_arxiv_atom_parsing_and_openalex_inverted_abstract(monkeypatch):
    feed = '''<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"><entry>
    <id>http://arxiv.org/abs/2106.09685v2</id><title>  LoRA:  Low-Rank\nAdaptation </title>
    <summary>A paper about adapters.</summary><published>2021-06-17T00:00:00Z</published>
    <author><name>Edward Hu</name></author>
    </entry></feed>'''
    raw_client = httpx.Client
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=feed.encode()))
    monkeypatch.setattr(academic.httpx, "Client", lambda **kwargs: raw_client(transport=transport, **kwargs))
    found = academic._arxiv_results("LoRA", 2)
    assert len(found) == 1
    assert found[0]["arxiv_id"] == "2106.09685"
    assert found[0]["authors"] == ["Edward Hu"]
    assert found[0]["pdf_url"].endswith("/2106.09685")
    assert academic._openalex_abstract({"quantization": [1], "Double": [0]}) == "Double quantization"


@pytest.mark.parametrize("url", [
    "http://arxiv.org/pdf/2106.09685",
    "https://127.0.0.1/pdf/2106.09685",
    "https://arxiv.org.evil.com/pdf/2106.09685",
    "https://arxiv.org/pdf/2106.09685?download=1",
    "https://arxiv.org/pdf/../admin",
    "https://user@arxiv.org/pdf/2106.09685",
    "https://aclanthology.org/../../../etc/passwd.pdf",
])
def test_import_url_allowlist_rejects_untrusted_targets(url):
    with pytest.raises(ValueError):
        academic._approved_pdf_url(url)


def test_paper_landing_pages_convert_to_approved_pdf():
    assert academic._approved_pdf_url("https://arxiv.org/abs/2106.09685") == (
        "https://arxiv.org/pdf/2106.09685", "arxiv_2106.09685.pdf")
    assert academic._approved_pdf_url("https://aclanthology.org/2020.acl-main.1/") == (
        "https://aclanthology.org/2020.acl-main.1.pdf", "acl_2020.acl-main.1.pdf")


def test_download_import_with_pdf_signature_and_versioning(factory, tmp_path, monkeypatch):
    pdf_bytes = ( __import__('pathlib').Path(__file__).parent / "fixtures/synthetic_paper.pdf" ).read_bytes()
    raw_client = httpx.Client
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=pdf_bytes))
    monkeypatch.setattr(academic.httpx, "Client", lambda **kwargs: raw_client(transport=transport, **kwargs))
    result = academic.import_approved_pdf(factory, tmp_path / "papers", "https://arxiv.org/abs/2106.09685")
    assert result["status"] == "ingested"
    assert result["chunks"] > 0
    again = academic.import_approved_pdf(factory, tmp_path / "papers", "https://arxiv.org/pdf/2106.09685")
    assert again["status"] == "skipped"


def test_download_redirect_and_html_are_rejected(factory, tmp_path, monkeypatch):
    raw_client = httpx.Client
    for response in [httpx.Response(302, headers={"location": "http://127.0.0.1/"}),
                     httpx.Response(200, content=b"<html>Not a PDF</html>")]:
        transport = httpx.MockTransport(lambda req: response)  # noqa: B023
        monkeypatch.setattr(academic.httpx, "Client", lambda **kwargs: raw_client(transport=transport, **kwargs))  # noqa: B023
        with pytest.raises(ValueError):
            academic.import_approved_pdf(factory, tmp_path / "papers", "https://arxiv.org/pdf/2106.09685")


def test_academic_api_routes(factory, tmp_path, monkeypatch):
    import aira.api as api_module
    monkeypatch.setattr(api_module, "search_academic", lambda *args, **kwargs:
                        {"query": "lora", "results": [], "errors": {}, "sources": ["arxiv"]})
    app = create_app(Settings(database_url=f"sqlite:///{tmp_path / 'testapi.db'}",
                              papers_dir=tmp_path, faithfulness_mode="extractive"))
    client = TestClient(app)
    response = client.post("/academic/search", json={"query": "lora", "top_k": 3})
    assert response.status_code == 200
    link = client.post("/academic/scholar-url", json={"query": "low rank"}).json()
    assert link["automatic_scholar_scraping"] is False
    assert "scholar.google.com" in link["url"]
    refused = client.post("/academic/import", json={"url": "http://localhost/admin"})
    assert refused.status_code == 422


def test_rag_only_queries_academic_apis_when_enabled(factory, monkeypatch):
    class Retriever:
        def search(self, *args, **kwargs):
            return []
    class Model:
        model = "mock"
        def invoke(self, messages):
            self.seen = messages
            return SimpleNamespace(content="Mock response")
    model = Model()
    calls = []
    monkeypatch.setattr(agent, "make_model", lambda *_: model)
    def academic_query(*_):
        calls.append(True)
        return {"results": [{"title": "LoRA", "url": "https://arxiv.org/abs/2106.09685",
                             "abstract": "Low rank adapters.", "evidence_id": "1" * 64,
                             "retrieved_at": "2026-10-13T00:00:00Z"}], "errors": {}}
    monkeypatch.setattr(agent, "_configured_academic_search", academic_query)
    context = ContextManager(factory)
    settings = Settings(faithfulness_mode="extractive")
    plain = agent.answer_with_local_context("LoRA?", "demo", settings, Retriever(), context)
    assert not calls
    assert plain["academic_used"] is False
    result = agent.answer_with_local_context("LoRA?", "demo", settings, Retriever(), context,
                                             session_factory=factory, use_academic=True)
    assert calls
    assert result["academic_used"] is True
    assert result["sources"][0]["source_kind"] == "academic_metadata_unverified"
    assert "UNVERIFIED ACADEMIC ABSTRACT" in model.seen[1][1]


def test_crossref_acl_metadata_points_to_importable_anthology(monkeypatch):
    calls = []

    def fake_get(self, url, **kwargs):
        calls.append((url, kwargs['params']))
        request = httpx.Request("GET", url)
        return httpx.Response(200, request=request, json={"message": {"items": [
            {"DOI": "10.18653/v1/2026.eacl-long.13", "title": ["Out of Style"],
             "author": [{"given": "Ada", "family": "Researcher"}],
             "published": {"date-parts": [[2026, 5, 1]]}},
            {"DOI": "10.1145/other", "title": ["Wrong provider"]},
        ]}})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    rows = _crossref_results("style variation", 8, 2025, "acl")
    assert len(rows) == 1
    assert rows[0]['url'] == 'https://aclanthology.org/2026.eacl-long.13/'
    assert rows[0]['pdf_url'] == 'https://aclanthology.org/2026.eacl-long.13.pdf'
    assert calls[0][1]['filter'] == 'prefix:10.18653,from-pub-date:2025-01-01'
    assert calls[0][1]['rows'] == 8


def test_crossref_acm_never_invents_pdf(monkeypatch):
    def fake_get(self, url, **kwargs):
        return httpx.Response(200, request=httpx.Request("GET", url),
                              json={"message": {"items": [
                                  {"DOI": "10.1145/123456.123457", "title": ["A conference paper"]}
                              ]}})
    monkeypatch.setattr(httpx.Client, "get", fake_get)
    row = _crossref_results("conference paper", 1, None, "acm")[0]
    assert row['pdf_url'] is None
    assert row['url'].startswith('https://doi.org/10.1145/')
    _validate_query('test', 50, ['acl', 'acm'], None)
    with pytest.raises(ValueError):
        _validate_query('test', 51, ['acl'], None)
    with pytest.raises(ValueError):
        _validate_query('test', 4, ['scholar'], None)
