"""Literature reviews: creation, selection, de-duplication, persistence and API validation."""
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from aira import literature
from aira.api import create_app
from aira.config import Settings
from aira.db import EvidenceChunk, make_session_factory


def paper(title, arxiv_id, year=2025):
    return {"provider":"arxiv", "provider_id":arxiv_id, "title":title,
        "authors":["A. Researcher", "B. Scientist"],"year":year,
        "abstract":f"The study {title} evaluates practical research techniques.",
        "url":f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url":f"https://arxiv.org/pdf/{arxiv_id}",
        "doi":None,"arxiv_id":arxiv_id,"found_in":["arxiv"],
        "source_kind":"academic_metadata_unverified", "retrieved_at":"2026-10-14T00:00:00Z",
        "evidence_id":"a"*64 if arxiv_id == "2501.12345" else "b"*64}


def test_literature_creation_selection_dedup_and_persistence(tmp_path, monkeypatch):
    db_url=f"sqlite:///{tmp_path/'lit.db'}"
    factory=make_session_factory(db_url)
    mock_papers=[paper("Context pruning", "2501.12345"), paper("Memory editing", "2501.45678")]
    def fake_search(_factory, query, **kwargs):
        from aira.ingestion import digest
        with _factory.begin() as session:
            for p in mock_papers:
                if session.get(EvidenceChunk, p["evidence_id"]) is None:
                    session.add(EvidenceChunk(id=p["evidence_id"],text=p["abstract"],text_sha256=digest(p["abstract"]), source_kind="academic_abstract",source_url=p["url"]))
        return {"results":mock_papers,"errors":{}}
    monkeypatch.setattr(literature,"search_academic",fake_search)
    created=literature.new_review(factory,"Long-horizon context compression",2024)
    first=literature.discover(factory,created["id"],None,Settings())
    assert first["added"]==2
    assert literature.discover(factory,created["id"],None,Settings())["added"]==0
    p=first["review"]["papers"][0]
    stored=literature.set_paper_status(factory,created["id"],p["id"],"included")
    assert stored["papers"][0]["status"]=="included"
    assert len(literature.review_list(make_session_factory(db_url)))==1
    assert literature.get_review(factory,created["id"])["papers"][0]["status"]=="included"
    class Model:
        def invoke(self, prompt):
            assert "ABSTRACT ONLY" in prompt[1][1]
            assert "[1]" in prompt[1][1]
            return SimpleNamespace(content="# Overview\nPaper [1] surveys context pruning.\n\n## Scope\nAbstract only.")
    monkeypatch.setattr("aira.agent.make_model",lambda *_: Model())
    survey=literature.make_survey(factory, created["id"],Settings())
    assert len(survey["sources"])==1
    assert "[1]" in survey["summary"]
    assert literature.get_review(factory,created["id"])["summary"]==survey["summary"]
    assert literature.delete_review(factory,created["id"])
    with factory() as session:
        assert session.scalar(select(func.count(EvidenceChunk.id)))==2 # evidence remains immutable


def test_literature_api_and_input_validation(tmp_path,monkeypatch):
    settings=Settings(database_url=f"sqlite:///{tmp_path/'api.db'}",papers_dir=tmp_path)
    client=TestClient(create_app(settings))
    assert client.post("/literature/reviews",json={"topic":"a"}).status_code==422
    created=client.post("/literature/reviews",json={"topic":"Context edits"}).json()
    assert created["topic"]=="Context edits"
    assert len(client.get("/literature/reviews").json()["reviews"])==1
    assert client.get("/literature/reviews/invalid").status_code==404
    assert client.patch(f"/literature/reviews/{created['id']}/papers/bogus",json={"status":"included"}).status_code==404
    assert client.post(f"/literature/reviews/{created['id']}/survey",json={}).status_code==422
    assert client.delete(f"/literature/reviews/{created['id']}").status_code==204
    assert client.get(f"/literature/reviews/{created['id']}").status_code==404
