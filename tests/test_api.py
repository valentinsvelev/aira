"""HTTP API basics: health, ingestion and document upload."""
from fastapi.testclient import TestClient

from aira.api import create_app
from aira.config import Settings


def test_api_health_ingest_and_missing_openai_key(workspace, tmp_path):
    app = create_app(Settings(database_url=f"sqlite:///{tmp_path / 'api.db'}",
                              papers_dir=workspace["papers"], llm_provider="openai"))
    client = TestClient(app)
    assert client.get("/health").json()["llm_enabled"] is False
    assert client.post("/chat", json={"question": "Hello"}).status_code == 503
    (workspace["papers"] / "brief.txt").write_text(
        "Evidence systems should preserve document provenance across revisions. " * 10)
    result = client.post("/ingest/sync")
    assert result.status_code == 200
    assert result.json()["summary"]["ingested"] == 1
    assert client.get("/context/test-sesh").json() == {
        "session_id": "test-sesh", "revision": 0, "notes": []}


def test_document_list_and_upload_keep_ingestion_idempotent(tmp_path):
    papers = tmp_path / "papers"
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'frontend.db'}", papers_dir=papers,
                        llm_provider="openai", faithfulness_mode="extractive")
    client = TestClient(create_app(settings))
    assert client.get("/documents").json() == {"documents": []}
    example = "Documents must retain immutable source evidence across revisions. " * 10

    response = client.post("/documents/upload", files={"file": ("test.txt", example, "text/plain")})
    assert response.status_code == 200
    assert response.json()["status"] == "ingested"
    inventory = client.get("/documents").json()["documents"]
    assert len(inventory) == 1
    assert inventory[0]["path"] == "test.txt"
    assert inventory[0]["active"] is True
    assert inventory[0]["version"] == 1
    assert inventory[0]["chunks"] > 0

    again = client.post("/documents/upload", files={"file": ("test.txt", example, "text/plain")})
    assert again.status_code == 200
    assert again.json()["status"] == "skipped"
    assert client.get("/documents").json()["documents"][0]["version"] == 1

    modified = client.post("/documents/upload", files={"file": (
        "test.txt", example + " Extra facts for version two.", "text/plain")})
    assert modified.status_code == 200
    assert modified.json()["status"] == "ingested"
    assert client.get("/documents").json()["documents"][0]["version"] == 2


def test_invalid_upload_is_rejected_without_creating_files(tmp_path):
    papers = tmp_path / "papers"
    client = TestClient(create_app(Settings(database_url=f"sqlite:///{tmp_path / 'frontend.db'}",
                                            papers_dir=papers)))
    bad = client.post("/documents/upload", files={"file": ("virus.exe", "bad", "text/plain")})
    assert bad.status_code == 400
    assert client.get("/documents").json() == {"documents": []}
    assert not list(papers.glob("*")) if papers.exists() else True


def test_upload_size_limit(tmp_path):
    client = TestClient(create_app(Settings(database_url=f"sqlite:///{tmp_path / 'frontend.db'}",
                                            papers_dir=tmp_path / "papers")))
    result = client.post("/documents/upload", files={"file": (
        "huge.txt", b"a" * (20 * 1024 * 1024 + 1), "text/plain")})
    assert result.status_code == 413
    assert not (tmp_path / "papers" / "huge.txt").exists()
