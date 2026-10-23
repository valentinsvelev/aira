"""Document ingestion: idempotency, versioning and path safety."""
from pathlib import Path

from sqlalchemy import func, select

from aira.db import Document, DocumentVersion, EvidenceChunk, IngestionRun

TEXT_V1 = ("Research pipelines must retain immutable evidence for later verification. "
           "All context notes should point back to stable source identifiers. ") * 10


TEXT_V2 = ("Updated research workflows require source version tracking and controlled reindexing. "
           "Evidence identifiers must retain original source provenance. ") * 10


def test_idempotent_then_versioned_then_deactivated(workspace):
    path = workspace["papers"] / "study.txt"
    path.write_text(TEXT_V1)
    ingestor = workspace["ingestor"]
    assert ingestor.sync()["summary"] == {"ingested": 1}
    assert ingestor.sync()["summary"] == {"skipped": 1}
    with workspace["factory"]() as db:
        assert db.scalar(select(func.count(DocumentVersion.id))) == 1
        initial_chunks = db.scalars(select(EvidenceChunk.id)).all()
        assert initial_chunks
    path.write_text(TEXT_V2)
    assert ingestor.sync()["summary"] == {"ingested": 1}
    with workspace["factory"]() as db:
        assert db.scalar(select(func.count(DocumentVersion.id))) == 2
        assert db.scalar(select(func.count(DocumentVersion.id)).where(DocumentVersion.active.is_(True))) == 1
        assert set(initial_chunks).issubset(set(db.scalars(select(EvidenceChunk.id)).all()))
    path.unlink()
    assert ingestor.sync()["summary"] == {"deactivated": 1}
    with workspace["factory"]() as db:
        assert db.scalar(select(func.count(Document.id)).where(Document.active.is_(True))) == 0
        assert db.scalar(select(func.count(EvidenceChunk.id))) > 0


def test_failed_document_does_not_poison_good_documents(workspace):
    (workspace["papers"] / "bad.pdf").write_bytes(b"not a real pdf")
    (workspace["papers"] / "good.txt").write_text(TEXT_V1)
    report = workspace["ingestor"].sync()
    assert report["summary"] == {"failed": 1, "ingested": 1}
    with workspace["factory"]() as db:
        assert db.scalar(select(func.count(IngestionRun.id)).where(IngestionRun.status == "failed")) == 1
        assert db.scalar(select(func.count(Document.id))) == 1


def test_ingest_outside_directory_rejected(workspace, tmp_path: Path):
    outside = tmp_path / "external.txt"
    outside.write_text(TEXT_V1)
    try:
        workspace["ingestor"].ingest_file(outside)
    except ValueError as exc:
        assert "inside PAPERS_DIR" in str(exc)
    else:
        raise AssertionError("Untrusted ingest path must be rejected")
