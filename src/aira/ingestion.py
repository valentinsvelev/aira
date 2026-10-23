"""Idempotent, versioned ingestion. DB commit makes each document atomically visible."""
import hashlib
from pathlib import Path

from pypdf import PdfReader
from sqlalchemy import select

from aira.db import Document, DocumentVersion, EvidenceChunk, IngestionRun


def digest(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def extract_pages(path: Path) -> list[str]:
    if path.suffix.lower() == ".pdf":
        # Only PDFs from the configured local folder are processed; no arbitrary URLs.
        reader = PdfReader(str(path))
        return [page.extract_text() or "" for page in reader.pages]
    if path.suffix.lower() in {".txt", ".md"}:
        return [path.read_text(encoding="utf-8")]
    raise ValueError(f"Unsupported extension: {path.suffix}")


def split_page(text: str) -> list[str]:
    """LlamaIndex sentence-aware splitting; deterministic fallback for minimal test envs."""
    if not text.strip():
        return []
    try:
        from llama_index.core.node_parser import SentenceSplitter
    except ImportError:
        # Lightweight fallback permits local unit tests without full optional AI stack.
        # Installed distributions use SentenceSplitter above.
        words = text.split()
        step = 420
        return [" ".join(words[i : i + 480]) for i in range(0, len(words), step)]
    return SentenceSplitter(chunk_size=512, chunk_overlap=64).split_text(text)


class Ingestor:
    def __init__(self, session_factory, papers_dir: Path):
        self.session_factory = session_factory
        self.root = papers_dir.resolve()

    def _log(self, path: str, status: str, detail: str = "", chunks: int = 0) -> None:
        with self.session_factory.begin() as db:
            db.add(IngestionRun(document_path=path, status=status, detail=detail[:1500], chunk_count=chunks))

    def ingest_file(self, path: Path) -> dict:
        path = path.resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Ingestion path must be inside PAPERS_DIR")
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            raise ValueError("Only PDF, TXT, and MD are supported")
        relative = str(path.relative_to(self.root))
        try:
            raw = path.read_bytes()
            checksum = digest(raw)
            with self.session_factory() as db:
                doc = db.scalar(select(Document).where(Document.path == relative))
                active = None if not doc else db.scalar(
                    select(DocumentVersion).where(
                        DocumentVersion.document_id == doc.id,
                        DocumentVersion.active.is_(True),
                    )
                )
                if doc and doc.active and active and active.sha256 == checksum:
                    self._log(relative, "skipped", "Unchanged checksum")
                    return {"path": relative, "status": "skipped", "chunks": 0}

            pages = extract_pages(path)
            if not pages or not any(p.strip() for p in pages):
                raise ValueError("No extractable text; scanned PDFs need OCR (not included)")
            page_chunks = [(page_no, chunk) for page_no, text in enumerate(pages, 1)
                           for chunk in split_page(text)]
            if not page_chunks:
                raise ValueError("No nonempty chunks")

            with self.session_factory.begin() as db:
                doc = db.scalar(select(Document).where(Document.path == relative))
                if doc is None:
                    doc = Document(path=relative, active=True)
                    db.add(doc)
                    db.flush()
                else:
                    doc.active = True
                existing = db.scalars(
                    select(DocumentVersion).where(DocumentVersion.document_id == doc.id)
                    .order_by(DocumentVersion.version.desc())
                ).all()
                current = next((v for v in existing if v.active), None)
                # Reuse a retained version when a deleted PDF returns unchanged.
                matching = next((v for v in existing if v.sha256 == checksum), None)
                if current:
                    current.active = False
                if matching:
                    matching.active = True
                    count = len(db.scalars(select(EvidenceChunk).where(
                        EvidenceChunk.document_version_id == matching.id)).all())
                    status = "restored"
                else:
                    version = DocumentVersion(
                        document_id=doc.id, version=(existing[0].version + 1 if existing else 1),
                        sha256=checksum, active=True,
                    )
                    db.add(version)
                    db.flush()
                    count = len(page_chunks)
                    for ordinal, (page, text) in enumerate(page_chunks):
                        chunk_id = digest(f"{version.id}:{page}:{ordinal}")
                        db.add(EvidenceChunk(
                            id=chunk_id, source_kind="pdf" if path.suffix.lower() == ".pdf" else "text",
                            document_version_id=version.id, page=page, ordinal=ordinal,
                            text=text, text_sha256=digest(text),
                        ))
                    status = "ingested"
                db.add(IngestionRun(document_path=relative, status=status, chunk_count=count))
            return {"path": relative, "status": status, "chunks": count}
        except Exception as exc:
            self._log(relative, "failed", str(exc))
            return {"path": relative, "status": "failed", "error": str(exc), "chunks": 0}

    def sync(self) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        paths = sorted(p for p in self.root.rglob("*") if p.is_file()
                       and p.suffix.lower() in {".pdf", ".txt", ".md"} and not p.is_symlink())
        results = [self.ingest_file(p) for p in paths]
        visible_paths = {str(p.relative_to(self.root)) for p in paths}
        # Delete/deactivate only after successfully scanning the directory.
        with self.session_factory.begin() as db:
            for doc in db.scalars(select(Document).where(Document.active.is_(True))):
                if doc.path not in visible_paths:
                    doc.active = False
                    for version in db.scalars(select(DocumentVersion).where(
                        DocumentVersion.document_id == doc.id,
                        DocumentVersion.active.is_(True),
                    )):
                        version.active = False
                    db.add(IngestionRun(document_path=doc.path, status="deactivated",
                                        detail="Source removed from local folder"))
                    results.append({"path": doc.path, "status": "deactivated", "chunks": 0})
        return {"results": results, "summary": {status: sum(r["status"] == status for r in results)
                for status in {r["status"] for r in results}}}
