"""Evidence capture and provenance inspection."""
from urllib.parse import urlparse

from aira.db import Document, DocumentVersion, EvidenceChunk
from aira.ingestion import digest


def capture_web_snippet(session_factory, url: str, snippet: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Invalid search-result URL")
    if not snippet.strip():
        raise ValueError("Empty search snippet")
    # Evidence represents the returned *snippet*, not authenticated page content.
    evidence_id = digest(f"web-snippet\n{url}\n{snippet}")
    with session_factory.begin() as db:
        if db.get(EvidenceChunk, evidence_id) is None:
            db.add(EvidenceChunk(id=evidence_id, source_kind="web_snippet", source_url=url,
                                 text=snippet, text_sha256=digest(snippet), ordinal=0))
    return evidence_id


def capture_web_page(session_factory, url: str, page_text: str) -> str:
    """Keep a durable, immutable snapshot of actually downloaded webpage text."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or not page_text.strip():
        raise ValueError("Invalid webpage capture")
    evidence_id = digest(f"web-page\n{url}\n{page_text}")
    with session_factory.begin() as db:
        if db.get(EvidenceChunk, evidence_id) is None:
            db.add(EvidenceChunk(id=evidence_id, source_kind="web_page", source_url=url,
                                 text=page_text, text_sha256=digest(page_text), ordinal=0))
    return evidence_id


def evidence_detail(session_factory, evidence_id: str) -> dict | None:
    with session_factory() as db:
        chunk = db.get(EvidenceChunk, evidence_id)
        if chunk is None:
            return None
        result = {"id": chunk.id, "source_kind": chunk.source_kind, "source_url": chunk.source_url,
                  "page": chunk.page, "text": chunk.text, "text_sha256": chunk.text_sha256,
                  "captured_at": chunk.created_at.isoformat() if chunk.created_at else None}
        if chunk.document_version_id:
            version = db.get(DocumentVersion, chunk.document_version_id)
            document = db.get(Document, version.document_id)
            result.update(document_path=document.path, document_version=version.version,
                          document_sha256=version.sha256,
                          source_active=bool(document.active and version.active))
        else:
            result.update(source_active=True, note=("Academic API abstract only; the full paper was not fetched or checked" if chunk.source_kind == "academic_abstract" else "Captured webpage text; claims and authenticity have not been independently verified" if chunk.source_kind == "web_page" else "Web search snippet only; the page itself was not fetched or authenticated"))
        return result
