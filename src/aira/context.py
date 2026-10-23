"""Conservative, verifiable, append-only context edit audit trail.

A 'supported' decision means extractively supported by stored evidence; it does
NOT imply that the source is scientifically true or that summaries are entailed.
"""
import re
from dataclasses import dataclass

from sqlalchemy import select

from aira.db import (
    ContextEvent,
    ContextNote,
    Document,
    DocumentVersion,
    EvidenceChunk,
    ResearchSession,
    utcnow,
)


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


@dataclass
class Verdict:
    accepted: bool
    reason: str
    similarity: float | None = None
    method: str = "extractive"


def check_extractively_supported(claim: str, quote: str, evidence: str) -> Verdict:
    """Strict MVP: claim must be an exact textual subspan of the stored quote.

    This intentionally rejects paraphrases. A later semantic checker must be
    evaluated against a human-labeled benchmark before auto-accepting them.
    """
    c, q, e = normalize(claim), normalize(quote), normalize(evidence)
    if not c or not q or not e:
        return Verdict(False, "Claim, quote and evidence must all be nonempty")
    if len(c) < 20:
        return Verdict(False, "Claim must contain at least 20 normalized characters")
    if q not in e:
        return Verdict(False, "The supplied quote is not present in immutable source evidence")
    if c not in q:
        return Verdict(False, "Only extractive claims are auto-accepted; paraphrases require future review")
    return Verdict(True, "Quote and claim match the stored source excerpt (extractive only)")


class ContextManager:
    def __init__(self, session_factory, checker=None):
        self.session_factory = session_factory
        self.checker = checker or check_extractively_supported

    @staticmethod
    def _get_or_create(db, session_id: str) -> ResearchSession:
        session = db.get(ResearchSession, session_id)
        if session is None:
            session = ResearchSession(id=session_id, revision=0)
            db.add(session)
            db.flush()
        return session

    @staticmethod
    def _evidence_active(db, evidence: EvidenceChunk) -> bool:
        if evidence.source_kind in {"web_snippet", "web_page", "academic_abstract"}:
            return True  # verified to a captured search snippet, NOT verified to full webpage
        version = db.get(DocumentVersion, evidence.document_version_id)
        if version is None or not version.active:
            return False
        document = db.get(Document, version.document_id)
        return bool(document and document.active)

    def notes(self, session_id: str) -> dict:
        with self.session_factory() as db:
            session = db.get(ResearchSession, session_id)
            notes = db.scalars(select(ContextNote).where(
                ContextNote.session_id == session_id).order_by(ContextNote.updated_at, ContextNote.id)
            ).all()
            return {"session_id": session_id, "revision": session.revision if session else 0,
                    "notes": [{"id": n.id, "claim": n.claim, "quote": n.quote,
                               "evidence_id": n.evidence_id,
                               "source_active": self._evidence_active(db, db.get(EvidenceChunk, n.evidence_id))}
                              for n in notes]}

    def events(self, session_id: str) -> list[dict]:
        with self.session_factory() as db:
            events = db.scalars(select(ContextEvent).where(
                ContextEvent.session_id == session_id)
                .order_by(ContextEvent.created_at, ContextEvent.id)).all()
            return [{"id": e.id, "operation": e.operation, "status": e.status,
                     "old_claim": e.old_claim, "new_claim": e.new_claim, "note_id": e.note_id,
                     "evidence_id": e.evidence_id, "reason": e.reason,
                     "previous_revision": e.previous_revision, "new_revision": e.new_revision}
                    for e in events]

    def propose(self, session_id: str, claim: str, quote: str, evidence_id: str,
                replace_note_id: str | None = None, expected_revision: int | None = None) -> dict:
        with self.session_factory.begin() as db:
            session = self._get_or_create(db, session_id)
            previous = session.revision
            existing = db.get(ContextNote, replace_note_id) if replace_note_id else None
            operation = "replace" if replace_note_id else "add"
            evidence = db.get(EvidenceChunk, evidence_id)
            if expected_revision is not None and previous != expected_revision:
                verdict = Verdict(False, "Revision conflict; reload the working context")
            elif replace_note_id and (existing is None or existing.session_id != session_id):
                verdict = Verdict(False, "Replacement note is not in this research session")
            elif evidence is None:
                verdict = Verdict(False, "Unknown evidence ID")
            elif not self._evidence_active(db, evidence):
                verdict = Verdict(False, "Evidence is from an inactive document version")
            else:
                verdict = (self.checker.check(claim, quote, evidence.text)
                           if hasattr(self.checker, "check")
                           else self.checker(claim, quote, evidence.text))
            old_claim = existing.claim if existing is not None else None
            if verdict.accepted:
                if existing:
                    existing.claim = claim
                    existing.quote = quote
                    existing.evidence_id = evidence_id
                    existing.updated_at = utcnow()
                    note = existing
                else:
                    note = ContextNote(session_id=session_id, claim=claim, quote=quote,
                                       evidence_id=evidence_id)
                    db.add(note)
                    db.flush()
                session.revision += 1
                note_id = note.id
            else:
                note_id = replace_note_id
            event = ContextEvent(
                session_id=session_id, operation=operation,
                status="accepted" if verdict.accepted else "rejected", note_id=note_id,
                old_claim=old_claim,
                new_claim=claim, evidence_id=evidence_id, reason=verdict.reason,
                previous_revision=previous, new_revision=session.revision,
            )
            db.add(event)
            db.flush()
            return {"accepted": verdict.accepted, "reason": verdict.reason,
                    "note_id": note_id, "revision": session.revision, "event_id": event.id,
                    "similarity": verdict.similarity, "verification_method": verdict.method}

    def remove(self, session_id: str, note_id: str, expected_revision: int | None = None) -> dict:
        with self.session_factory.begin() as db:
            session = self._get_or_create(db, session_id)
            old_rev = session.revision
            note = db.get(ContextNote, note_id)
            accepted = (note is not None and note.session_id == session_id
                        and (expected_revision is None or expected_revision == old_rev))
            prior_claim = note.claim if note and note.session_id == session_id else None
            prior_evidence = note.evidence_id if note and note.session_id == session_id else None
            if accepted:
                db.delete(note)
                session.revision += 1
            event = ContextEvent(session_id=session_id, operation="remove",
                                 status="accepted" if accepted else "rejected",
                                 note_id=note_id, old_claim=prior_claim, new_claim=None,
                                 evidence_id=prior_evidence,
                                 reason="Removed with audit trail" if accepted else "Unknown note or revision conflict",
                                 previous_revision=old_rev, new_revision=session.revision)
            db.add(event)
            db.flush()
            return {"accepted": accepted, "reason": event.reason,
                    "revision": session.revision, "event_id": event.id}
