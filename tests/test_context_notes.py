"""Working-context notes: verification, audit events, session isolation and staleness."""
from sqlalchemy import select

from aira.context import check_extractively_supported
from aira.db import EvidenceChunk
from aira.ingestion import digest

QUOTE = "The paper reports a 15 percent speedup on an A100 GPU at batch size one."


BAD = "The technique improves inference speed on all GPUs by 15 percent."


def seed_evidence(workspace):
    evidence_id = digest("dummy-evidence")
    with workspace["factory"].begin() as db:
        db.add(EvidenceChunk(id=evidence_id, source_kind="web_snippet",
                             source_url="https://example.org/study", page=None,
                             text=f"The study states: {QUOTE}", text_sha256=digest(QUOTE)))
    return evidence_id


def test_faithfulness_blocks_overgeneralization():
    assert check_extractively_supported(QUOTE, QUOTE, "..." + QUOTE).accepted
    assert not check_extractively_supported(BAD, QUOTE, "..." + QUOTE).accepted
    assert not check_extractively_supported(QUOTE, "a fake supporting quote", QUOTE).accepted


def test_accepted_rejected_replace_delete_events(workspace):
    evidence_id = seed_evidence(workspace)
    manager = workspace["context"]
    good = manager.propose("session-a", QUOTE, QUOTE, evidence_id, expected_revision=0)
    assert good["accepted"] and good["revision"] == 1
    rejected = manager.propose("session-a", BAD, QUOTE, evidence_id, expected_revision=1)
    assert not rejected["accepted"] and rejected["revision"] == 1
    stale = manager.propose("session-a", QUOTE, QUOTE, evidence_id, expected_revision=0)
    assert not stale["accepted"]
    assert len(manager.notes("session-a")["notes"]) == 1
    removed = manager.remove("session-a", good["note_id"], expected_revision=1)
    assert removed["accepted"] and removed["revision"] == 2
    assert manager.notes("session-a")["notes"] == []
    events = manager.events("session-a")
    assert [e["status"] for e in events] == ["accepted", "rejected", "rejected", "accepted"]


def test_cross_session_note_cannot_be_replaced(workspace):
    evidence_id = seed_evidence(workspace)
    manager = workspace["context"]
    first = manager.propose("alpha", QUOTE, QUOTE, evidence_id)
    cross = manager.propose("beta", QUOTE, QUOTE, evidence_id,
                            replace_note_id=first["note_id"])
    assert not cross["accepted"]
    assert manager.notes("alpha")["revision"] == 1
    assert manager.notes("beta")["revision"] == 0


def test_notes_become_stale_after_source_document_changes(workspace):
    from aira.db import EvidenceChunk
    original = ("Evidence provenance must remain linked to the original version. " * 8)
    file = workspace["papers"] / "versioned.txt"
    file.write_text(original)
    workspace["ingestor"].sync()
    with workspace["factory"]() as db:
        evidence_id = db.scalars(select(EvidenceChunk.id).where(
            EvidenceChunk.source_kind == "text")).first()
    phrase = "Evidence provenance must remain linked to the original version."
    saved = workspace["context"].propose("updated-session", phrase, phrase, evidence_id)
    assert saved["accepted"]
    file.write_text("The revised text changes source evidence in a new version. " * 8)
    workspace["ingestor"].sync()
    assert workspace["context"].notes("updated-session")["notes"][0]["source_active"] is False
    attempted = workspace["context"].propose("updated-session", phrase, phrase, evidence_id)
    assert not attempted["accepted"]
