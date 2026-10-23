"""Checking notes against sources: semantic similarity and the TEV1 classifier (no downloads)."""
import json

import httpx

from aira.context import ContextManager
from aira.db import EvidenceChunk
from aira.ingestion import digest
from aira.semantic import SemanticChecker, cosine_similarity
from aira.tev1_checker import Tev1Checker

ORIGINAL = "The paper reports a 15 percent speedup on an A100 GPU at batch size one."


REWRITE = "On an A100 with batch size one, the reported speed improvement is 15 percent."


WRONG = "A completely unrelated paper discusses protein synthesis in bacterial cells."


class FakeEncoder:
    def encode(self, texts):
        self.last_texts = texts
        mapping = {ORIGINAL: [1.0, 0.0], REWRITE: [0.99, 0.08], WRONG: [0.0, 1.0]}
        return [mapping[text] for text in texts]


SOURCE = "The base model uses NF4 quantization; double quantization reduces storage for quantization constants."


GOOD = "The model uses NF4 and double quantization to reduce memory."


BAD = "The approach always improves inference speed on every GPU."


def fake_tev1(guess="supported", probability=0.96):
    def handle(request):
        assert request.url.path == '/v1/systemone'
        data = json.loads(request.content)
        assert data['model'] == 'tev1:0.8b'
        assert data['questions']['faithfulness']['type'] == 'choice'
        assert 'source_evidence' in data['state']
        return httpx.Response(200, json={"answers": {"faithfulness": {
            "choice": guess,
            "probabilities": {"supported": probability, "contradicted": 0.02, "insufficient": 0.02},
        }}})
    return httpx.MockTransport(handle)


def test_cosine_known_vectors():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([1, 0], [-1, 0]) == -1.0


def test_semantic_checker_accepts_supported_paraphrase_and_rejects_irrelevant():
    checker = SemanticChecker(threshold=0.8, encoder=FakeEncoder())
    supported = checker.check(REWRITE, ORIGINAL, "Introduction\n" + ORIGINAL)
    assert supported.accepted and supported.similarity > 0.98
    assert supported.method == "semantic_cosine"
    irrelevant = checker.check(WRONG, ORIGINAL, ORIGINAL)
    assert not irrelevant.accepted and irrelevant.similarity == 0.0
    assert not checker.check(REWRITE, "invented quote", ORIGINAL).accepted


def test_new_number_guard_rejects_even_if_semantically_similar():
    checker = SemanticChecker(threshold=0.8, encoder=FakeEncoder())
    claim = "On an A100 GPU at batch size one, the paper shows a 90 percent speedup."
    verdict = checker.check(claim, ORIGINAL, ORIGINAL)
    assert not verdict.accepted
    assert "number" in verdict.reason.lower()


def test_persist_accepted_and_rejected_similarity_events(workspace):
    evidence_id = digest("semantic-fake")
    with workspace["factory"].begin() as db:
        db.add(EvidenceChunk(id=evidence_id, source_kind="web_snippet",
                             source_url="https://example.org/paper",
                             text=ORIGINAL, text_sha256=digest(ORIGINAL), ordinal=0))
    manager = ContextManager(workspace["factory"], checker=SemanticChecker(
        threshold=0.8, encoder=FakeEncoder()))
    accepted = manager.propose("s", REWRITE, ORIGINAL, evidence_id)
    assert accepted["accepted"]
    assert accepted["similarity"] > 0.98
    assert accepted["verification_method"] == "semantic_cosine"
    rejected = manager.propose("s", WRONG, ORIGINAL, evidence_id)
    assert not rejected["accepted"]
    assert manager.notes("s")["revision"] == 1
    assert manager.notes("s")["notes"][0]["claim"] == REWRITE
    events = manager.events("s")
    assert len(events) == 2
    assert "cosine" in events[0]["reason"].lower()
    assert events[1]["status"] == "rejected"


def test_tev1_accepts_supported_paraphrase_and_rejects_contradiction():
    supported = Tev1Checker('http://localhost:11434', transport=fake_tev1())
    good = supported.check(GOOD, SOURCE, 'paper text: ' + SOURCE)
    assert good.accepted and good.method == 'tev1_decision'
    bad = Tev1Checker('http://localhost:11434', transport=fake_tev1('contradicted', 0.01))
    assert not bad.check(BAD, SOURCE, SOURCE).accepted
    assert not supported.check('The reduction is 99 percent.', SOURCE, SOURCE).accepted
    assert not supported.check(GOOD, SOURCE, 'DIFFERENT source').accepted


def test_tev1_fails_closed_if_ollama_unavailable():
    def throw(_request):
        raise httpx.ConnectError('offline')
    checker = Tev1Checker('http://localhost:11434', transport=httpx.MockTransport(throw))
    outcome = checker.check(GOOD, SOURCE, SOURCE)
    assert not outcome.accepted
    assert 'unavailable' in outcome.reason


def test_tev1_requires_probability_threshold():
    checker = Tev1Checker('http://localhost:11434', threshold=.9, transport=fake_tev1('supported', .79))
    assert not checker.check(GOOD, SOURCE, SOURCE).accepted
