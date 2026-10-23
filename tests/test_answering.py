"""Direct RAG answering: prompt assembly, numbered citations and parameter flow (no real model)."""
from types import SimpleNamespace

from aira.agent import SYSTEM_PROMPT, answer_with_local_context, build_research_context
from aira.config import Settings
from aira.options import GenerationOptions, RetrievalOptions


class FakeRetriever:
    def search(self, query, top_k=4):
        return [{"id": "evidence-123", "path": "test.pdf", "page": 3, "version": 2,
                 "score": 0.9, "text": "NF4 reduces storage memory for frozen weights."}]


class FakeContext:
    def notes(self, session_id):
        return {"revision": 7, "notes": [
            {"claim": "Paged optimizers handle memory spikes.", "evidence_id": "memo-456",
             "source_active": True},
            {"claim": "Inactive stale fact", "evidence_id": "stale-999", "source_active": False},
        ]}


class FakeLLM:
    def __init__(self):
        self.messages = None
        self.model = "fake-local"

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content="Local mocked answer [evidence-123].")


class Retriever:
    def search(self, query, top_k=4, options=None):
        assert options.mode == 'hybrid'
        assert options.top_k == 3
        return [{'id': 'e1', 'text': 'A paper states that NF4 reduces memory.',
                 'path': 'QLoRA.pdf', 'version': 1, 'page': 2, 'score': .02}]


class Context:
    def notes(self, session_id):
        return {'revision': 0, 'notes': []}


def test_context_assembler_includes_notes_and_source_metadata_not_inactive_notes():
    source = FakeRetriever().search("quantization")
    snapshot = FakeContext().notes("demo")
    ctx = build_research_context(snapshot, source)
    assert "Paged optimizers handle memory spikes." in ctx
    assert "evidence-123" in ctx
    assert "test.pdf (version 2, page 3)" in ctx
    assert "Inactive stale fact" not in ctx


def test_answer_includes_explicit_context_and_citations(monkeypatch):
    import aira.agent as agent
    mock = FakeLLM()
    monkeypatch.setattr(agent, "make_model", lambda settings: mock)
    output = answer_with_local_context(
        "How does NF4 reduce memory?", "demo", Settings(), FakeRetriever(), FakeContext())
    assert output["context_revision"] == 7
    assert output["sources"][0]["id"] == "evidence-123"
    assert "How does NF4 reduce memory?" in mock.messages[1][1]
    assert "Paged optimizers handle memory spikes." in mock.messages[1][1]
    assert "NF4 reduces storage memory" in mock.messages[1][1]
    assert "stale fact" not in mock.messages[1][1]


def test_context_numbers_local_and_web_sources_in_display_order():
    snapshot = {"notes": [{"evidence_id": "old-id", "claim": "A preliminary note", "source_active": True}]}
    local = [{"id": "local-1", "path": "paper.pdf", "version": 1, "page": 4,
              "text": "NF4 is a four-bit NormalFloat format."}]
    web = [{"evidence_id": "web-1", "title": "An article", "url": "https://example.org/a",
            "retrieved_at": "2026-10-14", "excerpt": "An unverified web excerpt",
            "source_kind": "web_snippet_unverified"}]
    text = build_research_context(snapshot, local, web_results=web)
    assert "[1] evidence_id=local-1" in text
    assert "[2] evidence_id=web-1" in text
    assert "Provisional note" in text
    assert "[evidence:old-id]" not in text
    assert "Do not fabricate" in SYSTEM_PROMPT


def test_generation_and_retrieval_params_flow_into_answer(monkeypatch):
    from aira import agent
    seen = {}
    class Model:
        model = 'mock-ollama'
        def invoke(self, messages):
            assert 'QLoRA.pdf' in messages[1][1]
            return SimpleNamespace(content='NF4 reduces memory [e1]')
    def fake_model(settings, generation=None):
        seen['generation'] = generation
        return Model()
    monkeypatch.setattr(agent, 'make_model', fake_model)
    retrieval = RetrievalOptions(mode='hybrid', top_k=3)
    generation = GenerationOptions(temperature=.3, top_p=.8, num_predict=512)
    result = answer_with_local_context('How does it work?', 'test', Settings(),
                                       Retriever(), Context(), retrieval, generation, memory_route='research')
    assert seen['generation'] == generation
    assert result['parameters']['retrieval']['mode'] == 'hybrid'
    assert result['parameters']['generation']['num_predict'] == 512
