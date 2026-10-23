"""Source-aware routing of questions between local papers and the web."""
from types import SimpleNamespace

from aira import agent
from aira.config import Settings
from aira.memory import route_context


def test_general_knowledge_skips_local_but_explicit_research_still_works():
    assert route_context('Can you explain what AI is?') == 'neither'
    assert route_context('What is artificial intelligence?') == 'neither'
    assert route_context('Why is the sky blue?') == 'neither'
    assert route_context('What is the difference between LoRA and QLoRA?') == 'research'
    assert route_context('Summarize the papers in my library') == 'research'
    assert route_context('What did we decide about QLoRA earlier?') == 'conversation'
    assert route_context('Compare this with the paper we discussed yesterday') == 'both'
    assert route_context('What is AI?', requested='research') == 'research'


def test_web_enabled_question_skips_local_pdf_prefetch(monkeypatch):
    class Retriever:
        def __init__(self):
            self.calls = []
        def search(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return [{'id': 'paper-1', 'text': 'Irrelevant LoRA', 'page': 1,
                     'path': 'QLoRA.pdf', 'version': 1, 'score': .1}]

    class Context:
        def notes(self, _):
            return {'notes': [], 'revision': 0}

    class Model:
        model = 'mock'
        def invoke(self, messages):
            assert 'WEB SEARCH SNIPPET' in messages[-1][1]
            assert 'QLoRA.pdf' not in messages[-1][1]
            return SimpleNamespace(content='An answer [1].')

    web = {'evidence_id': 'web-1', 'url': 'https://example.org/ai',
           'title': 'Artificial intelligence', 'excerpt': 'An overview of AI.',
           'retrieved_at': '2026-10-14T12:00:00Z', 'source_kind': 'web_snippet_unverified'}
    monkeypatch.setattr(agent, '_configured_web_search', lambda *args, **kwargs: [web])
    monkeypatch.setattr(agent, 'read_top_results', lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(agent, 'make_model', lambda *args, **kwargs: Model())
    retriever = Retriever()
    result = agent.answer_with_local_context('Can you explain what AI is?', 'chat-1', Settings(),
        retriever, Context(), use_web=True, session_factory=object())
    assert retriever.calls == []
    assert result['memory_route'] == 'neither'
    assert [item['id'] for item in result['sources']] == ['web-1']
    assert result['web_used'] is True
