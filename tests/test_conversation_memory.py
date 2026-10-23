"""Conversation memory: provenance, chat isolation, routing and role-preserving prompts."""
import sys
from types import ModuleType, SimpleNamespace

from fastapi.testclient import TestClient

from aira.api import create_app
from aira.config import Settings
from aira.conversations import create_conversation, recent_history, save_exchange
from aira.db import make_session_factory
from aira.memory import (
    read_conversation_turns,
    render_conversation_evidence,
    route_context,
    search_conversation,
)
from aira.options import RetrievalOptions


def seed(factory, chat, message):
    save_exchange(factory, chat['id'], message, {'answer': f"Response: {message}", 'sources': []}, 'answer')


def test_auto_routing_and_manual_override():
    assert route_context('Hello!') == 'neither'
    assert route_context('What did we decide earlier?') == 'conversation'
    assert route_context('What does Dettmers report about NF4?') == 'research'
    assert route_context('Compare it with the paper we discussed yesterday') == 'both'
    assert route_context('How much faster was this?') == 'research'
    assert route_context('Which findings matter?', requested='both') == 'both'
    assert route_context('Hello!', requested='research') == 'research'


def test_conversation_search_preserves_chat_isolation_and_exact_turn_ids(tmp_path):
    factory = make_session_factory(f"sqlite:///{tmp_path/'history.db'}")
    first = create_conversation(factory)
    second = create_conversation(factory)
    seed(factory, first, 'We decided that QLoRA memory efficiency is the main topic.')
    seed(factory, first, 'We will benchmark multiple retrieval strategies.')
    seed(factory, second, 'Secret chat discusses QLoRA but is not accessible from first.')
    hits = search_conversation(factory, first['id'], 'QLoRA memory efficiency', top_k=3)
    assert hits and all(hit['conversation_id'] == first['id'] for hit in hits)
    assert not any('Secret' in hit['text'] for hit in hits)
    assert hits[0]['source_kind'] == 'conversation_turn'
    assert 'conversation-turn:' in render_conversation_evidence(hits)
    other = search_conversation(factory, second['id'], 'secret')
    assert other and other[0]['conversation_id'] == second['id']
    assert search_conversation(factory, 'missing', 'QLoRA') == []
    id_a = hits[0]['turn_id']
    id_b = other[0]['turn_id']
    assert [row['turn_id'] for row in read_conversation_turns(factory, first['id'], [id_a, id_b])] == [id_a]
    assert not read_conversation_turns(factory, 'missing', [id_a])
    assert not read_conversation_turns(factory, first['id'], [-5])


def test_rag_recalls_old_archived_turns_without_searching_papers(tmp_path, monkeypatch):
    from aira import agent
    factory = make_session_factory(f"sqlite:///{tmp_path/'rag.db'}")
    chat = create_conversation(factory)
    seed(factory, chat, 'We decided that QLoRA quantization should be studied first.')
    for i in range(6):
        seed(factory, chat, f"Unrelated conversation topic {i}")

    class Retriever:
        def __init__(self): self.queries = []
        def search(self, query, top_k=5, options=None):
            self.queries.append(query)
            return [{'id': 'source123', 'text': 'A scientific source paragraph', 'page': 1,
                     'path': 'paper.pdf', 'version': 1, 'score': 0.5}]

    class Context:
        def notes(self, session): return {'revision': 0, 'notes': []}

    class Model:
        model = 'mock'
        def __init__(self): self.messages = []
        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(content='Earlier we chose QLoRA.')

    model, retriever = Model(), Retriever()
    monkeypatch.setattr(agent, 'make_model', lambda *args, **kwargs: model)
    result = agent.answer_with_local_context(
        'What did we decide about QLoRA earlier?', chat['id'], Settings(),
        retriever, Context(), retrieval=RetrievalOptions(),
        session_factory=factory, history=recent_history(factory, chat['id']))
    assert result['memory_route'] == 'conversation'
    assert retriever.queries == []
    assert result['sources'] == []   # Conversation statements are NOT paper citations
    assert result['conversation_sources']
    # Proper user and assistant roles, not a single giant serialized history message.
    assert [role for role, _ in model.messages] == ['system', 'human', 'ai', 'human', 'ai', 'human']
    assert 'QLoRA quantization should be studied first' in model.messages[-1][1]
    assert '[conversation-turn:' in model.messages[-1][1]
    # A different request retrieves scientific evidence as before.
    result = agent.answer_with_local_context('What is quantization?', chat['id'], Settings(),
       retriever, Context(), session_factory=factory)
    assert result['memory_route'] == 'research' and retriever.queries
    assert result['sources'][0]['id'] == 'source123'


def test_agent_has_independent_conversation_tools(tmp_path, monkeypatch):
    import aira.agent as agent
    factory = make_session_factory(f"sqlite:///{tmp_path/'agent.db'}")
    chat = create_conversation(factory)
    seed(factory, chat, 'We settled on QLoRA as the primary benchmark target.')
    chosen = {}
    module = ModuleType('langchain.agents')
    tool_module = ModuleType('langchain.tools')
    def tool(fn):
        fn.name = fn.__name__
        return fn
    def create_agent(*, model, tools, system_prompt):
        chosen['tools'] = {t.name: t for t in tools}
        chosen['prompt'] = system_prompt
        return SimpleNamespace(invoke=lambda payload, config: {
            'messages': [SimpleNamespace(content='I can retrieve either archive')]})
    module.create_agent = create_agent
    tool_module.tool = tool
    monkeypatch.setitem(sys.modules, 'langchain.agents', module)
    monkeypatch.setitem(sys.modules, 'langchain.tools', tool_module)
    monkeypatch.setattr(agent, 'make_model', lambda *args, **kw: object())

    class Retriever:
        def search(self, query, top_k=5, options=None):
            return [{'id': 'paper123', 'text': 'Paper excerpt.', 'page': 2,
                     'path': 'paper.pdf', 'version': 1, 'score': 0.4}]

    class Context:
        def notes(self, session): return {'revision': 0, 'notes': []}

    result = agent.run_agent('What did we decide earlier?', chat['id'], Settings(),
                             factory, Retriever(), Context(), retrieval=RetrievalOptions())
    assert result['memory_route'] == 'conversation'
    assert 'search_conversation_history' in chosen['tools']
    assert 'read_conversation_history' in chosen['tools']
    assert 'search_local_papers' in chosen['tools']
    import json
    hits = json.loads(chosen['tools']['search_conversation_history']('QLoRA benchmark'))['results']
    assert hits and hits[0]['source_kind'] == 'conversation_turn'
    assert json.loads(chosen['tools']['read_conversation_history']([hits[0]['turn_id']]))['results']
    assert result['sources'] == []

    agent.run_agent('What did we decide earlier?', chat['id'], Settings(), factory,
        Retriever(), Context(), profile={'enabled_tools': ['local_search'], 'system_instructions': ''})
    assert 'search_conversation_history' not in chosen['tools']


def test_memory_routes_through_api_and_frontend_compatible(tmp_path, monkeypatch):
    from aira import api as api_module
    client = TestClient(create_app(Settings(database_url=f"sqlite:///{tmp_path/'api.db'}", papers_dir=tmp_path)))
    chat = client.post('/conversations', json={'title': 'New chat'}).json()
    received = []
    def answer(question, session_id, settings, retriever, context, options, generation,
               *, use_web, session_factory, use_academic, web_fetch_pages, history, memory_route=None):
        received.append(memory_route)
        return {'answer': 'We discussed NF4.', 'sources': [], 'memory_route': memory_route or 'conversation',
                'conversation_sources': [], 'model': 'mock'}
    monkeypatch.setattr(api_module, 'answer_with_local_context', answer)
    response = client.post('/answer', json={
        'question':'What did we discuss?', 'session_id':chat['id'], 'memory_route':'conversation'})
    assert response.status_code == 200, response.text
    assert received == ['conversation']
    assert client.post('/answer', json={'question':'test', 'memory_route':'other'}).status_code == 422
    assert client.get(f"/conversations/{chat['id']}/memory/search?q=NF4").status_code == 200
    stored = client.get(f"/conversations/{chat['id']}/messages").json()['messages'][-1]
    assert stored['metadata']['memory_route'] == 'conversation'
    assert client.get('/conversations/not-a-chat/memory/search?q=NF4').status_code == 404


def test_vague_historical_reference_returns_low_confidence_recent_turns(tmp_path):
    factory = make_session_factory(f"sqlite:///{tmp_path/'fallback.db'}")
    chat = create_conversation(factory)
    seed(factory, chat, 'I want to focus on the Dettmers study.')
    results = search_conversation(factory, chat['id'], 'the paper we discussed yesterday')
    assert results and all(hit['score'] == 0.0 for hit in results)
