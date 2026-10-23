"""Conversation persistence, cursor paging, isolation and API round-trips."""
from fastapi.testclient import TestClient
from sqlalchemy import select

from aira.api import create_app
from aira.config import Settings
from aira.conversations import (
    create_conversation,
    delete_conversation,
    get_messages,
    recent_history,
    save_exchange,
)
from aira.db import ChatConversation, ChatTurn, make_session_factory


def test_isolated_history_and_cursor_pagination(tmp_path):
    factory = make_session_factory(f"sqlite:///{tmp_path / 'chat.db'}")
    first = create_conversation(factory)
    second = create_conversation(factory)
    for i in range(5):
        save_exchange(factory, first['id'], f"user question {i}", {
            'answer': f"answer {i}", 'model': 'llama3.2:3b', 'sources': [
                {'id': f'evidence_{i}', 'path': 'research.pdf', 'page': i}]}, 'answer')
    save_exchange(factory, second['id'], "other paper", {'answer': 'Other answer'}, 'chat')
    result = get_messages(factory, first['id'], limit=3)
    assert [m['content'] for m in result['messages']] == ['answer 3', 'user question 4', 'answer 4']
    assert result['has_more'] and result['next_cursor']
    older = get_messages(factory, first['id'], limit=20, before=result['next_cursor'])
    assert len(older['messages']) == 7
    assert older['messages'][-1]['content'] == 'user question 3'
    assert all(m['conversation_id'] == first['id'] for m in older['messages'])
    assert [m['content'] for m in recent_history(factory, second['id'])] == ['other paper', 'Other answer']
    assert len(recent_history(factory, first['id'])) == 8
    assert get_messages(factory, 'missing') is None
    assert not delete_conversation(factory, 'missing')
    assert delete_conversation(factory, first['id'])
    assert get_messages(factory, first['id']) is None
    with factory() as db:
        assert not db.scalars(select(ChatTurn).where(ChatTurn.conversation_id == first['id'])).all()
        assert db.get(ChatConversation, second['id']) is not None


def test_conversation_api_and_model_history_roundtrip(tmp_path, monkeypatch):
    import aira.api as api_module

    captured = []

    def stub_answer(question, session_id, settings, retriever, context, options, generation,
                    *, use_web, session_factory, use_academic, web_fetch_pages, history):
        captured.append(list(history))
        return {'answer': f'From evidence: {question}', 'model': 'test-local-model',
                'sources': [{'id': 'a' * 64, 'path': 'paper.pdf', 'page': 2,
                             'version': 1, 'score': 2.0}], 'web_used': False}

    monkeypatch.setattr(api_module, 'answer_with_local_context', stub_answer)
    app = create_app(Settings(database_url=f"sqlite:///{tmp_path / 'api-chat.db'}",
                              papers_dir=tmp_path, llm_provider='ollama'))
    client = TestClient(app)
    a = client.post('/conversations', json={'title': 'New chat'})
    b = client.post('/conversations', json={'title': 'Second'})
    assert a.status_code == 201 and b.status_code == 201
    id_a, id_b = a.json()['id'], b.json()['id']
    one = client.post('/answer', json={'question': 'Explain quantization', 'session_id': id_a})
    two = client.post('/answer', json={'question': 'And memory?', 'session_id': id_a})
    assert one.status_code == two.status_code == 200
    assert captured[0] == []
    assert [m['role'] for m in captured[1]] == ['user', 'assistant']
    assert 'quantization' in captured[1][0]['content']
    assert client.get(f'/conversations/{id_b}/messages').json()['messages'] == []
    history = client.get(f'/conversations/{id_a}/messages?limit=2').json()
    assert [m['role'] for m in history['messages']] == ['user', 'assistant']
    assert history['messages'][1]['sources'][0]['id'] == 'a' * 64
    assert history['has_more']
    assert client.get('/conversations?limit=1&offset=0').json()['has_more']
    assert client.patch(f'/conversations/{id_a}', json={'title': 'New title'}).json()['title'] == 'New title'
    assert client.delete(f'/conversations/{id_a}').status_code == 204
    assert client.get(f'/conversations/{id_a}/messages').status_code == 404
    assert client.get('/conversations?limit=0').status_code == 422
