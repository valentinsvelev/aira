"""Saved RAG system presets, including the editable built-in preset."""

from fastapi.testclient import TestClient

from aira.api import create_app
from aira.config import Settings
from aira.db import make_session_factory
from aira.rag_systems import default_system, get_system, list_systems, update_system


def test_rag_presets_persist_and_reach_answer(tmp_path, monkeypatch):
    from aira import api as api_module
    settings = Settings(database_url=f"sqlite:///{tmp_path/'rag.db'}", papers_dir=tmp_path)
    client = TestClient(create_app(settings))
    assert client.get('/rag-systems').json()['systems'][0]['id'] == 'default'
    payload = {'name': 'Dense RAG', 'description': 'Custom searchable paper corpus',
               'system_instructions': 'Explain counterevidence.',
               'retrieval': {'mode': 'bm25', 'top_k': 3},
               'generation': {'model': 'qwen3.5:4b', 'top_k': 64, 'min_p': 0.05}}
    created = client.post('/rag-systems', json=payload)
    assert created.status_code == 201, created.text
    ident = created.json()['id']
    assert client.post('/rag-systems', json={**payload, 'generation': {'min_p': 1.01}}).status_code == 422
    # Built-in RAG is editable in place; its stable ID cannot be deleted.
    default_update = client.patch('/rag-systems/default', json=payload)
    assert default_update.status_code == 200, default_update.text
    assert next(item for item in client.get('/rag-systems').json()['systems'] if item['id'] == 'default')['name'] == 'Dense RAG'
    assert client.delete('/rag-systems/default').status_code == 403
    assert client.get('/rag-systems').json()['systems'][1]['id'] == ident
    called = []
    def fake_answer(question, session_id, settings, retriever, context, retrieval, generation, **kw):
        called.append((retrieval, generation, kw.get('profile')))
        return {'answer': 'Tested findings [1].', 'model': 'test', 'sources': []}
    monkeypatch.setattr(api_module, 'answer_with_local_context', fake_answer)
    result = client.post('/answer', json={'question': 'Assess retrieval', 'rag_system_id': ident})
    assert result.status_code == 200, result.text
    retrieval, generation, profile = called[0]
    assert retrieval.top_k == 3
    assert generation.top_k == 64 and generation.min_p == 0.05
    assert profile['system_instructions'] == 'Explain counterevidence.'
    reloaded = TestClient(create_app(settings))
    assert reloaded.get('/rag-systems').json()['systems'][1]['name'] == 'Dense RAG'
    assert client.patch(f'/rag-systems/{ident}', json={**payload, 'name': 'Renamed'}).json()['name'] == 'Renamed'
    assert client.delete(f'/rag-systems/{ident}').status_code == 204
    assert client.post('/answer', json={'question': 'Hello', 'rag_system_id': ident}).status_code == 404


def test_builtin_rag_edits_persist_without_duplicate(tmp_path):
    factory = make_session_factory(f'sqlite:///{tmp_path}/rag.db')
    original = default_system()
    edited = {k:v for k,v in original.items() if k not in {'builtin','id'}}
    edited['retrieval'] = {**edited['retrieval'], 'mode': 'dense', 'top_k': 7}
    edited['generation'] = {**edited['generation'], 'model': 'openbmb/minicpm-v4.7:1b'}
    assert update_system(factory, 'default', edited)['id'] == 'default'
    assert get_system(factory, 'default')['retrieval']['top_k'] == 7
    assert get_system(factory, 'default')['generation']['model'] == 'openbmb/minicpm-v4.7:1b'
    assert [p['id'] for p in list_systems(factory)] == ['default']
    factory2 = make_session_factory(f'sqlite:///{tmp_path}/rag.db')
    assert get_system(factory2, 'default')['retrieval']['mode'] == 'dense'


def test_builtin_rag_patch_allowed_but_delete_forbidden(tmp_path):
    settings = Settings(database_url=f'sqlite:///{tmp_path}/app.db', papers_dir=tmp_path,
                        llm_provider='openai')
    client = TestClient(create_app(settings))
    body = {k:v for k,v in default_system().items() if k not in {'builtin', 'id'}}
    body['description'] = 'Changed directly'
    assert client.patch('/rag-systems/default', json=body).status_code == 200
    result = client.get('/rag-systems').json()['systems'][0]
    assert result['description'] == 'Changed directly' and result['builtin']
    assert client.delete('/rag-systems/default').status_code == 403
