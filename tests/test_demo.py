"""Isolated demo stack: URL restrictions and repeatable seeding."""
import shutil
from pathlib import Path

import pytest

from scripts.seed_demo import check_safe_demo_url, seed

ROOT = Path(__file__).resolve().parents[1]


def test_demo_restricts_url_and_seed_is_repeatable(tmp_path):
    for unsafe in ('https://127.0.0.1:8100', 'http://127.0.0.1:8000',
                   'http://api:8100', 'http://127.0.0.1:8100@evil.test'):
        with pytest.raises(ValueError):
            check_safe_demo_url(unsafe)
    root = tmp_path
    (root / 'demo' / 'fixtures').mkdir(parents=True)
    (root / 'demo' / 'fixtures' / 'paper.txt').write_text('A fictional research paper.\n')
    state = {'agents': [], 'jobs': 0, 'workflows': {}}

    def mock_transport(_url, method, path, data=None):
        if path == '/ready': return {'status': 'ready'}
        if path == '/ingest/sync?index_vectors=false': return {'ingested': 1}
        if path == '/agents' and method == 'GET': return {'agents': state['agents']}
        if path == '/agents' and method == 'POST':
            result = {**data, 'id': 'demo-profile'}
            state['agents'].append(result)
            return result
        if path == '/agents/demo-profile/workflow':
            state['workflows']['demo-profile'] = data
            return {'workflow': data}
        if path == '/retrieval-evals/jobs':
            state['jobs'] += 1
            return {'job_id': 'job-1'}
        if path == '/retrieval-evals/jobs/job-1':
            return {'status': 'completed', 'run_id': 'run-1'}
        raise AssertionError((method, path))

    for _ in range(2):
        result = seed('http://127.0.0.1:8100', send=mock_transport, root=root)
        assert result['agent_id'] == 'demo-profile'
        assert result['evaluation']['run_id'] == 'run-1'
    assert len(state['agents']) == 1
    assert state['jobs'] == 2  # repeated evaluation is explicitly a NEW measured run
    assert state['workflows']['demo-profile']['nodes'][1]['type'] == 'local_search'
    assert (root / 'demo' / 'workspace' / 'papers' / 'paper.txt').is_file()


def test_demo_end_to_end_against_local_api_without_model(tmp_path, monkeypatch):
    """Smoke path creates real indexed documents, a graph and a scored synthetic run."""
    pytest.importorskip('llama_index', reason='Full BM25 inference requires optional LlamaIndex dependencies')

    from fastapi.testclient import TestClient

    from aira.api import create_app
    from aira.config import Settings
    demo_root = tmp_path / 'demo'
    for src in sorted((ROOT / 'demo' / 'fixtures').glob('*.txt')):
        target = demo_root / 'fixtures' / src.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    papers = demo_root / 'workspace' / 'papers'
    monkeypatch.setenv('AIRA_DATA_DIR', str(demo_root / 'workspace' / 'data'))
    client = TestClient(create_app(Settings(database_url=f'sqlite:///{tmp_path / "demo.db"}',
                                            papers_dir=papers)))
    def transport(_base, method, path, data=None):
        response = client.request(method, path, json=data)
        assert response.status_code < 400, (path, response.text)
        return response.json()
    result = seed('http://127.0.0.1:8100', send=transport, root=tmp_path)
    assert len(client.get('/documents').json()['documents']) == 3
    graph = client.get(f'/agents/{result["agent_id"]}/workflow').json()['workflow']
    assert graph['enabled']
    assert result['evaluation']['status'] == 'completed'
    assert result['evaluation']['run_id']
    assert client.get('/retrieval-evals/runs').json()['runs']
