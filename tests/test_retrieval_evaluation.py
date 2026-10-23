"""Deterministic offline retrieval evaluation: metrics, isolation, persistence and jobs."""
import json
import time
from threading import Event
from unittest.mock import patch

import pytest

from aira.benchmark_analytics import read_retrieval_runs
from aira.options import RetrievalOptions
from aira.retrieval_evaluation import (
    dataset_status,
    evaluate,
    job_status,
    list_runs,
    load_dataset,
    query_metrics,
    read_run,
    start_job,
)


def test_metrics_known_ranking_and_unjudged_docs():
    scores = query_metrics(['noise', 'd1', 'd2', 'noise2'], {'d1': 2, 'd2': 1}, 4)
    assert scores['recall'] == 1.0
    assert scores['mrr'] == 0.5
    assert scores['precision'] == 0.5
    assert 0 < scores['ndcg'] < 1
    assert query_metrics([], {'d1': 1}, 5) == {
        'recall': 0.0, 'mrr': 0.0, 'ndcg': 0.0, 'precision': 0.0,
    }
    assert query_metrics(['d1'], {'d1': 1}, 1) == {
        'recall': 1.0, 'mrr': 1.0, 'ndcg': 1.0, 'precision': 1.0,
    }


def test_official_scifact_files_loaded_without_adding_personal_papers(tmp_path):
    path = tmp_path / 'retrieval_eval' / 'datasets' / 'scifact'
    (path / 'qrels').mkdir(parents=True)
    (path / 'corpus.jsonl').write_text('\n'.join([
        json.dumps({'_id': 'D1', 'title': 'Study', 'text': 'Findings'}),
        json.dumps({'_id': 'D2', 'title': 'Control', 'text': 'No finding'}),
    ]))
    (path / 'queries.jsonl').write_text(json.dumps({'_id': 'Q1', 'text': 'question'}))
    (path / 'qrels' / 'test.tsv').write_text('query-id\tcorpus-id\tscore\nQ1\tD1\t1\n')
    dataset = load_dataset(tmp_path, 'scifact')
    assert dataset['documents']['D1'] == 'Study\nFindings'
    assert dataset['qrels']['Q1'] == {'D1': 1}
    assert dataset_status(tmp_path)[1]['available']
    assert not (tmp_path / 'papers').exists()


def test_seeded_sampling_and_persistent_run(tmp_path):
    options = RetrievalOptions(mode='sparse', top_k=5)

    def rank(_query):
        return [{'id': 'd2', 'score': 1.0}, {'id': 'd1', 'score': 0.5}]

    with patch('aira.retrieval_evaluation._benchmark_searcher', return_value=rank):
        first = evaluate(tmp_path, 'smoke', options, count=3, seed=101)
        again = evaluate(tmp_path, 'smoke', options, count=3, seed=101)
        assert first['query_ids'] == again['query_ids']
        assert first['sample_size'] == 3
        assert first['metrics']['p95_ms'] >= 0
        assert first['metrics']['recall'] >= 0
        jid = start_job(tmp_path, 'smoke', options, 3, 101)
        for _ in range(100):
            status = job_status(jid)
            if status['status'] not in ('queued', 'running'):
                break
            time.sleep(.01)
        assert status['status'] == 'completed', status
    saved = read_run(tmp_path, status['run_id'])
    assert saved['query_ids'] == first['query_ids']
    assert len(list_runs(tmp_path)) == 1
    runs, cases, hits = read_retrieval_runs(tmp_path)
    assert len(runs) == 1 and len(cases) == 3 and len(hits) == 6
    assert read_run(tmp_path, '../../not-a-uuid') is None


def test_cancel_event_does_not_save(tmp_path):
    flag = Event()
    flag.set()
    with patch('aira.retrieval_evaluation._benchmark_searcher', return_value=lambda q: []):
        with pytest.raises(InterruptedError):
            evaluate(tmp_path, 'smoke', RetrievalOptions(), count=3, seed=3, cancel=flag)
    assert list_runs(tmp_path) == []


def test_missing_dataset_and_validation(tmp_path):
    with pytest.raises(FileNotFoundError):
        start_job(tmp_path, 'scifact', RetrievalOptions(), 3, 42)
    with pytest.raises(ValueError):
        start_job(tmp_path, 'unknown', RetrievalOptions(), 3, 42)


def test_retrieval_eval_api_run_and_reload(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from aira.api import create_app
    from aira.config import Settings

    papers = tmp_path / 'papers'
    papers.mkdir()
    root = tmp_path / 'artifacts'
    monkeypatch.setenv('AIRA_DATA_DIR', str(root))
    app = create_app(Settings(database_url=f'sqlite:///{tmp_path / "api.db"}',
                              papers_dir=papers, llm_provider='openai'))
    client = TestClient(app)
    available = client.get('/retrieval-evals/datasets')
    assert available.status_code == 200
    assert available.json()['datasets'][0]['id'] == 'smoke'
    assert not available.json()['datasets'][1]['available']
    assert client.post('/retrieval-evals/jobs', json={
        'dataset': 'scifact', 'retrieval': RetrievalOptions().model_dump(),
    }).status_code == 422
    with patch('aira.retrieval_evaluation._benchmark_searcher',
               return_value=lambda q: [{'id': 'd2', 'score': 0.3}]):
        started = client.post('/retrieval-evals/jobs', json={
            'dataset': 'smoke', 'count': 3, 'seed': 123,
            'retrieval': RetrievalOptions().model_dump(),
        })
        assert started.status_code == 202
        job_id = started.json()['job_id']
        for _ in range(100):
            status = client.get(f'/retrieval-evals/jobs/{job_id}').json()
            if status['status'] in ('completed', 'failed', 'cancelled'):
                break
            time.sleep(.01)
        assert status['status'] == 'completed', status
    rid = status['run_id']
    full = client.get(f'/retrieval-evals/runs/{rid}')
    assert full.status_code == 200
    assert full.json()['sample_size'] == 3
    assert full.json()['dataset_fingerprint']
    assert len(client.get('/retrieval-evals/runs').json()['runs']) == 1
    assert client.get('/retrieval-evals/runs/not-a-uuid').status_code == 404
    # No personal documents or embeddings were created.
    assert client.get('/documents').json()['documents'] == []
