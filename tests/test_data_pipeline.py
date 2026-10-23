"""Airflow ingestion DAG and DuckDB benchmark analytics; no Docker or GPU required."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aira.api import create_app
from aira.benchmark_analytics import read_runs
from aira.config import Settings
from aira.experiment_store import save_experiment


def example_result():
    return {
        'sample_size': 1, 'metrics': {'answer_token_f1': .5},
        'results': [{'id': '1', 'question': 'Q?', 'prediction': 'A', 'reference': 'B',
                     'metrics': {'answer_f1': .5}}],
        'protocol_note': 'Diagnostic only',
    }


def test_artifacts_are_immutable_and_readable_without_duckdb(tmp_path):
    first = save_experiment(tmp_path, dataset='custom', provenance='example', sample_offset=3,
                            model='qwen', result=example_result())
    second = save_experiment(tmp_path, dataset='custom', provenance='example', sample_offset=3,
                             model='qwen', result=example_result())
    assert first['run_id'] != second['run_id']
    runs, cases, metrics = read_runs(tmp_path)
    assert len(runs) == len(cases) == len(metrics) == 2
    assert runs[0][2] == 'custom' and metrics[0][-1] == .5
    file = tmp_path/'benchmarks'/'runs'/f"{first['run_id']}.json"
    assert file.exists() and json.loads(file.read_text())['run_id'] == first['run_id']


def test_artifact_rejects_inconsistent_runs_and_invalid_scores(tmp_path):
    record = example_result()
    record['sample_size'] = 2
    with pytest.raises(ValueError, match='sample size'):
        save_experiment(tmp_path, dataset='custom', provenance='x', sample_offset=0, model='qwen', result=record)
    record = example_result()
    record['results'][0]['metrics']['answer_f1'] = 2.5
    save_experiment(tmp_path, dataset='custom', provenance='x', sample_offset=0, model='qwen', result=record)
    with pytest.raises(ValueError, match='Out-of-range'):
        read_runs(tmp_path)


def test_benchmark_api_persist_opt_in_and_airflow_staged_ingest(tmp_path, monkeypatch):
    from aira import benchmarks
    papers = tmp_path/'papers'
    papers.mkdir()
    (papers/'a.txt').write_text('A small example source about evidence and retrieval. ' * 15)
    client = TestClient(create_app(Settings(database_url=f'sqlite:///{tmp_path / "pipeline.db"}',
                          papers_dir=papers, llm_provider='openai', faithfulness_mode='extractive')))
    monkeypatch.setenv('AIRA_DATA_DIR', str(tmp_path/'local_artifacts'))
    monkeypatch.setattr(benchmarks, 'run_cases', lambda *a, **kw: example_result())
    payload = {'dataset': 'custom', 'count': 1, 'examples': [
        {'question': 'Q?', 'answer': 'B', 'context': 'example'}]}
    saved = client.post('/benchmarks/run', json=payload)
    assert saved.status_code == 200 and saved.json().get('run_id')
    assert len(list((tmp_path/'local_artifacts'/'benchmarks'/'runs').glob('*.json'))) == 1
    not_saved = client.post('/benchmarks/run', json={**payload, 'save_results': False})
    assert not_saved.status_code == 200 and 'run_id' not in not_saved.json()
    response = client.post('/ingest/sync?index_vectors=false')
    assert response.status_code == 200
    assert response.json()['summary']['ingested'] == 1
    assert client.post('/ingest/sync?index_vectors=false').json()['summary']['skipped'] == 1


def test_optional_compose_files_are_isolated_and_localhost_only():
    root = Path(__file__).resolve().parents[1]
    airflow = (root/'compose.airflow.yml').read_text()
    analytics = (root/'compose.analytics.yml').read_text()
    assert 'profiles: ["airflow"]' in airflow
    assert 'profiles: ["analytics"]' in analytics
    assert '127.0.0.1:8080:8080' in airflow
    assert 'AIRFLOW__CORE__EXECUTOR: LocalExecutor' in airflow
    assert 'restart: "no"' in analytics


def test_duckdb_end_to_end_if_installed(tmp_path):
    pytest.importorskip('duckdb')
    from aira.benchmark_analytics import build_analytics
    save_experiment(tmp_path, dataset='custom', provenance='example', sample_offset=0,
                    model='qwen', result=example_result())
    first = build_analytics(tmp_path)
    second = build_analytics(tmp_path)
    assert first['runs'] == second['runs'] == 1
    assert first['summary'][0]['mean_score'] == .5
    assert (tmp_path/'analytics'/'benchmark_summary.parquet').exists()
