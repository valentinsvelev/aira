"""Manually triggered AIRA PDF -> pgvector pipeline (Airflow 3).

Uses the application's existing ingestion API, not a second implementation of
PDF extraction, SQL mutations or embedding generation. Nothing is scheduled by
default, and each stage is retriable/idempotent.
"""
from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from airflow.sdk import dag, task

API_URL = os.getenv('AIRA_API_URL', 'http://api:8000').rstrip('/')
EMBEDDING_MODEL = os.getenv('AIRA_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
SCHEDULE = os.getenv('AIRA_INGEST_SCHEDULE', '').strip() or None  # opt-in only


def request_json(method: str, path: str, payload: dict | None = None, timeout: int = 900) -> dict:
    """Only hardcoded application endpoints are used; Airflow input cannot set URLs."""
    req = Request(API_URL + path, method=method,
                  data=json.dumps(payload).encode('utf-8') if payload is not None else None,
                  headers={'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=timeout) as response:
            data = json.load(response)
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError(f'AIRA API {method} {path} unavailable: {type(exc).__name__}') from exc
    if not isinstance(data, dict):
        raise ValueError('AIRA returned an unexpected response')
    return data


@dag(dag_id='aira_document_ingestion', description='Versioned PDFs -> chunks -> pgvector -> quality checks',
     start_date=datetime(2025, 1, 1, tzinfo=UTC), schedule=SCHEDULE,
     catchup=False, max_active_runs=1, tags=['aira', 'data-engineering', 'ingestion'])
def aira_ingestion_pipeline():
    @task(retries=2, retry_delay=timedelta(minutes=2))
    def check_api() -> bool:
        result = request_json('GET', '/health', timeout=20)
        if result.get('status') != 'ok':
            raise RuntimeError('AIRA backend health check failed')
        return True

    @task(retries=1, retry_delay=timedelta(minutes=2))
    def ingest_documents(_healthy: bool) -> dict:
        # API default still auto-indexes; Airflow explicitly decouples stages.
        result = request_json('POST', '/ingest/sync?index_vectors=false')
        if result.get('summary', {}).get('failed', 0):
            raise RuntimeError(f'PDF extraction failed for {result["summary"]["failed"]} documents; inspect API ingestion_runs')
        return result.get('summary', {})

    @task(retries=2, retry_delay=timedelta(minutes=3))
    def index_embeddings(_summary: dict) -> dict:
        result = request_json('POST', '/retrieval/reindex', {'embedding_model': EMBEDDING_MODEL})
        if not result.get('enabled', True) or result.get('missing', 0) != 0:
            raise RuntimeError('Indexing did not complete for all active chunks')
        return {'indexed': result.get('indexed'), 'embedded': result.get('embedded')}

    @task(retries=1)
    def check_index(_indexed: dict) -> dict:
        from urllib.parse import urlencode
        data = request_json('GET', '/retrieval/status?' + urlencode({'embedding_model': EMBEDDING_MODEL}), timeout=60)
        if not data.get('enabled') or data.get('missing') != 0:
            raise RuntimeError(f'pgvector data-quality check failed: {data.get("missing")} active chunks without embeddings')
        return {'indexed': data['indexed'], 'missing': data['missing'], 'model': data['model']}

    check_index(index_embeddings(ingest_documents(check_api())))


aira_ingestion_pipeline()
