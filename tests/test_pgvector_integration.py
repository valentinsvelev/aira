"""Run with TEST_POSTGRES_URL against a disposable empty pgvector database."""
import os

import pytest
from sqlalchemy import select

from aira.context import ContextManager
from aira.db import ContextEvent, ContextNote, EvidenceChunk, make_session_factory
from aira.ingestion import Ingestor
from aira.migrate import migrate
from aira.options import RetrievalOptions
from aira.retrieval import LocalRetriever
from aira.vector_store import DIMENSIONS, GTE_LARGE, EmbeddingIndex


@pytest.mark.integration
def test_sqlite_migration_and_pgvector_retrieval(tmp_path, monkeypatch):
    uri = os.getenv('TEST_POSTGRES_URL')
    if not uri:
        pytest.skip('Set TEST_POSTGRES_URL to run PostgreSQL/pgvector integration')
    # Source fixtures represent an existing SQLite corpus. PG must be empty.
    sqlite_path = tmp_path / 'previous.db'
    source = make_session_factory(f'sqlite:///{sqlite_path}')
    papers = tmp_path / 'papers'
    papers.mkdir()
    (papers / 'nf4.txt').write_text('NF4 quantization reduces memory footprint. ' * 5)
    assert Ingestor(source, papers).sync()['summary']['ingested'] == 1
    with source() as db:
        evidence_id = db.scalars(select(EvidenceChunk.id)).first()
    claim = 'NF4 quantization reduces memory footprint.'
    proposed = ContextManager(source).propose('migration-session', claim, claim, evidence_id)
    assert proposed['accepted']
    migrated = migrate(sqlite_path, uri)
    assert migrated['documents'] == 1
    assert migrated['evidence_chunks'] >= 1
    assert migrated['context_notes'] == 1
    assert migrated['context_events'] == 1
    factory = make_session_factory(uri)
    with factory() as db:
        assert db.get(EvidenceChunk, evidence_id) is not None
        note = db.scalars(select(ContextNote)).first()
        event = db.scalars(select(ContextEvent)).first()
        assert note.evidence_id == evidence_id
        assert note.id == proposed['note_id']
        assert event.id == proposed['event_id']

    # Validate real vector column, pgvector cosine operator, and hybrid fusion
    # without downloading a model in CI.
    def fake_embed(name, texts):
        return [[1.0] + [0.0] * (DIMENSIONS - 1) for _ in texts]
    monkeypatch.setattr('aira.vector_store.embed_texts', fake_embed)
    model = 'integration-test-local-model'
    indexer = EmbeddingIndex(factory, model)
    stats = indexer.sync()
    assert stats['enabled'] and stats['indexed'] >= 1
    assert stats['missing'] == 0
    results = indexer.dense_search('reduce memory', 5)
    assert results and results[0]['path'] == 'nf4.txt'
    combined = LocalRetriever(factory, model)
    hits = combined.search('quantization', options=RetrievalOptions(mode='hybrid'))
    assert hits and hits[0]['retrieval_method'] == 'hybrid'


@pytest.mark.integration
def test_gte_1024_vectors_use_separate_index(monkeypatch):
    """Run on disposable TEST_POSTGRES_URL populated by the migration test."""
    uri = os.getenv('TEST_POSTGRES_URL')
    if not uri:
        pytest.skip('Set TEST_POSTGRES_URL to run PostgreSQL/pgvector integration')
    factory = make_session_factory(uri)
    def fake_gte(model, texts):
        assert model == GTE_LARGE
        return [[1.0] + [0.0] * 1023 for _ in texts]
    monkeypatch.setattr('aira.vector_store.embed_texts', fake_gte)
    gte = EmbeddingIndex(factory, GTE_LARGE)
    info = gte.sync()
    assert info['enabled'] and info['dimensions'] == 1024 and info['indexed'] > 0
    hits = gte.dense_search('quantization', 5)
    assert hits and hits[0]['id']
    minilm = EmbeddingIndex(factory, 'integration-test-local-model')
    assert minilm.stats()['dimensions'] == 384
    assert gte.table != minilm.table
