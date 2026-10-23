"""Sparse, dense and hybrid retrieval, request options and separate embedding indexes."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from aira.api import create_app
from aira.config import Settings
from aira.options import GenerationOptions, RetrievalOptions
from aira.retrieval import LocalRetriever
from aira.vector_store import GTE_LARGE, MINILM, model_spec


@pytest.mark.integration
def test_llamaindex_bm25_can_retrieve_source(workspace):
    pytest.importorskip("llama_index.retrievers.bm25")
    (workspace["papers"] / "fixture.txt").write_text(
        "Context editing must preserve source provenance and immutable evidence. " * 10)
    workspace["ingestor"].sync()
    retriever = LocalRetriever(workspace["factory"])
    matches = retriever.search("immutable evidence context provenance")
    assert matches
    assert matches[0]["path"] == "fixture.txt"
    assert matches[0]["id"]
    assert matches[0]["page"] == 1


def test_two_page_pdf_is_extractable(workspace):
    import shutil
    from pathlib import Path
    fixture = Path(__file__).parent / "fixtures" / "synthetic_paper.pdf"
    shutil.copyfile(fixture, workspace["papers"] / "synthetic_paper.pdf")
    result = workspace["ingestor"].sync()
    assert result["summary"] == {"ingested": 1}
    from sqlalchemy import select

    from aira.db import EvidenceChunk
    with workspace["factory"]() as db:
        chunks = db.scalars(select(EvidenceChunk).order_by(EvidenceChunk.page)).all()
        assert {chunk.page for chunk in chunks} == {1, 2}
        assert all(chunk.source_kind == "pdf" for chunk in chunks)


def test_options_are_bounded_and_modes_explicit():
    assert RetrievalOptions(mode='hybrid').top_k == 5
    assert GenerationOptions().num_ctx == 0
    with pytest.raises(ValidationError):
        RetrievalOptions(mode='bm25', top_k=20)
    with pytest.raises(ValidationError):
        RetrievalOptions(mode='dense', dense_candidates=2, top_k=5)
    with pytest.raises(ValidationError):
        GenerationOptions(temperature=2.0)
    with pytest.raises(ValidationError):
        GenerationOptions(num_predict=0)


def test_hybrid_rrf_merges_sources_and_stable_ranks(workspace, monkeypatch):
    retriever = LocalRetriever(workspace['factory'])
    monkeypatch.setattr(retriever, '_bm25_search', lambda q, n: [
        {'id': 'a', 'text': 'A', 'score': 6.0, 'retrieval_method': 'bm25'},
        {'id': 'b', 'text': 'B', 'score': 5.0, 'retrieval_method': 'bm25'},
    ])
    monkeypatch.setattr(retriever.index, 'dense_search', lambda q, n, threshold: [
        {'id': 'b', 'text': 'B', 'score': .92, 'retrieval_method': 'dense'},
        {'id': 'c', 'text': 'C', 'score': .90, 'retrieval_method': 'dense'},
    ])
    result = retriever.search('query', options=RetrievalOptions(mode='hybrid', top_k=3))
    assert [h['id'] for h in result] == ['b', 'a', 'c']
    assert all(h['retrieval_method'] == 'hybrid' for h in result)
    assert result[0]['score'] == pytest.approx((1 / 62) + (1 / 61))


def test_dense_mode_does_not_call_bm25(workspace, monkeypatch):
    retriever = LocalRetriever(workspace['factory'])
    monkeypatch.setattr(retriever, '_bm25_search', lambda *a: pytest.fail('BM25 called'))
    monkeypatch.setattr(retriever.index, 'dense_search', lambda q, n, s: [
        {'id': 'dense', 'score': .8, 'retrieval_method': 'dense'}
    ])
    result = retriever.search('quantization', options=RetrievalOptions(mode='dense'))
    assert result[0]['id'] == 'dense'


def test_sqlite_vector_errors_are_explicit(workspace):
    retriever = LocalRetriever(workspace['factory'])
    with pytest.raises(RuntimeError, match='requires PostgreSQL'):
        retriever.search('some question', options=RetrievalOptions(mode='dense'))


def test_sparse_is_canonical_and_legacy_still_loads():
    assert RetrievalOptions().mode == "sparse"
    assert RetrievalOptions(mode="bm25").mode == "sparse"
    assert RetrievalOptions(mode="bm25").model_dump()["mode"] == "sparse"
    assert RetrievalOptions(mode="hybrid").mode == "hybrid"


def test_legacy_env_is_migrated_without_data_migration(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    assert Settings.from_env().retrieval_mode == "sparse"


def test_embedding_models_are_not_mixed():
    assert model_spec(MINILM) == ('chunk_embeddings', 384)
    assert model_spec(GTE_LARGE) == ('chunk_embeddings_1024', 1024)


def test_retrieval_status_and_model_validation(workspace, tmp_path):
    app = create_app(Settings(database_url=f'sqlite:///{tmp_path / "retrieval.db"}',
                              papers_dir=workspace['papers']))
    client = TestClient(app)
    assert client.get('/retrieval/status', params={'embedding_model': GTE_LARGE}).json()['dimensions'] == 1024
    assert client.get('/retrieval/status', params={'embedding_model': 'unknown'}).status_code == 422
    assert client.post('/retrieval/reindex', json={'embedding_model': 'unknown'}).status_code == 422
