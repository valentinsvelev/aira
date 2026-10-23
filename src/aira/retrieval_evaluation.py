"""Isolated, judged document-level retrieval benchmarking (NOT the production pgvector index).

Uses AIRA's LlamaIndex BM25, sentence-transformer embedding models, RRF rule and
optional reranker HTTP adapter against a fixed benchmark corpus. No LLM judge.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter
from uuid import uuid4

from aira.options import RetrievalOptions

_LOCK = Lock()
_JOBS: dict[str, dict] = {}
_CANCEL: dict[str, Event] = {}

# A tiny SYNTHETIC smoke test, never presented as an academic benchmark.
_DEMO = {
    'documents': {
        'd1': 'Low-rank adaptation LoRA trains small matrices while freezing a transformer backbone.',
        'd2': 'QLoRA uses 4-bit normal float NF4 and double quantization to reduce memory consumption.',
        'd3': 'Reciprocal rank fusion merges multiple ranked retrieval lists without score calibration.',
        'd4': 'BM25 is a lexical sparse retrieval model based on term frequency and document length.',
        'd5': 'Dense retrieval uses sentence embeddings to rank semantic similarity between queries and documents.',
        'd6': 'Cross-encoder rerankers jointly encode a query and candidate document to estimate relevance.',
        'd7': 'PostgreSQL pgvector supports vector distance search with approximate nearest neighbor indexes.',
        'd8': 'Scientific factuality evaluation can require gold evidence and document-level relevance judgments.',
    },
    'queries': {
        'q1': 'What does QLoRA use to save model memory?',
        'q2': 'Which ranking approach merges sparse and dense search results?',
        'q3': 'How does BM25 perform sparse retrieval?',
        'q4': 'What is a cross-encoder reranker?',
        'q5': 'How are embeddings stored and queried in PostgreSQL?',
    },
    'qrels': {'q1': {'d2': 1}, 'q2': {'d3': 1}, 'q3': {'d4': 1}, 'q4': {'d6': 1}, 'q5': {'d7': 1}},
}


def load_dataset(root: Path, name: str) -> dict:
    if name == 'smoke':
        return _DEMO
    if name != 'scifact':
        raise ValueError('Unknown retrieval dataset')
    base = Path(root) / 'retrieval_eval' / 'datasets' / 'scifact'
    corpus_file, query_file, qrels_file = base / 'corpus.jsonl', base / 'queries.jsonl', base / 'qrels' / 'test.tsv'
    if not all(p.is_file() for p in (corpus_file, query_file, qrels_file)):
        raise FileNotFoundError('SciFact is not installed. Run: python scripts/download_scifact.py')

    def read_jsonl(path: Path, is_corpus: bool) -> dict[str, str]:
        items = {}
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row['_id'])
                if key in items:
                    raise ValueError(f'Duplicate identifier in {path.name}: {key}')
                items[key] = (str(row.get('title') or '') + '\n' + str(row.get('text') or '')).strip() if is_corpus else str(row['text']).strip()
        return items

    corpus = read_jsonl(corpus_file, True)
    queries = read_jsonl(query_file, False)
    qrels: dict[str, dict[str, int]] = {}
    with qrels_file.open(encoding='utf-8') as stream:
        header = stream.readline().strip().split('\t')
        if header != ['query-id', 'corpus-id', 'score']:
            raise ValueError('Unsupported SciFact qrels header')
        for line in stream:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                raise ValueError('Malformed relevance judgment')
            qid, did, relevance = parts
            if int(relevance) > 0:
                qrels.setdefault(qid, {})[did] = int(relevance)
    if not corpus or not qrels or not all(q in queries for q in qrels):
        raise ValueError('Incomplete SciFact corpus/queries/relevance judgments')
    if any(d not in corpus for relevant in qrels.values() for d in relevant):
        raise ValueError('Relevance judgments refer to absent corpus documents')
    # SciFact test split only: no training/test label mixing.
    return {'documents': corpus, 'queries': {q: queries[q] for q in qrels}, 'qrels': qrels}


def dataset_status(root: Path) -> list[dict]:
    result = [{'id': 'smoke', 'name': 'Synthetic smoke test', 'available': True,
               'documents': len(_DEMO['documents']), 'queries': len(_DEMO['qrels']), 'official': False}]
    try:
        ds = load_dataset(root, 'scifact')
        result.append({'id': 'scifact', 'name': 'BEIR SciFact · test', 'available': True,
                       'documents': len(ds['documents']), 'queries': len(ds['qrels']), 'official': True})
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        result.append({'id': 'scifact', 'name': 'BEIR SciFact · test', 'available': False,
                       'documents': None, 'queries': None, 'official': True})
    return result


def query_metrics(ranked: list[str], relevance: dict[str, int], k: int) -> dict[str, float]:
    """Binary Recall/MRR/P and graded nDCG, with unjudged docs scored zero."""
    assert k >= 1
    visible = ranked[:k]
    positive = {doc for doc, rel in relevance.items() if rel > 0}
    hits = sum(doc in positive for doc in visible)
    rr = next((1.0 / i for i, doc in enumerate(visible, 1) if doc in positive), 0.0)
    gain = lambda rel: (2 ** max(0, rel) - 1)
    dcg = sum(gain(relevance.get(doc, 0)) / math.log2(pos + 2) for pos, doc in enumerate(visible))
    ideal = sorted((value for value in relevance.values() if value > 0), reverse=True)[:k]
    idcg = sum(gain(value) / math.log2(pos + 2) for pos, value in enumerate(ideal))
    return {'recall': hits / len(positive) if positive else 0.0,
            'mrr': rr, 'ndcg': dcg / idcg if idcg else 0.0, 'precision': hits / k}


def _percentile(values: list[float], percent: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    index = (len(sorted_values) - 1) * percent
    lo = int(index)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (index - lo)


def _benchmark_searcher(root: Path, data: dict, opts: RetrievalOptions):
    """Create isolated rankers for this corpus; production index remains untouched."""
    documents = data['documents']
    doc_ids = list(sorted(documents))
    lexical = None
    matrix = None
    np = None
    if opts.mode in ('sparse', 'hybrid'):
        from llama_index.core.schema import TextNode
        from llama_index.retrievers.bm25 import BM25Retriever
        nodes = [TextNode(id_=doc, text=documents[doc]) for doc in doc_ids]
        # Same BM25 retriever as LocalRetriever, with candidate limit at construction.
        lexical = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=min(len(nodes), opts.bm25_candidates))
    if opts.mode in ('dense', 'hybrid'):
        import numpy as numpy

        from aira.vector_store import embed_texts
        np = numpy
        # Cache only corpus embeddings; use content/model hashes to invalidate on change.
        corpus_hash = hashlib.sha256()
        for doc in doc_ids:
            corpus_hash.update(doc.encode()); corpus_hash.update(b'\0')
            corpus_hash.update(documents[doc].encode()); corpus_hash.update(b'\0')
        key = hashlib.sha256((opts.embedding_model + corpus_hash.hexdigest()).encode()).hexdigest()
        cache = root / 'retrieval_eval' / 'cache' / (key + '.npy')
        if cache.is_file():
            matrix = numpy.load(cache, allow_pickle=False)
            if matrix.shape != (len(doc_ids), 384 if 'MiniLM' in opts.embedding_model else 1024):
                raise ValueError('Cached embedding shape does not match corpus/model')
        else:
            # Large SciFact corpus is embedded once and cached for repeat experiments.
            batches = []
            for start in range(0, len(doc_ids), 32):
                batches.extend(embed_texts(opts.embedding_model, [documents[d] for d in doc_ids[start:start+32]]))
            matrix = numpy.asarray(batches, dtype=numpy.float32)
            cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=cache.parent, suffix='.npy', delete=False) as tmp:
                tmp_name = Path(tmp.name)
                numpy.save(tmp, matrix, allow_pickle=False)
            os.replace(tmp_name, cache)

    def rank(query: str) -> list[dict]:
        lex_hits = []
        dense_hits = []
        if lexical is not None:
            for item in lexical.retrieve(query):
                doc = item.node.node_id
                lex_hits.append({'id': doc, 'text': documents[doc][:1800],
                                 'score': float(item.score or 0), 'retrieval_method': 'sparse'})
        if matrix is not None:
            from aira.vector_store import embed_texts
            vec = np.asarray(embed_texts(opts.embedding_model, [query])[0], dtype=np.float32)
            similarities = matrix @ vec
            index = np.argsort(-similarities, kind='stable')
            for i in index:
                score = float(similarities[i])
                if score < opts.min_similarity:
                    break
                dense_hits.append({'id': doc_ids[i], 'text': documents[doc_ids[i]][:1800],
                                   'score': score, 'retrieval_method': 'dense'})
                if len(dense_hits) >= opts.dense_candidates:
                    break
        if opts.mode == 'sparse':
            ranked = lex_hits
        elif opts.mode == 'dense':
            ranked = dense_hits
        else:
            combined: dict[str, dict] = {}
            for stream in (lex_hits, dense_hits):
                for idx, hit in enumerate(stream, 1):
                    item = combined.setdefault(hit['id'], {**hit, 'score': 0.0,
                                                            'retrieval_method': 'hybrid'})
                    item['score'] += 1.0 / (opts.rrf_k + idx)
            ranked = sorted(combined.values(), key=lambda h: (-h['score'], h['id']))
        if opts.reranker != 'none':
            from aira.reranking import rerank_hits
            return rerank_hits(query, ranked[:opts.rerank_candidates], opts.reranker, opts.top_k)
        return ranked[:opts.top_k]

    return rank


def evaluate(root: Path, dataset: str, options: RetrievalOptions, *, count: int | None,
             seed: int, progress=None, cancel: Event | None = None) -> dict:
    data = load_dataset(root, dataset)
    # Fingerprint the complete judged corpus, queries and judgments to detect dataset drift.
    dataset_fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
    rng = random.Random(seed)
    qids = sorted(data['qrels'])
    if count is not None:
        qids = sorted(rng.sample(qids, min(len(qids), count)))
    if progress:
        progress(0, len(qids), 'Preparing isolated corpus / indexes')
    rank = _benchmark_searcher(root, data, options)
    cases = []
    times = []
    for i, qid in enumerate(qids):
        if cancel and cancel.is_set():
            raise InterruptedError('Evaluation cancelled')
        start = perf_counter()
        hits = rank(data['queries'][qid])
        latency = (perf_counter() - start) * 1000
        scores = query_metrics([hit['id'] for hit in hits], data['qrels'][qid], options.top_k)
        times.append(latency)
        cases.append({'query_id': qid, 'query': data['queries'][qid], 'metrics': scores,
                      'latency_ms': round(latency, 3), 'relevant_ids': sorted(data['qrels'][qid]),
                      'retrieved': [{'id': hit['id'], 'score': hit['score'],
                                     'first_stage_score': hit.get('first_stage_score')}
                                    for hit in hits]})
        if progress:
            progress(i+1, len(qids), 'Evaluating queries')
    metrics = {key: round(sum(case['metrics'][key] for case in cases) / len(cases), 6)
               for key in ('recall', 'mrr', 'ndcg', 'precision')}
    metrics.update({'p50_ms': round(_percentile(times, 0.50), 2),
                    'p95_ms': round(_percentile(times, 0.95), 2)})
    return {'dataset': dataset, 'dataset_fingerprint': dataset_fingerprint, 'corpus_size': len(data['documents']),
            'sample_size': len(qids), 'total_judged_queries': len(data['qrels']),
            'seed': seed, 'query_ids': qids, 'options': options.model_dump(),
            'metrics': metrics, 'cases': cases,
            'protocol': ('Synthetic smoke test; NOT a published benchmark.' if dataset == 'smoke'
                         else 'BEIR SciFact test judgments, document-level retrieval. Offline cosine scoring '
                         'uses the same embedding model but does NOT measure pgvector ANN index performance.'),
            'timing_note': 'Per-query retrieval plus reranking latency; corpus indexing/model initialization excluded.'}


def _save(root: Path, result: dict) -> dict:
    folder = root / 'retrieval_eval' / 'runs'
    folder.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid4())
    record = {'schema_version': 1, 'run_id': run_id,
              'created_at': datetime.now(UTC).isoformat(), **result}
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=folder,
                                     prefix='.run-', suffix='.tmp', delete=False) as stream:
        path = Path(stream.name)
        json.dump(record, stream, ensure_ascii=False, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.chmod(path, 0o600)
    os.replace(path, folder / (run_id + '.json'))
    return record


def list_runs(root: Path, limit: int = 30) -> list[dict]:
    rows = []
    for path in (root / 'retrieval_eval' / 'runs').glob('*.json'):
        try:
            item = json.loads(path.read_text(encoding='utf-8'))
            if item.get('schema_version') != 1 or item.get('run_id') != path.stem:
                continue
            rows.append({key: item[key] for key in ('run_id', 'created_at', 'dataset', 'sample_size',
                                                   'corpus_size', 'seed', 'options', 'metrics', 'protocol', 'dataset_fingerprint')})
        except (OSError, ValueError, KeyError):
            continue
    return sorted(rows, key=lambda r: r['created_at'], reverse=True)[:limit]


def read_run(root: Path, run_id: str) -> dict | None:
    from uuid import UUID
    try:
        safe_id = str(UUID(run_id))
    except (ValueError, TypeError):
        return None
    path = root / 'retrieval_eval' / 'runs' / f'{safe_id}.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def start_job(root: Path, dataset: str, options: RetrievalOptions, count: int | None, seed: int) -> str:
    # Explicit invocation only. No background polling outside job status.
    root = Path(root).resolve()
    # Validate dataset before making a thread, and bound concurrency.
    if dataset not in ('smoke', 'scifact'):
        raise ValueError('Unknown retrieval dataset')
    if dataset == 'scifact' and not dataset_status(root)[1]['available']:
        raise FileNotFoundError('SciFact dataset not installed')
    with _LOCK:
        if any(item['status'] in ('queued', 'running') for item in _JOBS.values()):
            raise RuntimeError('Another retrieval evaluation is running')
        job_id = str(uuid4())
        _JOBS[job_id] = {'job_id': job_id, 'status': 'queued', 'processed': 0, 'total': 0,
                         'phase': 'Queued', 'error': None, 'run_id': None}
        cancel = Event()
        _CANCEL[job_id] = cancel

    def update(**kwargs):
        with _LOCK:
            _JOBS[job_id].update(kwargs)

    def work():
        update(status='running', phase='Preparing dataset')
        try:
            result = evaluate(root, dataset, options, count=count, seed=seed,
                              progress=lambda done, total, phase: update(processed=done, total=total, phase=phase),
                              cancel=cancel)
            if cancel.is_set():
                update(status='cancelled', phase='Cancelled')
            else:
                record = _save(root, result)
                update(status='completed', run_id=record['run_id'], phase='Completed')
        except InterruptedError:
            update(status='cancelled', phase='Cancelled')
        except Exception as exc:
            update(status='failed', error=f'{type(exc).__name__}: {str(exc)[:350]}', phase='Failed')
        finally:
            with _LOCK:
                _CANCEL.pop(job_id, None)

    Thread(target=work, daemon=True, name='aira-retrieval-evaluation').start()
    return job_id


def job_status(job_id: str) -> dict | None:
    with _LOCK:
        return dict(_JOBS[job_id]) if job_id in _JOBS else None


def cancel_job(job_id: str) -> bool:
    with _LOCK:
        flag = _CANCEL.get(job_id)
        if flag is None:
            return False
        flag.set()
        return True
