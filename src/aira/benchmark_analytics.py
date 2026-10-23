"""Rebuild objective DuckDB/Parquet benchmark analytics from local JSON runs.

No LLM judges. Repeatable rebuild from immutable per-run artifacts. DuckDB is
optional and is intentionally not required in the API's dev container.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_runs(data_dir: Path) -> tuple[list[tuple], list[tuple], list[tuple]]:
    runs, cases, metrics = [], [], []
    for path in sorted((data_dir / 'benchmarks' / 'runs').glob('*.json')):
        item = json.loads(path.read_text(encoding='utf-8'))
        if item.get('schema_version') != 1 or not isinstance(item.get('results'), list):
            raise ValueError(f'Unsupported benchmark artifact: {path.name}')
        rid = item['run_id']
        if path.stem != rid:
            raise ValueError(f'Mismatched run ID in {path.name}')
        if len(item['results']) != item['sample_size']:
            raise ValueError(f'Sample count mismatch in {path.name}')
        runs.append((rid, item['created_at'], item['dataset'], item['model'],
                     item['source'], int(item['sample_offset']), int(item['sample_size']),
                     item.get('protocol_note', ''),
                     item.get('configuration', {}).get('llm_provider', ''),
                     item.get('configuration', {}).get('retrieval_mode', ''),
                     item.get('configuration', {}).get('embedding_model', '')))
        for position, record in enumerate(item['results']):
            cases.append((rid, position, str(record['id']), record['question'],
                          record['prediction'], record['reference']))
            for name, value in record['metrics'].items():
                # Never pretend absent/invalid scores equal zero.
                if isinstance(value, bool):
                    number = float(value)
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    number = float(value)
                else:
                    raise ValueError(f'Invalid {name} in {path.name}')
                if not 0.0 <= number <= 1.0:
                    raise ValueError(f'Out-of-range {name} in {path.name}')
                metrics.append((rid, position, name, number))
    return runs, cases, metrics



def read_retrieval_runs(data_dir: Path) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """Read versioned, judged retrieval runs for reproducible DuckDB analysis."""
    runs, cases, hits = [], [], []
    for path in sorted((data_dir / 'retrieval_eval' / 'runs').glob('*.json')):
        item = json.loads(path.read_text(encoding='utf-8'))
        if item.get('schema_version') != 1 or item.get('run_id') != path.stem:
            raise ValueError(f'Invalid retrieval run artifact {path.name}')
        options = item['options']
        metrics = item['metrics']
        if item['sample_size'] != len(item['cases']):
            raise ValueError(f'Retrieval case count mismatch: {path.name}')
        runs.append((item['run_id'], item['created_at'], item['dataset'],
                     item['corpus_size'], item['sample_size'], item['seed'],
                     options['mode'], options['embedding_model'], options['reranker'],
                     options['top_k'], options['bm25_candidates'], options['dense_candidates'],
                     options['rerank_candidates'], item['protocol'],
                     *[float(metrics[m]) for m in ('recall', 'mrr', 'ndcg', 'precision', 'p50_ms', 'p95_ms')],
                     item.get('dataset_fingerprint', '')))
        for case in item['cases']:
            cm = case['metrics']
            cases.append((item['run_id'], case['query_id'], case['query'], case['latency_ms'],
                          *[float(cm[m]) for m in ('recall', 'mrr', 'ndcg', 'precision')]))
            relevant = set(case['relevant_ids'])
            for rank, hit in enumerate(case['retrieved'], 1):
                hits.append((item['run_id'], case['query_id'], rank, hit['id'],
                             float(hit['score']), hit.get('first_stage_score'), hit['id'] in relevant))
    return runs, cases, hits


def build_analytics(data_dir: Path) -> dict:
    """Rebuild from source JSON, create idempotent tables, exports and report."""
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError('DuckDB is optional. Install pip install "duckdb>=1.3,<2" or run the analytics Compose service.') from exc

    data_dir = Path(data_dir).resolve()
    runs, cases, metrics = read_runs(data_dir)
    output = data_dir / 'analytics'
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    connection = duckdb.connect(str(output / 'aira_analytics.duckdb'))
    try:
        connection.execute('''CREATE OR REPLACE TABLE benchmark_runs (
            run_id VARCHAR PRIMARY KEY, created_at VARCHAR NOT NULL, dataset VARCHAR NOT NULL,
            model VARCHAR NOT NULL, source VARCHAR, sample_offset INTEGER, sample_size INTEGER,
            protocol_note VARCHAR, llm_provider VARCHAR,
            retrieval_mode VARCHAR, embedding_model VARCHAR)''')
        connection.execute('''CREATE OR REPLACE TABLE benchmark_cases (
            run_id VARCHAR, row_number INTEGER, example_id VARCHAR,
            question VARCHAR, prediction VARCHAR, reference VARCHAR,
            PRIMARY KEY (run_id, row_number))''')
        connection.execute('''CREATE OR REPLACE TABLE benchmark_case_metrics (
            run_id VARCHAR, row_number INTEGER, metric VARCHAR, score DOUBLE,
            PRIMARY KEY (run_id, row_number, metric))''')
        if runs:
            connection.executemany('INSERT INTO benchmark_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', runs)
        if cases:
            connection.executemany('INSERT INTO benchmark_cases VALUES (?, ?, ?, ?, ?, ?)', cases)
        if metrics:
            connection.executemany('INSERT INTO benchmark_case_metrics VALUES (?, ?, ?, ?)', metrics)
        connection.execute('''CREATE OR REPLACE VIEW benchmark_summary AS
            SELECT r.dataset, r.model, r.retrieval_mode, r.embedding_model, m.metric,
                   COUNT(DISTINCT r.run_id) AS runs,
                   COUNT(*) AS evaluated_cases, ROUND(AVG(m.score), 4) AS mean_score
            FROM benchmark_case_metrics m JOIN benchmark_runs r USING(run_id)
            GROUP BY r.dataset, r.model, r.retrieval_mode, r.embedding_model, m.metric''')
        # Retrieval evaluation: document-level qrels, configurations and rankings.
        # Separate tables avoid confusing retrieval quality with answer-generation scores.
        retrieval_runs, retrieval_cases, retrieval_hits = read_retrieval_runs(data_dir)
        connection.execute('''CREATE OR REPLACE TABLE retrieval_eval_runs (
            run_id VARCHAR PRIMARY KEY, created_at VARCHAR, dataset VARCHAR,
            corpus_size INTEGER, sample_size INTEGER, seed INTEGER,
            mode VARCHAR, embedding_model VARCHAR, reranker VARCHAR,
            top_k INTEGER, bm25_candidates INTEGER, dense_candidates INTEGER,
            rerank_candidates INTEGER, protocol VARCHAR,
            recall DOUBLE, mrr DOUBLE, ndcg DOUBLE, precision DOUBLE,
            p50_ms DOUBLE, p95_ms DOUBLE, dataset_fingerprint VARCHAR)''')
        connection.execute('''CREATE OR REPLACE TABLE retrieval_eval_cases (
            run_id VARCHAR, query_id VARCHAR, query VARCHAR, latency_ms DOUBLE,
            recall DOUBLE, mrr DOUBLE, ndcg DOUBLE, precision DOUBLE,
            PRIMARY KEY (run_id, query_id))''')
        connection.execute('''CREATE OR REPLACE TABLE retrieval_eval_hits (
            run_id VARCHAR, query_id VARCHAR, rank INTEGER, document_id VARCHAR,
            score DOUBLE, first_stage_score DOUBLE, relevant BOOLEAN,
            PRIMARY KEY (run_id, query_id, rank))''')
        if retrieval_runs:
            connection.executemany('INSERT INTO retrieval_eval_runs VALUES (' + ','.join('?' * 21) + ')', retrieval_runs)
        if retrieval_cases:
            connection.executemany('INSERT INTO retrieval_eval_cases VALUES (' + ','.join('?' * 8) + ')', retrieval_cases)
        if retrieval_hits:
            connection.executemany('INSERT INTO retrieval_eval_hits VALUES (' + ','.join('?' * 7) + ')', retrieval_hits)
        connection.execute('''CREATE OR REPLACE VIEW retrieval_eval_summary AS
            SELECT dataset, mode, embedding_model, reranker, top_k, seed, sample_size, dataset_fingerprint,
                   COUNT(*) AS runs, AVG(recall) AS recall, AVG(mrr) AS mrr,
                   AVG(ndcg) AS ndcg, AVG(precision) AS precision,
                   AVG(p50_ms) AS p50_ms, AVG(p95_ms) AS p95_ms
            FROM retrieval_eval_runs
            GROUP BY dataset, mode, embedding_model, reranker, top_k, seed, sample_size, dataset_fingerprint''')
        # Parquet snapshots for SQL notebooks / downstream tools.
        for table in ('benchmark_runs', 'benchmark_cases', 'benchmark_case_metrics', 'benchmark_summary', 'retrieval_eval_runs',
                      'retrieval_eval_cases', 'retrieval_eval_hits', 'retrieval_eval_summary'):
            # Table names fixed in code (never user-controlled).
            filename = output / f'{table}.parquet'
            temporary = output / f'.{table}.parquet.tmp'
            temporary.unlink(missing_ok=True)
            escaped = str(temporary).replace("'", "''")
            connection.execute(f"COPY (SELECT * FROM {table}) TO '{escaped}' (FORMAT PARQUET)")
            temporary.replace(filename)
        row_dicts = connection.execute('''SELECT dataset, model, retrieval_mode, embedding_model, metric, runs, evaluated_cases, mean_score
            FROM benchmark_summary ORDER BY dataset, model, retrieval_mode, embedding_model, metric''').fetchall()
    finally:
        connection.close()
    summary = [dict(zip(('dataset', 'model', 'retrieval_mode', 'embedding_model', 'metric', 'runs', 'evaluated_cases', 'mean_score'), row))
               for row in row_dicts]
    return {'runs': len(runs), 'cases': len(cases), 'individual_metrics': len(metrics),
            'analytics_dir': str(output), 'summary': summary,
            'retrieval_runs': len(retrieval_runs), 'retrieval_cases': len(retrieval_cases),
            'retrieval_hits': len(retrieval_hits)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Rebuild AIRA DuckDB + Parquet benchmark analytics')
    parser.add_argument('--data-dir', type=Path, default=Path('.data'))
    args = parser.parse_args(argv)
    print(json.dumps(build_analytics(args.data_dir), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
