"""Explicit, local background benchmark runs.

No scheduled jobs or implicit evaluation. One full run at a time per API process.
This dev-mode in-memory registry is intentionally not durable across server restarts.
Run outputs can optionally be saved as immutable JSON artifacts for DuckDB.
"""
from __future__ import annotations

import os
from pathlib import Path
from threading import Event, Lock, Thread
from uuid import uuid4

from aira import benchmarks
from aira.experiment_store import save_experiment

_LOCK = Lock()
_JOBS: dict[str, dict] = {}
_CANCEL: dict[str, Event] = {}


def run_all_job(*, dataset: str, examples: list[dict], settings, retriever, save_results: bool) -> str:
    with _LOCK:
        if any(job['status'] in ('queued', 'running') for job in _JOBS.values()):
            raise RuntimeError('A full benchmark run is already active in this API process.')
        run_id = str(uuid4())
        _CANCEL[run_id] = Event()
        _JOBS[run_id] = {'id': run_id, 'status': 'queued', 'processed': 0,
                         'total': len(examples) if dataset == 'custom' else None,
                         'result': None, 'error': None}

    def worker():
        with _LOCK:
            _JOBS[run_id]['status'] = 'running'
        all_predictions: list[dict] = []
        try:
            if dataset == 'custom':
                cases = [{'id': str(i + 1), 'question': row['question'],
                          'context': row.get('context', ''), 'gold': row['answer']}
                         for i, row in enumerate(examples)]
                all_cases = cases
                batches = ((pos, len(all_cases), all_cases[pos:pos + 10])
                           for pos in range(0, len(all_cases), 10))
                provenance = f'Entire local dataset · {len(cases)} labeled examples'
            else:
                batches = benchmarks.iter_all_public_cases(dataset)
                provenance = f'Full {dataset.upper()} selected Hugging Face split'
            if not (dataset != 'custom' or examples):
                raise ValueError('No labeled examples provided')
            for offset, total_rows, cases in batches:
                if _CANCEL[run_id].is_set():
                    with _LOCK:
                        _JOBS[run_id].update(status='cancelled',
                                             error=f'Cancelled after {len(all_predictions)} evaluated examples.')
                    return
                with _LOCK:
                    _JOBS[run_id]['total'] = total_rows if dataset == 'custom' else None
                    _JOBS[run_id]['rows_read'] = offset
                    _JOBS[run_id]['source_rows_total'] = total_rows
                if cases:
                    # run_cases uses the same model and deterministic metrics as small runs.
                    part = benchmarks.run_cases(dataset, cases, settings, retriever=retriever)
                    all_predictions.extend(part['results'])
                with _LOCK:
                    _JOBS[run_id]['processed'] = len(all_predictions)
                    _JOBS[run_id]['rows_read'] = min(total_rows, offset + (10 if dataset == 'custom' else 20))
            if not all_predictions:
                raise ValueError('The selected split had no labeled cases')
            keys = set().union(*(row['metrics'].keys() for row in all_predictions))
            labels = {'answer_f1': 'answer_token_f1',
                      'relevance_correct': 'relevance_accuracy', 'stance_correct': 'stance_accuracy'}
            metrics = {labels.get(key, key): round(sum(float(row['metrics'].get(key, 0))
                       for row in all_predictions) / len(all_predictions), 4) for key in keys}
            result = {'dataset': dataset, 'source': provenance, 'sample_size': len(all_predictions),
                      'metrics': metrics, 'results': all_predictions,
                      'protocol_note': ('Exhaustive evaluation of the selected rows/split with available labels. '
                                        'Scores remain diagnostics, not official leaderboard scores.')}
            if save_results:
                try:
                    result.update(save_experiment(Path(os.getenv('AIRA_DATA_DIR', '.data')),
                        dataset=dataset, provenance=provenance, sample_offset=0,
                        model=settings.ollama_model if settings.llm_provider == 'ollama' else settings.openai_model,
                        result=result, configuration={'sampling': 'all',
                                                      'generation_temperature': 0,
                                                      'generation_num_predict': 256}))
                except (OSError, ValueError, TypeError) as exc:
                    result['save_error'] = f'Finished but could not save ({type(exc).__name__})'
            with _LOCK:
                _JOBS[run_id].update(status='completed', result=result)
        except Exception as exc:
            with _LOCK:
                _JOBS[run_id].update(status='failed', error=(
                    f'Full run stopped after {len(all_predictions)} evaluated examples: '
                    f'{type(exc).__name__}. Check API logs, model connectivity and dataset access.'))

    Thread(target=worker, daemon=True, name=f'aira-benchmark-{run_id[:8]}').start()
    return run_id


def get_job(run_id: str) -> dict | None:
    with _LOCK:
        job = _JOBS.get(run_id)
        return dict(job) if job else None


def cancel_job(run_id: str) -> bool:
    with _LOCK:
        job = _JOBS.get(run_id)
        if job is None:
            return False
        if job['status'] in ('queued', 'running'):
            _CANCEL[run_id].set()
        return True
