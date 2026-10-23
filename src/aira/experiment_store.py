"""Local append-only benchmark artifacts, independent of optional DuckDB.

The API can persist runs without installing DuckDB in the serving container.
DuckDB analytics import these immutable JSON artifacts on demand.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def save_experiment(root: Path, *, dataset: str, provenance: str, sample_offset: int,
                    model: str, result: dict, configuration: dict | None = None) -> dict:
    """Atomically persist one completed run and return its public metadata.

    The UUID is generated on the server; no user-controlled filenames or paths.
    Predictions and questions remain on disk under the local-only .data directory.
    """
    if result['sample_size'] != len(result['results']):
        raise ValueError('Benchmark run sample size does not match per-case results')
    directory = Path(root).resolve() / 'benchmarks' / 'runs'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_id = str(uuid4())
    timestamp = datetime.now(UTC).isoformat()
    record = {
        'schema_version': 1,
        'run_id': run_id,
        'created_at': timestamp,
        'dataset': dataset,
        'source': provenance,
        'sample_offset': sample_offset,
        'model': model,
        'configuration': configuration or {},
        'sample_size': result['sample_size'],
        'metrics': result['metrics'],
        'results': result['results'],
        'protocol_note': result['protocol_note'],
    }
    target = directory / (run_id + '.json')
    # File is written within the same directory so os.replace is atomic.
    fd, name = tempfile.mkstemp(prefix='.run-', suffix='.tmp', dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            json.dump(record, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(name, 0o600)
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return {'run_id': run_id, 'created_at': timestamp}
