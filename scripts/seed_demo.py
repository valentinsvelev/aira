"""Repeatable seed for the intentionally isolated AIRA demo stack.

Copies synthetic fixtures to demo/workspace only and accesses loopback port 8100.
No model server, cloud services or personal project storage are required.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEMO_NAME = 'Demo · evidence workflow'


def request(base_url: str, method: str, path: str, data: dict | None = None) -> dict:
    payload = None if data is None else json.dumps(data).encode('utf-8')
    with urlopen(Request(base_url + path, data=payload, method=method,
                         headers={'Content-Type': 'application/json'} if payload else {}), timeout=12) as res:
        return json.loads(res.read())


def check_safe_demo_url(url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1')
            or parsed.port != 8100 or parsed.path not in ('', '/') or parsed.username):
        raise ValueError('Demo seeding is restricted to http://127.0.0.1:8100')


def seed(base_url: str, *, send=request, root: Path = ROOT, run_evaluation: bool = True) -> dict:
    check_safe_demo_url(base_url)
    # Wait for containers to become ready; report a useful error when unavailable.
    for attempt in range(35):
        try:
            if send(base_url, 'GET', '/ready')['status'] == 'ready':
                break
        except (OSError, KeyError, ValueError) as exc:
            if attempt == 34:
                raise RuntimeError('Demo API on port 8100 is not ready; inspect: docker compose -f compose.demo.yml logs api') from exc
            time.sleep(2)
    source = root / 'demo' / 'fixtures'
    target = root / 'demo' / 'workspace' / 'papers'
    target.mkdir(parents=True, exist_ok=True)
    for src in sorted(source.glob('*.txt')):
        dst = target / src.name
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            shutil.copyfile(src, dst)
    ingest = send(base_url, 'POST', '/ingest/sync?index_vectors=false')
    all_agents = send(base_url, 'GET', '/agents')['agents']
    found = next((a for a in all_agents if a.get('name') == DEMO_NAME), None)
    if found is None:
        found = send(base_url, 'POST', '/agents', {
            'name': DEMO_NAME,
            'description': 'Sample executable retrieval graph with synthetic demo documents',
            'system_instructions': 'Answer using retrieved evidence and cite sources. Demo documents are synthetic.',
            'enabled_tools': ['local_search'],
        })
    agent_id = found['id']
    graph = {'schema_version': 1, 'enabled': True, 'parallel': False,
             'nodes': [{'id': kind, 'type': kind, 'x': 80 + i * 240, 'y': 170}
                       for i, kind in enumerate(['question', 'local_search', 'synthesize', 'answer'])],
             'edges': [{'source': a, 'target': b} for a,b in [
                 ('question', 'local_search'), ('local_search', 'synthesize'), ('synthesize', 'answer')]]}
    send(base_url, 'PUT', f'/agents/{agent_id}/workflow', graph)
    result = {'ingestion': ingest, 'agent_id': agent_id}
    if run_evaluation:
        started = send(base_url, 'POST', '/retrieval-evals/jobs', {
            'dataset': 'smoke', 'count': 3, 'seed': 2026,
            'retrieval': {'mode': 'sparse', 'reranker': 'none', 'top_k': 5},
        })
        job_id = started['job_id']
        for _ in range(100):
            status = send(base_url, 'GET', f'/retrieval-evals/jobs/{job_id}')
            if status['status'] in {'completed', 'failed', 'cancelled'}:
                result['evaluation'] = status
                if status['status'] != 'completed':
                    raise RuntimeError(f'Demo retrieval evaluation did not complete: {status}')
                break
            time.sleep(.25)
        else:
            raise RuntimeError('Demo evaluation did not finish. View logs for retrieval errors.')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8100')
    parser.add_argument('--no-evaluation', action='store_true')
    args = parser.parse_args()
    outcome = seed(args.url.rstrip('/'), run_evaluation=not args.no_evaluation)
    print('Demo ready: http://localhost:3100')
    print('Synthetic evidence docs indexed; agent:', outcome['agent_id'])
    if 'evaluation' in outcome:
        print('Synthetic retrieval run:', outcome['evaluation']['run_id'])
    print('To execute a chat workflow, install the selected Ollama model separately.')

if __name__ == '__main__':
    main()
