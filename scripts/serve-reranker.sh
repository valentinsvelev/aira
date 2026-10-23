#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Run natively on the host. On macOS, torch can use MPS (Metal).
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m uvicorn aira.reranker_service:app --host "${RERANKER_HOST:-127.0.0.1}" --port "${RERANKER_PORT:-8010}"
