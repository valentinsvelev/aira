#!/usr/bin/env bash
# Generate REAL dependency locks with a resolver. Requires internet ONCE when dependencies change.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/' >&2; exit 1; }
command -v npm >/dev/null || { echo 'Node 22 and npm are required' >&2; exit 1; }
# uv pip compile resolves transitives; compiling target Python 3.12 matches Docker runtime.
# This lock is intentionally the container/API subset, not macOS MPS extras.
uv pip compile --python-version 3.12 --generate-hashes requirements-api.txt -o requirements-api.lock.txt
(cd web && npm install --package-lock-only --ignore-scripts --no-audit --no-fund)
echo 'Resolved requirements-api.lock.txt and web/package-lock.json. Commit both files.'
