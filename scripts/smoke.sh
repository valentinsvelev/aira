#!/usr/bin/env bash
# Smoke-test the already-started local Compose stack.
set -euo pipefail

echo "Checking FastAPI..."
curl --fail --silent --show-error "http://127.0.0.1:8000/health"
echo

echo "Checking PostgreSQL and pgvector..."
echo "SELECT extversion FROM pg_extension WHERE extname = 'vector'" \
  | docker compose exec -T db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tA' | grep -q .
docker compose exec -T api aira retrieval-status
echo "PostgreSQL + pgvector operational"

echo "Checking Next.js..."
curl --fail --silent --show-error --output /dev/null "http://127.0.0.1:3000/"
echo "Next.js responds on http://127.0.0.1:3000/"

echo "Checking container-to-Ollama connection..."
docker compose exec -T api python -c '
import json, os, urllib.request
base = os.environ["OLLAMA_BASE_URL"].rstrip("/")
try:
    with urllib.request.urlopen(base + "/api/tags", timeout=6) as response:
        names = [model["name"] for model in json.load(response).get("models", [])]
    configured = os.environ["OLLAMA_MODEL"]
    print("Ollama reachable at:", base)
    print("Available models:", ", ".join(names) if names else "(none)")
    if not any(name == configured or name.startswith(configured + ":") for name in names):
        print("WARNING: configured model missing:", configured)
except Exception as exc:
    print("ERROR: API container cannot reach Ollama:", type(exc).__name__, str(exc))
    print("See docs/docker.md for native macOS or CPU-container setup.")
    raise SystemExit(1)
'
echo "Smoke test complete."
