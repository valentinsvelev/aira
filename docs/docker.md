# Docker setup

The default stack runs three containers and uses Ollama on your machine:

```text
Browser ── localhost:3000 ── web (Next.js)
                                │
                     localhost:8000 ── api (FastAPI) ──► Ollama on the host
                                │
                               db (PostgreSQL 16 + pgvector, not published)
```

Ports are bound to `127.0.0.1`, so nothing is reachable from your network.
PDFs are read from `./papers`; everything else lives in Docker volumes.

## First start

```sh
cp .env.example .env          # set POSTGRES_PASSWORD
mkdir -p papers
docker compose up --build -d
./scripts/smoke.sh
```

## Connecting to Ollama

The API container reaches Ollama through `host.docker.internal:11434`
(`DOCKER_OLLAMA_BASE_URL` in `.env`). Compose maps that name to your machine on
Linux as well as on macOS and Windows.

- **macOS:** install the Ollama app and pull a model (`ollama pull qwen3.5:4b`).
  Running Ollama natively uses the Apple GPU, which Docker cannot. If the smoke
  test cannot reach it, quit the menu-bar app and start Ollama on all
  interfaces in a terminal: `OLLAMA_HOST=0.0.0.0:11434 ollama serve`.
- **Linux:** Ollama listens on `127.0.0.1` by default, which containers cannot
  reach. Set `OLLAMA_HOST=0.0.0.0:11434` for the Ollama service (for example
  with `systemctl edit ollama`) and restart it.

Ollama has no authentication. If it listens on all interfaces, make sure your
firewall blocks port 11434 from other machines, or use the containerised
option below.

### Ollama in Docker (CPU)

```sh
echo 'DOCKER_OLLAMA_BASE_URL=http://ollama:11434' >> .env
docker compose --profile cpu-ollama up -d
docker compose --profile cpu-ollama logs -f pull-model    # pulls OLLAMA_MODEL
```

This works everywhere but is slow, and on a Mac it cannot use the GPU.

## Dense retrieval

Sparse (BM25) retrieval works as soon as documents are synced. Dense and hybrid
retrieval need embeddings:

```sh
docker compose exec api aira index-embeddings
docker compose exec api aira index-embeddings --model Alibaba-NLP/gte-large-en-v1.5
docker compose exec api aira retrieval-status
```

The first run downloads the model into the `huggingface_cache` volume. Later
runs only embed chunks that do not have vectors yet. Uploads through the UI are
embedded automatically.

## Data and volumes

| Location | Contents |
|---|---|
| `./papers` | Your PDFs (bind mount) |
| `./.data` | Evaluation runs, benchmark results, backups (bind mount) |
| `aira_postgres_data` | The database: documents, evidence, embeddings, chats, notes |
| `aira_huggingface_cache` | Downloaded embedding and reranker models |
| `aira_phoenix_data` | Traces, when tracing is enabled |
| `aira_ollama_models` | Models, when using the CPU Ollama profile |

`docker compose down` keeps all of this. **`docker compose down -v` deletes the
volumes, including your database.** Back up first (see
[operations.md](operations.md)).

The volume prefix is the Compose project name (`aira`). To keep using volumes
created under another name, set `COMPOSE_PROJECT_NAME` in `.env`.

## Importing an existing SQLite database

If you used AIRA natively with SQLite and want to move to the Docker database,
run the import **before** syncing any documents into PostgreSQL:

```sh
cp .data/aira.db .data/aira.backup.db
docker compose exec api python scripts/migrate_sqlite.py --sqlite /app/.data/aira.db
docker compose exec api aira index-embeddings
```

The import refuses to run if the target already contains data, copies
everything in one transaction, keeps all IDs and leaves the SQLite file
unchanged.

## Troubleshooting

```sh
docker compose ps
docker compose logs --tail=100 api db web
docker compose exec api aira retrieval-status
```

- **Port 3000 or 8000 in use:** stop other Next.js or Uvicorn processes.
- **API healthy but answers fail:** run `./scripts/smoke.sh`; it prints the
  Ollama address the container uses and the models it can see.
- **Changed `.env` but nothing happened:** recreate the API container with
  `docker compose up -d api`.
