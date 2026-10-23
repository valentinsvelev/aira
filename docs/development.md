# Development

## Hot-reload stack

Building the API image installs PyTorch and the rest of the ML stack, so avoid
`docker compose up --build` for everyday changes. The development override
(`compose.dev.yml`) mounts the source into the containers instead:

```sh
make dev          # start; builds the images only if they do not exist yet
make dev-logs     # follow API and web logs
make dev-status
make dev-down     # stop; keeps all volumes
```

- Python changes in `src/aira/` reload Uvicorn automatically.
- Changes in `web/app`, `web/components` and `web/lib` refresh the browser.
- After changing `.env`, run `make dev` again to recreate the containers.
- After changing `requirements-api.txt` or a Dockerfile, run `make dev-rebuild`.
- After changing `web/package.json`, run `make dev-web-install` and commit the
  updated `web/package-lock.json`.

Development and production use the same project name and volumes, so do not
run both at once. `make build` builds and starts the production images.

## Running without Docker

```sh
python -m pip install -e '.[dev,local,postgres,web,observability]'
cp .env.example .env
uvicorn aira.api:app --reload            # http://127.0.0.1:8000

cd web
cp .env.local.example .env.local         # BACKEND_URL=http://127.0.0.1:8000
npm ci
npm run dev                              # http://127.0.0.1:3000
```

The `local` extra installs sentence-transformers (and PyTorch) for dense
retrieval and semantic note verification. Without `DATABASE_URL`, the API uses
SQLite in `.data/aira.db`, which supports sparse retrieval only; dense and
hybrid retrieval need PostgreSQL with pgvector.

## Tests and checks

All tests run offline. Model calls, web requests and Phoenix are mocked.

```sh
make lint            # ruff check .
make test-backend    # pytest
make coverage        # pytest with a line-by-line coverage report
make test-web        # node:test suite and TypeScript typecheck
make test            # all of the above
```

Backend tests live in `tests/`, one file per topic (`test_retrieval.py`,
`test_web_search.py`, `test_tracing.py`, ...). Frontend tests live in
`web/tests/` and are grouped the same way; `web/tests/helpers.mjs` reads source
files for the UI checks.

`tests/test_pgvector_integration.py` needs a disposable PostgreSQL database
with pgvector and runs only when `TEST_POSTGRES_URL` is set. CI provides one.

CI (`.github/workflows/ci.yml`) runs ruff, validates every Compose file, runs
the backend tests with coverage against PostgreSQL, checks the Alembic
migrations, and runs the frontend tests, typecheck and production build.
Coverage is uploaded to Codecov when a `CODECOV_TOKEN` secret is configured.

## Database migrations

Schema changes are managed with Alembic.

```sh
make db-migrate-container     # alembic upgrade head inside the API container
make db-check-container       # fails if the models and migrations differ
```

To add a migration, change the models in `src/aira/`, then run
`alembic revision --autogenerate -m "describe the change"` with `DATABASE_URL`
pointing at an up-to-date development database, review the generated file in
`alembic/versions/`, and commit it.

A database created before migrations were set up can be adopted without
changing its data; see [operations.md](operations.md#adopting-alembic).

## Dependency locks

`make lock-deps` resolves `requirements-api.lock.txt` (with hashes, used by the
API Dockerfile when present) and refreshes `web/package-lock.json`. It needs
[uv](https://docs.astral.sh/uv/) and network access.

## Project layout

```text
src/aira/           FastAPI app, retrieval, agents, verification, CLI
web/                Next.js frontend (app/, components/, lib/, tests/)
tests/              Backend tests, one file per topic
alembic/            Database migrations
scripts/            Smoke test, preflight, backups, demo seeding, model servers
airflow/dags/       Optional Airflow ingestion DAG
demo/               Synthetic fixtures for the isolated demo stack
docs/               Documentation
compose*.yml        Docker Compose stacks and overrides
```
