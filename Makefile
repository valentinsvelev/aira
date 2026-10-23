# Run from the repository root.
COMPOSE = docker compose
DEV = $(COMPOSE) -f compose.yml -f compose.dev.yml

.PHONY: up build down logs ingest smoke ps clean dev dev-down dev-logs dev-rebuild dev-status dev-web-install dev-web-repair tracing-up tracing-logs tracing-status tracing-test

# Normal production startup: reuse existing built images.
up:
	mkdir -p papers .data
	$(COMPOSE) up -d

# Explicit production rebuild (only needed for dependencies/build files or releases).
build:
	$(COMPOSE) up --build -d

# Dev stack: bind-mounted source and automatic backend/frontend reload.
# No --build: Compose builds a missing dev image once, then reuses it.
dev:
	mkdir -p papers .data
	$(DEV) up -d

dev-logs:
	$(DEV) logs -f api web

dev-down:
	$(DEV) down

dev-status:
	$(DEV) ps

# Use only after changing Python dependencies or Dockerfiles.
dev-rebuild:
	$(DEV) up --build -d api web

# Use after editing web/package.json; updates the persisted node_modules volume.
dev-web-install:
	$(DEV) exec web npm install --no-audit --no-fund

# Rebuild only the frontend dev image if its dependencies were never installed,
# e.g. after a canceled build; does not rebuild API or reset volumes.
dev-web-repair:
	$(DEV) build web
	$(DEV) up -d --no-deps --force-recreate web

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f api web

ingest:
	$(COMPOSE) exec api aira ingest

smoke:
	./scripts/smoke.sh

ps:
	$(COMPOSE) ps

# No volume deletion: preserves PDFs, database, model caches and npm packages.
clean:
	$(COMPOSE) down --remove-orphans

# Enable OTEL_TRACES_ENABLED=true in .env before running. Reuses existing API image.
tracing-up:
	$(DEV) --profile tracing up -d --no-deps phoenix
	$(DEV) up -d --no-build --no-deps --force-recreate api

tracing-logs:
	$(DEV) --profile tracing logs -f phoenix api

# Debug tracing without rebuilding or sending private research content.
tracing-status:
	curl -fsS http://127.0.0.1:8000/tracing/status && echo

tracing-test:
	curl -fsS -X POST http://127.0.0.1:8000/tracing/test && echo

# Optional local data-engineering services. AIRA hot-reload API/web stay unchanged.
AIRFLOW = $(COMPOSE) -f compose.yml -f compose.dev.yml -f compose.airflow.yml
ANALYTICS = $(COMPOSE) -f compose.yml -f compose.analytics.yml
.PHONY: airflow-up airflow-logs airflow-down analytics-build analytics-report

airflow-up:
	$(AIRFLOW) --profile airflow up -d --no-deps airflow

airflow-logs:
	$(AIRFLOW) --profile airflow logs -f airflow

airflow-down:
	$(AIRFLOW) --profile airflow stop airflow

analytics-build:
	$(ANALYTICS) build analytics

analytics-report:
	$(ANALYTICS) --profile analytics run --rm --no-deps analytics

# Checks. All are local/read-only except backup-db.
.PHONY: preflight preflight-hardened smoke-ready backup-db test test-backend test-web lint coverage
preflight:
	python3 scripts/preflight.py
preflight-hardened:
	python3 scripts/preflight.py --hardened-local
smoke-ready:
	python3 scripts/check_stack.py
backup-db:
	./scripts/backup-db.sh
test: lint test-backend test-web
test-backend:
	python -m pytest -q
test-web:
	cd web && npm test && npm run typecheck
lint:
	ruff check .
coverage:
	python -m pytest -q --cov --cov-report=term-missing

# Reproducible installs and explicit, non-destructive schema management.
.PHONY: lock-deps db-migrate db-check db-adopt demo-up demo-seed demo-down demo-status
lock-deps:
	./scripts/lock_dependencies.sh
# Run with DATABASE_URL pointing at the target DB. Alembic runs on the host,
# not in the hot-reloading application container.
db-migrate:
	alembic upgrade head
db-check:
	alembic current && alembic check
db-adopt:
	python scripts/adopt_alembic.py --yes-i-backed-up
DEMO = $(COMPOSE) -p aira-demo -f compose.demo.yml
demo-up:
	mkdir -p demo/workspace/papers demo/workspace/data
	$(DEMO) up -d
	python3 scripts/seed_demo.py --url http://127.0.0.1:8100

demo-seed:
	python3 scripts/seed_demo.py --url http://127.0.0.1:8100

demo-down:
	$(DEMO) down

demo-status:
	$(DEMO) ps

# Prefer these when PostgreSQL is private to the Compose network.
.PHONY: db-migrate-container db-check-container db-adopt-container
db-migrate-container:
	$(DEV) exec api alembic upgrade head

db-check-container:
	$(DEV) exec api alembic current
	$(DEV) exec api alembic check

db-adopt-container:
	$(DEV) exec api python /app/scripts/adopt_alembic.py --yes-i-backed-up

.PHONY: dev-api-rebuild
# Required only once for Alembic/package changes. Leaves frontend image untouched.
dev-api-rebuild:
	$(DEV) build api
	$(DEV) up -d --no-build --no-deps --force-recreate api
