# Operations

AIRA is a local, single-user application. It has no authentication, user
isolation or rate limiting, and it has not had a security review for network
exposure. Keep the published ports bound to `127.0.0.1`. Remote access would
need an authenticated TLS reverse proxy at the very least.

## Checks

```sh
make preflight            # configuration check; warnings only
make preflight-hardened   # also fails on a missing, short or default database password
make smoke-ready          # checks /health, /ready and the UI of a running stack
./scripts/smoke.sh        # also checks pgvector and the container's view of Ollama
```

`GET /health` only reports that the API process is up. `GET /ready` also opens a
database connection and, on PostgreSQL, checks the pgvector extension. It never
calls a model or Phoenix.

## Backups

```sh
make backup-db            # writes .data/backups/aira-postgres-<timestamp>.dump
pg_restore --list .data/backups/aira-postgres-<timestamp>.dump
```

The dump is written atomically with private file permissions, and a partial
file is removed if `pg_dump` fails. It contains the database only. Back up these
separately if you need them:

- `papers/` (your PDFs) and `.data/` (evaluation runs, benchmark results),
- Paper Explorer history, which is stored in the browser,
- model caches and Phoenix traces (Docker volumes), which can be recreated.

Test a restore in a new, disposable database, never over your live one:

```sh
createdb restore_test
psql -d restore_test -c 'CREATE EXTENSION vector'
pg_restore --no-owner --no-acl -d restore_test .data/backups/aira-postgres-<timestamp>.dump
```

## Changing the database password

`POSTGRES_PASSWORD` is only applied when the database volume is first created.
To change it later:

1. Run `make backup-db`.
2. Change the role's password inside PostgreSQL (replace `aira` if you set a
   different `POSTGRES_USER` or `POSTGRES_DB`):
   `docker compose exec db psql -U aira -d aira -c "ALTER ROLE aira PASSWORD 'new-password';"`
3. Put the same value in `.env` and recreate the API container:
   `docker compose up -d api`.

## Adopting Alembic

A database that was created by the application before migrations existed can
be marked as being at the baseline revision without touching its data:

```sh
make backup-db
make db-adopt-container
make db-check-container
```

The adoption script refuses to run if the schema is incomplete or if you have
not confirmed the backup.

With `AIRA_REQUIRE_MIGRATIONS=true` the API refuses to create tables itself and
expects `alembic upgrade head` to have been run. The demo stack uses this mode.
