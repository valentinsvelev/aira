#!/usr/bin/env bash
# PostgreSQL-only backup, safe to run while the dev stack is online.
# Does NOT back up PDF files, HF caches, Phoenix data, or local experiment artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."
umask 077
destination="${AIRA_BACKUP_DIR:-.data/backups}"
mkdir -p "$destination"
chmod 700 "$destination"
file="$destination/aira-postgres-$(date -u +%Y%m%dT%H%M%SZ)-$$.dump"
partial="${file}.partial"
trap 'rm -f "$partial"' EXIT
# Container's own database environment provides the password; none is printed here.
docker compose -f compose.yml exec -T db \
  sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc --no-owner --no-acl' > "$partial"
test -s "$partial"
mv "$partial" "$file"
trap - EXIT
printf 'PostgreSQL backup saved: %s\n' "$file"
echo 'Also back up papers/ and .data/ separately; see docs/operations.md.'
