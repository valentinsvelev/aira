"""Non-destructive Alembic adoption for AIRA databases created without migrations.

Requires a recent database backup. Verifies actual tables and columns, then
stamps the baseline revision without attempting CREATE TABLE or DROP TABLE.
"""
from __future__ import annotations

import argparse
import os

from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from aira import literature  # noqa: F401
from aira.db import Base
from alembic import command

BASELINE = '0001_aira_baseline'

def inspect_existing(url: str) -> None:
    engine = create_engine(url)
    try:
        inspector = inspect(engine)
        actual_tables = set(inspector.get_table_names())
        required_tables = set(Base.metadata.tables)
        missing_tables = required_tables - actual_tables
        if missing_tables:
            raise RuntimeError('Schema incomplete; missing tables: ' + ', '.join(sorted(missing_tables)))
        for name, table in Base.metadata.tables.items():
            actual_columns = {c['name'] for c in inspector.get_columns(name)}
            missing = {c.name for c in table.columns} - actual_columns
            if missing:
                raise RuntimeError(f'Schema incomplete; {name} missing columns: {sorted(missing)}')
        if engine.dialect.name == 'postgresql':
            for name in ('chunk_embeddings', 'chunk_embeddings_1024'):
                if name not in actual_tables:
                    raise RuntimeError(f'Schema incomplete; missing pgvector table {name}')
            with engine.connect() as connection:
                has_extension = connection.scalar(text("SELECT count(*) FROM pg_extension WHERE extname='vector'"))
                if not has_extension:
                    raise RuntimeError('Schema incomplete; pgvector extension missing')
        if 'alembic_version' in actual_tables:
            with engine.connect() as connection:
                revisions = connection.execute(text('SELECT version_num FROM alembic_version')).scalars().all()
            if revisions:
                raise RuntimeError(f'Database already has Alembic revision(s): {revisions}; do not stamp blindly')
    finally:
        engine.dispose()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yes-i-backed-up', action='store_true', help='Explicit confirmation of a completed backup')
    args = parser.parse_args()
    url = os.getenv('AIRA_MIGRATION_DATABASE_URL') or os.getenv('DATABASE_URL')
    if not url:
        parser.error('Set DATABASE_URL or AIRA_MIGRATION_DATABASE_URL')
    inspect_existing(url)
    if not args.yes_i_backed_up:
        parser.error('Schema validated. Back up the database, then rerun with --yes-i-backed-up')
    command.stamp(Config('alembic.ini'), BASELINE)
    print(f'Adopted Alembic revision {BASELINE} (no schema/data tables modified)')

if __name__ == '__main__':
    main()
