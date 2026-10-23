"""Migration context. This does not run when the API auto-reloads."""
from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from aira import literature  # noqa: F401 (register tables)
from aira.db import Base
from alembic import context

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)

def database_url() -> str:
    url = os.getenv('AIRA_MIGRATION_DATABASE_URL') or os.getenv('DATABASE_URL')
    if not url:
        raise RuntimeError('Set DATABASE_URL or AIRA_MIGRATION_DATABASE_URL to the target database URL')
    return url

def include_object(obj, name, kind, reflected, compare_to):
    # pgvector tables and custom PostgreSQL indexes belong to the baseline migration.
    # They are not mapped to ORM models but remain managed by migration SQL.
    if kind == 'table' and name in {'chunk_embeddings', 'chunk_embeddings_1024'}:
        return False
    if kind == 'index' and name == 'ix_chat_turns_fts':
        return False
    return True

def run_migrations_offline():
    context.configure(url=database_url(), target_metadata=Base.metadata, literal_binds=True, compare_type=True,
                      include_object=include_object)
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    engine = create_engine(database_url(), poolclass=pool.NullPool)
    try:
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True,
                              include_object=include_object)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
