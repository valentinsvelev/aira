"""Atomic, non-destructive SQLite to PostgreSQL migration utility."""
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from aira.db import (
    ContextEvent,
    ContextNote,
    Document,
    DocumentVersion,
    EvidenceChunk,
    IngestionRun,
    ResearchSession,
    make_session_factory,
)

TABLES = (Document, DocumentVersion, EvidenceChunk, IngestionRun,
          ResearchSession, ContextNote, ContextEvent)


def migrate(sqlite_path: Path, target_url: str) -> dict:
    if not sqlite_path.is_file():
        raise FileNotFoundError(f'SQLite file not found: {sqlite_path}')
    if not target_url.startswith('postgresql'):
        raise ValueError('Target DATABASE_URL must be PostgreSQL')
    # Only open the source read-only, to avoid modifying your original database.
    source = create_engine(f'sqlite:///file:{sqlite_path.resolve()}?mode=ro&uri=true')
    target_factory = make_session_factory(target_url)
    counts = {}
    with Session(source) as old, target_factory.begin() as new:
        for table in TABLES:
            if new.scalar(select(func.count()).select_from(table)):
                raise RuntimeError('Target is not empty; migration aborted without changes')
        for table in TABLES:
            rows = old.scalars(select(table)).all()
            for row in rows:
                fields = {column.key: getattr(row, column.key)
                          for column in table.__table__.columns}
                new.add(table(**fields))
            new.flush()
            counts[table.__tablename__] = len(rows)
    source.dispose()
    return counts


