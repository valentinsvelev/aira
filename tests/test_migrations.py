"""Schema migrations and safe, additive upgrades of existing databases."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

from aira.api import create_app
from aira.config import Settings
from aira.db import Document, SchemaMigration, make_session_factory

ROOT = Path(__file__).resolve().parents[1]


def alembic(tmp_path, url, *args):
    env = {**os.environ, 'DATABASE_URL': url, 'PYTHONPATH': str(ROOT / 'src')}
    return subprocess.run([sys.executable, '-m', 'alembic', '-c', str(ROOT / 'alembic.ini'), *args],
                          cwd=ROOT, env=env, capture_output=True, text=True)


def client(tmp_path: Path):
    return TestClient(create_app(Settings(database_url=f"sqlite:///{tmp_path / 'test.db'}",
                                         papers_dir=tmp_path, llm_provider="ollama")))


def test_fresh_schema_migration_and_strict_mode(tmp_path, monkeypatch):
    url = f'sqlite:///{tmp_path / "fresh.db"}'
    upgraded = alembic(tmp_path, url, 'upgrade', 'head')
    assert upgraded.returncode == 0, upgraded.stderr
    assert alembic(tmp_path, url, 'check').returncode == 0
    monkeypatch.setenv('AIRA_REQUIRE_MIGRATIONS', 'true')
    factory = make_session_factory(url)
    with factory.begin() as db:
        db.add(Document(path='demo.txt'))
    with factory() as db:
        assert db.query(Document).count() == 1
    monkeypatch.delenv('AIRA_REQUIRE_MIGRATIONS')


def test_adoption_refuses_incomplete_database_or_unbacked_adoption(tmp_path):
    url = f'sqlite:///{tmp_path / "old.db"}'
    create_engine(url).dispose()
    env = {**os.environ, 'DATABASE_URL': url, 'PYTHONPATH': str(ROOT / 'src')}
    command = [sys.executable, 'scripts/adopt_alembic.py']
    invalid = subprocess.run(command + ['--yes-i-backed-up'], cwd=ROOT, env=env,
                             capture_output=True, text=True)
    assert invalid.returncode != 0 and 'missing tables' in invalid.stderr
    f = make_session_factory(url)
    with f.begin() as db:
        db.add(Document(path='private_existing.pdf'))
    unconfirmed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    assert unconfirmed.returncode != 0 and 'Back up' in unconfirmed.stderr
    confirmed = subprocess.run(command + ['--yes-i-backed-up'], cwd=ROOT, env=env,
                               capture_output=True, text=True)
    assert confirmed.returncode == 0, confirmed.stderr
    with create_engine(url).connect() as connection:
        assert connection.scalar(text('SELECT version_num FROM alembic_version')) == '0001_aira_baseline'
        assert connection.scalar(text('SELECT path FROM documents')) == 'private_existing.pdf'
    rejected = subprocess.run(command + ['--yes-i-backed-up'], cwd=ROOT, env=env,
                              capture_output=True, text=True)
    assert rejected.returncode != 0 and 'already has Alembic revision' in rejected.stderr


def test_strict_mode_does_not_create_tables(tmp_path, monkeypatch):
    url = f'sqlite:///{tmp_path / "untouched.db"}'
    monkeypatch.setenv('AIRA_REQUIRE_MIGRATIONS', '1')
    with pytest.raises(RuntimeError, match='no Alembic revision'):
        make_session_factory(url)
    assert 'documents' not in inspect(create_engine(url)).get_table_names()


def test_repeated_startup_preserves_saved_data_and_migration_marker(tmp_path):
    from aira.db import ResearchSession
    url = f"sqlite:///{tmp_path/'upgrade.db'}"
    first = make_session_factory(url)
    with first.begin() as db:
        db.add(ResearchSession(id='persisted'))
        assert db.get(SchemaMigration, 'v1.5_conversation_tool') is not None
    second = make_session_factory(url)
    with second() as db:
        assert db.get(ResearchSession, 'persisted') is not None
        assert db.query(SchemaMigration).filter_by(name='v1.5_conversation_tool').count() == 1


def test_migration_adds_conversation_tool_once_but_respects_later_revocation(tmp_path):
    from sqlalchemy import select

    from aira.db import AgentProfile
    from aira.profiles import get_profile, update_profile
    url = f"sqlite:///{tmp_path/'migration.db'}"
    factory = make_session_factory(url)
    with factory.begin() as db:
        db.add(AgentProfile(name='Older profile', description='', system_instructions='',
                            enabled_tools=['local_search']))
    # Simulate an older database by clearing the migration marker.
    from aira.db import SchemaMigration
    with factory.begin() as db:
        db.delete(db.get(SchemaMigration, 'v1.5_conversation_tool'))
    factory = make_session_factory(url)
    with factory() as db:
        profile = db.scalars(select(AgentProfile).where(AgentProfile.name == 'Older profile')).first()
        profile_id = profile.id
    assert 'conversation_search' in get_profile(factory, profile_id)['enabled_tools']
    update_profile(factory, profile_id, enabled_tools=['local_search'])
    factory = make_session_factory(url)
    assert get_profile(factory, profile_id)['enabled_tools'] == ['local_search']


def test_additive_schema_change_preserves_existing_agent_row(tmp_path):
    from aira.db import make_session_factory
    url = f"sqlite:///{tmp_path / 'old.db'}"
    factory = make_session_factory(url)
    # Reproduce an older schema (no generation column) with persisted agent data.
    with factory.kw["bind"].begin() as conn:
        conn.execute(text("ALTER TABLE agent_profiles RENAME TO agent_profiles_current"))
        conn.execute(text("CREATE TABLE agent_profiles AS SELECT id,name,description,system_instructions,enabled_tools,created_at,updated_at FROM agent_profiles_current"))
        conn.execute(text("DROP TABLE agent_profiles_current"))
        conn.execute(text("INSERT INTO agent_profiles(id,name,description,system_instructions,enabled_tools,created_at,updated_at) "
                          "VALUES ('legacy','Old agent','','', '[\"local_search\"]',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"))
    new_factory = make_session_factory(url)
    with new_factory() as session:
        assert session.execute(text("SELECT name FROM agent_profiles WHERE id = 'legacy'")).scalar_one() == "Old agent"
        assert "generation" in {col["name"] for col in inspect(new_factory.kw["bind"]).get_columns("agent_profiles")}
    assert client(tmp_path).get("/agents").status_code == 200
