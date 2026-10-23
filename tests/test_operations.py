"""Operational checks that run offline: readiness, preflight, backups, dev launcher and Docker dependencies."""
import importlib.util
import os
import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from aira.api import create_app
from aira.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readiness_reports_db_without_model_calls(tmp_path, monkeypatch):
    from aira import api as api_module
    config = Settings(database_url=f"sqlite:///{tmp_path / 'ready.db'}", papers_dir=tmp_path)
    client = TestClient(create_app(config))
    assert client.get('/health').status_code == 200
    assert client.get('/ready').json() == {
        'status': 'ready', 'database': 'ok', 'vector_extension': 'not_applicable'}
    # The endpoint must be independent from inference and embedding packages.
    monkeypatch.setattr(api_module, 'answer_with_local_context', lambda *a, **k: 1 / 0)
    assert client.get('/ready').status_code == 200


def test_readiness_degrades_without_exposing_driver_errors(tmp_path, monkeypatch):
    config = Settings(database_url=f"sqlite:///{tmp_path / 'offline.db'}", papers_dir=tmp_path)
    client = TestClient(create_app(config))
    # Patch the engine's connect method at its actual SQLAlchemy instance.
    from sqlalchemy.engine import Engine
    def unavailable(self, *args, **kwargs):
        raise OSError('postgresql+psycopg://secret:password@hidden-host/private')
    monkeypatch.setattr(Engine, 'connect', unavailable)
    response = client.get('/ready')
    assert response.status_code == 503
    assert response.json() == {'detail': 'Database not ready'}
    assert 'password' not in response.text
    assert 'hidden-host' not in response.text


def test_preflight_checks_host_ports_password_and_secret_redaction(tmp_path):
    preflight = load_script('preflight')
    sample = 'ports: ["127.0.0.1:3000:3000", "127.0.0.1:8000:8000"]'
    errors, warnings = preflight.audit({'POSTGRES_PASSWORD': 'aira_local'}, sample, hardened_local=True)
    assert any('POSTGRES_PASSWORD' in item for item in errors)
    errors, warnings = preflight.audit({'POSTGRES_PASSWORD': 'long-local-example-value'}, sample, hardened_local=True)
    assert not errors
    assert any('authentication' in item for item in warnings)
    errors, _ = preflight.audit({'POSTGRES_PASSWORD': 'long-local-example-value'},
                                'ports: ["0.0.0.0:3000:3000"]', hardened_local=True)
    assert errors
    env = tmp_path / '.env'
    env.write_text('POSTGRES_PASSWORD="private-string"\nOTEL_CAPTURE_CONTENT=true\n')
    assert preflight.read_env(env)['POSTGRES_PASSWORD'] == 'private-string'


def test_smoke_endpoint_checks_do_not_require_running_models(monkeypatch):
    smoke = load_script('check_stack')
    class FakeResponse:
        def __init__(self, url):
            self.url = url
            self.status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return (b'{"status":"ready"}' if self.url.endswith('/ready') else
                    b'{"status":"ok"}' if self.url.endswith('/health') else b'<html>AIRA</html>')
    calls = []
    monkeypatch.setattr(smoke, 'urlopen', lambda url, timeout: (calls.append(url), FakeResponse(url))[1])
    ok, messages = smoke.check('http://127.0.0.1:8000', 'http://127.0.0.1:3000')
    assert ok and len(messages) == 3 and len(calls) == 3
    assert all('chat' not in url and 'tracing' not in url for url in calls)


def test_backup_script_passes_shell_syntax():
    subprocess.run(['bash', '-n', str(ROOT/'scripts'/'backup-db.sh')], check=True)


def test_backup_is_private_and_atomic_with_failure_cleanup(tmp_path):
    # Simulated docker CLI; never touches the user's real database.
    docker = tmp_path / 'docker'
    docker.write_text('#!/bin/sh\nif [ "$AIRA_FAKE_DUMP_FAILURE" = 1 ]; then echo partial; exit 7; fi\necho DATABASE-DUMP\n')
    docker.chmod(0o755)
    destination = tmp_path / 'saved'
    env = {**__import__('os').environ, 'PATH': f"{tmp_path}:{__import__('os').environ['PATH']}",
           'AIRA_BACKUP_DIR': str(destination)}
    command = ['bash', str(ROOT/'scripts'/'backup-db.sh')]
    success = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    assert success.returncode == 0, success.stderr
    backups = list(destination.glob('*.dump'))
    assert len(backups) == 1 and backups[0].read_bytes() == b'DATABASE-DUMP\n'
    assert (backups[0].stat().st_mode & 0o077) == 0
    env['AIRA_FAKE_DUMP_FAILURE'] = '1'
    failed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    assert failed.returncode != 0
    assert len(list(destination.glob('*.dump'))) == 1  # no corrupt published backup
    assert not list(destination.glob('*.partial'))


def test_dev_bootstrap_shell_valid():
    subprocess.run(["sh", "-n", str(ROOT / "scripts" / "start-api-dev.sh")], check=True)


def test_dev_launcher_is_conditional_and_preserves_api(monkeypatch, tmp_path):
    """Mock python and uvicorn to prove disabled tracing never invokes pip."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    python = bindir / "python"
    python.write_text('#!/bin/sh\necho "python-invoked" >&2\nexit 9\n')
    python.chmod(0o755)
    uvicorn = bindir / "uvicorn"
    uvicorn.write_text('#!/bin/sh\necho "uvicorn-started"\n')
    uvicorn.chmod(0o755)
    env = os.environ.copy()
    env.update({"PATH": str(bindir) + ":" + env["PATH"], "OTEL_TRACES_ENABLED": "false"})
    result = subprocess.run(["sh", str(ROOT / "scripts" / "start-api-dev.sh")], env=env,
                            capture_output=True, text=True, check=True)
    assert "uvicorn-started" in result.stdout
    assert "python-invoked" not in result.stderr


def test_dev_bootstrap_uses_persistent_volume():
    text = (ROOT / "compose.dev.yml").read_text()
    assert "tracing_python_deps:/opt/aira-tracing-deps" in text
    assert "./scripts/start-api-dev.sh:/app/scripts/start-api-dev.sh:ro" in text
    assert "PYTHONPATH: /app/src:/opt/aira-tracing-deps" in text


def test_dev_launcher_installs_only_if_tracing_enabled_and_sdk_missing(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    python = bindir / "python"
    python.write_text('#!/bin/sh\nif [ "$1" = "-c" ]; then exit 1; fi\nif [ "$1" = "-m" ] && [ "$2" = "pip" ]; then echo "pip-install-invoked"; exit 0; fi\nexit 2\n')
    python.chmod(0o755)
    uvicorn = bindir / "uvicorn"
    uvicorn.write_text("#!/bin/sh\necho uvicorn-started\n")
    uvicorn.chmod(0o755)
    env = os.environ.copy()
    env.update({"PATH": str(bindir) + ":" + env["PATH"], "OTEL_TRACES_ENABLED": "true"})
    result = subprocess.run(["sh", str(ROOT / "scripts" / "start-api-dev.sh")], env=env,
                            capture_output=True, text=True, check=True)
    assert "pip-install-invoked" in result.stdout
    assert "uvicorn-started" in result.stdout


def test_docker_runtime_dependencies_include_telemetry_and_match_project():
    import tomllib
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    expected = project["dependencies"][:]
    for name in ("local", "postgres", "web", "observability"):
        expected += project["optional-dependencies"][name]
    found = [line.strip() for line in (root / "requirements-api.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    assert set(found) == set(expected)
    assert "opentelemetry-sdk>=1.28,<2" in found
