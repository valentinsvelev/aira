import os

import dotenv

# Keep the suite independent of a developer's local .env. This must run before any
# aira import: aira.config would otherwise load .env at import time. The flags below
# also override values exported in the shell; tests that need them use monkeypatch.
dotenv.load_dotenv = lambda *args, **kwargs: False
os.environ["OTEL_TRACES_ENABLED"] = "false"
os.environ["OTEL_CAPTURE_CONTENT"] = "false"
os.environ["CHAT_TITLES_ENABLED"] = "false"

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from aira.context import ContextManager  # noqa: E402
from aira.db import make_session_factory  # noqa: E402
from aira.ingestion import Ingestor  # noqa: E402


@pytest.fixture
def workspace(tmp_path: Path):
    papers = tmp_path / "papers"
    papers.mkdir()
    factory = make_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    return {"papers": papers, "factory": factory,
            "ingestor": Ingestor(factory, papers),
            "context": ContextManager(factory)}
