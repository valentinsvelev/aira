"""Model selection, generation options and context-window defaults."""

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from aira.agent import make_model
from aira.api import create_app
from aira.config import Settings
from aira.model_context import ollama_model_max_context
from aira.options import GenerationOptions, RetrievalOptions
from aira.vector_store import MINILM


def test_retrieval_and_generation_defaults():
    assert RetrievalOptions().mode == 'sparse'
    assert RetrievalOptions().embedding_model == MINILM
    assert GenerationOptions().model is None
    for name in ('llama3.2:3b', 'qwen3.5:4b', 'granite4.1:3b'):
        assert GenerationOptions(model=name).model == name
    with pytest.raises(ValidationError):
        GenerationOptions(model='untrusted-model')
    with pytest.raises(ValidationError):
        RetrievalOptions(embedding_model='another-model')


def test_ollama_model_selection_forwarded_without_real_ollama(monkeypatch):
    from types import SimpleNamespace
    captured={}
    class FakeOllama:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    monkeypatch.setitem(__import__('sys').modules,'langchain_ollama',SimpleNamespace(ChatOllama=FakeOllama))
    make_model(Settings(),GenerationOptions(model='granite4.1:3b'))
    assert captured['model']=='granite4.1:3b'


def test_unlimited_is_default_but_limits_are_validated():
    assert GenerationOptions().num_predict == -1
    assert GenerationOptions(num_predict=32768).num_predict == 32768
    for wrong in (0, 1, 31, -2, 32769):
        try:
            GenerationOptions(num_predict=wrong)
            raise AssertionError(f"Accepted {wrong}")
        except ValueError:
            pass


def test_default_context_is_model_max_and_custom_remains_supported():
    assert GenerationOptions().num_ctx == 0
    assert GenerationOptions(num_ctx=131072).num_ctx == 131072


def test_model_max_from_ollama_show(monkeypatch):
    ollama_model_max_context.cache_clear()
    def fake_post(url, json, timeout):
        assert url.endswith('/api/show') and json['model'] == 'qwen3.5:4b'
        return httpx.Response(200, request=httpx.Request('POST', url), json={
            'model_info': {'qwen3.context_length': 131072, 'qwen3.embedding_length': 3000}})
    monkeypatch.setattr(httpx, 'post', fake_post)
    assert ollama_model_max_context('http://ollama:11434', 'qwen3.5:4b') == 131072


def test_default_ollama_model_is_qwen(monkeypatch):
    monkeypatch.delenv('OLLAMA_MODEL', raising=False)
    assert Settings().ollama_model == 'qwen3.5:4b'
    assert Settings.from_env().ollama_model == 'qwen3.5:4b'
    # A request can leave model unset, so the service-specific setting applies.
    assert GenerationOptions().model is None


def test_existing_explicit_model_configuration_is_respected(monkeypatch):
    monkeypatch.setenv('OLLAMA_MODEL', 'granite4.1:3b')
    assert Settings.from_env().ollama_model == 'granite4.1:3b'


def test_minicpm_model_accepted_by_validated_generation_options():
    assert GenerationOptions(model="openbmb/minicpm-v4.7:1b").model == "openbmb/minicpm-v4.7:1b"


def test_sampling_arguments_are_validated_and_included_in_agent_profile(tmp_path):
    client = TestClient(create_app(Settings(database_url=f"sqlite:///{tmp_path/'g.db'}", papers_dir=tmp_path)))
    assert GenerationOptions(top_k=40,min_p=0.1).model_dump()['min_p'] == 0.1
    assert client.post('/agents', json={'name': 'Sampler', 'generation': {'top_k': 21, 'min_p': 0.03}}).json()['generation']['top_k'] == 21
    assert client.post('/agents', json={'name': 'Bad', 'generation': {'top_k': -1}}).status_code == 422
    assert client.post('/agents', json={'name': 'Bad', 'generation': {'min_p': 2.0}}).status_code == 422
