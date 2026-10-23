"""Paper Explorer (ScholarQA): retrieval, answer attribution and local model routing (offline)."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aira import scholarqa
from aira.api import create_app
from aira.config import Settings
from aira.options import GenerationOptions
from aira.scholarqa import _synthesize
from aira.vllm_backend import model_status, model_url


def evidence(number: int):
    return {"provider": "openalex", "title": f"Paper {number}",
            "authors": ["Author, A"], "year": 2025,
            "url": f"https://doi.org/10.1234/{number}",
            "abstract": f"Paper {number} demonstrates an approach to evaluate retrieval and synthesis.",
            "evidence_id": str(number) * 64}


def test_question_search_citation_and_references(monkeypatch):
    queries = []
    calls = []

    def fake_search(_factory, query, **kwargs):
        queries.append(query)
        return {"results": [evidence(1), evidence(2), {**evidence(3), "abstract": "", "evidence_id": None}],
                "errors": {"arxiv": "temporarily unavailable"}}

    class Model:
        def invoke(self, prompt, **kwargs):
            calls.append(prompt)
            assert "PAPER" not in prompt[0][1]  # only system instructions in system prompt
            assert "Paper 1" in prompt[1][1]
            assert "Paper 2" in prompt[1][1]
            assert "Paper 3" not in prompt[1][1]
            return SimpleNamespace(content="## Findings\nPaper one [1] and two [2] support this. A hallucinated citation [99].\n## References\nFake ref")

    monkeypatch.setattr(scholarqa, "search_academic", fake_search)
    monkeypatch.setattr(scholarqa, "make_model", lambda *_: Model())
    monkeypatch.setattr(scholarqa, "callbacks", lambda: [])
    result = scholarqa.answer_scholar_question(None, "How are systems tested?", Settings(),
                                               context_questions=["retrieval performance"])
    assert queries == ["retrieval performance How are systems tested?"]
    assert result["scope"] == "abstracts_only"
    assert "[1]" in result["answer"] and "[2]" in result["answer"]
    assert "[unsupported citation]" in result["answer"]
    assert "Fake ref" not in result["answer"]
    assert len(result["papers"]) == 2
    assert all(p["cited"] for p in result["papers"])
    assert result["provider_errors"] == {"arxiv": "temporarily unavailable"}
    assert len(calls) == 1


def test_does_not_claim_unretrieved_content_and_requires_real_citations(monkeypatch):
    monkeypatch.setattr(scholarqa, "search_academic", lambda *_a, **_k:
                        {"results": [evidence(1)], "errors": {}})
    class Model:
        def invoke(self, *_a, **_k):
            return SimpleNamespace(content="No evidence [8]")
    monkeypatch.setattr(scholarqa, "make_model", lambda *_: Model())
    monkeypatch.setattr(scholarqa, "callbacks", lambda: [])
    import pytest
    with pytest.raises(RuntimeError, match="valid paper citations"):
        scholarqa.answer_scholar_question(None, "Any evidence?", Settings())


def test_endpoint_validates_and_returns_cited_answer(tmp_path, monkeypatch):
    settings = Settings(database_url=f"sqlite:///{tmp_path/'scholar.db'}", papers_dir=tmp_path)
    client = TestClient(create_app(settings))
    assert client.post("/scholar/answer", json={"question": "x"}).status_code == 422
    assert client.post("/scholar/answer", json={"question": "Valid question", "sources": ["nonsense"]}).status_code == 422
    monkeypatch.setattr("aira.api.answer_scholar_question", lambda *args, **kwargs:
                        {"question": args[1], "answer": "Finding [1]", "scope": "abstracts_only",
                         "papers": [dict(evidence(1), number=1, cited=True)], "provider_errors": {}})
    response = client.post("/scholar/answer", json={"question": "What is retrieved?"})
    assert response.status_code == 200
    assert response.json()["papers"][0]["cited"]


def test_model_specific_default_urls_and_old_override(monkeypatch):
    for name in ('SCHOLAR_MODEL_BASE_URL', 'SCHOLAR_VLLM_DR_TULU_URL', 'SCHOLAR_VLLM_ASTA_URL'):
        monkeypatch.delenv(name, raising=False)
    assert model_url('allenai/AstaBrief_8B') == 'http://host.docker.internal:8002/v1'
    assert model_url('rl-research/DR-Tulu-8B') == 'http://host.docker.internal:8001/v1'
    monkeypatch.setenv('SCHOLAR_MODEL_BASE_URL','http://127.0.0.1:8007/v1')
    assert model_url('rl-research/DR-Tulu-8B') == 'http://127.0.0.1:8007/v1'
    monkeypatch.setenv('SCHOLAR_VLLM_ASTA_URL','http://vllm-asta:8000/v1')
    assert model_url('allenai/AstaBrief_8B') == 'http://vllm-asta:8000/v1'


def test_inference_url_must_be_local(monkeypatch):
    monkeypatch.setenv('SCHOLAR_VLLM_ASTA_URL','https://hosted-inference.example/v1')
    with pytest.raises(ValueError, match='local|HTTP'):
        model_url('allenai/AstaBrief_8B')


def test_hf_model_posts_selected_id_to_local_vllm_endpoint(monkeypatch):
    import aira.vllm_backend as backend
    sent=[]
    monkeypatch.setenv('SCHOLAR_VLLM_DR_TULU_URL','http://127.0.0.1:8001/v1')
    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {'choices':[{'message':{'content':'Grounded findings [1].'}}]}
    class FakeClient:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def post(self,url,json,headers):
            sent.append((url,json,headers))
            return FakeResponse()
    monkeypatch.setattr(backend.httpx,'Client',FakeClient)
    answer = _synthesize('System','Papers',GenerationOptions(),Settings(), 'rl-research/DR-Tulu-8B')
    assert answer == 'Grounded findings [1].'
    assert sent[0][0] == 'http://127.0.0.1:8001/v1/chat/completions'
    assert sent[0][1]['model'] == 'rl-research/DR-Tulu-8B'
    assert sent[0][1]['messages'][0]['role'] == 'system'


def test_user_check_rejects_server_running_wrong_model(monkeypatch):
    import aira.vllm_backend as backend
    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {'data':[{'id':'another/model'}]}
    class FakeClient:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def get(self,*_args,**_kwargs): return FakeResponse()
    monkeypatch.setattr(backend.httpx,'Client',FakeClient)
    status=model_status('allenai/AstaBrief_8B')
    assert status['ready'] is False
    assert 'different model' in status['detail']


def test_astabrief_uses_numbered_references_without_claiming_full_text(monkeypatch):
    calls=[]
    monkeypatch.setattr(scholarqa, 'search_academic', lambda *args, **kwargs: {'results':[{
        'evidence_id':'abc', 'title':'Verified title', 'url':'https://example.org/1',
        'abstract':'Summary in an abstract.', 'year':2025, 'authors':['Smith'], 'provider':'openalex'}],
        'errors':{}})
    monkeypatch.setattr(scholarqa, '_synthesize', lambda system, request, generation, settings, model_id: (
        calls.append((request,model_id)) or 'Synthesis [1].'))
    result = scholarqa.answer_scholar_question(None, 'Research question?', Settings(),
                                                model_id='allenai/AstaBrief_8B')
    assert result['papers'][0]['cited'] is True
    assert '"[1]"' in calls[0][0]
    assert 'Abstract: Summary in an abstract.' in calls[0][0]
    assert 'NOT verified full-text' in calls[0][0]


def test_scholar_model_validation(tmp_path):
    client = TestClient(create_app(Settings(database_url=f"sqlite:///{tmp_path/'s.db'}", papers_dir=tmp_path)))
    bad = client.post('/scholar/answer', json={'question': 'Which methods help?', 'model_id': 'unapproved/model'})
    assert bad.status_code == 422
