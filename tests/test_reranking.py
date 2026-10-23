"""Reranking contracts with zero model downloads or GPU requirements."""
import pytest
from pydantic import ValidationError

from aira import reranker_service, reranking
from aira.options import RetrievalOptions
from aira.retrieval import LocalRetriever

HITS = [
    {'id':'a','text':'wrong result', 'score':0.9, 'path':'a.pdf', 'page':1},
    {'id':'b','text':'correct result','score':0.8, 'path':'b.pdf', 'page':2},
    {'id':'c','text':'related result','score':0.5, 'path':'c.pdf', 'page':3},
]


def test_options_and_validation():
    assert RetrievalOptions().reranker == 'none'
    assert RetrievalOptions.model_validate({'mode':'bm25'}).mode == 'sparse'
    assert RetrievalOptions(reranker='jinaai/jina-reranker-v3.5',rerank_candidates=9).reranker.endswith('v3.5')
    with pytest.raises(ValidationError):
        RetrievalOptions(reranker='fake')
    with pytest.raises(ValidationError):
        RetrievalOptions(reranker='jinaai/jina-reranker-v3.5',rerank_candidates=3,top_k=5)


def test_reranker_reorders_and_keeps_provenance(monkeypatch):
    captured = {}
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'results':[{'index':2,'relevance_score':0.1},
                                            {'index':1,'relevance_score':0.9},
                                            {'index':0,'relevance_score':0.5}]}
    def mock_post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()
    monkeypatch.setattr(reranking.httpx,'post',mock_post)
    output = reranking.rerank_hits('question', HITS, reranking.RERANK_MODELS[0], 2)
    assert [v['id'] for v in output] == ['b','a']
    assert output[0]['path'] == 'b.pdf' and output[0]['page']==2
    assert output[0]['first_stage_score'] == 0.8
    assert output[0]['score'] == 0.9
    assert captured['json']['documents'] == [x['text'] for x in HITS]


@pytest.mark.parametrize('ranking', [
    [{'index':0,'relevance_score':float('nan')},{'index':1,'relevance_score':0}, {'index':2,'relevance_score':1}],
    [{'index':0,'relevance_score':1},{'index':0,'relevance_score':2},{'index':2,'relevance_score':3}],
    [{'index':0,'relevance_score':1}],
])
def test_invalid_rerank_results_rejected(monkeypatch,ranking):
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'results':ranking}
    monkeypatch.setattr(reranking.httpx,'post',lambda *args,**kw:Response())
    with pytest.raises(RuntimeError):
        reranking.rerank_hits('q',HITS,reranking.RERANK_MODELS[1],2)


def test_service_gte_pairwise_contract(monkeypatch):
    class FakeGTE:
        def predict(self,pairs):
            assert pairs==[('query','first'),('query','second')]
            return [0.1,0.7]
    monkeypatch.setattr(reranker_service,'load_model',lambda model:FakeGTE())
    assert reranker_service.score_documents('query',['first','second'],reranker_service.GTE) == [
        {'index':1,'relevance_score':0.7},{'index':0,'relevance_score':0.1}]


def test_service_jina_listwise_contract(monkeypatch):
    class FakeJina:
        def rerank(self,q,documents):
            assert q=='query' and documents==['first','second']
            return [{'index':1,'relevance_score':.8},{'index':0,'relevance_score':.2}]
    monkeypatch.setattr(reranker_service,'load_model',lambda model:FakeJina())
    assert [v['index'] for v in reranker_service.score_documents('query',['first','second'],reranker_service.JINA)] == [1,0]


def test_retrieval_applies_rerank_after_first_stage(monkeypatch):
    r=LocalRetriever.__new__(LocalRetriever)
    r.index=type('DummyIndex',(),{'model_name':'sentence-transformers/all-MiniLM-L6-v2'})()
    r.session_factory=object()
    monkeypatch.setattr(r,'_bm25_search',lambda q,k:HITS[:k])
    captured=[]
    def fake_rerank(query,hits,model,k):
        captured.append((query,[x['id'] for x in hits],model,k))
        return [hits[1]]
    monkeypatch.setattr(reranking,'rerank_hits',fake_rerank)
    opts=RetrievalOptions(mode='sparse',top_k=1,bm25_candidates=3,
                          reranker=reranking.RERANK_MODELS[1],rerank_candidates=3)
    assert r._search('my query',options=opts)[0]['id']=='b'
    assert captured==[('my query',['a','b','c'],reranking.RERANK_MODELS[1],1)]
    opts=RetrievalOptions(mode='sparse',top_k=1,bm25_candidates=3)
    assert r._search('my query',options=opts)[0]['id']=='a'
