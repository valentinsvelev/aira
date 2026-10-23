"""Benchmark runs: dataset loading, seeded sampling, real predictions, scoring and the job API."""
import time
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from aira import benchmarks
from aira.api import create_app
from aira.config import Settings


def test_hf_qasper_parse_question_gold_and_context():
    paper = {
        'title': 'Low rank language models', 'abstract': 'Low rank adapters reduce trainable parameters.',
        'full_text': {'section_name': ['Introduction'], 'paragraphs': [['LoRA learns low rank adapters.']]},
        'qas': {'question_id': ['abc'], 'question': ['How are adapters trained?'],
                'answers': [[{'answer': {'unanswerable': False, 'free_form_answer': 'With low rank matrices',
                                          'extractive_spans': [], 'yes_no': None}}]]},
    }
    cases = benchmarks.cases_for_benchmark('qasper', [paper], 1)
    assert len(cases) == 1
    assert cases[0]['gold'] == ['With low rank matrices']
    assert 'low rank adapters' in cases[0]['context']


def test_druid_and_real_predictions(monkeypatch):
    rows = [{'id': 'row1', 'claim': 'Mars is a planet', 'evidence': 'Mars is the fourth planet.',
             'relevant': True, 'evidence_stance': 'supports'}]
    cases = benchmarks.cases_for_benchmark('druid', rows, 1)
    class Model:
        def invoke(self, prompt):
            assert 'Mars is a planet' in prompt
            return SimpleNamespace(content='{"relevant": true, "stance": "supports"}')
    monkeypatch.setattr('aira.agent.make_model', lambda *args: Model())
    result = benchmarks.run_cases('druid', cases, SimpleNamespace())
    assert result['metrics'] == {'relevance_accuracy': 1.0, 'stance_accuracy': 1.0}
    assert result['sample_size'] == 1


def test_custom_scoring_and_retrieval(monkeypatch):
    class Model:
        def invoke(self, prompt):
            assert 'From my PDF' in prompt
            return SimpleNamespace(content='Low rank adapters')
    class Retriever:
        def search(self, question, top_k):
            assert top_k == 4
            return [{'text': 'From my PDF: LoRA uses low rank adapters.'}]
    monkeypatch.setattr('aira.agent.make_model', lambda *args: Model())
    result = benchmarks.run_cases('custom', [{'id': '1', 'question': 'How does LoRA work?',
                         'gold': 'low rank adapters'}], SimpleNamespace(), Retriever())
    assert result['metrics']['answer_token_f1'] == 1.0


def test_dataset_download_uses_allowlisted_url_and_bounded_rows(monkeypatch):
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={'rows': [{'row': {'claim': 'a', 'evidence': 'b'}}]})
    client = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: client(
        transport=httpx.MockTransport(respond), **{k: v for k, v in kwargs.items() if k != 'timeout'}))
    rows, source = benchmarks.dataset_rows('druid', 11, 40)
    assert len(rows) == 1
    assert requests[0].url.host == 'datasets-server.huggingface.co'
    assert requests[0].url.params['length'] == '20'
    assert requests[0].url.params['config'] == 'DRUID'
    assert '11' in source


def test_benchmark_api_explicitly_runs_custom_and_returns_503_on_dataset_failure(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from aira.api import create_app
    from aira.config import Settings
    papers = tmp_path / 'papers'
    papers.mkdir()
    app = create_app(Settings(database_url=f'sqlite:///{tmp_path / "bench.db"}',
                              papers_dir=papers, llm_provider='openai'))
    client = TestClient(app)
    def fake_run(source, cases, settings, retriever=None):
        assert source == 'custom' and len(cases) == 1
        assert cases[0]['question'] == 'What is LoRA?'
        return {'sample_size': 1, 'metrics': {'answer_token_f1': 1.0}, 'results': [],
                'protocol_note': 'Mock scored one example'}
    monkeypatch.setattr(benchmarks, 'run_cases', fake_run)
    response = client.post('/benchmarks/run', json={
        'dataset': 'custom', 'count': 1,
        'examples': [{'question': 'What is LoRA?', 'answer': 'Low-rank adaptation', 'context': 'A paper'}],
    })
    assert response.status_code == 200
    assert response.json()['metrics']['answer_token_f1'] == 1.0
    assert client.post('/benchmarks/run', json={'dataset': 'custom', 'count': 1}).status_code == 422
    def down(*args):
        raise RuntimeError('Could not load public dataset sample')
    monkeypatch.setattr(benchmarks, 'dataset_rows', down)
    response = client.post('/benchmarks/run', json={'dataset':'qasper', 'count':1})
    assert response.status_code == 503
    assert 'dataset' in response.json()['detail']


def test_random_public_sample_uses_seed_and_not_first_row(monkeypatch):
    monkeypatch.setattr(benchmarks, 'dataset_size', lambda dataset: 100)
    called = []
    def rows(dataset, offset, length):
        called.append(offset)
        return ([{'id': str(offset), 'claim': 'claim', 'evidence': 'evidence',
                  'relevant': True, 'evidence_stance':'supports'}], 'page')
    monkeypatch.setattr(benchmarks, 'dataset_rows', rows)
    cases, provenance, seed = benchmarks.random_dataset_cases('druid', 3, seed=12345)
    assert len(cases) == 3 and all(index > 0 for index in called)
    assert [c['id'] for c in cases] == [str(i) for i in called[:3]]
    assert '12345' in provenance and seed == 12345


def test_dataset_size_uses_split_num_rows(monkeypatch):
    orig_client = httpx.Client
    def respond(request):
        return httpx.Response(200, json={'size': {'splits': [
            {'config':'qasper','split':'train','num_rows':600},
            {'config':'qasper','split':'validation','num_rows':42}]}})
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: orig_client(
        transport=httpx.MockTransport(respond),
        **{k:v for k,v in kwargs.items() if k != 'timeout'}))
    assert benchmarks.dataset_size('qasper') == 42


def test_full_custom_job_all_examples(monkeypatch):
    from aira.benchmark_jobs import get_job, run_all_job
    def score(src, cases, settings, retriever=None):
        return {'results': [{'id': c['id'], 'question': c['question'],
                             'prediction': 'answer', 'reference':c['gold'],
                             'metrics':{'answer_f1':.75}} for c in cases]}
    monkeypatch.setattr(benchmarks, 'run_cases', score)
    examples = [{'question':f'Q{i}', 'answer':f'A{i}', 'context':'context'} for i in range(13)]
    settings = SimpleNamespace(ollama_model='qwen3.5:4b', llm_provider='ollama')
    job_id = run_all_job(dataset='custom', examples=examples, settings=settings,
                         retriever=None, save_results=False)
    for _ in range(100):
        state = get_job(job_id)
        if state['status'] in ('completed','failed'):
            break
        time.sleep(.02)
    assert state['status'] == 'completed', state['error']
    assert state['processed'] == state['total'] == 13
    assert state['result']['sample_size'] == 13
    assert state['result']['metrics']['answer_token_f1'] == .75


def test_full_benchmark_api_exposes_progress_and_cancel(tmp_path, monkeypatch):
    from aira import benchmarks
    monkeypatch.setattr(benchmarks, 'run_cases', lambda _src, cases, _settings, retriever=None: {
        'results': [{'id': c['id'], 'question': c['question'], 'prediction': 'A',
                     'reference': c['gold'], 'metrics': {'answer_f1': 1.0}} for c in cases]})
    client = TestClient(create_app(Settings(database_url=f'sqlite:///{tmp_path}/jobs.db',
                            papers_dir=tmp_path, llm_provider='openai')))
    response = client.post('/benchmarks/jobs', json={'dataset': 'custom', 'save_results': False,
        'examples': [{'question':'Why?', 'answer':'A', 'context':'Some paper text'}]})
    assert response.status_code == 202
    job_id = response.json()['job_id']
    state = {}
    for _ in range(100):
        state = client.get(f'/benchmarks/jobs/{job_id}').json()
        if state['status'] in ('completed','failed'):
            break
        time.sleep(.02)
    assert state['status'] == 'completed', state.get('error')
    assert state['result']['sample_size'] == 1
    assert client.delete(f'/benchmarks/jobs/{job_id}').status_code == 200
    assert client.get('/benchmarks/jobs/not-a-job').status_code == 404
