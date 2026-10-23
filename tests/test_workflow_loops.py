"""Nested gates, explicit loop edges, safety budgets and schema compatibility."""
from types import SimpleNamespace

import pytest

from aira.agent_workflows import WorkflowDefinition, execute_workflow, starter_workflow
from aira.config import Settings
from aira.options import GenerationOptions, RetrievalOptions


def nested():
    # A condition nested inside an outer loop; an inner loop retries retrieval
    # and resets each time the outer loop returns to its repeat branch.
    nodes = [
        ("q", "question"), ("outer", "loop"), ("gate", "evidence_gate"),
        ("inner", "loop"), ("lookup", "local_search"),
        ("write", "synthesize"), ("done", "answer"),
    ]
    edges = [
        ("q", "outer", None),
        ("outer", "gate", "repeat"), ("outer", "write", "done"),
        ("gate", "outer", "enough"), ("gate", "inner", "insufficient"),
        ("inner", "lookup", "repeat"), ("inner", "outer", "done"),
        ("lookup", "inner", None), ("write", "done", None),
    ]
    return {"schema_version":2,"enabled":True,"parallel":True,"max_steps":80,
        "nodes":[{"id":id,"type":kind,"x":i*70,"y":0,
                  "min_sources":18,"max_iterations":2,
                  "query_template":"{question} attempt {iteration}"} for i,(id,kind) in enumerate(nodes)],
        "edges":[{"source":a,"target":b,"condition":c} for a,b,c in edges]}


def run(graph,monkeypatch,tmp_path, hits=None):
    import aira.agent as agent
    class Model:
        model="offline"
        def invoke(self, messages, **kw):
            return SimpleNamespace(content="Supported conclusion [1].")
    monkeypatch.setattr(agent,"make_model",lambda *a,**kw:Model())
    requests=[]
    class Retriever:
        def search(self, *,query,options):
            requests.append(query)
            if hits is not None:
                return hits
            return [{"id":f"paper-{len(requests)}","path":"science.pdf","text":"Useful scientific evidence", "page":len(requests)}]
    result=execute_workflow(graph,question="Evaluate RAG",session_id="test",settings=Settings(papers_dir=tmp_path),
        factory=None,retriever=Retriever(),retrieval=RetrievalOptions(),generation=GenerationOptions(),
        profile={"enabled_tools":["local_search"]},use_web=False,use_academic=False)
    return result,requests


def test_old_schema_unchanged():
    saved=starter_workflow()
    assert WorkflowDefinition.model_validate(saved).schema_version==1


def test_nested_gate_loop_validation_and_execution(monkeypatch,tmp_path):
    graph=nested()
    assert WorkflowDefinition.model_validate(graph).schema_version==2
    result,queries=run(graph,monkeypatch,tmp_path)
    assert len(queries)==4, queries  # 2 inner attempts in each of 2 outer cycles
    assert len({x['id'] for x in result['sources']})==4
    assert result['answer'].endswith('[1].')
    assert [s['branch'] for s in result['workflow_steps'] if s['node']=='outer']==['repeat','repeat','done']
    assert len([s for s in result['workflow_steps'] if s['node']=='gate'])==2
    assert result['workflow_graph']['schema_version']==2
    assert [s['iteration'] for s in result['workflow_steps'] if s['node']=='inner']==[1,2,2,1,2,2]


def test_gate_takes_enough_branch(monkeypatch,tmp_path):
    graph=nested()
    graph['nodes'][2]['min_sources']=1
    # Add an early search before outer gate so the condition sees evidence.
    graph['nodes'].append({"id":"pre","type":"local_search"})
    graph['edges'].remove({"source":"q","target":"outer","condition":None})
    graph['edges'] += [{"source":"q","target":"pre"},{"source":"pre","target":"outer"}]
    result,queries=run(graph,monkeypatch,tmp_path)
    assert queries==["Evaluate RAG"]
    assert [s['branch'] for s in result['workflow_steps'] if s['node']=='gate']==['enough','enough'] # short-circuit inner search


def test_invalid_unbounded_cycle_rejected():
    graph={"schema_version":2,"nodes":[{"id":"q","type":"question"},
           {"id":"a","type":"local_search"}, {"id":"b","type":"local_search"},
           {"id":"s","type":"synthesize"}, {"id":"z","type":"answer"}],
           "edges":[{"source":"q","target":"a"}, {"source":"a","target":"b"},
                    {"source":"b","target":"a"}, {"source":"a","target":"s"},
                    {"source":"s","target":"z"}]}
    with pytest.raises(ValueError,match="Cycles must"):
        WorkflowDefinition.model_validate(graph)


def test_loop_must_have_both_branch_labels():
    graph=nested()
    graph['edges']=[e for e in graph['edges'] if not (e['source']=='inner' and e['condition']=='done')]
    with pytest.raises(ValueError,match="exactly one link"):
        WorkflowDefinition.model_validate(graph)


def test_overall_limit_aborts_before_infinite_execution(monkeypatch,tmp_path):
    graph=nested()
    graph['max_steps']=8
    with pytest.raises(RuntimeError,match="safety limit"):
        run(graph,monkeypatch,tmp_path,hits=[])


def test_graph_bad_templates_rejected():
    graph=nested()
    next(n for n in graph['nodes'] if n['id']=='lookup')['query_template']='{question} {secret}'
    with pytest.raises(ValueError,match="query supports only"):
        WorkflowDefinition.model_validate(graph)


def test_nested_graph_api_save_reload_and_tools_permission(tmp_path):
    from fastapi.testclient import TestClient

    from aira.api import create_app
    base=Settings(database_url=f"sqlite:///{tmp_path/'loops.sqlite'}", papers_dir=tmp_path)
    c=TestClient(create_app(base))
    agent=c.post("/agents",json={"name":"Nested test", "enabled_tools":["local_search"]}).json()
    g=nested()
    saved=c.put(f"/agents/{agent['id']}/workflow",json=g)
    assert saved.status_code==200,saved.text
    assert saved.json()['workflow']['schema_version']==2
    assert c.get(f"/agents/{agent['id']}/workflow").json()['workflow']==saved.json()['workflow']
    g['nodes'].append({"id":"other","type":"web_search"})
    g['edges'].extend([{"source":"q","target":"other"},{"source":"other","target":"outer"}])
    response=c.put(f"/agents/{agent['id']}/workflow",json=g)
    assert response.status_code==422 and 'permitted' in response.text


def test_duplicate_conditions_and_broken_loop_exit():
    g=nested()
    g['edges'].append({"source":"gate","target":"outer","condition":"enough"})
    with pytest.raises(ValueError,match="Duplicate"):
        WorkflowDefinition.model_validate(g)
    g=nested()
    next(e for e in g['edges'] if e['source']=='outer' and e['condition']=='done')['target']='inner'
    with pytest.raises(ValueError,match="reachable|done branch"):
        WorkflowDefinition.model_validate(g)
