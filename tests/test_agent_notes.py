"""Agent note edits go through the real LangGraph tool path, verification and audit log."""
import json
import sys
import types
from types import SimpleNamespace

from aira.agent import _explicit_note_edit_request, run_agent
from aira.config import Settings
from aira.context import ContextManager
from aira.db import EvidenceChunk
from aira.ingestion import digest

QUOTE = 'The experiment reports a 15 percent speedup for batch size one.'


def test_note_tool_edits_are_requested_verified_audited_and_reused(workspace, monkeypatch):
    import aira.agent as agent_module
    factory = workspace['factory']
    source_id = digest('tool-edit-document')
    with factory.begin() as session:
        session.add(EvidenceChunk(id=source_id, source_kind='web_snippet',
                                  source_url='https://example.org/paper', page=None,
                                  ordinal=0, text=QUOTE, text_sha256=digest(QUOTE)))
    context = ContextManager(factory)
    class Retriever:
        def search(self, query, top_k=5, options=None):
            return [{'id':source_id,'path':'paper.pdf','version':1,'page':1,
                     'score':1.0,'text':QUOTE}]
    seen = []
    def fake_create_agent(model, tools, system_prompt):
        seen.append({fn.__name__:fn for fn in tools})
        class FakeAgent:
            def invoke(self, payload, config=None):
                if 'propose_context_note' in seen[-1]:
                    saved = json.loads(seen[-1]['propose_context_note'](
                        claim=QUOTE, exact_source_quote=QUOTE, evidence_id=source_id))
                    assert saved['accepted'] is True
                return {'messages':[SimpleNamespace(content='Grounded response [1].')]}
        return FakeAgent()
    agents = types.ModuleType('langchain.agents')
    agents.create_agent = fake_create_agent
    tools = types.ModuleType('langchain.tools')
    tools.tool = lambda fn:fn
    root = types.ModuleType('langchain')
    root.agents = agents
    root.tools = tools
    monkeypatch.setitem(sys.modules,'langchain',root)
    monkeypatch.setitem(sys.modules,'langchain.agents',agents)
    monkeypatch.setitem(sys.modules,'langchain.tools',tools)
    monkeypatch.setattr(agent_module,'make_model',lambda *a,**kw:object())
    monkeypatch.setattr(agent_module,'callbacks',lambda:[])
    params = {'session_id':'session-1', 'settings':Settings(), 'session_factory':factory,
              'retriever':Retriever(), 'context':context}
    # Ordinary research prompts must not expose mutation tools to the model.
    run_agent('What is the observed speedup?', **params)
    assert 'propose_context_note' not in seen[-1]
    assert context.notes('session-1')['revision'] == 0
    run_agent('Remember this paper finding as a note', **params)
    assert 'propose_context_note' in seen[-1]
    assert context.notes('session-1')['revision'] == 1
    assert len(context.notes('session-1')['notes']) == 1
    assert context.events('session-1')[-1]['status'] == 'accepted'
    # The next turn sees that previously saved note as provisional working memory.
    _,_,snapshot = agent_module.prepare_question('Tell me about the finding', 'session-1',
                                                  Retriever(), context)
    assert snapshot['revision'] == 1
    assert QUOTE in agent_module.build_research_context(snapshot,[])
    assert not _explicit_note_edit_request('Just explain this abstract')
    assert _explicit_note_edit_request('Please delete the note')
