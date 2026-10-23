// Behind the Scenes: trace timeline, graph layout and Phoenix sessions.
import test from 'node:test';
import assert from 'node:assert/strict';
import { layoutResearchTrace } from '../lib/trace-layout.ts';
import { source } from './helpers.mjs';

const tracesWorkspace = source('components/traces-workspace.tsx');

test("graph collapses internal chain spans and permits root input-output selection", () => {
  const graph = layoutResearchTrace({id:"tr", duration_us:2000, spans: [
    {id:"root",name:"research.chat",parent_id:null,kind:"agent",start_us:0,duration_us:2000,input:"Hello",output:"World"},
    {id:"chain",name:"agent.chain",parent_id:"root",kind:"agent",start_us:1,duration_us:1900},
    {id:"llm",name:"llm.chat",parent_id:"chain",kind:"model",start_us:2,duration_us:1800}
  ]});
  assert.ok(!graph.nodes.some(n=>n.label==="agent.chain"));
  assert.equal(graph.nodes.find(n=>n.key==="input")?.spanId, "root");
  assert.equal(graph.nodes.find(n=>n.key==="output")?.spanId, "root");
  assert.ok(graph.links.some(link => link.from==="input" && link.to==="llm"));
  assert.ok(graph.links.some(link => link.from==="llm" && link.to==="output" && link.relation==="timeline"));
  assert.ok(!graph.nodes.some(n=>n.key==="agent"));
});

test('trace layout hides root wrapper and its input/output previews', () => {
  const graph = source('lib/trace-layout.ts');
  const css = source('app/globals.css');
  assert.match(graph,/key: "input"/);
  assert.match(graph,/Click to inspect input/);
  assert.match(graph,/Click to inspect output/);
  assert.match(graph,/s\.id !== root\?\.id/);
  assert.match(graph,/relation: isParent \? "parent" : "timeline"/);
  assert.match(css,/\.trace-graph-output\{border:2px solid #f472b6/);
});

test('research timeline shows human-readable chronological steps with optional details', () => {
  assert.match(tracesWorkspace, /trace-journey-steps/);
  assert.match(tracesWorkspace, /layoutResearchTrace/);
  assert.match(tracesWorkspace, /How your answer was made/);
  assert.match(tracesWorkspace, /Searching your papers/);
  assert.match(tracesWorkspace, /Recorded operation:/);
  assert.doesNotMatch(tracesWorkspace, /Fit restores full view/);
  assert.doesNotMatch(tracesWorkspace, /Tracing is local and opt-in/);
});

test('trace graph orders actual spans in time and omits the root as a step', () => {
  const spans = [
    {id:'root',parent_id:null,name:'research.chat',kind:'agent',start_us:0,duration_us:8000000},
    {id:'web',parent_id:'root',name:'web.search',kind:'tool',start_us:200,duration_us:1500000},
    {id:'pdf',parent_id:'root',name:'retrieval.search',kind:'retrieval',start_us:500,duration_us:2000000},
    {id:'llm',parent_id:'root',name:'llm.chat',kind:'model',start_us:1000,duration_us:4000000},
  ];
  const graph = layoutResearchTrace({id:'t1',duration_us:8000000,spans});
  assert.equal(graph.nodes.length,5);
  assert.ok(!graph.nodes.some(n=>n.key==='agent'));
  assert.ok(graph.links.some(e=>e.from==='input' && e.to==='web' && e.relation==='timeline'));
  assert.ok(graph.links.some(e=>e.from==='web' && e.to==='pdf' && e.relation==='timeline'));
  assert.ok(graph.links.some(e=>e.from==='llm' && e.to==='output'));
  assert.ok(graph.nodes.find(n=>n.key==='web').x < graph.nodes.find(n=>n.key==='pdf').x);
  assert.ok(graph.nodes.find(n=>n.key==='pdf').x < graph.nodes.find(n=>n.key==='llm').x);
  assert.equal(graph.nodes.find(n=>n.key==='output').detail,'Click to inspect output');
  assert.equal(graph.nodes.find(n=>n.key==='input').detail,'Click to inspect input');
});

test('trace graph preserves nesting and avoids cycles from malformed parents', () => {
  const spans=[{id:'root',parent_id:null,name:'research.chat',kind:'agent',start_us:0,duration_us:100000},
    {id:'tool',parent_id:'model',name:'search',kind:'tool',start_us:500,duration_us:1000},
    {id:'model',parent_id:'root',name:'llm.chat',kind:'model',start_us:200,duration_us:90000}];
  const graph=layoutResearchTrace({id:'t2',duration_us:100000,spans});
  assert.ok(graph.links.some(e=>e.from==='model'&&e.to==='tool'));
  assert.ok(graph.nodes.find(n=>n.key==='tool').x>graph.nodes.find(n=>n.key==='model').x);
});

test('trace workspace exposes session chooser and the new session API', () => {
  const view=source('components/traces-workspace.tsx');
  assert.match(view,/Conversation session/);
  assert.match(view,/\/sessions\?limit=60/);
  assert.match(source('app/api/backend/[...path]/route.ts'), /\/\^sessions\$/);
});
