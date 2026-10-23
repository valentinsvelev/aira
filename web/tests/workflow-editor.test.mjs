// Visual workflow editor: executable graphs, gates, loops and playback.
import test from 'node:test';
import assert from 'node:assert/strict';
import { source } from './helpers.mjs';

const executableAgentFlow = source('components/executable-agent-flow.tsx');
const advancedAgentFlow = source('components/advanced-agent-flow.tsx');
const proxyRoute = source('app/api/backend/[...path]/route.ts');

test('workflow canvas is independently selectable from tool-permissions editor',()=>{
  const ui=source('components/agent-workspace.tsx');
  assert.match(ui,/Executable workflow/);
  assert.match(ui,/onClick=\{\(\) => edit\(profile, "workflow"\)\}/);
  assert.match(ui,/ExecutableAgentFlow workflow=\{workflow\}/);
  assert.match(ui,/method:"PUT",body:JSON\.stringify\(workflow\)/);
});

test('graph nodes render in React Flow and support safe edge editing',()=>{
  const ui=source('components/executable-agent-flow.tsx');
  assert.match(ui,/ReactFlow<CanvasNode>/);
  assert.match(ui,/isValidConnection/);
  assert.match(ui,/onEdgesDelete/);
  assert.match(ui,/onNodesChange/);
  assert.match(ui,/Execute this graph in Agentic RAG/);
});

test('proxy accepts workflow get/save, chat displays saved execution steps',()=>{
  const proxy=source('app/api/backend/[...path]/route.ts');
  assert.match(proxy,/PUT: \[new RegExp\(`/);
  assert.match(proxy,/agents\/\$\{agentId\}\/workflow/);
  const chat=source('components/research-workspace.tsx');
  assert.match(chat,/Executed visual workflow/);
  assert.match(chat,/workflowSteps: turn\.metadata\.workflow_steps/);
  assert.match(chat,/setMode\("chat"\); setView\("research"\)/);
});

test('workflow visual editor supports gating, concurrent retrieval and playback stages',()=>{
  const ui=source('components/executable-agent-flow.tsx');
  assert.match(ui,/Add conditional evidence check/);
  assert.match(ui,/Minimum evidence sources/);
  assert.match(ui,/Parallel independent searches/);
  assert.match(ui,/Recorded workflow playback/);
  assert.match(ui,/Load Phoenix spans/);
  assert.match(ui,/sourceHandle/);
  assert.match(ui,/onNodesChange/);
});

test('Phoenix playback is manually requested, scoped to saved run and proxied',()=>{
  const proxy=source('app/api/backend/[...path]/route.ts');
  assert.match(proxy,/agents\/\$\{agentId\}\/workflow\/runs\/\[0-9\]\+\/phoenix/);
  const builder=source('components/agent-workspace.tsx');
  assert.match(builder,/profileId=\{editing\}/);
});

test("advanced mode is separate from the legacy editor",()=>{
  assert.match(executableAgentFlow,/schema_version === 2/);
  assert.match(executableAgentFlow,/Switch to advanced editor/);
  assert.match(executableAgentFlow,/LegacyExecutableAgentFlow/);
});

test("advanced editor provides repeat and condition handles plus limits",()=>{
  assert.match(advancedAgentFlow,/id=\{data\.kind==="loop"\?"repeat":"enough"\}/);
  assert.match(advancedAgentFlow,/id=\{data\.kind==="loop"\?"done":"insufficient"\}/);
  assert.match(advancedAgentFlow,/Maximum repetitions/);
  assert.match(advancedAgentFlow,/Maximum workflow node executions/);
  assert.match(advancedAgentFlow,/query_template/);
});

test("backend proxy exports PUT for saving agent workflows",()=>{
  assert.match(proxyRoute,/export const PUT = forward/);
  assert.match(proxyRoute,/PUT:\s*\[new RegExp/);
});

test("advanced playback highlights repeated node visits",()=>{
  assert.match(advancedAgentFlow,/recent\.get\(n\.id\)/);
  assert.match(advancedAgentFlow,/step\.iteration/);
  assert.match(advancedAgentFlow,/Load Phoenix spans/);
});
