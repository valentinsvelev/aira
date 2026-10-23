// Research-agent editor, tool permissions and generation defaults.
import test from 'node:test';
import assert from 'node:assert/strict';
import { AGENT_TOOL_CATALOG, disableAgentTool, enableAgentTool, normalizeAgentTools, toolNodeId, toolFromNodeId, validAgentToolConnection, readAgentFlowLayout, writeAgentFlowLayout } from '../lib/agent-flow-config.ts';
import { source } from './helpers.mjs';

const agentWorkspace = source('components/agent-workspace.tsx');
const researchWorkspace = source('components/research-workspace.tsx');
const globalStyles = source('app/globals.css');

test('required local retrieval cannot be disabled; unknown tools are never persisted', () => {
  assert.deepEqual(normalizeAgentTools(['__arbitrary_code__']), ['local_search']);
  const enabled = enableAgentTool([], 'web_search');
  assert.deepEqual(enabled, ['local_search', 'web_search']);
  assert.deepEqual(disableAgentTool(enabled, 'local_search'), enabled);
  assert.deepEqual(disableAgentTool(enabled, 'web_search'), ['local_search']);
});

test('only agent-to-known-tool links are valid (no arbitrary execution graphs)', () => {
  assert.equal(AGENT_TOOL_CATALOG.length, 6);
  assert.equal(validAgentToolConnection('agent', toolNodeId('context_edit')), true);
  assert.equal(validAgentToolConnection('question', toolNodeId('context_edit')), false);
  assert.equal(validAgentToolConnection('agent', 'answer'), false);
  assert.equal(validAgentToolConnection('agent', 'tool:exec_code'), false);
  assert.equal(toolFromNodeId('tool:academic_search'), 'academic_search');
  assert.equal(toolFromNodeId('tool:evil'), null);
});

test('layout storage validates untrusted browser data, never changes backend tool permissions', () => {
  const originalWindow = globalThis.window;
  const memory = new Map();
  globalThis.window = { localStorage: {
    getItem: id => memory.get(id) ?? null,
    setItem: (id, value) => memory.set(id, value),
  } };
  try {
    writeAgentFlowLayout('default', { positions: { agent: {x:123,y:456} }, visibleTools: ['local_search'] });
    assert.deepEqual(readAgentFlowLayout('default'), { positions: { agent: { x:123, y:456 } }, visibleTools: ['local_search'] });
    memory.set('aira.agent-flow-layout.corrupt', '{bad json');
    assert.equal(readAgentFlowLayout('corrupt'), null);
    memory.set('aira.agent-flow-layout.bad', JSON.stringify({visibleTools:['context_edit','unknown'],positions:{agent:{x:'abc',y:0}}}));
    assert.deepEqual(readAgentFlowLayout('bad'), {visibleTools:['local_search','context_edit'],positions:{}});
  } finally { globalThis.window = originalWindow; }
});

test('both editors save through existing agent API and canvas uses React Flow', () => {
  const agent = source('components/agent-workspace.tsx');
  const flow = source('components/agent-flow-builder.tsx');
  const pkg = JSON.parse(source('package.json'));
  assert.match(agent, /Form editor/);
  assert.match(agent, /Visual builder/);
  assert.match(agent, /postJSON<AgentProfile>\("\/agents", draft\)/);
  assert.match(agent, /patchJSON<AgentProfile>\(`\/agents\/\$\{editing\}`, draft\)/);
  assert.match(agent, /builderMode === "form"/);
  assert.match(agent, /onEnabledToolsChange=\{updateFlowToolPermissions\}/);
  assert.match(flow, /ReactFlow<FlowNode>/);
  assert.match(flow, /onEdgesDelete=/);
  assert.match(flow, /application\/x-aira-agent-tool/);
  assert.equal(pkg.dependencies['@xyflow/react'], '^12.12.0');
  assert.match(source('app/layout.tsx'), /@xyflow\/react\/dist\/style\.css/);
});

test('agent editor context maximum mode and right-content modal alignment',()=>{
 assert.match(source('components/agent-workspace.tsx'), /Model maximum \(auto\)/);
 const css=source('app/globals.css');
 assert.match(css, /var\(--expanded-sidebar-width,254px\)/);
 assert.match(css, /::-webkit-scrollbar-thumb/);
 assert.match(css, /\[data-theme="dark"\] \.agent-modal-overlay \.agent-editor-heading/);
});

test('default generation editor and sidebar resize remain available', () => {
  assert.match(agentWorkspace,/<h3>Generation settings for this agent<\/h3>/);
  assert.doesNotMatch(agentWorkspace,/editing !== "default" && <><h3>Generation settings/);
  assert.match(researchWorkspace,/sidebar-resize-handle/);
  assert.match(globalStyles,/min-width:254px/);
});

test('agent cards have no colored tool tags', () => {
  assert.doesNotMatch(source('components/agent-workspace.tsx'), /agent-tool-pills/);
});

test('agent sampler settings and paper model choices are configurable',()=>{
  const agent=source('components/agent-workspace.tsx');
  const scholar=source('components/literature-workspace.tsx');
  assert.match(agent,/"top_k", "Top-k sampling"/);
  assert.match(agent,/"min_p", "Min-p sampling"/);
  assert.match(scholar,/rl-research\/DR-Tulu-8B/);
  assert.match(scholar,/allenai\/AstaBrief_8B/);
  assert.match(scholar,/paper-chat-conversation/);
  assert.match(scholar,/paper-chat-composer/);
  assert.match(scholar,/aria-label="Configure Paper Explorer"/);
});

test('agent proxy explicitly allows built-in default profile', () => {
  const proxy=source('app/api/backend/[...path]/route.ts');
  assert.match(proxy, /\(\?:default\|/);
  assert.match(proxy, /agentId/);
});

test('Qwen 3.5 4B is the default for fresh research agents, RAG profiles and paper exploration', () => {
  const agent = source('components/agent-workspace.tsx');
  const rag = source('components/rag-workspace.tsx');
  const paper = source('components/literature-workspace.tsx');
  const controls = source('components/parameter-controls.tsx');
  assert.match(agent, /model: "qwen3\.5:4b", temperature:/);
  assert.match(agent, /draft\.generation\.model \|\| "qwen3\.5:4b"/);
  assert.match(rag, /draft\.generation\.model\?\?"qwen3\.5:4b"/);
  assert.match(controls, /generation\.model \?\? "qwen3\.5:4b"/);
  assert.match(paper, /useState<string>\("qwen3\.5:4b"\)/);
});
