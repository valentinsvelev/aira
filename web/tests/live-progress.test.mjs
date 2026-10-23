// Live request progress and the live workflow canvas.
import test from 'node:test';
import assert from 'node:assert/strict';
import { source } from './helpers.mjs';

const researchWorkspace = source('components/research-workspace.tsx');
const liveWorkflowCanvas = source('components/live-workflow-canvas.tsx');
const proxyRoute = source('app/api/backend/[...path]/route.ts');

test("research request polls server operation progress without revealing chain-of-thought", () => {
  const ui = source('components/research-workspace.tsx');
  const proxy = source('app/api/backend/[...path]/route.ts');
  assert.match(ui, /progress_id: progressId/);
  assert.match(ui, /chat\/progress\/\$\{progressId\}/);
  assert.match(proxy, /chat\\\/progress/);
});

test('research chat attaches live workflow visual to actual request progress', () => {
  assert.match(researchWorkspace, /setLiveWorkflow\(progress\.workflow\)/);
  assert.match(researchWorkspace, /<LiveWorkflowCanvas progress=\{liveWorkflow\}/);
  assert.match(researchWorkspace, /setLiveWorkflow\(null\)/);
});

test('read-only graph has per-node statuses, loop visits, and explicit branch edges', () => {
  assert.match(liveWorkflowCanvas, /status === "running"/);
  assert.match(liveWorkflowCanvas, /data\.visits > 1/);
  assert.match(liveWorkflowCanvas, /edge\.condition === origin\.branch/);
  assert.match(liveWorkflowCanvas, /nodesDraggable=\{false\}/);
  assert.match(liveWorkflowCanvas, /nodesConnectable=\{false\}/);
});

test('live React Flow nodes are inspectable without changing the graph itself', () => {
  assert.match(liveWorkflowCanvas, /onNodeClick=\{/);
  assert.match(liveWorkflowCanvas, /nodesDraggable=\{false\}/);
  assert.match(liveWorkflowCanvas, /aira-node-inspector/);
  assert.match(liveWorkflowCanvas, /selectedEvent\.latency_ms/);
  assert.match(liveWorkflowCanvas, /selectedEvent\.sources/);
  assert.match(liveWorkflowCanvas, /selectedEvent\.branch/);
  assert.match(liveWorkflowCanvas, /selectedEvent\.span_id/);
});

test('saved conversation graphs expose per-node inspector without auto-polling Phoenix', () => {
  assert.match(researchWorkspace, /progressFromSavedRun\(message\.workflowGraph/);
  assert.match(liveWorkflowCanvas, /async function loadPhoenix\(\)/);
  assert.match(liveWorkflowCanvas, /onClick=\{\(\) => void loadPhoenix\(\)\}/);
  assert.match(proxyRoute, /workflow\/runs\/\[0-9\]\+\/phoenix/);
});

test('repeated workflow node visits retain exact span identity', () => {
  assert.match(liveWorkflowCanvas, /nodeEvents\.map\(/);
  assert.match(liveWorkflowCanvas, /span\.span_id === selectedEvent\.span_id/);
});
