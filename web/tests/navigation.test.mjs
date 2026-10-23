// Sidebar, breadcrumbs and the Configuration hub.
import test from 'node:test';
import assert from 'node:assert/strict';
import { source } from './helpers.mjs';

const researchWorkspace = source('components/research-workspace.tsx');
const configurationWorkspace = source('components/configuration-workspace.tsx');

test('configuration is a standalone page without a sidebar dropdown', () => {
  const bottom = researchWorkspace.slice(researchWorkspace.indexOf('<nav className="sidebar-bottom-nav"'), researchWorkspace.indexOf('<div className="sidebar-footer">'));
  assert.doesNotMatch(bottom, /nav-section-toggle|sidebar-configuration|aria-expanded|configurationExpanded|evaluationExpanded/);
  assert.match(researchWorkspace, /type WorkspaceView =[^;]*"configuration"/);
  assert.match(researchWorkspace, /view === "configuration" \? <ConfigurationWorkspace/);
  assert.match(researchWorkspace, /setView\("configuration"\)/);
  for (const section of ['Research agents', 'Your library', 'Evaluation and Debugging', 'Behind the scenes', 'Experiments on Benchmark Datasets']) {
    assert.ok(configurationWorkspace.includes(section), `${section} missing from Configuration page`);
  }
  for (const dest of ['agents', 'library', 'traces', 'experiments']) {
    assert.ok(configurationWorkspace.includes(`onNavigate("${dest}")`));
  }
});

test('redundant Workspace labels are gone, but branding is retained', () => {
  assert.doesNotMatch(researchWorkspace, /className="side-section-caption">WORKSPACE/);
  assert.doesNotMatch(researchWorkspace, /className="breadcrumb">Workspace/);
  assert.match(researchWorkspace, /<strong>AIRA<\/strong>/);
});

test('configuration and about remain docked above the sidebar footer', () => {
  const topNav = researchWorkspace.indexOf('<nav className="nav-list"');
  const bottomNav = researchWorkspace.indexOf('<nav className="sidebar-bottom-nav"');
  const footer = researchWorkspace.indexOf('<div className="sidebar-footer">');
  assert.ok(topNav >= 0 && topNav < bottomNav && bottomNav < footer);
  const bottom = researchWorkspace.slice(bottomNav, footer);
  assert.match(bottom, /title="Configuration"/);
  assert.match(bottom, /title="About AIRA"/);
  assert.ok(bottom.indexOf('title="Configuration"') < bottom.indexOf('title="About AIRA"'));
 });

test('topbar breadcrumbs show Research chat and Configuration parent', () => {
  assert.match(researchWorkspace, /topbar-parent-link[^>]+onClick=\{showFreshChat\}>Research chat/);
  assert.match(researchWorkspace, /topbar-parent-link[^>]+onClick=\{\(\) => setView\("configuration"\)\}>Configuration/);
  assert.match(researchWorkspace, /conversations\.find\(\(item\) => item\.id === sessionId\)\?\.title \|\| "New chat"/);
});

test('config children all share a back button', () => {
  assert.match(researchWorkspace, /\["agents", "rag", "library", "traces", "experiments", "retrieval-eval"\]\.includes\(view\) && <div className="config-back-container"/);
  assert.match(researchWorkspace, /Back to Configuration/);
});

test('configuration shows dynamic library and agent counts', () => {
  const code=source('components/configuration-workspace.tsx');
  assert.match(code, /Papers available: \{documentCount\} · Indexed chunks: \{chunkCount\}/);
  assert.match(code, /Agents available: \{agentCount\}/);
  assert.match(code, /Systems available: \{ragCount\}/);
});
