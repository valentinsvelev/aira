// Research Chat composer and the chat configuration dialog.
import test from 'node:test';
import assert from 'node:assert/strict';
import { source } from './helpers.mjs';

const researchWorkspace = source('components/research-workspace.tsx');

test('composer shows only one configuration button instead of the old toolbar', () => {
  const workspace = source('components/research-workspace.tsx');
  const composer = workspace.slice(workspace.indexOf('<div className="composer-area shared-chat-composer">'), workspace.indexOf('    {recentsDialog &&'));
  assert.ok(composer.length > 0, 'composer present');
  assert.match(composer,/aria-label="Configure research chat"/);
  assert.match(composer,/chat-config-overlay/);
  assert.match(composer,/aria-modal="true"/);
  assert.match(composer,/Agentic RAG/);
  assert.match(composer,/Academic search/);
  assert.match(composer,/RAG systems/);
  assert.match(composer,/research agents/);
  assert.doesNotMatch(composer,/Advanced retrieval and generation settings/);
  assert.doesNotMatch(composer, /className="composer-mode-group"/);
  assert.doesNotMatch(composer, /className="composer-search-options"/);
});

test('research agent selector is only shown for Agentic RAG', () => {
  assert.match(researchWorkspace, /\{mode === "chat" && <div className="chat-config-section"><label[^>]*htmlFor="chat-config-agent"/);
  assert.match(researchWorkspace, /const currentChatAgentId = mode === "chat" \? agentProfileId : "default"/);
  assert.match(researchWorkspace, /agent_profile_id: currentChatAgentId/);
});

test('advanced settings are replaced by a working configuration link', () => {
  const dialog = researchWorkspace.slice(researchWorkspace.indexOf('{settingsOpen && <div className="chat-config-overlay"'), researchWorkspace.indexOf('    {inspectorOpen &&'));
  assert.match(dialog, /Configure \{mode === "answer"/);
  assert.match(dialog, /setView\(mode === "answer" \? "rag" : "agents"\)/);
  assert.doesNotMatch(dialog, /Advanced retrieval and generation settings|<ParameterControls/);
});

test('chat context selector has automatic model maximum option',()=>{
 assert.match(source('components/parameter-controls.tsx'), /Model maximum \(auto\)/);
});

test('Research Chat and Paper Explorer share identical new-chat composer geometry', () => {
  const chat = source('components/research-workspace.tsx');
  const paper = source('components/literature-workspace.tsx');
  const css = source('app/globals.css');
  assert.match(chat, /className="composer-area shared-chat-composer"/);
  assert.match(paper, /className="composer-area paper-chat-composer shared-chat-composer"/);
  assert.match(chat, /composer-new-chat/);
  assert.match(paper, /composer-new-chat/);
  assert.match(chat, /rows=\{1\} aria-label="Research question"/);
  assert.match(paper, /rows=\{1\} maxLength=\{500\}/);
  assert.match(css, /\.shared-chat-composer \.composer-new-chat\{[\s\S]*?max-width:740px/);
  assert.match(css, /\.shared-chat-composer \.composer-new-chat textarea\{/);
});

test('research chat and Paper Explorer hide the inset divider', () => {
  const css = source('app/globals.css');
  assert.match(css, /\.composer-footer-simplified::before\{content:none!important/);
});
