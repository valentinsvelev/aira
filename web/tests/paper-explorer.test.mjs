// Paper Explorer: question-first literature answers with cited abstracts.
import test from 'node:test';
import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { source } from './helpers.mjs';



test('Paper Explorer stays discoverable from navigation with question-first heading', () => {
  const shell=source('components/research-workspace.tsx');
  const view=source('components/literature-workspace.tsx');
  assert.match(shell, /<span className="nav-label">Paper explorer<\/span>/);
  assert.match(view, /className="welcome paper-chat-welcome"/);
  assert.match(view, /let's dive into the <em>research literature\.<\/em>/);
});

test('one request retrieves, synthesizes and cites; shortlist workflow has gone', () => {
  const view=source('components/literature-workspace.tsx');
  assert.match(view, /\/scholar\/answer/);
  assert.match(view, /context_questions:/);
  assert.doesNotMatch(view, /\/literature\/reviews|Shortlist|Import PDF|Generate synthesis/);
});

test('source abstracts are browsable and honest about scope', () => {
  const view=source('components/literature-workspace.tsx');
  assert.match(view, /Papers consulted/);
  assert.match(view, /Read source abstract/);
  assert.match(view, /not verified full-text readings/);
});

test('chat composer divider and earlier trace timeline are retained', () => {
  const css=source('app/globals.css');
  assert.match(css, /\.composer-footer-simplified::before/);
  assert.match(css, /\.scholar-query-row/);
  assert.match(css, /\.trace-journey/);
});

test("scholar answer is a single question-first workflow, not a shortlist", () => {
  const ui = source('components/literature-workspace.tsx');
  assert.match(ui, /postJSON<ScholarAnswer>\("\/scholar\/answer"/);
  assert.match(ui, /context_questions: turns\.slice\(-3\)/);
  assert.doesNotMatch(ui, /\/literature\/reviews/);
  assert.doesNotMatch(ui, /Shortlist|Import PDF|Generate synthesis/);
});

test("citations resolve to a paper list with original source urls", () => {
  const ui = source('components/literature-workspace.tsx');
  assert.match(ui, /#scholar-paper-\$\{turnIndex\}-\$\{number\}/);
  assert.match(ui, /href=\{p\.url\}/);
  assert.match(ui, /p\.cited/);
  assert.match(ui, /abstracts, not verified full-text readings/);
  assert.match(source('app/api/backend/[...path]/route.ts'), /\^scholar\\\/answer\$/);
});

test("Scholar mode uses local model with source-grounded question requests", () => {
  const ui=source('components/literature-workspace.tsx');
  assert.match(ui, /postJSON<ScholarAnswer>/);
  assert.match(ui, /scope: "abstracts_only"/);
  assert.match(ui, /context_questions/);
});

test('scholar question state is declared before its first use',()=>{
 const src=source('components/literature-workspace.tsx');
 assert.ok(src.indexOf('const [question, setQuestion]') < src.indexOf('async function ask('));
 assert.match(src, /question: next/);
});

test('scholar QA offers bounded results and selectable academic sources', () => {
  const view=source('components/literature-workspace.tsx');
  assert.match(view,/"arxiv", label: "arXiv"/);
  assert.match(view,/"openalex", label: "OpenAlex"/);
  assert.match(view,/"acl", label: "ACL"/);
  assert.match(view,/"acm", label: "ACM"/);
  assert.match(view,/setTopK/);
});

test('paper explorer uses Research Chat welcome and compact composer patterns',()=>{
  const paper=source('components/literature-workspace.tsx');
  const chat=source('components/research-workspace.tsx');
  const css=source('app/globals.css');
  assert.match(paper, /className="welcome paper-chat-welcome"/);
  assert.match(paper, /composer-new-chat/);
  assert.match(chat, /composer-new-chat/);
  assert.match(paper, /composer-footer-simplified/);
  assert.match(css, /\.paper-chat-workspace\{background:#fff\}/);
  assert.match(css, /\.paper-chat-welcome \.prompts-grid\{margin-top:38px\}/);
});

test('vLLM model check is manual and model specific',()=>{
  const paper=source('components/literature-workspace.tsx');
  assert.match(paper,/Check local vLLM servers/);
  assert.match(paper,/\/scholar\/model-status/);
  assert.ok(source('app/api/backend/[...path]/route.ts').includes('model-status'));
  assert.match(paper,/rl-research\/DR-Tulu-8B/);
  assert.match(paper,/allenai\/AstaBrief_8B/);
});

test('Paper Explorer displays one model label and defaults to all four academic sources', () => {
  const paper = source('components/literature-workspace.tsx');
  const css = source('app/globals.css');
  assert.doesNotMatch(paper, /Synthesis model/);
  assert.match(paper, /<h3>Language model<\/h3>/);
  assert.match(paper, /useState<string\[\]>\(PROVIDERS\.map\(provider => provider\.id\)\)/);
  for (const source of ['arxiv', 'openalex', 'acl', 'acm']) assert.match(paper, new RegExp(`id: "${source}"`));
  assert.match(paper, /placeholder="Any year"/);
  assert.match(css, /\.paper-chat-setting-label \.paper-chat-year-input\{[\s\S]*?min-width:80px/);
});
