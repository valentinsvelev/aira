// Citation numbering, source labels and the Sources line.
import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeCitations } from '../lib/citations.ts';
import { sourceLabel, citedSourceLabel } from '../lib/source-labels.ts';
import { resolve } from 'node:path';
import { source } from './helpers.mjs';

const researchWorkspace = source('components/research-workspace.tsx');

test('numbers actual evidence and rewrites validated aliases', () => {
  const result = normalizeCitations('NF4 [evidence:abc]. DQ [2]. Source [evidence:abc].', ['abc', 'def']);
  assert.match(result.markdown, /\[1\]\(#research-source-1\)/);
  assert.match(result.markdown, /\[2\]\(#research-source-2\)/);
  assert.deepEqual(result.citedSourceIds, ['abc', 'def']);
  assert.equal(result.unverifiedCount, 0);
});

test('unknown citations remain explicitly unverified', () => {
  const result = normalizeCitations('Claims [evidence:fiction] [9].', ['real']);
  assert.equal(result.unverifiedCount, 2);
  assert.deepEqual(result.citedSourceIds, []);
  assert.ok(!result.markdown.includes('#research-source-'));
});

test('ignores model-written bibliography and shows authoritative list separately', () => {
  const result = normalizeCitations('Evidence [1].\n\nReferences:\n[1] Invented citation 1990', ['real']);
  assert.equal(result.modelBibliographyRemoved, true);
  assert.ok(!result.markdown.includes('Invented citation'));
});

test('renumbers sparse citations to contiguous numbers, preserving original source IDs', () => {
  const ids = Array.from({ length: 14 }, (_, i) => `source-${i + 1}`);
  const response = 'AI is defined by these sources [11], [12], [13] and [14]. [11]';
  const result = normalizeCitations(response, ids);
  assert.deepEqual(result.citedSourceNumbers, [1, 2, 3, 4]);
  assert.deepEqual(result.citedSourceIds, ids.slice(10, 14));
  assert.equal(result.markdown.includes('[11]'), false);
  assert.match(result.markdown, /\[1\]\(#research-source-1\)/);
  assert.match(result.markdown, /\[4\]\(#research-source-4\)/);
  assert.equal((result.markdown.match(/#research-source-1/g) || []).length, 2);
});

test('citation order follows first mention, not server retrieval order', () => {
  const result = normalizeCitations('First [5], then [2], again [5].', ['a','b','c','d','e']);
  assert.deepEqual(result.citedSourceIds, ['e', 'b']);
  assert.deepEqual(result.citedSourceNumbers, [1, 2]);
  assert.equal(result.markdown, 'First [1](#research-source-1), then [2](#research-source-2), again [1](#research-source-1).');
});

test('unsupported citations and model bibliography do not create citation chips', () => {
  const result = normalizeCitations('Supported [2]. Unsupported [99].\n\nReferences:\n[1] Invented', ['a','b','c']);
  assert.deepEqual(result.citedSourceNumbers, [1]);
  assert.deepEqual(result.citedSourceIds, ['b']);
  assert.equal(result.unverifiedCount, 1);
});

test('explicit evidence IDs resolve and compact numbers without changing IDs', () => {
  const result = normalizeCitations('Use [evidence:last] and not the other retrieved hits.', ['first','middle','last']);
  assert.deepEqual(result.citedSourceNumbers, [1]);
  assert.deepEqual(result.citedSourceIds, ['last']);
});

test('old saved responses keep source identity after UI renumbering', () => {
  const ids = ['unused', 'web-a', 'web-b'];
  const result = normalizeCitations('Source [3] and [2].', ids);
  assert.deepEqual(result.citedSourceIds, ['web-b','web-a']);
  assert.deepEqual(result.sourceIds, ids);
  assert.equal(result.markdown, 'Source [1](#research-source-1) and [2](#research-source-2).');
});

test('multiple passages from same PDF carry distinct page references', () => {
  const a = sourceLabel({id:'one',path:'Dettmers et al. (2023) QLoRA.pdf',page:4});
  const b = sourceLabel({id:'two',path:'Dettmers et al. (2023) QLoRA.pdf',page:26});
  assert.equal(a,'Dettmers et al. (2023), p. 4');
  assert.equal(b,'Dettmers et al. (2023), p. 26');
});

test('academic metadata can use first author and year', () => {
  assert.equal(sourceLabel({id:'x',title:'Efficient training',authors:['Ada Researcher','Bob Scientist'],year:2026}),
               'Researcher et al. (2026)');
});

test("Sources entries retain separate compact clickable buttons", () => {
  const ui = source('components/research-workspace.tsx');
  const css = source('app/globals.css');
  assert.match(ui, /className="answer-sources"/);
  assert.match(css, /\.answer-sources>button\{border:1px solid/);
});

test('generated Sources footer is suppressed, verified inline citations remain', () => {
  const normalized = normalizeCitations('NF4 improves efficiency [2].\n\nSources: [1], [2]', ['a','b']);
  assert.deepEqual(normalized.citedSourceIds,['b']);
  assert.match(normalized.markdown,/research-source-1/);
  assert.doesNotMatch(normalized.markdown,/Sources:/);
});

test('web citation uses full canonical URL; paper includes page', () => {
  assert.equal(citedSourceLabel({id:'a',url:'https://example.org/paper',title:'A Page',source_kind:'web_page'}),'https://example.org/paper');
  assert.equal(citedSourceLabel({id:'a',path:'Dettmers et al. (2023) QLoRA.pdf',page:4}),'Dettmers et al. (2023), p. 4');
});

test('sources appear before response metadata; old chip row removed', () => {
  assert.ok(researchWorkspace.indexOf('className="answer-sources"') < researchWorkspace.indexOf('className="message-meta"'));
  assert.doesNotMatch(researchWorkspace,/className="citation-chips"/);
});
