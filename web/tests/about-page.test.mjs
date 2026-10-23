// The About page.
import test from 'node:test';
import assert from 'node:assert/strict';
import { source } from './helpers.mjs';

const infoWorkspace = source('components/info-workspace.tsx');

test('About the Project describes retrieval and research inspirations', () => {
  assert.match(infoWorkspace, /How retrieval works/);
  assert.match(infoWorkspace, /Research inspiration/);
});

test('research inspirations are sorted by first-author surname', () => {
  assert.match(infoWorkspace, /97NEP1pyS3/);
  assert.match(infoWorkspace, /2026\.eacl-long\.13\//);
  assert.match(infoWorkspace, /\.sort\(\(a, b\) =>/);
  assert.match(infoWorkspace, /surname\(a\)\.localeCompare\(surname\(b\)/);
});
