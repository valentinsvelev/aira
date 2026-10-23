// Light and dark theme.
import test from 'node:test';
import assert from 'node:assert/strict';
import { source } from './helpers.mjs';

const globalStyles = source('app/globals.css');

test('theme is user-toggleable and persists in browser storage', () => {
  const view=source('components/research-workspace.tsx');
  assert.match(view,/evidence-lab\.theme/);
  assert.match(view,/toggleTheme/);
  assert.match(source('app/globals.css'), /\[data-theme="dark"\]/);
});

test('lighter dark mode is applied to the main pane, modal, and article surfaces', () => {
  for (const fragment of ['--bg:#1c293b','--paper:#27384e','.chat-config-agent-link','background:#29394f!important','.sidebar{background:#202d44}','.config-back-container{background:#1d2a3c}']) {
    assert.ok(globalStyles.includes(fragment), `${fragment} missing`);
  }
});
