// The Next.js server-side proxy only forwards known API routes.
import test from 'node:test';
import assert from 'node:assert/strict';
import { source } from './helpers.mjs';



test('API proxy accepts default RAG edits and benchmark jobs', () => {
  const src = source('app/api/backend/[...path]/route.ts');
  assert.match(src, /ragId = "\(\?:default/);
  assert.match(src, /benchmarks\\\/jobs/);
});
