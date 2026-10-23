// Retrieval Evaluation dashboard.
import test from 'node:test';
import assert from 'node:assert/strict';
import { source } from './helpers.mjs';

const configurationWorkspace = source('components/configuration-workspace.tsx');
const researchWorkspace = source('components/research-workspace.tsx');
const retrievalEvaluationWorkspace = source('components/retrieval-evaluation-workspace.tsx');
const proxyRoute = source('app/api/backend/[...path]/route.ts');

test("retrieval evaluation is available through Configuration and navigation", () => {
  assert.match(configurationWorkspace, /onNavigate\("retrieval-eval"\)/);
  assert.match(researchWorkspace, /<RetrievalEvaluationWorkspace\/>/);
});

test("dashboard includes objective metrics, document rankings, run comparison and optional rerankers", () => {
  for (const token of ["Recall", "nDCG", "MRR", "Precision", "p95 latency", "relevant_ids", "Comparable saved runs", "gte-reranker-modernbert-base", "jina-reranker-v3.5", "Random seed"]) {
    assert.ok(retrievalEvaluationWorkspace.includes(token), token);
  }
  assert.match(retrievalEvaluationWorkspace, /seed===\(chosen\?\.seed\?\?seed\)/);
});

test("proxy lists every retrieval evaluation route", () => {
  for (const route of ["retrieval-evals\\/datasets", "retrieval-evals\\/runs", "retrieval-evals\\/jobs"]) {
    assert.ok(proxyRoute.includes(route), route);
  }
});
