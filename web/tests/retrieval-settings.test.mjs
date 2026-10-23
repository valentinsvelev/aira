// Retrieval options, reranking and saved RAG systems.
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { source } from './helpers.mjs';



test('research chat exposes GTE and Jina options with candidate pool', () => {
  const s = source('components/parameter-controls.tsx');
  assert.match(s,/gte-reranker-modernbert-base/);
  assert.match(s,/jina-reranker-v3\.5/);
  assert.match(s,/Rerank candidate pool/);
  assert.match(s,/retrieval\.rerank_candidates/);
});

test('saved RAG system editor supports reranking', () => {
  const s = source('components/rag-workspace.tsx');
  assert.match(s,/Second-stage reranker/);
  assert.match(s,/Rerank candidate pool/);
  assert.match(s,/reranker: "none"/);
});

test('retrieval options preserve no-reranker defaults', () => {
  const s = source('components/research-workspace.tsx');
  const t = source('lib/types.ts');
  assert.match(s,/reranker: "none", rerank_candidates: 20/);
  assert.match(t,/rerank_candidates: number/);
  assert.match(s,/chat-config-reranker/);
  assert.match(s,/Candidate pool to rerank/);
});

test('saved RAG configs have matching UI, API selection and proxy routes',()=>{
  const config=source('components/configuration-workspace.tsx');
  const chat=source('components/research-workspace.tsx');
  const editor=source('components/rag-workspace.tsx');
  const proxy=source('app/api/backend/[...path]/route.ts');
  assert.match(config, /<strong>RAG systems<\/strong>/);
  assert.match(config, /Papers available: \{documentCount\} · Indexed chunks: \{chunkCount\}/);
  assert.match(editor,/postJSON<RAGSystem>\("\/rag-systems"/);
  assert.match(chat,/rag_system_id: mode === "answer" \? ragSystemId/);
  assert.match(chat,/view === "rag" \? <RAGWorkspace/);
  assert.match(proxy,/\^rag-systems\$/);
  assert.match(proxy,/ragId/);
});

test('RAG editing in place and conditional fields', () => {
  const src = source('components/rag-workspace.tsx');
  assert.doesNotMatch(src, /Duplicate & edit|agent-tool-pills/);
  assert.match(src, /draft\.retrieval\.mode !== "sparse"/);
  assert.match(src, /draft\.retrieval\.mode === "hybrid"/);
  assert.match(src, /Edit \$\{system\.name\}/);
  assert.match(src, /openbmb\/minicpm-v4\.7:1b/);
});

test('the user-facing retrieval value is sparse, not bm25',()=>{
  const texts=[source('components/research-workspace.tsx'),source('components/rag-workspace.tsx'),
    source('components/parameter-controls.tsx'),source('lib/types.ts')];
  for (const text of texts){
    assert.doesNotMatch(text, /value="bm25"|mode: "bm25"/);
  }
  assert.match(source('components/rag-workspace.tsx'), /value="sparse">Sparse/);
});
