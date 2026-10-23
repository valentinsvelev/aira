# Architecture

AIRA is a human-directed research assistant. The model helps you find, read and
summarise evidence; you decide what to keep. Two ideas shape the design:

1. **Evidence is immutable.** Text extracted from a source is stored once, with a
   stable ID, and never edited. Citations always point at that exact text.
2. **Working context is editable but checked.** Research notes can be added,
   replaced and removed, but every change must quote a source passage and pass a
   verification step. Accepted and rejected changes are both logged.

## Components

```text
Next.js 16 UI (React 19, React Flow)
   │
   │  /api/backend/*  server-side proxy, forwards only allow-listed routes
   ▼
FastAPI backend (src/aira/api.py)
   ├── Answering
   │     ├── RAG: retrieve → build prompt → one model call      (agent.py)
   │     ├── Agentic RAG: LangGraph agent with permitted tools   (agent.py)
   │     └── Visual workflows: bounded DAGs with gates and loops (agent_workflows.py,
   │                                                              advanced_workflows.py)
   ├── Retrieval (retrieval.py, vector_store.py, reranking.py)
   │     ├── Sparse: LlamaIndex BM25 over active chunks
   │     ├── Dense: sentence-transformer embeddings in pgvector (HNSW, cosine)
   │     ├── Hybrid: reciprocal-rank fusion of both lists
   │     └── Optional reranker service (reranker_service.py)
   ├── External sources, opt-in (websearch.py, webbrowse.py, academic.py, scholarqa.py)
   ├── Notes and verification (context.py, semantic.py, tev1_checker.py)
   ├── Conversation memory (conversations.py, memory.py)
   └── Tracing (observability.py, trace_reader.py)
   │
   ├── Ollama (local models) or an OpenAI-compatible API
   └── SQLAlchemy → PostgreSQL 16 + pgvector (SQLite for tests and native runs)
```

## Data model

| Table | Contents |
|---|---|
| `documents`, `document_versions` | One logical document per path; one version per distinct file content (SHA-256) |
| `evidence_chunks` | Immutable page-level passages, plus captured web pages, snippets and academic abstracts |
| `chunk_embeddings`, `chunk_embeddings_1024` | Vectors per chunk and model, kept separate from the evidence |
| `ingestion_runs` | Each sync, with counts and failures |
| `research_sessions`, `context_notes`, `context_events` | Notes per chat and the full audit log of proposed, accepted and rejected edits |
| `chat_conversations`, `chat_turns` | Saved chats, including the sources and settings used for each answer |
| `agent_profiles`, `agent_workflows`, `rag_system_profiles` | Saved agents, their workflows and retrieval/generation presets |
| `literature_reviews`, `literature_review_papers` | Saved literature reviews (API) |

Schema changes go through Alembic migrations (`alembic/`). See
[development.md](development.md#database-migrations).

## Ingestion

- PDFs (and `.txt` / `.md` files) are read from `papers/` or uploaded through the API.
- Each file is hashed. Unchanged files are skipped; a changed file becomes a new
  version, and only active versions are searched. Old versions stay in the
  database so earlier citations still resolve.
- One failing file does not stop the rest of the sync; failures are recorded in
  the ingestion run.
- Dense vectors are computed per model and can be backfilled at any time with
  `aira index-embeddings`. Re-embedding never changes evidence IDs.

## Retrieval

- **Sparse** ranks passages by BM25 keyword matching.
- **Dense** embeds the query and finds the nearest passages by cosine similarity.
  MiniLM (384 dimensions) and GTE-large (1024 dimensions) use separate tables,
  so their scores are never mixed.
- **Hybrid** merges the two ranked lists with reciprocal-rank fusion,
  `score = Σ 1 / (rrf_k + rank)`, rather than adding raw scores.
- An optional cross-encoder reranker reorders the first-stage candidates.

Scores from different modes mean different things, so compare rankings, not
numbers. All parameters (`top_k`, candidate counts, `rrf_k`, minimum similarity,
reranker) are set per request.

## Answering and citations

The prompt contains the retrieved passages, each with a number, plus any saved
notes and recent chat turns. Web pages, snippets and abstracts are passed as
untrusted data, clearly separated from your library. After generation, AIRA:

- renumbers the citations the model used so they start at `[1]`,
- links each number to its evidence ID, document version and page,
- marks citations that do not match a retrieved passage as unverified, and
- replaces any bibliography the model wrote with the real source list.

This proves that a citation exists, not that the passage supports the claim.

In Agentic RAG, the agent only receives the tools its profile allows, and web or
academic tools also require the per-chat toggle. Tool calls are bounded and
recorded.

## Note verification

A proposed note must include an exact quote from an evidence chunk. Depending on
`FAITHFULNESS_MODE` it is then checked by:

- `semantic`: embedding similarity between the note and the quote, plus a guard
  that rejects numbers which do not appear in the source;
- `extractive`: the exact quote only;
- `tev1`: an experimental classifier served by Ollama, which fails closed when
  unavailable.

These are heuristics. A high similarity score does not prove entailment.

## Web and academic sources

- Web search uses DDGS (no key) or Tavily. Reading a page is limited to
  public HTTP(S) hosts, with DNS checks against private addresses, redirect and
  size limits, and timeouts.
- Academic search queries arXiv, OpenAlex and Crossref and stores abstracts as
  metadata evidence. Google Scholar is only opened in your browser; it is never
  scraped.
- PDF import is limited to an allow-list of open-access hosts (arXiv, ACL
  Anthology), checks the PDF signature, caps the size at 20 MiB and goes through
  normal versioned ingestion.

## Further reading

- [Conversation memory](conversation-memory.md)
- [Tracing](tracing.md)
- [Evaluation](evaluation.md)
