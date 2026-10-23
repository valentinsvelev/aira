# Roadmap

AIRA 0.1.0 is a working local research assistant. These are the directions
that would make it more reliable and more useful as a research tool. They are
ideas, not commitments.

## Evidence and verification

- Split answers into individual claims and check each against its cited passage.
- Build a labelled set (supported, contradicted, insufficient evidence, omitted
  qualifier, changed numbers) and compare the note verifiers with an NLI model
  and human judgement.
- Compare append-only history, summarisation and verified editable notes on
  long research sessions.
- An optional critical-review mode for a single manuscript; see the
  [design note](paper-assistant-tool-ideas.md).

## Retrieval

- Token-aware budgeting across papers, web results and conversation memory.
- Semantic search over conversation history.
- Restrict a chat to a selected set of papers.
- Publish reproducible retrieval results (BM25, dense, hybrid, reranking) with
  cost and latency.

## Data and operations

- Store Paper Explorer history on the server, with export and import.
- Content-addressed storage for raw PDFs and extracted text.
- Ingestion quality checks: empty documents, chunk and page consistency,
  extraction failures.
- An opt-in integration suite that runs real models and Phoenix.

## Before any shared or remote use

- Authentication, per-user isolation and rate limiting.
- An authenticated TLS reverse proxy in front of the web and API services.
