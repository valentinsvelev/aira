# Conversation memory

AIRA keeps two kinds of material apart: what your **sources** say, and what was
**said in the chat**. A sentence from an earlier turn only shows that someone
said it, not that it is true, so it is never cited as scientific evidence.

```text
                     New question
                          │
          Last two exchanges (as chat messages)
                          │
          Router (RAG) or agent tools (Agentic RAG)
                ┌─────────┴─────────┐
     Conversation archive     Research evidence
     chat_turns, this chat    PDFs, web, academic metadata
     full-text search         BM25 / pgvector / hybrid
                └─────────┬─────────┘
       Prompt: search results + verified notes
                          │
                      Local model
                          │
   Answer with numbered paper citations and [conversation-turn:ID] references
```

## What is stored and searched

- Every user and assistant message is stored in `chat_turns`. Search uses
  PostgreSQL full-text search (with a simple fallback on SQLite) and only looks
  inside the current chat.
- Search results are turn IDs, roles, dates and excerpts. You can open the exact
  earlier message from the answer.
- At most two recent exchanges go into the prompt as normal chat messages, plus
  up to four matching older turns as quoted data. The full transcript is never
  re-inserted.

## Routing

In **RAG** mode a simple rule-based router picks the initial sources:

| Example question | Route |
|---|---|
| "What did we decide about QLoRA earlier?" | Conversation |
| "What does the QLoRA paper report about NF4?" | Research |
| "Compare with the paper we discussed yesterday." | Both |
| "Hello!" | Neither |

The chat settings let you override the route (`auto`, `conversation`,
`research`, `both`, `neither`) for debugging and experiments.

In **Agentic RAG** mode the agent can additionally call:

- `search_conversation_history(query)`: search the current chat,
- `read_conversation_history(turn_ids)`: read up to 12 earlier turns in full.

Conversation search is enabled for new agents by default and can be switched off
per agent.

## CLI and API

```sh
docker compose exec api aira memory-search "QLoRA memory" --session <chat-uuid> --top-k 5
docker compose exec api aira answer "What did we decide earlier?" \
  --session <chat-uuid> --memory-route conversation
```

- `GET /conversations/{id}/memory/search?q=...&limit=5`
- `GET /conversations/{id}/memory/turns?ids=4,7,15`
- `POST /answer` and `POST /chat` accept an optional `memory_route` (default `auto`).

## Limitations

1. Search is lexical, not semantic. Vague references without shared words fall
   back to the most recent turns, with a score of 0.
2. The RAG router is rule-based, and small local models sometimes skip useful
   tool calls.
3. The prompt budget is limited by characters per source, not by a joint token
   budget, and `num_ctx` still caps what the model sees.
4. The CLI reads saved chat context but does not save new turns; the web app does.
