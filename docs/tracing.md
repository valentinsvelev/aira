# Tracing with Phoenix

AIRA can record each research request as an OpenTelemetry trace and send it to
a local [Phoenix](https://github.com/Arize-ai/phoenix) instance. Nothing leaves
your machine, and no account is needed.

## Enable it

1. In `.env`, set `OTEL_TRACES_ENABLED=true`.
2. Start Phoenix and recreate the API container:

   ```sh
   make tracing-up
   make tracing-status      # shows whether the exporter is configured
   ```

3. Ask a new question in Research chat, then open <http://localhost:6006> and
   select the `aira` project (or your `PHOENIX_PROJECT_NAME`). Spans can take a
   few seconds to appear.

Inside AIRA, **Configuration → Evaluation and Debugging → Behind the scenes**
shows the same runs as a timeline. Each answer also has a *Copy trace ID* action
that you can search for in Phoenix.

## What is recorded

Each chat turn creates a `research.chat` trace with `session.id` set to the chat
ID, so Phoenix groups the turns of one conversation. Depending on what actually
happened, it contains spans such as:

- `retrieval.search`: mode, ranked evidence IDs and scores,
- `llm.chat` / `llm.answer`: model name and token counts reported by Ollama,
- `tool.*`, `web.search`, `academic.search`, `memory.conversation.search`,
- `context.verify`: verification method and result.

Health checks, dashboard polling and other HTTP traffic are filtered out before
export, so Phoenix only shows research runs.

## Privacy

By default **no prompts, answers, tool inputs or document text** are exported:
only timings, IDs, scores, settings and statuses.

Setting `OTEL_CAPTURE_CONTENT=true` additionally stores bounded excerpts of
prompts, answers, tool data and retrieved passages in Phoenix's volume. Only do
this on your own machine, and not with sensitive documents. Even then, the
in-app timeline only shows an allow-listed set of fields and at most 600
characters of input and output.

## Troubleshooting

```sh
make tracing-test        # sends a content-free test trace
make tracing-logs        # Phoenix and API logs
```

- **Empty project:** check that the API container sees the setting, then ask a
  new question:
  `docker compose exec api python -c 'import os; print(os.getenv("OTEL_TRACES_ENABLED"))'`
- **Changed `.env`:** recreate the API container (`make tracing-up` does this).
- **No retrieval span:** general-knowledge questions may legitimately skip local
  retrieval.

Phoenix runs from the `arizephoenix/phoenix:latest` image. For reproducible
setups, pin a tested tag in `compose.yml`.
