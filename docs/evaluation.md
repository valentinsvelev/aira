# Evaluation

AIRA includes three ways to measure what it does. None of them uses an LLM as
a judge, and none of them claims to reproduce official leaderboard results.

## Retrieval evaluation (BEIR SciFact)

**Configuration → Evaluation and Debugging → Retrieval Evaluation** measures how
well each retrieval setup finds the relevant documents for judged queries.

Download the dataset once from the official BEIR mirror:

```sh
python scripts/download_scifact.py      # into .data/retrieval_eval/datasets/scifact
```

Then choose sparse, dense or hybrid retrieval, an embedding model, an optional
reranker, the number of queries and a random seed. A run reports Recall@k, MRR,
nDCG@k and query latency, and stores an immutable JSON artifact in
`.data/retrieval_eval/runs/` so runs can be compared later.

The evaluation builds its own index over the SciFact corpus. It never adds
those documents to your library or uses your pgvector index.

## Benchmark experiments

**Configuration → Evaluation and Debugging → Experiments on Benchmark Datasets**
runs the configured local model on a small sample of a QA benchmark. Nothing
starts until you press *Start run*; the page shows every prediction next to the
reference answer.

| Option | Context given to the model | Score |
|---|---|---|
| [QASPER](https://huggingface.co/datasets/allenai/qasper) | Paragraphs selected from the paper by a simple lexical match | Token F1 against the best reference answer |
| [DRUID](https://huggingface.co/datasets/copenlu/druid) | The claim's evidence passage | Relevance and stance accuracy |
| Your own file | The row's `context`, or your indexed papers if it has none | Token F1 |

Public datasets are fetched on demand from the Hugging Face dataset viewer.
Samples can be the first rows, a seeded random sample, or the whole split.
BrowseComp-Plus and DeepResearch Bench are listed for reference only, because
they need large corpora or external judges; AIRA does not pretend to run them.

**These are small-sample diagnostics.** QASPER's official evidence scoring and
DRUID's full protocol are not reproduced, and a handful of examples is not
representative.

### Your own dataset

Upload a UTF-8 `.json`, `.jsonl` or `.csv` file:

```json
[
  {"question": "What is LoRA?", "answer": "Low-rank adaptation",
   "context": "LoRA freezes the base weights and trains low-rank matrices."},
  {"question": "What is NF4?", "answer": "A 4-bit normal-float quantization format"}
]
```

Only the selected rows are sent to your local API. Results are saved to
`.data/benchmarks/runs/` only if you tick the save option.

## Analytics with DuckDB

Saved benchmark runs can be rebuilt into a DuckDB database and Parquet tables:

```sh
make analytics-build
make analytics-report      # reads .data/benchmarks/runs, writes .data/analytics/
```

## Airflow ingestion (optional)

`compose.airflow.yml` adds a local Airflow instance with a DAG that calls the
API to sync the papers folder and backfill embeddings. It is paused by default
and only runs on a schedule if you set `AIRA_INGEST_SCHEDULE`.

```sh
make airflow-up     # UI on http://localhost:8080
make airflow-down
```

## Ideas for further evaluation

- Label questions by the source they need (papers, conversation, both, neither)
  and measure routing accuracy, retrieval recall and citation correctness.
- Build a set of faithful paraphrases, omitted qualifiers, contradictions and
  changed numbers, then sweep `SIMILARITY_THRESHOLD` and compare the note
  verifiers by precision and false-accept rate.
- Compare RAG, Agentic RAG and visual workflows on the same questions for
  answer quality, tool calls, tokens and latency.
