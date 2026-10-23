# AIRA local demo data

These are **synthetic fixtures**, not real published papers or benchmark measurements.
Do not cite them as scientific evidence outside the demo. The demo creates its own Docker
project, database volume, and `demo/workspace` files; it does not touch your normal AIRA data.

`make demo-up` starts the three services, uploads these fixtures, saves a sample React Flow
agent, and runs a real three-query **synthetic** BM25 retrieval evaluation.
The synthetic benchmark metrics reflect that tiny benchmark only. It never connects to an LLM
judge. For full research-chat inference, independently install the selected Ollama model.

Open http://localhost:3100 (API http://localhost:8100) and browse:
- Your library: three indexed synthetic documents (sparse retrieval usable immediately)
- Research agents: **Demo · evidence workflow**, with its saved React Flow graph
- Configuration / Evaluation: saved retrieval run on the synthetic benchmark

Run `make demo-seed` again safely: fixtures and agent are not duplicated. `make demo-down`
stops the stack without deleting the isolated demo database; add `-v` to the demo Compose
command only if you explicitly want to erase its isolated volumes.
