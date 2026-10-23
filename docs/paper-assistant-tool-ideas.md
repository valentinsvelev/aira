# Design note: critical review of a manuscript

Source: Rajesh Jayaram et al. (2026), [Towards Automating Scientific Review with Google’s Paper Assistant Tool](https://arxiv.org/abs/2606.28277). This is a design note, **not an implemented feature**.

The paper focuses on verification, not bibliographic discovery: it segments manuscripts by logical themes, allocates additional model calls to difficult segments, runs specialist reviews with access to the complete paper, and synthesizes findings while using search to reduce hallucinations. The authors report 89.7% error detection on a 26-paper mathematical/proof-error subset of SPOT (29 labeled errors), compared to 55.2% with a single-call Gemini 3.1 Pro baseline; the reported gains are *not* transferable to small Ollama models without evaluation. Their pilots also relied on much stronger proprietary models.

## Possible feature: Critical review

Offer an optional **Critical review** action on a user-selected manuscript:

1. **Segment** PDF pages into methods, theory, experiments, conclusions and cited evidence, preserving page/section IDs and the underlying original text.
2. **Targeted passes**, run separately: (a) claim ↔ evidence consistency, (b) experimental controls / baseline / statistical reporting checklist, and (c) cited-reference existence using existing arXiv/OpenAlex lookup. Reserve proof/algorithm verification for specialized evaluators; don't claim to prove correctness.
3. **Verification pass**: each proposed critique must quote a supporting excerpt, specify a page and classify issue severity as *possible error*, *unclear/missing evidence*, or *suggested improvement*. Drop unsupported critiques.
4. **Human-facing report**: aggregate deduplicated, evidence-linked findings with a clear *needs human review* label and an option to save them as research notes.

Start with a single model and strict limits (e.g. 1–2 segments, no more than 3 additional calls) for affordable Apple-silicon local inference, then benchmark against a single-pass prompt. Log all checks as Phoenix spans and compare human-reviewed false-positive rates, evidence-grounding, latency and token consumption.

**Important limitations:** hallucinated critiques and PDF parsing errors are explicitly reported in PAT; local 3B–4B models should not be marketed as reliable proof checkers. Keep all outputs non-authoritative and ask humans to verify them.
