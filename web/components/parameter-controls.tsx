"use client";

import type { GenerationOptions, RetrievalOptions, VectorStatus, MemoryRoute } from "@/lib/types";

type Props = {
  verificationMode: "semantic" | "tev1" | "extractive";
  setVerificationMode: (value: "semantic" | "tev1" | "extractive") => void;
  memoryRoute: MemoryRoute;
  setMemoryRoute: (value: MemoryRoute) => void;
  retrieval: RetrievalOptions;
  setRetrieval: (next: RetrievalOptions) => void;
  generation: GenerationOptions;
  generationLocked?: boolean;
  setGeneration: (next: GenerationOptions) => void;
  status: VectorStatus | null;
  reindexing: boolean;
  reindex: () => void;
};

function NumInput({ label, value, min, max, step = 1, onChange }: {
  label: string; value: number; min: number; max: number; step?: number;
  onChange: (value: number) => void;
}) {
  return <label className="param-field">
    <span>{label}</span>
    <input type="number" value={value} min={min} max={max} step={step}
      onChange={(e) => {
        const n = Number(e.target.value);
        if (Number.isFinite(n)) onChange(n);
      }} />
  </label>;
}

export function ParameterControls({ memoryRoute, setMemoryRoute, retrieval, setRetrieval, generation, setGeneration, verificationMode, setVerificationMode,
  status, reindexing, reindex, generationLocked = false }: Props) {
  return <div className="parameter-panel">
    <div className="param-sections">
      <section>
        <h4>Dual context memory</h4>
        <label className="param-field"><span>Which archive should be consulted?</span>
          <select aria-label="Context routing" value={memoryRoute} onChange={e => setMemoryRoute(e.target.value as MemoryRoute)}>
            <option value="auto">Auto (RAG: heuristic; Agentic RAG: tool choice)</option>
            <option value="conversation">Conversation history only</option>
            <option value="research">Research evidence only</option>
            <option value="both">Both context archives</option>
            <option value="neither">Neither (no archive search)</option>
          </select>
        </label>
        <p className="param-help">The last two exchanges remain available for continuity. Older turns stay in PostgreSQL and can be retrieved on demand. Conversation statements are never treated as paper evidence.</p>
        <h4>Research retrieval</h4>
        <div className="param-grid">
          <label className="param-field"><span>Strategy</span>
            <select value={retrieval.mode} onChange={(e) =>
              setRetrieval({ ...retrieval, mode: e.target.value as RetrievalOptions["mode"] })}>
              <option value="sparse">Sparse</option>
              <option value="dense">Dense (pgvector)</option>
              <option value="hybrid">Hybrid (RRF)</option>
            </select>
          </label>
          <label className="param-field"><span>Dense embedding model</span>
            <select value={retrieval.embedding_model} onChange={(e) =>
              setRetrieval({ ...retrieval, embedding_model: e.target.value as RetrievalOptions["embedding_model"] })}>
              <option value="sentence-transformers/all-MiniLM-L6-v2">MiniLM-L6-v2 · 384D</option>
              <option value="Alibaba-NLP/gte-large-en-v1.5">GTE-large-en-v1.5 · 1024D</option>
            </select>
          </label>
          <label className="param-field"><span>Second-stage reranker</span>
            <select value={retrieval.reranker ?? "none"} onChange={(e) => setRetrieval({ ...retrieval, reranker: e.target.value as RetrievalOptions["reranker"] })}>
              <option value="none">None</option>
              <option value="Alibaba-NLP/gte-reranker-modernbert-base">GTE ModernBERT · 149M</option>
              <option value="jinaai/jina-reranker-v3.5">Jina v3.5 · 0.6B (non-commercial)</option>
            </select>
          </label>
          {(retrieval.reranker ?? "none") !== "none" && <NumInput label="Rerank candidate pool" value={retrieval.rerank_candidates ?? 20} min={retrieval.top_k} max={100}
            onChange={(v) => setRetrieval({ ...retrieval, rerank_candidates: v })} />}
          <NumInput label="Final top-k" value={retrieval.top_k} min={1} max={12}
            onChange={(v) => setRetrieval({ ...retrieval, top_k: v })} />
          <NumInput label="Sparse candidates" value={retrieval.bm25_candidates} min={1} max={100}
            onChange={(v) => setRetrieval({ ...retrieval, bm25_candidates: v })} />
          <NumInput label="Dense candidates" value={retrieval.dense_candidates} min={1} max={100}
            onChange={(v) => setRetrieval({ ...retrieval, dense_candidates: v })} />
          <NumInput label="RRF k" value={retrieval.rrf_k} min={1} max={200}
            onChange={(v) => setRetrieval({ ...retrieval, rrf_k: v })} />
          <NumInput label="Minimum cosine similarity" value={retrieval.min_similarity} min={0} max={1} step={0.05}
            onChange={(v) => setRetrieval({ ...retrieval, min_similarity: v })} />
        </div>
        <div className="index-status">
          <span>pgvector: {status?.enabled ? `${status.indexed} indexed · ${status.missing ?? "?"} pending` : "not connected"}</span>
          <button type="button" disabled={reindexing || !status?.enabled} onClick={reindex}>
            {reindexing ? "Indexing…" : "Index missing embeddings"}
          </button>
        </div>
      </section>
      <section>
        <h4>Local generation (Ollama)</h4>
        {generationLocked && <p className="param-help">This agent uses its saved generation settings. Edit them in Research agents. Only the default agent supports per-chat generation customization.</p>}
        <fieldset disabled={generationLocked} className="generation-fieldset">
        <div className="param-grid">
          <label className="param-field"><span>Model</span>
            <select value={generation.model ?? "qwen3.5:4b"} onChange={(e) =>
              setGeneration({ ...generation, model: e.target.value as NonNullable<GenerationOptions["model"]> })}>
              <option value="qwen3.5:4b">Qwen 3.5 · 4B</option>
              <option value="llama3.2:3b">Llama 3.2 · 3B</option>
              <option value="granite4.1:3b">Granite 4.1 · 3B</option>
            </select>
          </label>
          <NumInput label="Temperature" value={generation.temperature} min={0} max={1} step={0.05}
            onChange={(v) => setGeneration({ ...generation, temperature: v })} />
          <NumInput label="Top-p" value={generation.top_p} min={0.01} max={1} step={0.05}
            onChange={(v) => setGeneration({ ...generation, top_p: v })} />
          <label className="param-field"><span>Max output tokens</span>
            <select value={generation.num_predict === -1 ? "unlimited" : "limited"}
              onChange={e => setGeneration({ ...generation, num_predict: e.target.value === "unlimited" ? -1 : 1024 })}>
              <option value="unlimited">Model decides (default)</option>
              <option value="limited">Set a limit</option>
            </select>
          </label>
          {generation.num_predict !== -1 && <NumInput label="Token limit" value={generation.num_predict} min={32} max={32768}
            onChange={v => setGeneration({ ...generation, num_predict: v })} />}
          <label className="param-field"><span>Context window</span><select aria-label="Context window mode" value={generation.num_ctx === 0 ? "auto" : "manual"} onChange={event=>setGeneration({ ...generation, num_ctx:event.target.value === "auto" ? 0 : 4096 })}><option value="auto">Model maximum (auto)</option><option value="manual">Custom limit</option></select></label>
          {generation.num_ctx !== 0 && <NumInput label="Context window (tokens)" value={generation.num_ctx} min={2048} max={1048576} step={1024}
            onChange={(v) => setGeneration({ ...generation, num_ctx: v })} />}
          <NumInput label="Repeat penalty" value={generation.repeat_penalty} min={0.8} max={2} step={0.05}
            onChange={(v) => setGeneration({ ...generation, repeat_penalty: v })} />
          <label className="param-field"><span>Seed (optional)</span>
            <input type="number" placeholder="random" min={0} max={2147483647}
              value={generation.seed ?? ""} onChange={(e) => setGeneration({ ...generation,
                seed: e.target.value === "" ? null : Number(e.target.value) })} />
          </label>
        </div>
        </fieldset>
        <h4>Verification (context edits)</h4>
        <label className="param-field"><span>Checker</span>
          <select value={verificationMode} onChange={(e) => setVerificationMode(e.target.value as typeof verificationMode)}>
            <option value="semantic">Cosine similarity (baseline)</option>
            <option value="tev1">TEV1 decision model (experimental)</option>
            <option value="extractive">Exact text (strict)</option>
          </select>
        </label>
      </section>
    </div>
    <p className="param-help">Request-level settings. Index each embedding model separately; first GTE-large download can take time. TEV1 requires Ollama 0.35+ and a separately pulled model. Dense/hybrid search requires pgvector embeddings;
      cosine and sparse scores are not directly comparable. Small local LLMs may not obey every sampling preference exactly.</p>
  </div>;
}
