"use client";

import { useState } from "react";
import { ArrowRight, Database, Plus, Save, Trash2 } from "lucide-react";
import { postJSON, patchJSON, deleteAPI } from "@/lib/api";
import type { RAGSystem, RetrievalOptions, GenerationOptions } from "@/lib/types";
import { DEFAULT_GENERATION } from "@/components/agent-workspace";

const DEFAULT_RETRIEVAL: RetrievalOptions = {
  mode: "sparse", embedding_model: "sentence-transformers/all-MiniLM-L6-v2",
  top_k: 5, bm25_candidates: 20, dense_candidates: 20, rrf_k: 60, min_similarity: 0,
  reranker: "none", rerank_candidates: 20,
};
const fresh = (): Omit<RAGSystem, "id"> => ({ name: "", description: "", system_instructions: "",
  retrieval: {...DEFAULT_RETRIEVAL}, generation: {...DEFAULT_GENERATION} });

export function RAGWorkspace({ systems, selectedId, onSelect, onChanged }: {
  systems: RAGSystem[]; selectedId: string; onSelect: (id:string)=>void; onChanged: ()=>Promise<void>;
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState(fresh());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  function edit(system?: RAGSystem) {
    setEditing(system?.id ?? "new"); setError("");
    setDraft(system ? { name: system.name, description: system.description,
      system_instructions: system.system_instructions, retrieval: {...DEFAULT_RETRIEVAL, ...system.retrieval},
      generation: {...DEFAULT_GENERATION, ...system.generation} } : fresh());
  }
  function setRet<K extends keyof RetrievalOptions>(key: K, value: RetrievalOptions[K]) {
    setDraft(v=>({...v,retrieval:{...v.retrieval,[key]:value}}));
  }
  function setGen<K extends keyof GenerationOptions>(key: K, value: GenerationOptions[K]) {
    setDraft(v=>({...v,generation:{...v.generation,[key]:value}}));
  }
  async function save() {
    if (!draft.name.trim() || busy) return;
    setBusy(true);setError("");
    try {
      const result = editing === "new" ? await postJSON<RAGSystem>("/rag-systems", draft)
        : await patchJSON<RAGSystem>(`/rag-systems/${editing}`, draft);
      await onChanged(); onSelect(result.id); setEditing(null);setNotice("RAG system saved and selected.");
    } catch (e) {setError(e instanceof Error ? e.message : "Save failed");}
    finally {setBusy(false);}
  }
  async function remove(id:string) {
    if (!window.confirm("Delete this RAG system?")) return;
    try { await deleteAPI(`/rag-systems/${id}`); if (selectedId===id) onSelect("default");
      await onChanged();setNotice("RAG system deleted."); }
    catch(e) {setError(e instanceof Error ? e.message : "Delete failed");}
  }
  return <div className="agent-workspace">
    <div className="agent-workspace-title"><div><div className="eyebrow">LOCAL RETRIEVAL WORKFLOWS</div><h1>RAG systems</h1>
      <p>Choose how Research chat retrieves papers and generates source-grounded answers. Saved in your local database.</p></div>
      <button className="agent-primary" onClick={()=>edit()}><Plus size={16}/> New RAG system</button></div>
    {error && <p className="agent-alert" role="alert">{error}</p>}{notice && <p className="agent-notice" role="status">{notice}</p>}
    <div className="agent-card-list">{systems.map(system=><article key={system.id} className="agent-profile-card">
      <div className="agent-card-icon"><Database size={21}/></div><div className="agent-card-content"><h2>{system.name} {system.builtin && <span>Built-in</span>}</h2>
      <p>{system.description || "Local evidence-grounded RAG"}</p></div>
      <div className="agent-card-actions"><button className={selectedId===system.id ? "active":""} onClick={()=>onSelect(system.id)}>{selectedId===system.id ? "Selected":"Use RAG"}</button>
      <button onClick={()=>edit(system)} aria-label={`Edit ${system.name}`}>Edit</button>
      {!system.builtin && <button className="agent-danger" aria-label={`Delete ${system.name}`} onClick={()=>void remove(system.id)}><Trash2 size={15}/></button>}
      </div></article>)}</div>
    {editing && <div className="agent-modal-overlay" onPointerDown={e=>{if(e.target===e.currentTarget && !busy)setEditing(null);}}>
      <div className="agent-editor" role="dialog" aria-modal="true" aria-label="RAG system editor">
        <div className="agent-editor-heading"><h2>{editing==="new" ? "New RAG system":"Edit RAG system"}</h2><button aria-label="Close RAG editor" onClick={()=>setEditing(null)}>×</button></div>
        {error && <p className="agent-alert" role="alert">{error}</p>}
        <label>Name<input maxLength={100} value={draft.name} onChange={e=>setDraft(v=>({...v,name:e.target.value}))}/></label>
        <label>Description<input maxLength={500} value={draft.description} onChange={e=>setDraft(v=>({...v,description:e.target.value}))}/></label>
        <label>Additional system instructions<textarea rows={4} maxLength={6000} value={draft.system_instructions} onChange={e=>setDraft(v=>({...v,system_instructions:e.target.value}))}/></label>
        <p className="agent-note">Instructions specialize the answer style; mandatory evidence and citation rules remain in force.</p>
        <h3>Retrieval settings</h3><div className="param-grid agent-generation-grid">
          <label className="param-field"><span>Retrieval mode</span><select value={draft.retrieval.mode} onChange={e=>setRet("mode",e.target.value as RetrievalOptions["mode"])}>
            <option value="sparse">Sparse</option><option value="dense">Dense</option><option value="hybrid">Hybrid RRF</option></select></label>
          {draft.retrieval.mode !== "sparse" && <label className="param-field"><span>Embedding model</span><select value={draft.retrieval.embedding_model} onChange={e=>setRet("embedding_model",e.target.value as RetrievalOptions["embedding_model"])}>
            <option value="sentence-transformers/all-MiniLM-L6-v2">MiniLM-L6</option><option value="Alibaba-NLP/gte-large-en-v1.5">GTE Large</option></select></label>}
          <label className="param-field"><span>Second-stage reranker</span><select value={draft.retrieval.reranker ?? "none"} onChange={e=>setRet("reranker",e.target.value as RetrievalOptions["reranker"])}>
            <option value="none">None (first-stage only)</option>
            <option value="Alibaba-NLP/gte-reranker-modernbert-base">GTE ModernBERT · 149M</option>
            <option value="jinaai/jina-reranker-v3.5">Jina reranker v3.5 · 0.6B</option>
          </select></label>
          {(draft.retrieval.reranker ?? "none") !== "none" && <label className="param-field"><span>Rerank candidate pool</span>
            <input type="number" min={draft.retrieval.top_k} max={100} value={draft.retrieval.rerank_candidates ?? 20} onChange={e=>setRet("rerank_candidates",Number(e.target.value))}/></label>}
          {([["top_k","Retrieved chunks",1,12,1],
            ...(draft.retrieval.mode !== "dense" ? [["bm25_candidates", draft.retrieval.mode === "sparse" ? "Candidates" : "Sparse candidates",1,100,1]] as const : []),
            ...(draft.retrieval.mode !== "sparse" ? [["dense_candidates", draft.retrieval.mode === "dense" ? "Candidates" : "Dense candidates",1,100,1],["min_similarity","Min similarity",0,1,0.05]] as const : []),
            ...(draft.retrieval.mode === "hybrid" ? [["rrf_k","RRF smoothing",1,200,1]] as const : [])
          ] as const).map(([k,label,min,max,step])=><label className="param-field" key={k}><span>{label}</span><input type="number" min={min} max={max} step={step} value={draft.retrieval[k]} onChange={e=>setRet(k,Number(e.target.value))}/></label>)}
        </div><p className="agent-note">Reranking runs on a separate local inference service and changes passage order before the final top-k selection. No API key is required. The Jina model is CC BY-NC 4.0 (non-commercial).</p><h3>Generation settings</h3><div className="param-grid agent-generation-grid">
          <label className="param-field"><span>Model</span><select value={draft.generation.model??"qwen3.5:4b"} onChange={e=>setGen("model",e.target.value as GenerationOptions["model"])}>
            <option value="qwen3.5:4b">Qwen 3.5 · 4B</option><option value="llama3.2:3b">Llama 3.2 · 3B</option><option value="granite4.1:3b">Granite 4.1 · 3B</option><option value="openbmb/minicpm-v4.7:1b">MiniCPM-V 4.7 · 1B</option></select></label>
          {([["temperature","Temperature",0,1,0.05],["top_p","Top-p",0.01,1,0.05],["top_k","Top-k sampling",1,200,1],["min_p","Min-p sampling",0,1,0.01],["repeat_penalty","Repeat penalty",0.8,2,0.05]] as const).map(([k,label,min,max,step])=><label className="param-field" key={k}><span>{label}</span><input type="number" min={min} max={max} step={step} value={draft.generation[k]} onChange={e=>setGen(k,Number(e.target.value))}/></label>)}
          <label className="param-field"><span>Context tokens (0 = auto)</span><input type="number" min="0" step="1024" value={draft.generation.num_ctx} onChange={e=>setGen("num_ctx",Number(e.target.value))}/></label>
          <label className="param-field"><span>Max output tokens (-1 = auto)</span><input type="number" min="-1" value={draft.generation.num_predict} onChange={e=>setGen("num_predict",Number(e.target.value))}/></label>
          <label className="param-field"><span>Seed (optional)</span><input type="number" min="0" value={draft.generation.seed??""} onChange={e=>setGen("seed",e.target.value===""?null:Number(e.target.value))}/></label>
        </div><button className="agent-primary" disabled={busy||!draft.name.trim()} onClick={()=>void save()}><Save size={15}/>{busy?"Saving…":"Save RAG system"}</button>
      </div></div>}
  </div>;
}
