"use client";

import { useCallback, useEffect, useState } from "react";
import { Activity, AlertCircle, ArrowRight, Database, Play, RefreshCw, StopCircle } from "lucide-react";
import { api, postJSON } from "@/lib/api";
import type { RetrievalOptions } from "@/lib/types";

type Dataset = {id:string;name:string;available:boolean;documents:number|null;queries:number|null;official:boolean};
type Metric = {recall:number;mrr:number;ndcg:number;precision:number;p50_ms:number;p95_ms:number};
type Run = {run_id:string;created_at:string;dataset:string;dataset_fingerprint:string;sample_size:number;corpus_size:number;seed:number;
  options:RetrievalOptions;metrics:Metric;protocol:string;query_ids?:string[];
  timing_note?:string;cases?:{query_id:string;query:string;metrics:Record<string,number>;latency_ms:number;relevant_ids:string[];
    retrieved:{id:string;score:number;first_stage_score?:number|null}[]}[]};
type Job = {status:string;processed:number;total:number;phase:string;error?:string|null;run_id?:string|null};
const DEFAULT: RetrievalOptions = {mode:"sparse",embedding_model:"sentence-transformers/all-MiniLM-L6-v2",top_k:5,
  bm25_candidates:20,dense_candidates:20,rrf_k:60,min_similarity:0,reranker:"none",rerank_candidates:20};
const MODES = [{value:"sparse",name:"Sparse"},{value:"dense",name:"Dense"},{value:"hybrid",name:"Hybrid"}] as const;
const MODEL_NAMES:Record<string,string> = {none:"No reranker","jinaai/jina-reranker-v3.5":"Jina v3.5",
  "Alibaba-NLP/gte-reranker-modernbert-base":"GTE ModernBERT"};
const percentage = (value:number) => `${(value * 100).toFixed(1)}%`;
const label = (run:Run) => `${run.options.mode}${run.options.reranker === "none" ? "" : ` + ${MODEL_NAMES[run.options.reranker]}`}`;

export function RetrievalEvaluationWorkspace() {
  const [datasets,setDatasets]=useState<Dataset[]>([]);
  const [dataset,setDataset]=useState("smoke");
  const [options,setOptions]=useState<RetrievalOptions>(DEFAULT);
  const [sample,setSample]=useState("all");
  const [seed,setSeed]=useState(42);
  const [runs,setRuns]=useState<Run[]>([]);
  const [chosen,setChosen]=useState<Run|null>(null);
  const [jobId,setJobId]=useState<string|null>(null);
  const [progress,setProgress]=useState<Job|null>(null);
  const [error,setError]=useState("");
  const [busy,setBusy]=useState(false);

  const refresh=useCallback(async()=>{
    const [available,past]=await Promise.all([api<{datasets:Dataset[]}>("/retrieval-evals/datasets"),api<{runs:Run[]}>("/retrieval-evals/runs")]);
    setDatasets(available.datasets);setRuns(past.runs);
  },[]);
  useEffect(()=>{void refresh().catch(e=>setError(e instanceof Error?e.message:"Could not load evaluation datasets"));},[refresh]);
  useEffect(()=>{
    if(!jobId)return;
    let active=true;
    let timer:ReturnType<typeof setTimeout>;
    async function poll(){
      try{
        const state=await api<Job>(`/retrieval-evals/jobs/${jobId}`);
        if(!active)return;
        setProgress(state);
        if(["completed","failed","cancelled"].includes(state.status)){
          setBusy(false);setJobId(null);
          if(state.error)setError(state.error);
          if(state.run_id){
            const full=await api<Run>(`/retrieval-evals/runs/${state.run_id}`);
            if(active){setChosen(full);await refresh();}
          }
        }else timer=setTimeout(poll,1800);
      }catch(e){if(active){setBusy(false);setJobId(null);setError(e instanceof Error?e.message:"Status unavailable");}}
    }
    void poll();
    return ()=>{active=false;clearTimeout(timer);};
  },[jobId,refresh]);
  async function start(){
    if(busy)return;
    setError("");setChosen(null);setProgress(null);setBusy(true);
    try{
      const job=await postJSON<{job_id:string}>("/retrieval-evals/jobs",{dataset,retrieval:options,count:sample==="all"?null:Number(sample),seed});
      setJobId(job.job_id);
    }catch(e){setBusy(false);setError(e instanceof Error?e.message:"Could not start evaluation");}
  }
  async function open(run:Run){
    try{setChosen(await api<Run>(`/retrieval-evals/runs/${run.run_id}`));setError("");}
    catch(e){setError(e instanceof Error?e.message:"Could not open run");}
  }
  const currentDataset=datasets.find(d=>d.id===dataset);
  // Only compare metric bars for identical datasets, sample sets, and k.
  const comparable=runs.filter(r=>r.dataset===(chosen?.dataset??dataset) &&
    r.seed===(chosen?.seed??seed) && (!chosen || r.dataset_fingerprint===chosen.dataset_fingerprint) && r.sample_size===(chosen?.sample_size??(sample==="all"?currentDataset?.queries:Number(sample))) &&
    r.options.top_k===(chosen?.options.top_k??options.top_k)).slice(0,8);
  return <div className="feature-page retrieval-evaluation-page">
    <header className="feature-head"><div><div className="eyebrow">OBJECTIVE RETRIEVAL QUALITY</div>
      <h1>Retrieval evaluation</h1>
      <p>Compare Sparse, Dense, Hybrid and second-stage rerankers on a fixed, relevance-judged corpus. No LLM-as-a-judge.</p>
    </div></header>
    <div className="retrieval-eval-grid">
      <section className="experiments-simple-panel retrieval-eval-config" aria-label="Evaluation configuration">
        <div className="experiments-simple-heading"><h2>New retrieval run</h2><span>Local evaluation</span></div>
        <div className="retrieval-eval-fields">
          <label>Dataset<select value={dataset} disabled={busy} onChange={e=>{setDataset(e.target.value);setChosen(null);}}>
            {datasets.map(d=><option value={d.id} disabled={!d.available} key={d.id}>{d.name}{d.available?"":" · not installed"}</option>)}
          </select></label>
          <label>Retrieval mode<select value={options.mode} disabled={busy} onChange={e=>setOptions(p=>({...p,mode:e.target.value as RetrievalOptions["mode"]}))}>
            {MODES.map(m=><option key={m.value} value={m.value}>{m.name}</option>)}
          </select></label>
          <label>Second-stage reranker<select value={options.reranker} disabled={busy} onChange={e=>setOptions(p=>({...p,reranker:e.target.value as RetrievalOptions["reranker"]}))}>
            {Object.entries(MODEL_NAMES).map(([id,name])=><option key={id} value={id}>{name}</option>)}
          </select></label>
          {options.mode!=="sparse"&&<label>Embedding model<select value={options.embedding_model} disabled={busy} onChange={e=>setOptions(p=>({...p,embedding_model:e.target.value as RetrievalOptions["embedding_model"]}))}>
            <option value="sentence-transformers/all-MiniLM-L6-v2">MiniLM-L6</option>
            <option value="Alibaba-NLP/gte-large-en-v1.5">GTE Large</option>
          </select></label>}
          <label>Evaluation cutoff (k)<select disabled={busy} value={options.top_k} onChange={e=>setOptions(p=>({...p,top_k:Number(e.target.value)}))}>
            {[1,3,5,10].map(k=><option key={k} value={k}>{k}</option>)}
          </select></label>
          <label>Queries<select value={sample} disabled={busy} onChange={e=>setSample(e.target.value)}>
            {[3,10,30].map(n=><option key={n} value={n}>{n} random queries</option>)}<option value="all">All judged queries</option>
          </select></label>
          <label>Random seed<input type="number" min={0} max={2147483647} value={seed} disabled={busy} onChange={e=>setSeed(Math.max(0,Math.min(2147483647,Number(e.target.value)||0)))}/></label>
          <label>Candidate pool<input type="number" min={options.top_k} max={100} value={options.mode==="dense"?options.dense_candidates:options.bm25_candidates} disabled={busy} onChange={e=>setOptions(p=>({...p,bm25_candidates:Math.max(p.top_k,Number(e.target.value)||p.top_k),dense_candidates:Math.max(p.top_k,Number(e.target.value)||p.top_k)}))}/></label>
          {options.reranker!=="none"&&<label>Reranking candidates<input type="number" min={options.top_k} max={100} value={options.rerank_candidates} disabled={busy} onChange={e=>setOptions(p=>({...p,rerank_candidates:Math.max(p.top_k,Number(e.target.value)||p.top_k)}))}/></label>}
          {options.mode==="hybrid"&&<label>RRF constant<input type="number" min={1} max={200} value={options.rrf_k} disabled={busy} onChange={e=>setOptions(p=>({...p,rrf_k:Math.max(1,Number(e.target.value)||1)}))}/></label>}
        </div>
        <p className="retrieval-eval-hint">{currentDataset?.official?"Official SciFact test relevance judgments. The complete benchmark corpus is always searched, even for a small query sample.":"Synthetic examples for verifying the evaluation pipeline—not scientifically meaningful benchmark scores."}</p>
        {dataset==="scifact"&&!currentDataset?.available&&<p className="retrieval-eval-hint">Install once from the project root: <code>python scripts/download_scifact.py</code></p>}
        <button type="button" className="feature-action primary" disabled={!currentDataset?.available||busy} onClick={()=>void start()}><Play size={16}/> {busy?"Evaluating…":"Run retrieval evaluation"}</button>
        {busy&&<div className="retrieval-eval-progress" role="status"><Activity size={15}/>{progress?.phase||"Starting…"} · {progress?.processed??0}/{progress?.total??"?"} queries
          <button type="button" onClick={()=>{if(jobId)void api(`/retrieval-evals/jobs/${jobId}`,{method:"DELETE"}).catch(e=>setError(String(e)));}}><StopCircle size={14}/> Cancel</button></div>}
        {error&&<p className="retrieval-eval-error" role="alert"><AlertCircle size={15}/>{error}</p>}
      </section>
      <section className="experiments-simple-panel retrieval-eval-history" aria-label="Previous retrieval runs">
        <div className="experiments-simple-heading"><h2>Saved experiments</h2><button type="button" className="retrieval-eval-refresh" onClick={()=>void refresh().catch(e=>setError(String(e)))}><RefreshCw size={15}/> Refresh</button></div>
        {runs.length===0?<p className="retrieval-eval-hint">No runs yet. Execute the smoke test or install SciFact to start comparing retrieval systems.</p>:
          <div className="retrieval-eval-runlist">{runs.map(run=><button type="button" key={run.run_id} className={chosen?.run_id===run.run_id?"retrieval-eval-selected":""} onClick={()=>void open(run)}>
            <span><strong>{label(run)}</strong><small>{run.dataset} · {run.sample_size} queries · seed {run.seed} · k={run.options.top_k}</small></span>
            <span><strong>{percentage(run.metrics.ndcg)}</strong><small>nDCG@{run.options.top_k}</small></span><ArrowRight size={14}/>
          </button>)}</div>}
      </section>
    </div>
    {chosen&&<section className="experiments-simple-panel retrieval-eval-results" aria-label="Retrieval evaluation results">
      <div className="experiments-simple-heading"><h2>Result · {label(chosen)}</h2><small>{chosen.dataset} · {chosen.sample_size} queries · seed {chosen.seed}</small></div>
      <div className="retrieval-eval-metrics">{([
        ["Recall",percentage(chosen.metrics.recall)],["MRR",percentage(chosen.metrics.mrr)],
        ["nDCG",percentage(chosen.metrics.ndcg)],["Precision",percentage(chosen.metrics.precision)],
        ["p50 latency",`${chosen.metrics.p50_ms.toFixed(1)} ms`],["p95 latency",`${chosen.metrics.p95_ms.toFixed(1)} ms`]
      ] as const).map(([name,value])=><div key={name}><small>{name}{["Recall","MRR","nDCG","Precision"].includes(name)?`@${chosen.options.top_k}`:""}</small><strong>{value}</strong></div>)}</div>
      <p className="retrieval-eval-hint">{chosen.protocol} {chosen.timing_note}</p>
      <h3>Comparable saved runs</h3>
      <p className="retrieval-eval-hint">Only matching dataset, sample size, random seed and evaluation k are compared. Different embedding and reranking models are allowed.</p>
      <div className="retrieval-eval-bars">{comparable.map(run=><div key={run.run_id} className="retrieval-eval-bar-row"><span title={label(run)}>{label(run)}</span><div className="retrieval-eval-track"><div style={{width:`${Math.max(0,Math.min(100,run.metrics.ndcg*100))}%`}}/></div><b>{percentage(run.metrics.ndcg)}</b></div>)}</div>
      {chosen.cases&&<details className="retrieval-eval-cases"><summary>Inspect per-query rankings and relevance judgments ({chosen.cases.length})</summary>
        {chosen.cases.map(c=><article key={c.query_id}><strong>{c.query_id} · {c.query}</strong><small>Recall {percentage(c.metrics.recall)} · nDCG {percentage(c.metrics.ndcg)} · {c.latency_ms} ms</small>
          <p>Relevant documents: {c.relevant_ids.join(", ")}</p>
          <div>{c.retrieved.map((hit,i)=><span key={`${c.query_id}-${hit.id}`} className={c.relevant_ids.includes(hit.id)?"relevant":""}>{i+1}. {hit.id}{c.relevant_ids.includes(hit.id)?" ✓":""}</span>)}</div>
        </article>)}
      </details>}
    </section>}
    <p className="feature-footnote"><Database size={14}/> Evaluation data is saved under <code>.data/retrieval_eval/runs</code> and can be exported through DuckDB. These are document-level offline rankings, not production pgvector ANN latency measurements.</p>
  </div>;
}
