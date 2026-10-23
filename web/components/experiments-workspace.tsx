"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowUpRight, Check, CircleAlert, Database, FileUp, FlaskConical, Play, X } from "lucide-react";
import { BENCHMARK_CATALOG } from "@/lib/benchmark-catalog";
import { api, postJSON } from "@/lib/api";
import { parseBenchmarkFile, type BenchmarkRow } from "@/lib/benchmark-upload";

type RunResult = {
  dataset: string; source: string; sample_size: number; run_id?: string; save_error?: string;
  metrics: Record<string, number>; protocol_note: string;
  results: { id:string; question:string; prediction:string; reference:string; metrics:Record<string, number|boolean> }[];
};

/** Explicitly run a small sample against the configured local model; never auto-runs. */
export function ExperimentsWorkspace() {
  const fileRef = useRef<HTMLInputElement>(null);
  const [dataset, setDataset] = useState<File | null>(null);
  const [uploadedRows, setUploadedRows] = useState<BenchmarkRow[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [notice, setNotice] = useState("");
  const [count, setCount] = useState<number | "all">(3);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState<{processed:number;rows_read?:number;source_rows_total?:number;total?:number|null}|null>(null);
  const [running, setRunning] = useState(false);
  const [saveResults, setSaveResults] = useState(true);
  const [result, setResult] = useState<RunResult | null>(null);
  const selected = BENCHMARK_CATALOG.find(item => item.id === selectedId);
  const runnable = Boolean(dataset ? uploadedRows.length > 0 : selectedId === "qasper" || selectedId === "druid");

  function selectBenchmark(id: string) {
    setSelectedId(current => current === id ? null : id);
    setDataset(null); setUploadedRows([]);
    if (fileRef.current) fileRef.current.value = "";
    setNotice(""); setResult(null); setJobId(null);setProgress(null);
  }

  async function stageUpload(file: File | null) {
    setNotice(""); setResult(null);
    setSelectedId(null); setDataset(null); setUploadedRows([]); setJobId(null);setProgress(null);
    if (!file) return;
    try {
      const rows = parseBenchmarkFile(file.name, await file.text());
      setDataset(file); setUploadedRows(rows);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not read the dataset.");
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  useEffect(() => {
    if (!jobId) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const job = await api<{status:string;result:RunResult|null;error?:string|null;processed:number;rows_read?:number;source_rows_total?:number;total?:number|null}>(`/benchmarks/jobs/${jobId}`);
        if (!alive) return;
        setProgress(job);
        if (job.status === "completed" || job.status === "failed" || job.status === "cancelled") {
          if (job.result) setResult(job.result);
          if (job.error) setNotice(job.error);
          setRunning(false);setJobId(null);
        } else timer = setTimeout(poll, 2500);
      } catch (e) {
        if (alive) {setNotice(e instanceof Error ? e.message : "Lost benchmark job status.");setRunning(false);setJobId(null);}
      }
    }
    void poll();
    return () => {alive = false;clearTimeout(timer);};
  }, [jobId]);

  async function cancelRun() {
    if (!jobId) return;
    try {await api(`/benchmarks/jobs/${jobId}`, {method:"DELETE"});setNotice("Cancellation requested. The current batch will finish first.");}
    catch (error) {setNotice(error instanceof Error ? error.message : "Could not cancel run.");}
  }

  function randomRows(rows: BenchmarkRow[], n: number): BenchmarkRow[] {
    // Fresh cryptographic randomness: no starting-row bias for local files.
    const copy = [...rows];
    for (let i=copy.length-1; i>0; i--) {
      const j = Math.floor((crypto.getRandomValues(new Uint32Array(1))[0] / 4294967296) * (i+1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy.slice(0,n);
  }

  async function startRun() {
    if (!runnable || running) return;
    setRunning(true); setResult(null); setNotice("");setProgress(null);
    let launchedJob = false;
    try {
      const source = dataset ? "custom" : selectedId;
      if (count === "all") {
        if (dataset && uploadedRows.length > 10000) {
          throw new Error("Full local runs currently support up to 10,000 uploaded examples per run.");
        }
        const response = await postJSON<{job_id:string}>("/benchmarks/jobs", {
          dataset: source, save_results: saveResults,
          examples: dataset ? uploadedRows : [],
        });
        launchedJob = true;
        setJobId(response.job_id);
        return;
      }
      const response = await postJSON<RunResult>("/benchmarks/run", {
        dataset: source, count, random_sample: true, save_results: saveResults,
        examples: dataset ? randomRows(uploadedRows, count) : [],
      });
      setResult(response);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Benchmark failed. Check the API and local model.");
    } finally { if (!launchedJob) setRunning(false); }
  }

  return <div className="feature-page experiments-page experiments-simple">
    <header className="feature-head"><div>
      <div className="eyebrow">EVALUATION & DEBUGGING</div>
      <h1>Benchmark experiments</h1>
      <p>Run small, real evaluations using your configured LLM. Nothing runs until you press Start run.</p>
    </div></header>

    <section className="experiments-simple-panel" aria-labelledby="benchmark-choose-title">
      <div className="experiments-simple-heading"><h2 id="benchmark-choose-title">Choose a dataset</h2><span>Step 1 of 2</span></div>
      <div className="experiments-simple-options">
        {BENCHMARK_CATALOG.map(item => <button type="button" key={item.id}
          className={`experiments-option ${selectedId === item.id ? "selected" : ""}`}
          aria-pressed={selectedId === item.id} disabled={running}
          onClick={() => selectBenchmark(item.id)}>
          <span className="experiments-option-indicator">{selectedId === item.id ? <Check size={16}/> : <Database size={16}/>}</span>
          <span className="experiments-option-body"><strong>{item.name}</strong><small>{item.track} · {item.metrics}</small>
            <small>{item.id === "qasper" || item.id === "druid" ? "Small sample available" : "Reference only · advanced setup required"}</small></span>
        </button>)}
        <button type="button" className={`experiments-option experiments-option-custom ${dataset ? "selected" : ""}`} disabled={running} onClick={() => fileRef.current?.click()}>
          <span className="experiments-option-indicator"><FileUp size={16}/></span>
          <span className="experiments-option-body"><strong>Your own dataset</strong><small>{dataset ? `${dataset.name} · ${uploadedRows.length} valid examples` : "JSON, JSONL or CSV · question, answer, optional context"}</small></span>
        </button>
      </div>
      <input ref={fileRef} className="visually-hidden" aria-label="Select local benchmark dataset" type="file" accept=".json,.jsonl,.csv" onChange={event => void stageUpload(event.target.files?.[0] ?? null)}/>
      {dataset && <div className="experiments-selection-inline"><FileUp size={15}/><span>{dataset.name} · {(dataset.size / 1024).toFixed(1)} KB · the selected random sample (or all rows) is sent when you run</span><button type="button" onClick={() => { setDataset(null);setUploadedRows([]);setResult(null);if (fileRef.current) fileRef.current.value = ""; }} aria-label="Clear dataset"><X size={15}/></button></div>}
      {selected && <div className="experiments-protocol">
        <div className="experiments-protocol-name"><strong>{selected.name}</strong><span>{selected.year}</span></div>
        <p>{selected.summary}</p>
        <details><summary>Evaluation protocol and requirements</summary>
          <p><strong>Official protocol</strong> · {selected.protocol}</p>
          <p><strong>Official metrics</strong> · {selected.metrics}</p>
          <p className="experiments-protocol-note"><CircleAlert size={14}/>{selected.caveat}</p>
        </details>
        <div className="experiments-protocol-links"><a href={selected.datasetUrl} target="_blank" rel="noopener noreferrer">Dataset <ArrowUpRight size={14}/></a><a href={selected.officialUrl} target="_blank" rel="noopener noreferrer">Protocol / paper <ArrowUpRight size={14}/></a></div>
      </div>}
    </section>
    <section className="experiments-simple-panel experiments-launch" aria-labelledby="benchmark-run-title">
      <div className="experiments-simple-heading"><h2 id="benchmark-run-title">Run evaluation</h2><span>Step 2 of 2</span></div>
      <div className="experiments-run-config"><label>Sample size <select aria-label="Benchmark sample size" value={count} disabled={running} onChange={event=>setCount(event.target.value === "all" ? "all" : Number(event.target.value))}>{[1,3,5,10].map(n=><option key={n} value={n}>{n} random examples</option>)}<option value="all">All examples</option></select></label></div>
      <p>{count === "all" ? "Full runs evaluate every labeled example of the selected dataset split. Public datasets may take hours or longer, depending on the model and dataset size; the API must stay running." : "A new random sample is drawn for every run. Public samples record their random seed and selected row indices in the results."}</p>
      <p>{runnable ? (dataset ? "Custom rows without context will search your indexed papers. Only selected rows leave your browser for your local API." : "The selected public sample is fetched on demand. Your local model answers using the dataset-provided evidence; no papers are imported.") : "Choose QASPER, DRUID or a local labeled dataset to run. Larger aira benchmarks still require their official corpora and evaluators."}</p>
      <label className="experiments-save-option" style={{display:"flex",alignItems:"center",gap:8,margin:"14px 0",fontSize:12}}>
        <input type="checkbox" checked={saveResults} disabled={running} onChange={event=>setSaveResults(event.target.checked)}/>
        Save benchmark predictions locally for reproducible DuckDB analysis
      </label>
      <button type="button" className="feature-action primary experiments-run" disabled={!runnable || running} onClick={() => void startRun()}><Play size={16}/>{running ? "Running model evaluations…" : "Start run"}</button>
      {jobId && <button type="button" className="feature-action secondary" onClick={()=>void cancelRun()}>Cancel full run</button>}
      {running && <p role="status" className="experiments-notice">{count === "all" ? `Full run in progress: ${progress?.processed ?? 0} examples evaluated${progress?.source_rows_total ? ` · ${progress.rows_read ?? 0}/${progress.source_rows_total} source rows read` : ""}.` : `Evaluating ${count} random examples…`}</p>}
      {notice && <p role="alert" className="experiments-notice"><CircleAlert size={15}/>{notice}</p>}
      {result && <div className="experiments-results" role="region" aria-label="Benchmark results">
        <h3>Results · {result.sample_size} evaluated</h3><p className="experiments-results-source">{result.source}</p>
        {result.run_id && <p className="experiments-results-source">Saved locally · run ID: <code>{result.run_id}</code></p>}
        {result.save_error && <p role="alert" className="experiments-notice">{result.save_error}</p>}
        <div className="experiments-metric-cards">{Object.entries(result.metrics).map(([key,value])=><div key={key}><strong>{Math.round(value * 100)}%</strong><span>{key.replace(/_/g," ")}</span></div>)}</div>
        <p className="experiments-results-note">{result.protocol_note}</p>
        <details><summary>Inspect individual predictions</summary>{result.results.map(item=><article className="experiments-result-row" key={item.id}><strong>{item.question}</strong><p><b>Model:</b> {item.prediction || "(empty response)"}</p><p><b>Reference:</b> {item.reference}</p><small>{Object.entries(item.metrics).map(([key,value])=>`${key.replace(/_/g," ")}: ${typeof value === "boolean" ? (value?"correct":"incorrect") : `${Math.round(value*100)}%`}`).join(" · ")}</small></article>)}</details>
      </div>}
    </section>
  </div>;
}
