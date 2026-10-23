"use client";

import { Activity, ArrowRight, BarChart3, Bot, Database, FileText, Settings2, Workflow, Layers3 } from "lucide-react";

export type ConfigurationDestination = "agents" | "rag" | "library" | "traces" | "experiments" | "retrieval-eval";

type Props = {
  documentCount: number;
  agentCount: number;
  ragCount: number;
  chunkCount: number;
  onNavigate: (destination: ConfigurationDestination) => void;
};

/** Landing page for the three configuration areas. No sidebar dropdowns. */
export function ConfigurationWorkspace({ documentCount, agentCount, ragCount, chunkCount, onNavigate }: Props) {
  return <div className="feature-page configuration-page">
    <header className="feature-head"><div>
      <div className="eyebrow">WORKSPACE SETTINGS</div>
      <h1>Configuration</h1>
      <p>Manage your research assistants, indexed documents and evaluation tools from one place.</p>
    </div></header>
    <div className="configuration-grid">
      <button className="configuration-card" type="button" onClick={() => onNavigate("agents")}>
        <span className="configuration-card-icon"><Bot size={22}/></span>
        <span className="configuration-card-copy"><strong>Research agents</strong><small>Configure your custom research agents.</small><span className="configuration-card-link">Agents available: {agentCount} <ArrowRight size={15}/></span></span>
      </button>
      <button className="configuration-card" type="button" onClick={() => onNavigate("rag")}>
        <span className="configuration-card-icon"><Layers3 size={22}/></span>
        <span className="configuration-card-copy"><strong>RAG systems</strong><small>Configure your custom retrieval-augmented generation (RAG) systems.</small><span className="configuration-card-link">Systems available: {ragCount} <ArrowRight size={15}/></span></span>
      </button>
      <button className="configuration-card configuration-library" type="button" onClick={() => onNavigate("library")}>
        <span className="configuration-card-icon"><FileText size={22}/></span>
        <span className="configuration-card-copy"><strong>Your library</strong><small>Upload, index and browse the local documents used for research evidence.</small><span className="configuration-card-link">Papers available: {documentCount} · Indexed chunks: {chunkCount} <ArrowRight size={15}/></span></span>
      </button>
      <section className="configuration-card configuration-evaluation" aria-labelledby="configuration-evaluation-title">
        <span className="configuration-card-icon"><Activity size={22}/></span>
        <div className="configuration-card-copy">
          <strong id="configuration-evaluation-title">Evaluation and Debugging</strong>
          <small>Inspect real agent traces and explore benchmark protocols.</small>
          <div className="configuration-evaluation-links">
            <button type="button" onClick={() => onNavigate("traces")}><Workflow size={17}/><span>Behind the scenes</span><ArrowRight size={15}/></button>
            <button type="button" onClick={() => onNavigate("retrieval-eval")}><BarChart3 size={17}/><span>Retrieval Evaluation</span><ArrowRight size={15}/></button>
            <button type="button" onClick={() => onNavigate("experiments")}><Database size={17}/><span>Experiments on Benchmark Datasets</span><ArrowRight size={15}/></button>
          </div>
        </div>
      </section>
    </div>
  </div>;
}
