"use client";

import { useEffect, useMemo, useState } from "react";
import { ArrowUpRight, CircleAlert, RefreshCw, Workflow } from "lucide-react";
import { api } from "@/lib/api";
import type { TraceList, TraceRun, TraceSpan } from "@/lib/types";
import { layoutResearchTrace } from "@/lib/trace-layout";

type SessionList = {available:boolean; sessions:{session_id:string;num_traces:number;start_time:string}[];message?:string|null};
const micro = (us: number) => us >= 1000000 ? `${(us / 1000000).toFixed(1)} s` : `${(us / 1000).toFixed(0)} ms`;

/** Friendly labels are presentation only: the recorded order and timing are unchanged. */
function friendlyStep(kind: string, span: TraceSpan | undefined, fallback: string) {
  if (kind === "input") return {icon: "✎", title: "Your question", description: "The question that started this research"};
  if (kind === "output") return {icon: "✓", title: "Answer ready", description: "The assistant's final response"};
  const name = span?.name || fallback;
  const count = span?.evidence_count;
  if (name.startsWith("web.")) return {icon: "⌕", title: "Searching the web", description: count != null ? `${count} results found` : "Looking for relevant sources"};
  if (name.startsWith("academic.")) return {icon: "▤", title: "Finding academic papers", description: count != null ? `${count} papers found` : "Checking research sources"};
  if (name.startsWith("memory.")) return {icon: "◷", title: "Checking earlier context", description: "Looking at previous conversations"};
  if (name.startsWith("retrieval.")) return {icon: "▤", title: "Searching your papers", description: count != null ? `${count} passages retrieved` : "Looking for relevant passages"};
  if (name.includes("read_web_page") || name.includes("read_page")) return {icon: "▧", title: "Reading a source", description: "Opening a webpage for more detail"};
  if (kind === "model" || name.startsWith("llm.")) return {icon: "✦", title: "Consulting the AI model", description: "Working with the collected context"};
  if (kind === "tool") return {icon: "⌁", title: "Using a research tool", description: "Gathering or checking information"};
  return {icon: "●", title: "Processing evidence", description: "A recorded research operation"};
}

function SpanGraph({ trace }: { trace: TraceRun }) {
  const layout = useMemo(() => layoutResearchTrace(trace), [trace]);
  const [selected, setSelected] = useState<string | null>(null);
  const spansById = useMemo(() => new Map(trace.spans.map(span => [span.id, span])), [trace]);
  const selectedNode = layout.nodes.find(item => item.key === selected);
  const span = selectedNode?.spanId ? spansById.get(selectedNode.spanId) : undefined;
  const run = trace.spans.find(item => item.name === "research.chat" || item.name === "literature.survey");
  const contentCaptured = Boolean(run?.input || run?.output);
  const numberOfModels = layout.nodes.filter(node => node.kind === "model").length;
  const numberOfTools = layout.nodes.filter(node => node.kind === "tool" || node.kind === "retrieval").length;

  return <section className="trace-diagram trace-journey" aria-label="How the assistant produced this answer">
    <header className="trace-journey-header"><div>
      <strong><Workflow size={20}/> How your answer was made</strong>
      <p>Follow the steps in the order they were recorded.</p>
    </div><span className="trace-journey-duration">{micro(trace.duration_us)} total</span></header>
    <div className="trace-journey-summary"><span>{Math.max(0,layout.nodes.length-2)} recorded steps</span>{numberOfTools>0 && <span>{numberOfTools} searches & tools</span>}{numberOfModels>0 && <span>{numberOfModels} model calls</span>}</div>
    <ol className="trace-journey-steps">
      {layout.nodes.map((node, index) => {
        const recorded = node.spanId ? spansById.get(node.spanId) : undefined;
        const display = friendlyStep(node.kind, recorded, node.label);
        return <li key={node.key} className={`trace-journey-item trace-journey-${node.kind}`}>
          <span className="trace-journey-connector" aria-hidden="true"><span>{index === layout.nodes.length - 1 ? "✓" : index + 1}</span></span>
          <button className={`trace-graph-node trace-graph-${node.kind} trace-journey-step ${selected === node.key ? "selected" : ""}`}
            type="button" onClick={() => setSelected(current => current === node.key ? null : node.key)}
            aria-expanded={selected === node.key} aria-label={`${display.title}. Click for details.`}>
            <span className="trace-journey-icon" aria-hidden="true">{display.icon}</span>
            <span className="trace-journey-copy"><strong>{display.title}</strong><small>{display.description}</small></span>
            {node.kind !== "input" && node.kind !== "output" && recorded && <span className="trace-journey-time">{micro(recorded.duration_us)}</span>}
            <span className="trace-journey-chevron" aria-hidden="true">{selected === node.key ? "−" : "+"}</span>
          </button>
          {selected === node.key && <div className="trace-journey-detail" role="region" aria-label={`${display.title} details`}>
            {span && <><p><strong>Recorded operation:</strong> {span.name}</p>
              <p><strong>Duration:</strong> {micro(span.duration_us)}{span.error ? " · An error occurred" : ""}</p>
              {span.model && <p><strong>Model:</strong> {span.model}</p>}{span.tool && <p><strong>Tool:</strong> {span.tool}</p>}
              {span.strategy && <p><strong>Search strategy:</strong> {span.strategy}</p>}
              {span.evidence_count != null && <p><strong>Evidence found:</strong> {span.evidence_count}</p>}
              {node.kind === "input" && <p><strong>Question:</strong> {span.input || "Question text was not captured."}</p>}
              {node.kind === "output" && <p><strong>Answer:</strong> {span.output || "Answer text was not captured."}</p>}
              {node.kind !== "input" && node.kind !== "output" && span.input && <p><strong>Input:</strong> {span.input}</p>}
              {node.kind !== "input" && node.kind !== "output" && span.output && <p><strong>Output:</strong> {span.output}</p>}
            </>}
          </div>}
        </li>;
      })}
    </ol>
    <footer className="trace-journey-footer"><span>Tap a step to see what happened.</span><details><summary>About the sequence</summary><p>Steps show observed start-time order, not necessarily dependencies. Detailed technical relationships are available in Phoenix.</p>{!contentCaptured && <p>Question and answer text were not saved in this trace. Content capture is optional and applies only to new runs.</p>}</details></footer>
  </section>;
}

export function TracesWorkspace({ conversationTitles = {} }: { conversationTitles?: Record<string, string> }) {
  const [result, setResult] = useState<TraceList | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeId, setActiveId] = useState("");
  const [sessions, setSessions] = useState<SessionList | null>(null);
  const [selectedSession, setSelectedSession] = useState("");
  async function refresh(sessionFilter = selectedSession) {
    setLoading(true);setError("");
    try {
      const data = await api<TraceList>(`/traces?limit=12${sessionFilter ? `&session_id=${encodeURIComponent(sessionFilter)}` : ""}`);
      setResult(data);
      try { setSessions(await api<SessionList>("/sessions?limit=60")); } catch { setSessions(null); }
      setActiveId(old=>data.traces.some(t=>t.id===old)?old:data.traces[0]?.id||"");
    }
    catch(err){setError(err instanceof Error?err.message:"Could not load traces");}
    finally{setLoading(false);}
  }
  useEffect(()=>{void refresh("");},[]);
  const current = result?.traces.find(item=>item.id===activeId);
  return <div className="feature-page trace-page">
    <header className="feature-head"><div><div className="eyebrow">OBSERVABILITY</div><h1>Behind the scenes</h1><p>See what the research agent actually did: model calls, retrieval, tool calls, and timing.</p></div>
      <div className="feature-actions"><button className="feature-action secondary" onClick={()=>void refresh()} disabled={loading}><RefreshCw className={loading?"spin":""} size={15}/> Refresh</button><a className="feature-action primary" href="http://localhost:6006" target="_blank" rel="noopener noreferrer">Open Phoenix <ArrowUpRight size={16}/></a></div>
    </header>
    {(sessions?.sessions?.length || 0) > 0 && <div className="trace-picker trace-session-picker"><label>Conversation session <select aria-label="Phoenix conversation session" value={selectedSession} onChange={event=>{const value=event.target.value; setSelectedSession(value);void refresh(value);}}>
      <option value="">All recent research sessions</option>
      {(sessions?.sessions || []).map(session=><option key={session.session_id} value={session.session_id}>{conversationTitles[session.session_id] || `${session.session_id.slice(0,13)}…`} · {session.num_traces} trace(s)</option>)}
      </select></label><small>{sessions?.sessions.length} session(s) in Phoenix</small></div>}
    {error ? <div className="feature-error"><CircleAlert size={16}/>{error}</div> : !result?.available ? <div className="feature-empty"><Workflow size={28}/><h2>No Phoenix connection</h2><p>Start Phoenix and enable OpenTelemetry on the API, then send a research question.</p><code>docker compose --profile tracing up -d phoenix</code></div>
      : !result.traces.length ? <div className="feature-empty"><Workflow size={28}/><h2>No recorded traces yet</h2><p>{result.message || "Send a new question in RAG or Agentic RAG with tracing enabled, then refresh. Diagnostic and health traces are hidden."}</p></div>
      : <><div className="trace-picker"><label>Recorded request <select value={activeId} onChange={e=>setActiveId(e.target.value)}>{result.traces.map((trace,i)=><option key={trace.id} value={trace.id}>Research run {i+1} · {micro(trace.duration_us)} · {trace.spans.length} spans · {trace.id.slice(0,10)}</option>)}</select></label><small>Latest {result.traces.length} traces {sessions && !sessions.available ? "· Phoenix session API unavailable" : ""}</small></div>
      {current && <SpanGraph key={current.id} trace={current}/>}</>}
  </div>;
}
