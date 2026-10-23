"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ArrowRight, BookOpen, ExternalLink, LoaderCircle, MessageSquareText, RotateCcw, Search, SlidersHorizontal, Send, Settings2, X, ShieldCheck } from "lucide-react";
import { api, postJSON } from "@/lib/api";
import type { GenerationOptions } from "@/lib/types";
import { paperSessionTitle } from "@/lib/paper-history";
import type { PaperAnswerResult, PaperSession, PaperTurn } from "@/lib/paper-history";

type ScholarAnswer = PaperAnswerResult & { scope: "abstracts_only" };

const SUGGESTIONS = [
  "How do researchers evaluate the reliability of RAG systems?",
  "What are the main differences between LoRA and QLoRA?",
  "How do AI agents retrieve and maintain long-term memory?",
];
const PROVIDERS = [
  { id: "arxiv", label: "arXiv" },
  { id: "openalex", label: "OpenAlex" },
  { id: "acl", label: "ACL" },
  { id: "acm", label: "ACM" },
];

function PaperAnswer({ turn, turnIndex }: { turn: PaperTurn; turnIndex: number }) {
  const [expanded, setExpanded] = useState<number | null>(null);
  // Only numeric citations linked to real paper records become clickable.
  const linkedAnswer = turn.result.answer.replace(/\[(\d{1,4})\](?!\()/g, (text, raw: string) => {
    const number = Number(raw);
    return turn.result.papers.some(p => p.number === number)
      ? `[${number}](#scholar-paper-${turnIndex}-${number})` : text;
  });
  const citedCount = turn.result.papers.filter(p => p.cited).length;
  return <article className="scholar-turn">
    <div className="scholar-question"><MessageSquareText size={18}/><h2>{turn.question}</h2></div>
    <section className="scholar-answer" aria-label="Evidence-based synthesis">
      <div className="scholar-answer-heading"><BookOpen size={18}/><strong>Research synthesis</strong><span>{citedCount} paper{citedCount === 1 ? "" : "s"} cited</span></div>
      <div className="scholar-answer-prose answer-markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
          a: ({ href, children }) => {
            const match = /^#scholar-paper-\d+-(\d+)$/.exec(href || "");
            if (match) {
              const number = Number(match[1]);
              return <a className="scholar-inline-citation" href={href} onClick={() => setExpanded(number)}
                title={`Inspect paper ${number}`} aria-label={`Inspect cited paper ${number}`}>{children}</a>;
            }
            return <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>;
          },
        }}>{linkedAnswer}</ReactMarkdown>
      </div>
      <p className="scholar-evidence-note">Based on publicly indexed abstracts, not verified full-text readings. Follow citations to inspect the original papers.</p>
      {Object.keys(turn.result.provider_errors).length > 0 && <p className="scholar-provider-warning" role="note">Some academic sources were unavailable: {Object.keys(turn.result.provider_errors).join(", ")}. Results may be incomplete.</p>}
    </section>
    <section className="scholar-references" aria-label="Papers used for synthesis">
      <div className="scholar-section-head"><div><span className="eyebrow">TRACEABLE EVIDENCE</span><h3>Papers consulted</h3></div><span>{turn.result.papers.length} abstracts</span></div>
      <p className="scholar-references-intro">All retrieved abstracts used as model context are listed here. A “cited” label means the answer explicitly refers to that paper; the others are contextual references.</p>
      <div className="scholar-paper-list">{turn.result.papers.map(p => <article className="scholar-paper" id={`scholar-paper-${turnIndex}-${p.number}`} key={p.number}>
        <div className="scholar-paper-number">[{p.number}]</div>
        <div className="scholar-paper-details">
          <div className="scholar-paper-title"><a href={p.url} target="_blank" rel="noopener noreferrer">{p.title} <ExternalLink size={13}/></a>{p.cited && <span className="scholar-cited-tag">Cited</span>}</div>
          <p>{p.authors.slice(0,3).join(", ") || "Authors unavailable"}{p.authors.length>3 ? " et al." : ""} · {p.year ?? "Year unknown"} · {p.provider}</p>
          <button className="scholar-abstract-toggle" type="button" aria-expanded={expanded === p.number} onClick={() => setExpanded(expanded === p.number ? null : p.number)}>{expanded === p.number ? "Hide abstract" : "Read source abstract"}</button>
          {expanded === p.number && <p className="scholar-source-abstract">{p.abstract}</p>}
        </div>
      </article>)}</div>
    </section>
  </article>;
}

export function LiteratureWorkspace({ generation, session, onSave, onNewTopic }: {
  generation: GenerationOptions; session: PaperSession | null; onSave: (session: PaperSession) => void; onNewTopic: () => void;
}) {
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<PaperTurn[]>(session?.turns ?? []);
  const [sources, setSources] = useState<string[]>(PROVIDERS.map(provider => provider.id));
  const [topK, setTopK] = useState(8);
  const [year, setYear] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [chosenModel, setChosenModel] = useState<string>("qwen3.5:4b");
  const [serverResults, setServerResults] = useState<{model:string;ready:boolean;detail:string}[]>([]);
  const [checkingServers, setCheckingServers] = useState(false);
  async function checkServers() {
    setCheckingServers(true);
    try {
      const data = await api<{models: {model:string;ready:boolean;detail:string}[]}>('/scholar/model-status');
      setServerResults(data.models);
    } catch (reason) {
      setServerResults([{model:'Local vLLM',ready:false,detail:reason instanceof Error ? reason.message : 'Unable to check inference servers'}]);
    } finally {setCheckingServers(false);}
  }
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => { endRef.current?.scrollIntoView({behavior:"smooth",block:"end"}); }, [turns.length]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function ask(override?: string) {
    const next = (override ?? question).trim();
    if (busy || next.length < 3 || !sources.length) return;
    setQuestion(""); setBusy(true); setError("");
    try {
      const result = await postJSON<ScholarAnswer>("/scholar/answer", {
        question: next,
        sources, top_k: topK, year_from: year ? Number(year) : null,
        model_id: chosenModel, generation: {...generation, model: ["llama3.2:3b", "qwen3.5:4b", "granite4.1:3b"].includes(chosenModel) ? chosenModel : generation.model},
        context_questions: turns.slice(-3).map(t => t.question),
      });
      const nextTurns = [...turns, { question: next, result }];
      setTurns(nextTurns);
      const timestamp = new Date().toISOString();
      onSave({
        id: session?.id ?? crypto.randomUUID(), title: session?.title ?? paperSessionTitle(next),
        created_at: session?.created_at ?? timestamp, updated_at: timestamp, turns: nextTurns,
      });
    } catch (reason) {
      setQuestion(next);
      setError(reason instanceof Error ? reason.message : "Could not generate a paper synthesis.");
    } finally { setBusy(false); }
  }

  return <section className="paper-chat-workspace">
    <div className="paper-chat-conversation" role="log" aria-label="Paper Explorer conversation">
      {!turns.length && <div className="welcome paper-chat-welcome">
        <h1>Yo, no time to waste, let's dive into the <em>research literature.</em></h1>
        <div className="prompts-grid paper-chat-prompts">{SUGGESTIONS.map(example=><button className="prompt-card" key={example} onClick={()=>void ask(example)}>
          <span className="prompt-icon"><ArrowRight size={17}/></span><strong>{example}</strong><span>Search papers and synthesize their findings</span>
        </button>)}</div>
      </div>}
      {turns.length>0 && <div className="paper-chat-results">
        <div className="paper-chat-results-head"><span className="eyebrow">EVIDENCE-GROUNDED PAPER CHAT</span>
          <button type="button" className="scholar-new" disabled={busy} onClick={onNewTopic}><RotateCcw size={15}/> New topic</button></div>
        {turns.map((turn,index)=><PaperAnswer key={index} turn={turn} turnIndex={index}/>)}
        <div ref={endRef}/>
      </div>}
      {error && <p className="scholar-error" role="alert">{error}</p>}
      {busy && <div className="scholar-working" role="status"><LoaderCircle className="spin" size={18}/><div><strong>Finding academic evidence and synthesizing papers…</strong><p>Answers use retrieved abstracts and must include valid citations.</p></div></div>}
    </div>
    <div className="composer-area paper-chat-composer shared-chat-composer">
      <form className={`composer ${turns.length === 0 ? "composer-new-chat" : ""}`} onSubmit={event=>{event.preventDefault();void ask();}}>
        <textarea rows={1} maxLength={500} aria-label="Ask research papers" placeholder="Ask a question about the research literature…" value={question}
          onChange={e=>setQuestion(e.target.value)} onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();void ask();}}}/>
        <div className="composer-footer composer-footer-simplified">
          <button type="button" className="chat-config-trigger" title="Configure paper search" aria-label="Configure Paper Explorer" aria-haspopup="dialog" onClick={()=>setSettingsOpen(true)}><Settings2 size={21}/></button>
          <button className="send-button" type="submit" aria-label="Search papers" disabled={busy||question.trim().length<3||!sources.length}>
            {busy?<LoaderCircle className="spin" size={17}/>:<Send size={17}/>}</button>
        </div>
      </form><p className="composer-note"><ShieldCheck size={12}/> Responses may contain mistakes. Verify key claims against the linked evidence.</p>
    </div>
    {settingsOpen && <div className="chat-config-overlay" onMouseDown={e=>{if(e.target===e.currentTarget)setSettingsOpen(false);}}>
      <section className="chat-config-dialog" role="dialog" aria-modal="true" aria-label="Paper Explorer settings">
        <header className="chat-config-header"><div><div className="eyebrow">PAPER EXPLORER</div><h2>Search and model settings</h2>
          <p>Choose where to find papers and which LLM synthesizes the abstracts.</p></div>
          <button className="chat-config-close" aria-label="Close Paper Explorer settings" onClick={()=>setSettingsOpen(false)}><X size={19}/></button>
        </header><div className="chat-config-body">
          <div className="chat-config-section"><h3>Language model</h3>
            <select id="paper-chat-model" aria-label="Language model" value={chosenModel} onChange={e=>setChosenModel(e.target.value)}>
              <optgroup label="Existing local Ollama models">
                <option value="qwen3.5:4b">Qwen 3.5 · 4B</option><option value="llama3.2:3b">Llama 3.2 · 3B</option><option value="granite4.1:3b">Granite 4.1 · 3B</option>
              </optgroup>
              <optgroup label="Optional Hugging Face models (local vLLM/SGLang)">
                <option value="rl-research/DR-Tulu-8B">DR-Tulu-8B</option>
                <option value="allenai/AstaBrief_8B">AstaBrief 8B</option>
              </optgroup>
            </select>
            <p className="agent-note">vLLM serves these optional models locally. AstaBrief is designed for cited synthesis; DR-Tulu requires its separate agent workflow for optimal tool use. The model is not downloaded by selecting it.</p>
            <div className="scholar-server-check">
              <button type="button" disabled={checkingServers} onClick={()=>void checkServers()}>{checkingServers ? "Checking…" : "Check local vLLM servers"}</button>
              {serverResults.map(server=><p key={server.model} role="status"><span className={server.ready?"scholar-server-ready":"scholar-server-unready"}>{server.ready?"● Ready":"○ Offline"}</span> · {server.model.split('/').at(-1)} · {server.detail}</p>)}
            </div>
          </div>
          <div className="chat-config-section"><h3>Academic search</h3><div className="paper-chat-provider-grid">
            {PROVIDERS.map(provider=><label key={provider.id}><input type="checkbox" checked={sources.includes(provider.id)} disabled={busy} onChange={e=>setSources(list=>e.target.checked?[...list,provider.id]:list.filter(id=>id!==provider.id))}/>{provider.label}</label>)}
          </div><label className="paper-chat-setting-label">Maximum papers <select value={topK} disabled={busy} onChange={e=>setTopK(Number(e.target.value))}>
            <option value={5}>5</option><option value={8}>8</option><option value={12}>12</option></select></label>
          <label className="paper-chat-setting-label">Published since <input className="paper-chat-year-input" type="number" min={1900} max={2100} placeholder="Any year" value={year} disabled={busy} onChange={e=>setYear(e.target.value)}/></label>
          <p className="agent-note">Only sources returning actual abstracts can contribute to the answer. Provider availability varies.</p></div>
        </div><footer className="chat-config-footer"><button onClick={()=>setSettingsOpen(false)}>Done</button></footer>
      </section></div>}
  </section>;
}
