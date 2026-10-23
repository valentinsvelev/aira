"use client";

import { useState, useEffect } from "react";
import { BrainCircuit, Check, CircleAlert, GitBranch, ListChecks, Plus, Save, ShieldCheck, Trash2 } from "lucide-react";
import { deleteAPI, patchJSON, postJSON } from "@/lib/api";
import { AgentFlowBuilder } from "@/components/agent-flow-builder";
import { ExecutableAgentFlow, starterExecutableWorkflow, type WorkflowDefinition } from "@/components/executable-agent-flow";
import { api } from "@/lib/api";
import { AGENT_TOOL_CATALOG, normalizeAgentTools, readAgentFlowLayout, writeAgentFlowLayout, type AgentFlowTool, type AgentFlowPosition } from "@/lib/agent-flow-config";
import type { AgentProfile, GenerationOptions } from "@/lib/types";

const TOOL_CHOICES = AGENT_TOOL_CATALOG;
export const DEFAULT_GENERATION: GenerationOptions = {
  model: "qwen3.5:4b", temperature: 0, top_p: 0.9, top_k: 40, min_p: 0,
  num_predict: -1, num_ctx: 0, repeat_penalty: 1.1, seed: null,
};
const fresh = () => ({ name: "", description: "", system_instructions: "", enabled_tools: TOOL_CHOICES.map(t => t.id), generation: {...DEFAULT_GENERATION} });
type Draft = ReturnType<typeof fresh>;

export function AgentWorkspace({ profiles, onChanged, selectedId, onSelect }: {
  profiles: AgentProfile[];
  onChanged: () => Promise<void>;
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState<Draft>(fresh());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [builderMode, setBuilderMode] = useState<"form" | "visual" | "workflow">("form");
  const [workflow, setWorkflow] = useState<WorkflowDefinition>(starterExecutableWorkflow());
  const [visibleFlowTools, setVisibleFlowTools] = useState<AgentFlowTool[]>(normalizeAgentTools(fresh().enabled_tools));
  const [flowPositions, setFlowPositions] = useState<Record<string, AgentFlowPosition>>({});
  const updateFlowToolPermissions = (tools: AgentFlowTool[]) => {
    const normalized = normalizeAgentTools(tools);
    setDraft(current => ({...current, enabled_tools: normalized}));
    setVisibleFlowTools(current => normalizeAgentTools([...current, ...normalized]));
  };
  const updateFlowPosition = (id: string, point: AgentFlowPosition) =>
    setFlowPositions(current => ({...current, [id]: point}));

  useEffect(() => {
    if (!editing) return;
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape" && !busy) setEditing(null); };
    const dialog = document.querySelector<HTMLElement>('[role="dialog"][aria-label="Agent profile editor"]');
    dialog?.querySelector<HTMLElement>('button:not([disabled])')?.focus();
    const trap = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const focusable = Array.from(dialog?.querySelectorAll<HTMLElement>('button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled])') || []);
      if (!focusable.length) return;
      const first = focusable[0], last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {event.preventDefault(); last.focus();}
      else if (!event.shiftKey && document.activeElement === last) {event.preventDefault(); first.focus();}
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("keydown", trap);
    return () => {document.removeEventListener("keydown", onKey);document.removeEventListener("keydown", trap);};
  }, [editing, busy]);

  function edit(profile?: AgentProfile, mode: "form" | "visual" | "workflow" = "form") {
    setEditing(profile?.id || "new");
    const permissions = normalizeAgentTools(profile?.enabled_tools || fresh().enabled_tools);
    const stored = profile ? readAgentFlowLayout(profile.id) : null;
    setDraft(profile ? { name: profile.name, description: profile.description, system_instructions: profile.system_instructions,
      enabled_tools: permissions, generation: {...DEFAULT_GENERATION, ...(profile.generation || {})} } : (mode === "visual" || mode === "workflow") ? {...fresh(), enabled_tools: normalizeAgentTools(["local_search"])} : fresh());
    setVisibleFlowTools(profile ? normalizeAgentTools([...(stored?.visibleTools || []), ...permissions]) : (mode === "visual" || mode === "workflow") ? ["local_search"] : permissions);
    setFlowPositions(stored?.positions || {});
    setBuilderMode(mode);
    setWorkflow(starterExecutableWorkflow());
    if (profile) {
      void api<{workflow:WorkflowDefinition | null}>(`/agents/${profile.id}/workflow`).then(response=>{
        if (response.workflow) setWorkflow(response.workflow);
      }).catch(()=>{});
    }
    setError(""); setNotice("");
  }

  function setGen<K extends keyof GenerationOptions>(key: K, value: GenerationOptions[K]) {
    setDraft(d => ({...d, generation: {...d.generation, [key]: value}}));
  }

  async function save() {
    if (!draft.name.trim() || busy) return;
    setBusy(true); setError("");
    try {
      if (editing === "new") {
        const created = await postJSON<AgentProfile>("/agents", draft);
        writeAgentFlowLayout(created.id, { positions: flowPositions, visibleTools: visibleFlowTools });
        if (builderMode === "workflow") await api(`/agents/${created.id}/workflow`, {method:"PUT",body:JSON.stringify(workflow)});
        await onChanged();
        onSelect(created.id);
      } else if (editing) {
        await patchJSON<AgentProfile>(`/agents/${editing}`, draft);
        writeAgentFlowLayout(editing, { positions: flowPositions, visibleTools: visibleFlowTools });
        if (builderMode === "workflow") await api(`/agents/${editing}/workflow`, {method:"PUT",body:JSON.stringify(workflow)});
        await onChanged();
      }
      setEditing(null); setNotice("Agent profile saved.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save agent");
    } finally { setBusy(false); }
  }

  async function remove(id: string) {
    if (!confirm("Delete this agent profile? Existing chat messages will remain.")) return;
    setBusy(true); setError("");
    try {
      await deleteAPI(`/agents/${id}`);
      await onChanged();
      if (selectedId === id) onSelect("default");
      setEditing(null); setNotice("Agent profile deleted.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete agent");
    } finally { setBusy(false); }
  }

  return <div className="agent-workspace">
    <div className="agent-workspace-title"><div><div className="eyebrow">CUSTOM RESEARCH WORKFLOWS</div><h1>Research agents</h1><p>Create instruction profiles and specify which tools each agent can use. Profiles are saved locally in PostgreSQL.</p></div><div className="agent-workspace-create"><button type="button" className="agent-visual-launch" onClick={() => edit(undefined, "workflow")}><GitBranch size={16}/> New workflow</button><button type="button" className="agent-visual-launch" onClick={() => edit(undefined, "visual")}><GitBranch size={16}/> New visual agent</button><button className="agent-primary" onClick={() => edit()}><Plus size={16}/> New agent</button></div></div>
    {error && <p className="agent-alert" role="alert"><CircleAlert size={15}/>{error}</p>}
    {notice && <p className="agent-notice" role="status"><Check size={15}/>{notice}</p>}
    <div className="agent-card-list">{profiles.map(profile => <article className="agent-profile-card" key={profile.id}>
      <div className="agent-card-icon"><BrainCircuit size={21}/></div>
      <div className="agent-card-content"><h2>{profile.name} {profile.id === "default" && <span>Built-in</span>}</h2><p>{profile.description || "General research assistant"}</p></div>
      <div className="agent-card-actions"><button onClick={() => onSelect(profile.id)} className={selectedId === profile.id ? "active" : ""}>{selectedId === profile.id ? "Selected" : "Use agent"}</button><button onClick={() => edit(profile)} aria-label={`Edit ${profile.name}`}>Edit</button><button onClick={() => edit(profile, "visual")} aria-label={`Visual builder for ${profile.name}`}><GitBranch size={14}/> Visual builder</button><button onClick={() => edit(profile, "workflow")} aria-label={`Executable workflow for ${profile.name}`}><GitBranch size={14}/> Workflow</button>{profile.id !== "default" && <><button className="agent-danger" aria-label={`Delete ${profile.name}`} onClick={() => void remove(profile.id)}><Trash2 size={15}/></button></>}</div>
    </article>)}</div>
    {editing && <div className="agent-modal-overlay" onPointerDown={event => { if (event.target === event.currentTarget && !busy) setEditing(null); }}><div className={`agent-editor ${builderMode !== "form" ? "agent-editor-visual" : ""}`} role="dialog" aria-modal="true" aria-label="Agent profile editor"><div className="agent-editor-heading"><h2>{editing === "new" ? "Create agent" : "Edit agent"}</h2><button onClick={() => setEditing(null)} aria-label="Close agent editor">×</button></div>
      {error && <p role="alert" className="agent-alert">{error}</p>}
      <div className="agent-editor-mode" role="group" aria-label="Agent editor mode">
        <button type="button" aria-pressed={builderMode === "form"} className={builderMode === "form" ? "active" : ""} onClick={() => setBuilderMode("form")}><ListChecks size={16}/> Form editor</button>
        <button type="button" aria-pressed={builderMode === "visual"} className={builderMode === "visual" ? "active" : ""} onClick={() => setBuilderMode("visual")}><GitBranch size={16}/> Visual builder</button><button type="button" aria-pressed={builderMode === "workflow"} className={builderMode === "workflow" ? "active" : ""} onClick={() => setBuilderMode("workflow")}><GitBranch size={16}/> Executable workflow</button>
      </div>
      <label>Agent name<input maxLength={100} value={draft.name} onChange={e => setDraft(d => ({ ...d, name: e.target.value }))} placeholder="e.g. Critical Literature Reviewer"/></label>
      <label>Description<input maxLength={500} value={draft.description} onChange={e => setDraft(d => ({ ...d, description: e.target.value }))} placeholder="What this agent specializes in"/></label>
      {builderMode === "visual" && <section aria-label="Visual tool permissions" className="agent-flow-section">
        <div className="agent-flow-section-heading"><h3>Tool permissions canvas</h3><span>Drag · connect · configure</span></div>
        <AgentFlowBuilder enabledTools={normalizeAgentTools(draft.enabled_tools)} visibleTools={visibleFlowTools} positions={flowPositions}
          onEnabledToolsChange={updateFlowToolPermissions} onVisibleToolsChange={setVisibleFlowTools} onPositionChange={updateFlowPosition}/>
        <p className="agent-note">Only enabled tool connections and the agent settings are saved in PostgreSQL. Node positions are saved in this browser. This is a permission editor, not an executable step-by-step workflow.</p>
      </section>}
      {builderMode === "workflow" && <section className="agent-flow-section" aria-label="Executable agent workflow">
        <div className="agent-flow-section-heading"><h3>Executable research workflow</h3><span>Connected graph · validated on save</span></div>
        <ExecutableAgentFlow workflow={workflow} onChange={setWorkflow} permittedTools={draft.enabled_tools} onAllowTool={tool=>updateFlowToolPermissions([...draft.enabled_tools,tool])} profileId={editing}/>
      </section>}
      <label>System instructions<textarea rows={builderMode === "visual" ? 4 : 8} maxLength={6000} value={draft.system_instructions} onChange={e => setDraft(d => ({ ...d, system_instructions: e.target.value }))} placeholder="You specialize in comparing scientific evidence. Explicitly distinguish study findings from speculation…"/></label>
      <p className="agent-note">{editing === "default" ? "Changes to the built-in Research assistant are stored as a local override. Existing chats remain intact." : "These instructions specialize the assistant but cannot override evidence rules, per-request search toggles, or mandatory note verification."}</p>
      <h3>Generation settings for this agent</h3>
        <p className="agent-note">These settings are saved to this agent. For the default assistant they become the starting values for new chats, with optional per-chat overrides.</p>
        <div className="param-grid agent-generation-grid">
          <label className="param-field"><span>Model</span><select value={draft.generation.model || "qwen3.5:4b"} onChange={e=>setGen("model", e.target.value as GenerationOptions["model"])}>
            <option value="qwen3.5:4b">Qwen 3.5 · 4B</option><option value="llama3.2:3b">Llama 3.2 · 3B</option><option value="granite4.1:3b">Granite 4.1 · 3B</option></select></label>
          {([ ["temperature", "Temperature", 0, 1, 0.05], ["top_p", "Top-p", 0.01, 1, 0.05], ["top_k", "Top-k sampling", 1, 200, 1], ["min_p", "Min-p sampling", 0, 1, 0.01], ["repeat_penalty", "Repeat penalty", 0.8, 2, 0.05] ] as const).map(([key,label,min,max,step]) =>
            <label className="param-field" key={key}><span>{label}</span><input type="number" min={min} max={max} step={step} value={draft.generation[key]} onChange={e=>setGen(key, Number(e.target.value))}/></label>)}
          <label className="param-field"><span>Context window</span><select aria-label="Agent context window mode" value={draft.generation.num_ctx === 0 ? "auto" : "manual"} onChange={e=>setGen("num_ctx", e.target.value === "auto" ? 0 : 4096)}><option value="auto">Model maximum (auto)</option><option value="manual">Custom limit</option></select></label>
          {draft.generation.num_ctx !== 0 && <label className="param-field"><span>Context tokens</span><input type="number" min="2048" max="1048576" step="1024" value={draft.generation.num_ctx} onChange={e=>setGen("num_ctx", Number(e.target.value))}/></label>}
          <label className="param-field"><span>Max output tokens</span><select value={draft.generation.num_predict === -1 ? "unlimited" : "limited"} onChange={e=>setGen("num_predict", e.target.value === "unlimited" ? -1 : 1024)}><option value="unlimited">Model decides</option><option value="limited">Set a limit</option></select></label>
          {draft.generation.num_predict !== -1 && <label className="param-field"><span>Token limit</span><input type="number" min="32" max="32768" value={draft.generation.num_predict} onChange={e=>setGen("num_predict", Number(e.target.value))}/></label>}
          <label className="param-field"><span>Seed (optional)</span><input type="number" min="0" placeholder="Random" value={draft.generation.seed ?? ""} onChange={e=>setGen("seed", e.target.value === "" ? null : Number(e.target.value))}/></label>
        </div>
      {builderMode === "form" && <><h3>Available tools</h3><div className="agent-tools-grid">{TOOL_CHOICES.map(tool => <label className="agent-tool-choice" key={tool.id}><input type="checkbox" checked={draft.enabled_tools.includes(tool.id)} disabled={tool.id === "local_search"} onChange={e => {
          const tools = e.target.checked ? [...draft.enabled_tools, tool.id] : draft.enabled_tools.filter(t => t !== tool.id);
          updateFlowToolPermissions(normalizeAgentTools(tools));
        }}/><span><strong>{tool.label}</strong><small>{tool.help}</small></span></label>)}</div></>}
      <button className="agent-primary" disabled={!draft.name.trim() || busy} onClick={() => void save()}><Save size={15}/>{busy ? "Saving…" : "Save agent"}</button>
    </div></div>}
  </div>;
}
