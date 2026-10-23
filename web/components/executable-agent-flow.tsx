"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ReactFlow, Background, BackgroundVariant, Controls, Handle, MiniMap, Position,
  applyNodeChanges, type Connection, type Edge, type Node, type NodeProps, type OnNodesChange,
} from "@xyflow/react";
import { ArrowRight, BookOpen, BrainCircuit, Check, ChevronLeft, ChevronRight, Clipboard, Globe, MessageSquare, Pause, Play, RefreshCw, Search, ShieldCheck, Trash2, GitBranch } from "lucide-react";
import { api } from "@/lib/api";
import { AdvancedAgentFlow } from "@/components/advanced-agent-flow";
import type { AgentTool, WorkflowStep } from "@/lib/types";

export type WorkflowNodeId = "question" | "local_search" | "academic_search" | "web_search" | "evidence_gate" | "synthesize" | "answer";
export type WorkflowDefinition = {
  enabled: boolean;
  parallel?: boolean;
  schema_version?: 1 | 2;
  max_steps?: number;
  nodes: { id: WorkflowNodeId; type: WorkflowNodeId; x: number; y: number; min_sources?: number; max_iterations?: number; query_template?: string }[];
  edges: { source: WorkflowNodeId; target: WorkflowNodeId; condition?: "enough" | "insufficient" | "repeat" | "done" | null }[];
};
type PhoenixSpan = { node:string; span_id:string; latency_ms:number; status:"completed"|"failed" };
type WorkflowRun = { id:number; conversation_id:string; created_at:string; trace_id:string|null;
  steps:WorkflowStep[]; workflow:WorkflowDefinition|null };

const titles: Record<WorkflowNodeId,string> = {
  question: "Research question", local_search: "Local PDF search", academic_search: "Academic search",
  web_search: "Web search", evidence_gate: "Evidence check", synthesize: "LLM synthesis", answer: "Cited answer",
};
const required: WorkflowNodeId[] = ["question", "synthesize", "answer"];
const sources: WorkflowNodeId[] = ["local_search", "academic_search", "web_search"];
const permissible = new Set(["question:synthesize", "synthesize:answer",
  ...sources.flatMap(tool => [`question:${tool}`, `${tool}:synthesize`, `${tool}:evidence_gate`, `evidence_gate:${tool}`]),
  "evidence_gate:synthesize"]);

export function starterExecutableWorkflow(): WorkflowDefinition {
  return { enabled: true, parallel:true,
    nodes: [
      { id:"question", type:"question", x:20, y:160 },
      { id:"local_search", type:"local_search", x:320, y:160 },
      { id:"synthesize", type:"synthesize", x:630, y:160 },
      { id:"answer", type:"answer", x:940, y:160 },
    ],
    edges: [{source:"question",target:"local_search"},{source:"local_search",target:"synthesize"},{source:"synthesize",target:"answer"}],
  };
}
function buildConnections(primary: WorkflowNodeId[], fallback: WorkflowNodeId[], gate:boolean): WorkflowDefinition["edges"] {
  return [
    ...primary.flatMap(tool => [
      {source:"question" as WorkflowNodeId,target:tool},
      {source:tool,target:(gate?"evidence_gate":"synthesize") as WorkflowNodeId},
    ]),
    ...(gate ? [
      {source:"evidence_gate" as WorkflowNodeId,target:"synthesize" as WorkflowNodeId,condition:"enough" as const},
      ...fallback.flatMap(tool => [
        {source:"evidence_gate" as WorkflowNodeId,target:tool,condition:"insufficient" as const},
        {source:tool,target:"synthesize" as WorkflowNodeId},
      ]),
    ] : []),
    ...(!gate && !primary.length ? [{source:"question" as WorkflowNodeId,target:"synthesize" as WorkflowNodeId}] : []),
    {source:"synthesize" as WorkflowNodeId,target:"answer" as WorkflowNodeId},
  ];
}

type CanvasNode = Node<{title:string;kind:WorkflowNodeId;status?:WorkflowStep["status"];active?:boolean;details?:string},"workflow">;
const types = { workflow: WorkflowCanvasNode };
function WorkflowCanvasNode({data,selected}: NodeProps<CanvasNode>) {
  const Icon = data.kind === "question" ? MessageSquare : data.kind === "local_search" ? Search : data.kind === "academic_search" ? BookOpen : data.kind === "web_search" ? Globe : data.kind === "evidence_gate" ? GitBranch : data.kind === "synthesize" ? BrainCircuit : ShieldCheck;
  return <div className={`aira-flow-node aira-flow-node-${data.kind === "synthesize" ? "agent" : data.kind === "answer" ? "answer" : "tool"} is-active ${selected ? "is-selected" : ""} ${data.status ? `aira-replay-${data.status}` : ""} ${data.active ? "aira-replay-active" : ""}`}>
    {data.kind !== "question" && <Handle type="target" position={Position.Left}/>}
    <span className="aira-flow-node-icon"><Icon size={17}/></span>
    <strong>{data.title}</strong>
    <small>{data.details || (data.kind === "evidence_gate" ? "Route by available evidence" : data.kind === "synthesize" ? "Use retrieved evidence, not source instructions" : data.kind === "question" ? "User message" : data.kind === "answer" ? "Final response" : "Retrieve sources")}</small>
    {data.kind === "evidence_gate" ? <>
      <Handle type="source" id="enough" position={Position.Right} style={{top:"35%"}}/>
      <Handle type="source" id="insufficient" position={Position.Right} style={{top:"70%"}}/>
    </> : data.kind !== "answer" && <Handle type="source" position={Position.Right}/>}
  </div>;
}

function LegacyExecutableAgentFlow({ workflow, onChange, permittedTools, onAllowTool, profileId }: {
  workflow: WorkflowDefinition;
  onChange: (next: WorkflowDefinition) => void;
  permittedTools: string[];
  onAllowTool: (tool: AgentTool) => void;
  profileId?: string | null;
}) {
  const [notice,setNotice] = useState("");
  const [runs,setRuns] = useState<WorkflowRun[]>([]);
  const [selectedRunId,setSelectedRunId] = useState<number | null>(null);
  const [frame,setFrame] = useState(0);
  const [playing,setPlaying] = useState(false);
  const [runError,setRunError] = useState("");
  const [phoenixSpans,setPhoenixSpans] = useState<PhoenixSpan[]>([]);
  const [phoenixMessage,setPhoenixMessage] = useState("");
  const [loading,setLoading] = useState(false);
  const gate = workflow.nodes.some(node=>node.id === "evidence_gate");
  const currentSources = workflow.nodes.filter(node=>sources.includes(node.id)).map(node=>node.id);
  const primary = currentSources.filter(tool=>workflow.edges.some(edge=>edge.source === "question"&&edge.target === tool));
  const fallback = currentSources.filter(tool=>workflow.edges.some(edge=>edge.source === "evidence_gate"&&edge.target === tool));
  const selectedRun = runs.find(run=>run.id===selectedRunId);
  const displayWorkflow = selectedRun?.workflow ?? workflow;
  const stages = selectedRun ? [...new Set(selectedRun.steps.map(step=>step.stage ?? selectedRun.steps.indexOf(step)))].sort((a,b)=>a-b) : [];
  const selectedStage = stages[Math.min(frame,stages.length-1)] ?? -1;
  const activeSteps = selectedRun ? selectedRun.steps.filter(step=>(step.stage??selectedRun.steps.indexOf(step))<=selectedStage).map(step=>{
    const phoenix = phoenixSpans.find(span=>span.node===step.node && (!step.span_id || step.span_id===span.span_id)) ;
    return phoenix ? {...step,latency_ms:phoenix.latency_ms,span_id:phoenix.span_id,status:phoenix.status} : step;
  }) : [];
  const latestSteps = new Map(activeSteps.map(step=>[step.node,step]));
  const atStage = new Set(activeSteps.filter(step=>(step.stage??selectedRun?.steps.indexOf(step))===selectedStage).map(step=>step.node));
  async function loadPhoenix() {
    if (!selectedRun || !profileId || profileId === "new") return;
    setPhoenixMessage("Loading Phoenix workflow spans…");
    try {
      const response=await api<{available:boolean;spans:PhoenixSpan[];message?:string|null}>(`/agents/${profileId}/workflow/runs/${selectedRun.id}/phoenix`);
      setPhoenixSpans(response.spans || []);
      setPhoenixMessage(response.message || (response.available ? `Confirmed ${response.spans.length} workflow spans in Phoenix.` : "Phoenix is unavailable."));
    } catch (err) {
      setPhoenixMessage(err instanceof Error?err.message:"Phoenix is unavailable; recorded execution events remain visible.");
    }
  }
  const onReload = useCallback(async () => {
    if (!profileId || profileId === "new") return;
    setLoading(true); setRunError("");
    try {
      const data = await api<{runs:WorkflowRun[]}>(`/agents/${profileId}/workflow/runs`);
      setRuns(data.runs);
    } catch (err) { setRunError(err instanceof Error?err.message:"Could not load workflow runs"); }
    finally {setLoading(false);}
  },[profileId]);
  useEffect(()=>{void onReload();},[onReload]);
  useEffect(()=>{
    if (!playing || !selectedRun) return;
    if (frame >= stages.length-1) return;
    const timer = window.setTimeout(()=>setFrame(current=>current+1),1100);
    return ()=>window.clearTimeout(timer);
  },[playing,frame,selectedRun,stages.length]);
  const nodes = useMemo<CanvasNode[]>(() => displayWorkflow.nodes.map(node => {
    const status = latestSteps.get(node.id);
    return { id:node.id, type:"workflow", position:{x:node.x,y:node.y},
      deletable: !required.includes(node.id) && !selectedRun,
      data:{title:titles[node.id],kind:node.id,status:status?.status,active:atStage.has(node.id),
        details:status ? `${status.status} · ${status.sources} sources · ${status.latency_ms} ms${status.branch ? ` · ${status.branch}` : ""}` : undefined},
    };
  }),[displayWorkflow,selectedRun,activeSteps,selectedStage]);
  const edges = useMemo<Edge[]>(() => displayWorkflow.edges.map(edge=>({
    id:`${edge.source}:${edge.target}`, source:edge.source,target:edge.target,
    type:"smoothstep", animated: !selectedRun && workflow.enabled,
    sourceHandle:edge.source === "evidence_gate" ? edge.condition || undefined : undefined,
    label:edge.condition === "enough" ? "enough" : edge.condition === "insufficient" ? "too little" : undefined,
    className: selectedRun && edge.source==="evidence_gate" && latestSteps.get("evidence_gate")?.branch !== edge.condition ? "aira-replay-unselected-edge" : "",
  })),[displayWorkflow,workflow.enabled,selectedRun,selectedStage]);
  const onNodesChange: OnNodesChange<CanvasNode> = useCallback(changes => {
    if (selectedRun) return;
    const moved = applyNodeChanges(changes.filter(item=>item.type!=="remove"),nodes);
    let updated = workflow.nodes.map(node => {
      const next = moved.find(item=>item.id===node.id);
      return next ? {...node,x:next.position.x,y:next.position.y} : node;
    });
    const deleted = changes.filter(item=>item.type==="remove").map(item=>item.id);
    if (deleted.length) updated = updated.filter(node=>!deleted.includes(node.id) || required.includes(node.id));
    onChange({...workflow,nodes:updated,edges:workflow.edges.filter(edge=>updated.some(n=>n.id===edge.source)&&updated.some(n=>n.id===edge.target))});
  },[workflow,nodes,onChange,selectedRun]);
  const connect = useCallback((edge:Connection) => {
    if (selectedRun) return;
    const key = `${edge.source}:${edge.target}`;
    if (!permissible.has(key) || (!gate && key.includes("evidence_gate"))) { setNotice("Connect first-pass search → evidence check → optional fallback → synthesis → answer."); return; }
    if (workflow.edges.some(item=>`${item.source}:${item.target}`===key)) return;
    const condition = edge.source === "evidence_gate" ? edge.sourceHandle === "enough" ? "enough" : "insufficient" : undefined;
    onChange({...workflow,edges:[...workflow.edges,{source:edge.source as WorkflowNodeId,target:edge.target as WorkflowNodeId,condition}]});
    setNotice("");
  },[workflow,onChange,gate,selectedRun]);
  function add(tool: WorkflowNodeId, asFallback=false) {
    if (workflow.nodes.some(node=>node.id===tool)) return;
    if (!permittedTools.includes(tool)) onAllowTool(tool as AgentTool);
    const number = workflow.nodes.filter(node=>sources.includes(node.id)).length;
    const nextPrimary = asFallback ? primary : [...primary,tool];
    const nextFallback = asFallback ? [...fallback,tool] : fallback;
    onChange({...workflow,nodes:[...workflow.nodes,{id:tool,type:tool,x:gate?320:315,y:50+number*155}],
      edges:buildConnections(nextPrimary,nextFallback,gate)});
  }
  function remove(tool: WorkflowNodeId) {
    onChange({...workflow,nodes:workflow.nodes.filter(node=>node.id!==tool),
      edges:buildConnections(primary.filter(item=>item!==tool),fallback.filter(item=>item!==tool),gate)});
  }
  function changeRole(tool:WorkflowNodeId, toFallback:boolean) {
    if (toFallback && primary.length<=1 || !toFallback && fallback.length<=1) {
      setNotice("Keep at least one first-pass and one fallback search when the evidence check is enabled.");return;
    }
    const nextPrimary = toFallback ? primary.filter(item=>item!==tool) : [...primary,tool];
    const nextFallback = toFallback ? [...fallback,tool] : fallback.filter(item=>item!==tool);
    onChange({...workflow,edges:buildConnections(nextPrimary,nextFallback,true)});
    setNotice("");
  }
  function toggleGate() {
    if (gate) {
      const all = [...primary,...fallback];
      onChange({...workflow,nodes:workflow.nodes.filter(node=>node.id!=="evidence_gate"),edges:buildConnections(all,[],false)});
      return;
    }
    const pool = sources.filter(tool=>!currentSources.includes(tool));
    const fallbackTool = pool.includes("academic_search") ? "academic_search" : pool[0] || primary.at(-1);
    if (!fallbackTool) {setNotice("Add a search tool before enabling the evidence check.");return;}
    const nextPrimary = primary.filter(tool=>tool!==fallbackTool);
    if (!nextPrimary.length) {setNotice("Keep one first-pass search to enable an evidence check.");return;}
    if (!currentSources.includes(fallbackTool) && !permittedTools.includes(fallbackTool)) onAllowTool(fallbackTool as AgentTool);
    onChange({...workflow,nodes:[...workflow.nodes.map(node=>node.id==="synthesize"?{...node,x:930,y:160}:node.id==="answer"?{...node,x:1220,y:160}:node),
      {id:"evidence_gate",type:"evidence_gate",x:610,y:160,min_sources:1},
      ...(!currentSources.includes(fallbackTool)?[{id:fallbackTool,type:fallbackTool,x:820,y:330}]:[])],
      edges:buildConnections(nextPrimary,[fallbackTool],true)});
  }
  const missing = gate ? (!primary.length || !fallback.length ? ["initial and fallback searches"] : []) : workflow.nodes.filter(node=>sources.includes(node.id) &&
    (!workflow.edges.some(edge=>edge.source==="question"&&edge.target===node.id)||
     !workflow.edges.some(edge=>edge.source===node.id&&edge.target==="synthesize"))).map(node=>titles[node.id]);
  return <div className="aira-executable-builder">
    <div className="aira-executable-toolbar">
      <label><input type="checkbox" checked={workflow.enabled} onChange={event=>onChange({...workflow,enabled:event.target.checked})}/>
        Execute this graph in Agentic RAG</label>
      <label><input type="checkbox" checked={workflow.parallel ?? true} onChange={event=>onChange({...workflow,parallel:event.target.checked})}/>
        Parallel independent searches</label>
      <span>External searches still require permission in Research Chat.</span>
    </div>
    <div className="aira-executable-palette">
      <button type="button" onClick={toggleGate} disabled={Boolean(selectedRun)}><GitBranch size={14}/>{gate?"Remove evidence check":"Add conditional evidence check"}</button>
      {sources.map(tool => <button key={tool} type="button" disabled={Boolean(selectedRun)||currentSources.includes(tool)} onClick={()=>add(tool)}>
        <ArrowRight size={14}/> Add {titles[tool]}
      </button>)}
    </div>
    {gate && !selectedRun && <div className="aira-evidence-settings">
      <label>Minimum sources to skip fallback <input aria-label="Minimum evidence sources" type="number" min={1} max={18}
        value={workflow.nodes.find(node=>node.id==="evidence_gate")?.min_sources??1}
        onChange={event=>onChange({...workflow,nodes:workflow.nodes.map(node=>node.id==="evidence_gate"?{...node,min_sources:Number(event.target.value)}:node)})}/></label>
      {currentSources.map(tool=><label key={tool}>{titles[tool]} <select aria-label={`Stage for ${titles[tool]}`}
        value={fallback.includes(tool)?"fallback":"initial"}
        onChange={event=>changeRole(tool,event.target.value==="fallback")}>
          <option value="initial">First-pass search</option><option value="fallback">If insufficient evidence</option>
        </select></label>)}
    </div>}
    <div className="aira-flow-canvas aira-executable-canvas" aria-label="Executable workflow graph">
      <ReactFlow<CanvasNode> nodes={nodes} edges={edges} nodeTypes={types}
        onNodesChange={onNodesChange} onConnect={connect}
        onEdgesDelete={removed=>!selectedRun && onChange({...workflow,
          edges:workflow.edges.filter(edge=>!removed.some(item=>item.source===edge.source&&item.target===edge.target))})}
        isValidConnection={edge=>permissible.has(`${edge.source}:${edge.target}`)}
        nodesDraggable={!selectedRun} nodesConnectable={!selectedRun} elementsSelectable
        fitView minZoom={0.3} maxZoom={1.7} fitViewOptions={{padding:0.15,maxZoom:1.05}}>
        <Background variant={BackgroundVariant.Dots} color="#cbd3e5"/><Controls showInteractive={false}/><MiniMap/>
      </ReactFlow>
    </div>
    {!selectedRun && <div className="aira-executable-toolbar">
      {sources.filter(tool=>workflow.nodes.some(node=>node.id===tool)).map(tool=><button key={tool} type="button" onClick={()=>remove(tool)}><Trash2 size={13}/> Remove {titles[tool]}</button>)}
    </div>}
    <section className="aira-workflow-replay" aria-label="Recorded workflow playback">
      <div className="aira-replay-title"><strong>Run playback</strong><button type="button" disabled={loading||!profileId||profileId==="new"} onClick={()=>void onReload()}><RefreshCw size={14}/> Refresh runs</button></div>
      <p>Replay recorded node stages, timings and Phoenix span IDs from saved Research Chat runs. No automatic Phoenix polling.</p>
      <select aria-label="Select recorded workflow run" value={selectedRunId??""} onChange={event=>{
        setSelectedRunId(event.target.value?Number(event.target.value):null);setFrame(0);setPlaying(false);setPhoenixSpans([]);setPhoenixMessage("");
      }}><option value="">Edit current workflow (no playback)</option>
        {runs.map(run=><option key={run.id} value={run.id}>{new Date(run.created_at).toLocaleString()} · Run {run.id}</option>)}
      </select>
      {runError && <p role="alert">{runError}</p>}
      {selectedRun && <>
        {!selectedRun.workflow && <p>Older run: graph snapshot unavailable; the current graph is displayed.</p>}
        <div className="aira-replay-phoenix"><button type="button" onClick={()=>void loadPhoenix()} disabled={!selectedRun.trace_id}><RefreshCw size={14}/> Load Phoenix spans</button>
          {phoenixMessage && <small role="status">{phoenixMessage}</small>}
        </div>
        <div className="aira-replay-controls">
          <button type="button" aria-label="Previous stage" disabled={frame<=0} onClick={()=>{setPlaying(false);setFrame(v=>Math.max(0,v-1));}}><ChevronLeft size={17}/></button>
          <button type="button" onClick={()=>{if(frame>=stages.length-1)setFrame(0);setPlaying(value=>!value);}}>{playing?<Pause size={15}/>:<Play size={15}/>} {playing?"Pause":"Play"}</button>
          <button type="button" aria-label="Next stage" disabled={frame>=stages.length-1} onClick={()=>{setPlaying(false);setFrame(v=>Math.min(stages.length-1,v+1));}}><ChevronRight size={17}/></button>
          <input aria-label="Playback stage" type="range" min="0" max={Math.max(0,stages.length-1)} value={frame} onChange={event=>{setPlaying(false);setFrame(Number(event.target.value));}}/>
          <small>Stage {frame+1} / {stages.length}</small>
        </div>
        <div className="aira-replay-step-list">{activeSteps.filter(step=>(step.stage??selectedRun.steps.findIndex(item=>item.node===step.node))===selectedStage).map(step=><div key={step.node}>
          <strong>{titles[step.node as WorkflowNodeId]||step.node}</strong> · {step.status} · {step.latency_ms} ms · {step.sources} sources
          {step.branch?` · ${step.branch==="enough"?"Evidence sufficient":"Fallback required"}`:""}
          {step.reason?` · ${step.reason}`:""}
          {step.span_id && <button type="button" title="Copy Phoenix span ID" onClick={()=>void navigator.clipboard.writeText(step.span_id!)}><Clipboard size={12}/> Span ID</button>}
        </div>)}</div>
        {selectedRun.trace_id && <button type="button" className="aira-replay-trace" onClick={()=>void navigator.clipboard.writeText(selectedRun.trace_id!)}><Check size={13}/> Copy Phoenix trace ID</button>}
      </>}
    </section>
    {(notice || missing.length>0) && <p className="agent-note" role="status">{notice || `Fix graph before saving: ${missing.join(", ")}`}</p>}
    <p className="agent-note">Searches can execute concurrently. The evidence check routes to fallback search when the configured source threshold is not met. Playback uses saved run metadata correlated with instrumented Phoenix spans. The original LLM-directed agent remains available when this graph is disabled.</p>
  </div>;
}


/** Advanced graphs use a separate editor; legacy profiles retain their original canvas. */
export function ExecutableAgentFlow(props: {
  workflow: WorkflowDefinition;
  onChange: (next: WorkflowDefinition) => void;
  permittedTools: string[];
  onAllowTool: (tool: AgentTool) => void;
  profileId?: string | null;
}) {
  if (props.workflow.schema_version === 2) return <AdvancedAgentFlow {...props}/>;
  return <div className="aira-executable-builder">
    <div className="aira-advanced-intro">
      <div><strong>Advanced executable workflows</strong><small>Multiple conditions, repeated tools, nested loops, and configurable safety limits.</small></div>
      <button type="button" onClick={() => props.onChange({...props.workflow, schema_version: 2, max_steps: 96})}>
        Switch to advanced editor <ArrowRight size={14}/>
      </button>
    </div>
    <LegacyExecutableAgentFlow {...props}/>
  </div>;
}
