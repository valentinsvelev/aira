"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ReactFlow, Background, BackgroundVariant, Controls, MiniMap, Handle, Position,
  applyNodeChanges, type Node, type Edge, type Connection, type NodeProps, type OnNodesChange,
} from "@xyflow/react";
import {
  ArrowRight, BookOpen, BrainCircuit, GitBranch, Globe, MessageSquare,
  Search, ShieldCheck, Repeat2, Plus, Trash2, RefreshCw, Play, Pause,
  ChevronLeft, ChevronRight, Clipboard,
} from "lucide-react";
import { api } from "@/lib/api";
import type { AgentTool, WorkflowStep } from "@/lib/types";
import type { WorkflowDefinition } from "@/components/executable-agent-flow";

type Kind = "question" | "local_search" | "academic_search" | "web_search" | "evidence_gate" | "loop" | "synthesize" | "answer";
type Choice = "enough" | "insufficient" | "repeat" | "done";
type Item = { id: string; type: Kind; x: number; y: number; min_sources?: number;
  max_iterations?: number; query_template?: string };
type Wire = {source:string;target:string;condition?:Choice|null};
type Graph = {enabled:boolean;parallel?:boolean;schema_version:2;max_steps?:number;nodes:Item[];edges:Wire[]};
type Step = WorkflowStep & {iteration?:number;branch?:string;span_id?:string};
type Run = {id:number;conversation_id:string;created_at:string;trace_id:string|null;steps:Step[];workflow:Graph|null};
type PhoenixSpan = {node:string;span_id:string;latency_ms:number;status:"completed"|"failed"};
const labels:Record<Kind,string> = {
  question:"Research question",local_search:"Local search",academic_search:"Academic search",
  web_search:"Web search",evidence_gate:"Evidence condition",loop:"Repeat until",
  synthesize:"LLM synthesis",answer:"Cited answer",
};
const tools:Kind[] = ["local_search","academic_search","web_search"];
const palette:Kind[] = [...tools,"evidence_gate","loop"];
const locked:Kind[] = ["question","synthesize","answer"];
const name = (kind:Kind) => labels[kind] ?? kind;

type FlowNode = Node<{title:string;kind:Kind;subtitle:string;status?:string;active?:boolean},"aira-advanced">;
function AdvancedNode({data,selected}:NodeProps<FlowNode>) {
  const Icon = data.kind === "question" ? MessageSquare : data.kind === "local_search" ? Search
    : data.kind === "academic_search" ? BookOpen : data.kind === "web_search" ? Globe
    : data.kind === "evidence_gate" ? GitBranch : data.kind === "loop" ? Repeat2
    : data.kind === "synthesize" ? BrainCircuit : ShieldCheck;
  return <div className={`aira-flow-node aira-advanced-node ${selected?"is-selected":""} ${data.status?`aira-replay-${data.status}`:""} ${data.active?"aira-replay-active":""}`}>
    {data.kind!=="question" && <Handle type="target" position={Position.Left}/>}
    <span className="aira-flow-node-icon"><Icon size={18}/></span>
    <strong>{data.title}</strong>
    <small>{data.subtitle}</small>
    {data.kind==="evidence_gate" || data.kind==="loop" ? <>
      <Handle type="source" id={data.kind==="loop"?"repeat":"enough"} position={Position.Right} style={{top:"36%"}}/>
      <Handle type="source" id={data.kind==="loop"?"done":"insufficient"} position={Position.Right} style={{top:"78%"}}/>
    </> : data.kind!=="answer" && <Handle type="source" position={Position.Right}/>}
  </div>;
}
const nodeTypes = {"aira-advanced": AdvancedNode};

export function AdvancedAgentFlow({workflow,onChange,permittedTools,onAllowTool,profileId}:{
  workflow:WorkflowDefinition;onChange:(graph:WorkflowDefinition)=>void;
  permittedTools:string[];onAllowTool:(tool:AgentTool)=>void;profileId?:string|null;
}) {
  const graph = workflow as unknown as Graph;
  const [picked,setPicked] = useState<string|null>(null);
  const [notice,setNotice] = useState("");
  const [runs,setRuns] = useState<Run[]>([]);
  const [selectedRun,setSelectedRun] = useState<number|null>(null);
  const [frame,setFrame] = useState(0);
  const [playing,setPlaying] = useState(false);
  const [loading,setLoading] = useState(false);
  const [phoenixSpans,setPhoenixSpans] = useState<PhoenixSpan[]>([]);
  const [phoenixMessage,setPhoenixMessage] = useState("");
  const run = runs.find(r=>r.id===selectedRun);
  const display = run?.workflow || graph;
  const stages = [...new Set((run?.steps || []).map(s=>s.stage ?? 0))].sort((a,b)=>a-b);
  const currentStage = stages[Math.min(frame,stages.length-1)] ?? -1;
  const elapsed = (run?.steps || []).filter(s=>(s.stage??0)<=currentStage);
  // A node can occur multiple times in a loop. Show its most recent visit.
  const recent = new Map(elapsed.map(s=>[s.node,s]));
  const current = new Set((run?.steps||[]).filter(s=>(s.stage??0)===currentStage).map(s=>s.node));
  const selected = graph.nodes.find(n=>n.id===picked);
  const update = (value:Graph) => onChange(value as unknown as WorkflowDefinition);
  const loadRuns = useCallback(async () => {
    if (!profileId || profileId === "new") return;
    setLoading(true);
    try {
      const result = await api<{runs:Run[]}>(`/agents/${profileId}/workflow/runs`);
      setRuns(result.runs ?? []);
    } catch (e) { setNotice(e instanceof Error?e.message:"Could not load workflow runs"); }
    finally {setLoading(false);}
  },[profileId]);
  useEffect(()=>{void loadRuns();},[loadRuns]);
  useEffect(()=>{
    if(!playing || !run || frame>=stages.length-1) return;
    const timer=window.setTimeout(()=>setFrame(v=>v+1),1100);
    return ()=>window.clearTimeout(timer);
  },[playing,run,frame,stages.length]);

  const nodes = useMemo<FlowNode[]>(()=>display.nodes.map(n=>{
    const entry=recent.get(n.id);
    return {id:n.id,type:"aira-advanced",position:{x:n.x,y:n.y},deletable:false,
      data:{kind:n.type,title:labels[n.type] || n.type,
        subtitle:entry ? `${entry.status} · ${entry.sources} sources · ${entry.latency_ms} ms${entry.branch?` · ${entry.branch}`:""}`
          : n.type==="loop" ? `Max ${n.max_iterations??3} repetitions · target ${n.min_sources??1} sources`
          : n.type==="evidence_gate" ? `At least ${n.min_sources??1} unique sources`
          : tools.includes(n.type) ? n.query_template && n.query_template!=="{question}"?n.query_template:"Retrieve evidence"
          : n.type==="question"?"Start here":n.type==="synthesize"?"One synthesis after all branches settle":"Final response",
        status:entry?.status,active:current.has(n.id)},
    };
  }),[display,elapsed,currentStage]);
  const edges = useMemo<Edge[]>(()=>display.edges.map((edge,i)=>({
    id:`edge-${i}-${edge.source}-${edge.target}-${edge.condition||""}`,source:edge.source,target:edge.target,
    type:"smoothstep",sourceHandle:edge.condition||undefined,
    label:edge.condition||undefined,animated:!run && graph.enabled,
    className:run && recent.get(edge.source)?.branch && edge.condition &&
      recent.get(edge.source)?.branch!==edge.condition?"aira-replay-unselected-edge":"",
  })),[display,run,currentStage,graph.enabled]);
  const onNodesChange:OnNodesChange<FlowNode> = useCallback(changes=>{
    if(run) return;
    const moved=applyNodeChanges(changes.filter(c=>c.type!=="remove"),nodes);
    if(!changes.some(c=>c.type==="position")) return;
    update({...graph,nodes:graph.nodes.map(n=>{
      const match=moved.find(m=>m.id===n.id);
      return match?{...n,x:match.position.x,y:match.position.y}:n;
    })});
  },[run,nodes,graph,onChange]);
  const connect=useCallback((c:Connection)=>{
    if(run || !c.source || !c.target || c.source===c.target) return;
    const source=graph.nodes.find(n=>n.id===c.source);
    const target=graph.nodes.find(n=>n.id===c.target);
    if(!source||!target||source.type==="answer"||target.type==="question"||source.type==="synthesize"&&target.type!=="answer") return;
    const condition=(source.type==="loop"||source.type==="evidence_gate")?c.sourceHandle as Choice|null:null;
    if((source.type==="loop"||source.type==="evidence_gate")&&!condition) return;
    if(graph.edges.some(e=>e.source===c.source&&e.condition===condition)) {
      if(source.type==="loop" || source.type==="evidence_gate") {setNotice("Each control-flow branch may have only one outgoing connection.");return;}
    }
    if(graph.edges.some(e=>e.source===c.source&&e.target===c.target&&e.condition===condition)) return;
    update({...graph,edges:[...graph.edges,{source:c.source,target:c.target,condition}]});
    setNotice("");
  },[graph,run,onChange]);
  function addNode(kind:Kind) {
    if(tools.includes(kind)&&!permittedTools.includes(kind)) onAllowTool(kind as AgentTool);
    const count=graph.nodes.filter(n=>n.type===kind).length+1;
    const id=`${kind}_${count}_${Math.random().toString(36).slice(2,6)}`;
    update({...graph,nodes:[...graph.nodes,{id,type:kind,x:300+((graph.nodes.length%4)*220),y:65+(graph.nodes.length%5)*115,
      min_sources:kind==="loop"?18:1,max_iterations:3,query_template:"{question}"}]});
    setPicked(id);
    setNotice("Connect this node to the graph before saving. Loop 'repeat' edges must lead back to the same loop.");
  }
  function removeNode(id:string) {
    const node=graph.nodes.find(n=>n.id===id);
    if(!node||locked.includes(node.type)) return;
    update({...graph,nodes:graph.nodes.filter(n=>n.id!==id),edges:graph.edges.filter(e=>e.source!==id&&e.target!==id)});
    setPicked(null);
  }
  function updateNode(value:Partial<Item>) {
    if(!picked) return;
    update({...graph,nodes:graph.nodes.map(n=>n.id===picked?{...n,...value}:n)});
  }
  async function loadPhoenix() {
    if(!run||!profileId||profileId==="new") return;
    setPhoenixMessage("Loading Phoenix spans…");
    try {
      const result=await api<{available:boolean;spans:PhoenixSpan[];message?:string|null}>(`/agents/${profileId}/workflow/runs/${run.id}/phoenix`);
      setPhoenixSpans(result.spans||[]);
      setPhoenixMessage(result.message||(result.available?`${result.spans.length} Phoenix spans found.`:"Phoenix is unavailable; saved steps are shown."));
    } catch(e) {setPhoenixMessage(e instanceof Error?e.message:"Could not load Phoenix spans");}
  }
  return <div className="aira-executable-builder aira-advanced-builder">
    <div className="aira-executable-toolbar">
      <label><input type="checkbox" checked={graph.enabled} onChange={e=>update({...graph,enabled:e.target.checked})}/> Execute workflow</label>
      <label><input type="checkbox" checked={graph.parallel??true} onChange={e=>update({...graph,parallel:e.target.checked})}/> Parallel independent searches</label>
      <label>Max steps <input aria-label="Maximum workflow node executions" type="number" min={4} max={180} value={graph.max_steps??96}
        onChange={e=>update({...graph,max_steps:Number(e.target.value)})}/></label>
    </div>
    <div className="aira-advanced-intro"><div><strong>Advanced graph builder</strong>
      <small>Compose unlimited nesting patterns within the 45-node graph, 10-repetition-per-loop, and 180-step safety caps.</small></div>
    </div>
    <div className="aira-executable-palette aira-advanced-palette">
      {palette.map(kind=><button type="button" key={kind} onClick={()=>addNode(kind)} disabled={!!run||graph.nodes.length>=45}>
        <Plus size={13}/>{name(kind)}
      </button>)}
    </div>
    <div className="aira-advanced-layout">
      <div className="aira-flow-canvas aira-executable-canvas" aria-label="Advanced executable workflow graph">
        <ReactFlow<FlowNode> nodes={nodes} edges={edges} nodeTypes={nodeTypes}
          onNodesChange={onNodesChange} onConnect={connect} onNodeClick={(_,node)=>setPicked(node.id)}
          onEdgesDelete={removed=>!run && update({...graph,edges:graph.edges.filter((edge,i)=>!removed.some(r=>r.id===`edge-${i}-${edge.source}-${edge.target}-${edge.condition||""}`))})}
          nodesDraggable={!run} nodesConnectable={!run} edgesReconnectable={false} fitView
          fitViewOptions={{padding:0.15,maxZoom:1.1}} minZoom={0.25} maxZoom={1.7}>
          <Background variant={BackgroundVariant.Dots} color="#cbd3e5"/><Controls showInteractive={false}/><MiniMap/>
        </ReactFlow>
      </div>
      <aside className="aira-advanced-inspector">
        <strong>Node settings</strong>
        {selected ? <><small>{name(selected.type)} · {selected.id}</small>
          {selected.type==="loop" && <>
            <label>Maximum repetitions <input type="number" min={1} max={10} value={selected.max_iterations??3}
              onChange={e=>updateNode({max_iterations:Number(e.target.value)})} disabled={!!run}/></label>
            <label>Stop once source count reaches <input type="number" min={1} max={18} value={selected.min_sources??18}
              onChange={e=>updateNode({min_sources:Number(e.target.value)})} disabled={!!run}/></label>
            <p>Connect <b>repeat</b> to the body and return to this loop. Connect <b>done</b> toward synthesis.</p>
          </>}
          {selected.type==="evidence_gate" && <>
            <label>Minimum unique sources <input type="number" min={1} max={18} value={selected.min_sources??1}
              onChange={e=>updateNode({min_sources:Number(e.target.value)})} disabled={!!run}/></label>
            <p>Use the <b>enough</b> and <b>insufficient</b> output handles to route each branch.</p>
          </>}
          {tools.includes(selected.type) && <label>Search query template
            <textarea rows={3} value={selected.query_template??"{question}"} disabled={!!run}
              onChange={e=>updateNode({query_template:e.target.value})}/>
            <small>Allowed placeholders: {"{question}"}, {"{iteration}"}.</small>
          </label>}
          {!locked.includes(selected.type) && <button type="button" disabled={!!run} onClick={()=>removeNode(selected.id)}><Trash2 size={14}/> Remove node</button>}
        </> : <p>Select a node to edit its parameters. Add conditions and loop nodes using the toolbar above. Connect both output handles of each decision node.</p>}
        <p>Only loop <b>repeat</b> edges may create cycles. The backend rejects unbounded or unreachable graphs.</p>
      </aside>
    </div>
    <section className="aira-workflow-replay" aria-label="Recorded advanced workflow playback">
      <div className="aira-replay-title"><strong>Run playback</strong><button type="button" disabled={loading||!profileId||profileId==="new"} onClick={()=>void loadRuns()}>
        <RefreshCw size={14}/> Refresh runs</button></div>
      <p>Replay repeated visits, nested decisions, timings, and source counts from your saved conversations.</p>
      <select aria-label="Select recorded workflow run" value={selectedRun??""} onChange={e=>{
        setSelectedRun(e.target.value?Number(e.target.value):null);setFrame(0);setPlaying(false);setPhoenixSpans([]);setPhoenixMessage("");
      }}><option value="">Edit current workflow</option>{runs.map(item=><option key={item.id} value={item.id}>
        {new Date(item.created_at).toLocaleString()} · Run {item.id}</option>)}</select>
      {run && <>
        <div className="aira-replay-phoenix"><button type="button" onClick={()=>void loadPhoenix()} disabled={!run.trace_id}>
          <RefreshCw size={14}/> Load Phoenix spans</button>{phoenixMessage&&<small role="status">{phoenixMessage}</small>}</div>
        <div className="aira-replay-controls">
          <button type="button" aria-label="Previous stage" disabled={frame<=0} onClick={()=>{setPlaying(false);setFrame(v=>Math.max(0,v-1));}}><ChevronLeft size={17}/></button>
          <button type="button" onClick={()=>{if(frame>=stages.length-1)setFrame(0);setPlaying(p=>!p);}}>
            {playing?<Pause size={15}/>:<Play size={15}/>} {playing?"Pause":"Play"}</button>
          <button type="button" aria-label="Next stage" disabled={frame>=stages.length-1} onClick={()=>{setPlaying(false);setFrame(v=>Math.min(stages.length-1,v+1));}}><ChevronRight size={17}/></button>
          <input aria-label="Playback stage" type="range" min={0} max={Math.max(0,stages.length-1)} value={frame}
            onChange={e=>{setPlaying(false);setFrame(Number(e.target.value));}}/>
          <small>Stage {frame+1} / {stages.length}</small>
        </div>
        <div className="aira-replay-step-list">{(run.steps||[]).filter(s=>(s.stage??0)===currentStage).map((step,i)=>{
          const matched=phoenixSpans.find(span=>span.span_id===step.span_id);
          return <div key={`${step.node}-${i}`}><strong>{display.nodes.find(n=>n.id===step.node)?.type||step.node}</strong> · {matched?.status||step.status} · {matched?.latency_ms??step.latency_ms} ms · {step.sources} sources
            {step.branch&&` · ${step.branch}`}{step.iteration!==undefined&&` · iteration ${step.iteration}`}
            {step.span_id&&<button type="button" onClick={()=>void navigator.clipboard.writeText(step.span_id!)}><Clipboard size={12}/> Copy span ID</button>}</div>;
        })}</div>
      </>}
    </section>
    {notice&&<p className="agent-note" role="status">{notice}</p>}
    <p className="agent-note">Save to validate the graph. Independent searches run concurrently per wave; loop branches are bounded and evidence sources are deduplicated. The original LLM-directed agent still runs when workflows are disabled.</p>
  </div>;
}
