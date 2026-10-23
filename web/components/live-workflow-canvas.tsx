"use client";

import { useMemo, useState } from "react";
import {
  ReactFlow, Background, BackgroundVariant, Controls, Handle, Position,
  type Edge, type Node, type NodeProps,
} from "@xyflow/react";
import { BookOpen, BrainCircuit, Check, Copy, Database, GitBranch, Globe, MessageSquare, Repeat2 } from "lucide-react";
import { api } from "@/lib/api";
import type { WorkflowStep, SavedWorkflowGraph } from "@/lib/types";

/** Operational metadata only: no questions, retrieved passages, or model responses. */
export type LiveWorkflowEvent = {
  sequence: number;
  node: string;
  status: "running" | "completed" | "skipped" | "failed";
  stage: number;
  sources: number;
  latency_ms?: number;
  span_id?: string;
  reason?: string;
  branch?: "enough" | "insufficient" | "repeat" | "done";
  iteration?: number;
};
export type LiveWorkflowProgress = {
  graph: {
    schema_version: number;
    nodes: { id: string; type: string; x: number; y: number }[];
    edges: { source: string; target: string; condition?: string | null }[];
  };
  events: LiveWorkflowEvent[];
  sequence: number;
  trace_id?: string | null;
};

type LiveNode = Node<{ kind: string; status: string; branch?: string; visits: number; iteration?: number }, "live">;
const names: Record<string, string> = {
  question: "Question", local_search: "Local search", academic_search: "Academic search", web_search: "Web search",
  evidence_gate: "Evidence check", loop: "Repeat until", synthesize: "LLM synthesis", answer: "Answer",
};
function LiveNodeView({ data, selected }: NodeProps<LiveNode>) {
  const Icon = data.kind === "question" ? MessageSquare : data.kind === "local_search" ? Database
    : data.kind === "academic_search" ? BookOpen : data.kind === "web_search" ? Globe
    : data.kind === "evidence_gate" ? GitBranch : data.kind === "loop" ? Repeat2
    : data.kind === "synthesize" ? BrainCircuit : Check;
  return <div className={`aira-live-node aira-live-${data.status} ${selected ? "aira-live-selected" : ""}`}>
    {data.kind !== "question" && <Handle type="target" position={Position.Left} isConnectable={false} />}
    <div className="aira-live-node-head"><Icon size={17}/><strong>{names[data.kind] || data.kind}</strong></div>
    <small>{data.status === "running" ? "Running…" : data.status === "idle" ? "Pending" : data.status}</small>
    {(data.visits > 1 || data.iteration !== undefined || data.branch) && <small>
      {data.visits > 1 ? `${data.visits} visits` : ""}{data.iteration !== undefined ? ` · iteration ${data.iteration}` : ""}{data.branch ? ` · ${data.branch}` : ""}
    </small>}
    {data.kind === "evidence_gate" || data.kind === "loop" ? <>
      <Handle type="source" id={data.kind === "loop" ? "repeat" : "enough"} position={Position.Right} style={{ top: "35%" }} isConnectable={false}/>
      <Handle type="source" id={data.kind === "loop" ? "done" : "insufficient"} position={Position.Right} style={{ top: "75%" }} isConnectable={false}/>
    </> : data.kind !== "answer" && <Handle type="source" position={Position.Right} isConnectable={false} />}
  </div>;
}
const nodeTypes = { live: LiveNodeView };
type PhoenixSpan = { node: string; span_id: string; latency_ms: number; status: string };
type PhoenixResponse = { available: boolean; spans: PhoenixSpan[]; message?: string | null };
type SavedRun = { profileId: string; turnId: number };

/** Reconstruct read-only playback from the graph + bounded steps saved in PostgreSQL. */
export function progressFromSavedRun(graph: SavedWorkflowGraph, steps: WorkflowStep[], traceId?: string | null): LiveWorkflowProgress {
  return {
    graph: {
      schema_version: graph.schema_version || 1,
      nodes: graph.nodes.map(node => ({ id: node.id, type: node.type, x: node.x, y: node.y })),
      edges: graph.edges.map(edge => ({ source: edge.source, target: edge.target, condition: edge.condition })),
    },
    events: steps.map((step, index) => ({
      sequence: index + 1, node: step.node, status: step.status,
      stage: step.stage || 0, sources: step.sources, latency_ms: step.latency_ms,
      span_id: step.span_id, reason: step.reason, branch: step.branch, iteration: step.iteration,
    })),
    sequence: steps.length,
    trace_id: traceId,
  };
}

export function LiveWorkflowCanvas({ progress, failed = false, savedRun }: {
  progress: LiveWorkflowProgress; failed?: boolean; savedRun?: SavedRun;
}) {
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [selectedSequence, setSelectedSequence] = useState<number | null>(null);
  const [phoenix, setPhoenix] = useState<PhoenixResponse | null>(null);
  const [loadingPhoenix, setLoadingPhoenix] = useState(false);
  const [phoenixError, setPhoenixError] = useState("");
  const [copyMessage, setCopyMessage] = useState("");
  const latest = useMemo(() => new Map(progress.events.map(event => [event.node, event])), [progress.events]);
  const visits = useMemo(() => {
    const result = new Map<string, number>();
    // Live runs count "running" events; a saved run counts every recorded event.
    for (const event of progress.events) if (event.status === "running" || savedRun) {
      result.set(event.node, (result.get(event.node) || 0) + 1);
    }
    return result;
  }, [progress.events, savedRun]);
  const nodes = useMemo<LiveNode[]>(() => progress.graph.nodes.map(item => {
    const event = latest.get(item.id);
    return {
      id: item.id, type: "live", position: { x: item.x, y: item.y },
      data: { kind: item.type, status: event?.status || "idle", branch: event?.branch,
              visits: visits.get(item.id) || 0, iteration: event?.iteration },
      draggable: false, selectable: true, selected: item.id === selectedNode,
    };
  }), [progress.graph.nodes, latest, visits, selectedNode]);
  const edges = useMemo<Edge[]>(() => progress.graph.edges.map((edge, index) => {
    // Explicit branch labels prevent showing an untaken branch as active.
    const origin = latest.get(edge.source);
    const branchChosen = !!edge.condition && !!origin?.branch && edge.condition === origin.branch;
    const branchRejected = !!edge.condition && !!origin?.branch && edge.condition !== origin.branch;
    const destination = latest.get(edge.target);
    const current = !branchRejected && (branchChosen || (!edge.condition && !!origin && !!destination));
    return {
      id: `live-${index}`, source: edge.source, target: edge.target, sourceHandle: edge.condition || undefined,
      label: edge.condition || undefined, type: "smoothstep", animated: current && destination?.status === "running",
      className: branchRejected ? "aira-live-edge-muted" : current ? "aira-live-edge-active" : "aira-live-edge-idle",
      style: { strokeWidth: current ? 2.5 : 1.5 },
    };
  }), [progress.graph.edges, latest]);
  const running = progress.events.filter(event => event.status === "running" && latest.get(event.node)?.sequence === event.sequence);
  const nodeEvents = selectedNode ? progress.events.filter(event => event.node === selectedNode) : [];
  // One panel per visit: live events include running/completed transitions; persisted
  // playback contains one terminal event per visit. Avoid collapsing loop iterations.
  const selectedEvent = nodeEvents.find(event => event.sequence === selectedSequence) || nodeEvents.at(-1);
  const nodeKind = progress.graph.nodes.find(node => node.id === selectedNode)?.type || "";
  // Prefer exact span identity for repeated nodes. Do not incorrectly associate a
  // previous loop iteration with the latest Phoenix span from the same node.
  const correlatedSpan = selectedEvent?.span_id
    ? phoenix?.spans.find(span => span.span_id === selectedEvent.span_id)
    : phoenix?.spans.filter(span => span.node === selectedNode).at(-1);
  const traceId = progress.trace_id;

  async function copy(value: string) {
    try { await navigator.clipboard.writeText(value); setCopyMessage("Copied to clipboard"); }
    catch { setCopyMessage("Clipboard unavailable"); }
  }
  async function loadPhoenix() {
    if (!savedRun) return;
    setLoadingPhoenix(true); setPhoenixError("");
    try {
      const response = await api<PhoenixResponse>(`/agents/${encodeURIComponent(savedRun.profileId)}/workflow/runs/${savedRun.turnId}/phoenix`);
      setPhoenix(response);
      if (!response.available) setPhoenixError(response.message || "Phoenix is currently unavailable.");
    } catch { setPhoenixError("Could not load Phoenix spans for this saved run."); }
    finally { setLoadingPhoenix(false); }
  }

  return <section className="aira-live-workflow" aria-label={savedRun ? "Saved workflow execution" : "Live agent workflow"} aria-live="off">
    <div className="aira-live-workflow-title"><strong>{savedRun ? "Executed workflow" : "Live workflow"}</strong>
      {!savedRun && <span className="aira-live-pulse" />}
      <small>{failed ? "Request failed" : savedRun ? "Select a node to inspect its visits" : running.length ? `${running.length} active node${running.length === 1 ? "" : "s"}` : "Following execution"}</small>
    </div>
    <div className="aira-live-canvas" aria-label={`Workflow graph: ${progress.graph.nodes.length} nodes. ${running.length} running.`}>
      <ReactFlow<LiveNode> nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView
        fitViewOptions={{ padding: 0.24, maxZoom: 1.1 }} minZoom={0.2} maxZoom={1.5}
        nodesDraggable={false} nodesConnectable={false} elementsSelectable={true}
        onNodeClick={(_, node) => { setSelectedNode(node.id); setSelectedSequence(null); setCopyMessage(""); }}
        onPaneClick={() => { setSelectedNode(null); setSelectedSequence(null); }}
        panOnDrag={true} zoomOnScroll={false} zoomOnPinch={true} zoomOnDoubleClick={false}
        proOptions={{hideAttribution:true}}>
        <Background variant={BackgroundVariant.Dots} color="#cdd5e6" gap={18}/>
        <Controls showInteractive={false}/>
      </ReactFlow>
    </div>
    <div className="aira-live-legend"><span>● Running</span><span>● Completed</span><span>● Skipped</span><span>● Failed</span></div>
    {selectedNode && <aside className="aira-node-inspector" aria-label="Selected workflow node details">
      <div className="aira-node-inspector-head"><strong>{names[nodeKind] || nodeKind || selectedNode}</strong>
        <button type="button" aria-label="Close node inspector" onClick={() => setSelectedNode(null)}>×</button></div>
      <small className="aira-node-inspector-id">Node ID: {selectedNode}</small>
      {nodeEvents.length === 0 ? <p>No execution recorded for this node yet.</p> : <>
        {nodeEvents.length > 1 && <div className="aira-node-visits" aria-label="Node execution events">
          {nodeEvents.map((event, index) => <button type="button" key={event.sequence}
            aria-pressed={event.sequence === selectedEvent?.sequence}
            className={event.sequence === selectedEvent?.sequence ? "selected" : ""}
            onClick={() => setSelectedSequence(event.sequence)}>
            {savedRun ? `Visit ${index+1}` : `Event ${index+1}`} · {event.status}{event.iteration !== undefined ? ` · loop ${event.iteration}` : ""}
          </button>)}
        </div>}
        {selectedEvent && <dl className="aira-node-metrics">
          <div><dt>Status</dt><dd>{selectedEvent.status}</dd></div>
          <div><dt>Latency</dt><dd>{selectedEvent.latency_ms !== undefined ? `${selectedEvent.latency_ms} ms` : selectedEvent.status === "running" ? "In progress" : "Not recorded"}</dd></div>
          <div><dt>{savedRun ? "Accepted sources" : "Retrieved sources"}</dt><dd>{selectedEvent.sources}</dd></div>
          <div><dt>Stage</dt><dd>{selectedEvent.stage}</dd></div>
          {selectedEvent.branch && <div><dt>Branch taken</dt><dd>{selectedEvent.branch}</dd></div>}
          {selectedEvent.iteration !== undefined && <div><dt>Loop iteration</dt><dd>{selectedEvent.iteration}</dd></div>}
        </dl>}
        {selectedEvent?.reason && <p className="aira-node-inspector-note">{selectedEvent.reason}</p>}
        {(selectedEvent?.span_id || correlatedSpan?.span_id) && <div className="aira-node-span">
          <span>Phoenix span</span>
          <code>{selectedEvent?.span_id || correlatedSpan?.span_id}</code>
          <button type="button" onClick={() => void copy((selectedEvent?.span_id || correlatedSpan?.span_id)!) } title="Copy Phoenix span ID"><Copy size={13}/> Copy span</button>
        </div>}
        {traceId && <div className="aira-node-span"><span>Trace ID</span><code>{traceId}</code>
          <button type="button" onClick={() => void copy(traceId)} title="Copy Phoenix trace ID"><Copy size={13}/> Copy trace</button></div>}
        {savedRun && <div className="aira-node-phoenix">
          <button type="button" onClick={() => void loadPhoenix()} disabled={loadingPhoenix}>{loadingPhoenix ? "Checking Phoenix…" : "Load Phoenix span details"}</button>
          {phoenixError && <small role="status">{phoenixError}</small>}
          {phoenix?.available && <small role="status">{correlatedSpan ? `Phoenix: ${correlatedSpan.status}, ${correlatedSpan.latency_ms} ms` : "No matching span in Phoenix for this node."}</small>}
        </div>}
        {copyMessage && <small role="status">{copyMessage}</small>}
      </>}
    </aside>}
    <ol className="aira-live-events" aria-label="Recent workflow execution events">
      {progress.events.slice(-6).map(event => <li key={event.sequence}>
        <strong>{names[progress.graph.nodes.find(node => node.id === event.node)?.type || ""] || event.node}</strong>
        <span>{event.status}{event.branch ? ` · ${event.branch}` : ""}{event.iteration !== undefined ? ` · loop ${event.iteration}` : ""}</span>
      </li>)}
    </ol>
  </section>;
}
