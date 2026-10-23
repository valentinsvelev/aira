"use client";

import { useCallback, useEffect, useMemo, useState, type DragEvent } from "react";
import {
  ReactFlow, Background, BackgroundVariant, Controls, Handle, MarkerType,
  MiniMap, Position, applyNodeChanges, type Connection, type Edge,
  type Node, type NodeProps, type OnNodesChange, type ReactFlowInstance,
} from "@xyflow/react";
import { BookOpen, BrainCircuit, Check, CircleHelp, MessageCircle, Plus, ShieldCheck, Trash2 } from "lucide-react";
import {
  AGENT_TOOL_CATALOG, defaultAgentPositions, disableAgentTool,
  enableAgentTool, normalizeAgentTools, toolFromNodeId, toolNodeId,
  validAgentToolConnection, type AgentFlowPosition, type AgentFlowTool,
} from "@/lib/agent-flow-config";

type FlowKind = "question" | "agent" | "answer" | "tool";
type FlowNode = Node<{ title: string; description: string; kind: FlowKind; enabled: boolean }, "airaAgentNode">;
const nodeTypes = { airaAgentNode: AgentFlowNode };

function AgentFlowNode({ data, selected }: NodeProps<FlowNode>) {
  const Icon = data.kind === "question" ? MessageCircle : data.kind === "agent" ? BrainCircuit : data.kind === "answer" ? ShieldCheck : BookOpen;
  return <div className={`aira-flow-node aira-flow-node-${data.kind} ${data.enabled ? "is-active" : "is-inactive"} ${selected ? "is-selected" : ""}`}>
    {data.kind !== "question" && <Handle type="target" position={Position.Left} />}
    <div className="aira-flow-node-top"><span className="aira-flow-node-icon"><Icon size={17}/></span>
      {data.kind === "tool" && <span className="aira-flow-node-state">{data.enabled ? "Allowed" : "Not connected"}</span>}
    </div>
    <strong>{data.title}</strong><small>{data.description}</small>
    {(data.kind === "question" || data.kind === "agent") && <Handle type="source" position={Position.Right} />}
  </div>;
}

function makeNodes(tools: readonly AgentFlowTool[], enabled: readonly AgentFlowTool[], positions: Record<string, AgentFlowPosition>): FlowNode[] {
  const defaults = defaultAgentPositions();
  const basics: FlowNode[] = [
    { id: "question", type: "airaAgentNode", position: positions.question || defaults.question, deletable: false, data: { kind: "question", title: "Research question", description: "Incoming chat request", enabled: true } },
    { id: "agent", type: "airaAgentNode", position: positions.agent || defaults.agent, deletable: false, data: { kind: "agent", title: "Research agent", description: "LLM chooses which tools to call", enabled: true } },
    { id: "answer", type: "airaAgentNode", position: positions.answer || defaults.answer, deletable: false, data: { kind: "answer", title: "Cited answer", description: "Evidence-grounded response", enabled: true } },
  ];
  return [...basics, ...AGENT_TOOL_CATALOG.filter(tool => tools.includes(tool.id)).map(tool => ({
    id: toolNodeId(tool.id), type: "airaAgentNode" as const,
    position: positions[toolNodeId(tool.id)] || defaults[toolNodeId(tool.id)],
    deletable: tool.id !== "local_search",
    data: { kind: "tool" as const, title: tool.label, description: tool.help, enabled: enabled.includes(tool.id) },
  }))];
}

function makeEdges(enabled: readonly AgentFlowTool[]): Edge[] {
  const fixed: Edge[] = [
    { id: "question-agent", source: "question", target: "agent", selectable: false, deletable: false, type: "smoothstep" },
    { id: "agent-answer", source: "agent", target: "answer", selectable: false, deletable: false, type: "smoothstep" },
  ];
  return [...fixed, ...enabled.map(tool => ({
    id: `allowed:${tool}`, source: "agent", target: toolNodeId(tool),
    type: "smoothstep" as const, animated: false, deletable: tool !== "local_search",
    style: { stroke: "#757fe0", strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color: "#757fe0" },
  }))];
}

export function AgentFlowBuilder({ enabledTools, visibleTools, positions, onEnabledToolsChange, onVisibleToolsChange, onPositionChange }: {
  enabledTools: AgentFlowTool[];
  visibleTools: AgentFlowTool[];
  positions: Record<string, AgentFlowPosition>;
  onEnabledToolsChange: (tools: AgentFlowTool[]) => void;
  onVisibleToolsChange: (tools: AgentFlowTool[]) => void;
  onPositionChange: (id: string, position: AgentFlowPosition) => void;
}) {
  const [selectedTool, setSelectedTool] = useState<AgentFlowTool | null>(null);
  const [instance, setInstance] = useState<ReactFlowInstance<FlowNode> | null>(null);
  const [nodes, setNodes] = useState<FlowNode[]>(() => makeNodes(visibleTools, enabledTools, positions));
  const edges = useMemo(() => makeEdges(enabledTools), [enabledTools]);

  useEffect(() => {
    setNodes(oldNodes => makeNodes(visibleTools, enabledTools, positions).map(node => ({
      ...node,
      // Keep current dragged positions when tool permissions change.
      position: oldNodes.find(previous => previous.id === node.id)?.position || node.position,
    })));
  }, [visibleTools, enabledTools, positions]);

  const onNodesChange: OnNodesChange<FlowNode> = useCallback(changes => {
    // Delete is handled explicitly to ensure backend permissions cannot get out of sync.
    setNodes(current => applyNodeChanges(changes.filter(change => change.type !== "remove"), current));
    for (const change of changes) {
      if (change.type !== "remove") continue;
      const tool = toolFromNodeId(change.id);
      if (!tool || tool === "local_search") continue;
      onVisibleToolsChange(normalizeAgentTools(visibleTools.filter(item => item !== tool)));
      onEnabledToolsChange(disableAgentTool(enabledTools, tool));
      if (selectedTool === tool) setSelectedTool(null);
    }
  }, [enabledTools, onEnabledToolsChange, onVisibleToolsChange, selectedTool, visibleTools]);

  const enable = (tool: AgentFlowTool) => {
    onVisibleToolsChange(enableAgentTool(visibleTools, tool));
    onEnabledToolsChange(enableAgentTool(enabledTools, tool));
    setSelectedTool(tool);
  };
  const disable = (tool: AgentFlowTool) => {
    if (tool !== "local_search") onEnabledToolsChange(disableAgentTool(enabledTools, tool));
  };
  const remove = (tool: AgentFlowTool) => {
    if (tool === "local_search") return;
    onVisibleToolsChange(normalizeAgentTools(visibleTools.filter(t => t !== tool)));
    disable(tool);
    setSelectedTool(null);
  };
  const connect = (connection: Connection) => {
    if (!validAgentToolConnection(connection.source, connection.target)) return;
    const tool = toolFromNodeId(connection.target!);
    if (tool && visibleTools.includes(tool)) onEnabledToolsChange(enableAgentTool(enabledTools, tool));
  };
  const drop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const tool = event.dataTransfer.getData("application/x-aira-agent-tool") as AgentFlowTool;
    if (!AGENT_TOOL_CATALOG.some(item => item.id === tool)) return;
    const point = instance?.screenToFlowPosition({ x: event.clientX, y: event.clientY });
    if (point) onPositionChange(toolNodeId(tool), point);
    if (!visibleTools.includes(tool)) onVisibleToolsChange(normalizeAgentTools([...visibleTools, tool]));
    setSelectedTool(tool);
  };
  const item = AGENT_TOOL_CATALOG.find(tool => tool.id === selectedTool);
  return <div className="aira-agent-flow-builder">
    <div className="aira-flow-helper"><CircleHelp size={17}/><p>Connect <strong>Research agent → Tool</strong> to grant permission. Disconnect to revoke it. The LLM chooses tools dynamically during a chat; these arrows do not set a fixed execution order.</p></div>
    <div className="aira-flow-layout">
      <aside className="aira-flow-palette" aria-label="Available agent tools">
        <h4>Tool library</h4><p>Drag a tool onto the canvas, or click Add to enable it.</p>
        {AGENT_TOOL_CATALOG.map(tool => <div className="aira-flow-palette-tool" key={tool.id} draggable={tool.id !== "local_search"}
          onDragStart={event => {event.dataTransfer.setData("application/x-aira-agent-tool", tool.id);event.dataTransfer.effectAllowed="copy";}}>
          <strong>{tool.label}</strong><small>{tool.id === "local_search" ? "Required" : enabledTools.includes(tool.id) ? "Enabled" : "Optional"}</small>
          {tool.id !== "local_search" && <button type="button" onClick={() => enable(tool.id)} title={`Enable ${tool.label}`} disabled={enabledTools.includes(tool.id)}><Plus size={14}/> Add</button>}
        </div>)}
      </aside>
      <div className="aira-flow-canvas" role="region" aria-label="Visual agent tool permissions" onDragOver={event => event.preventDefault()} onDrop={drop}>
        <ReactFlow<FlowNode> nodes={nodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onConnect={connect}
          onInit={setInstance} isValidConnection={connection => validAgentToolConnection(connection.source, connection.target)}
          onEdgesDelete={deleted => { for (const edge of deleted) {const tool = toolFromNodeId(edge.target);if (tool) disable(tool);} }}
          onNodeClick={(_, node) => setSelectedTool(toolFromNodeId(node.id))}
          onNodeDragStop={(_, node) => onPositionChange(node.id, node.position)}
          onPaneClick={() => setSelectedTool(null)}
          fitView fitViewOptions={{ padding: 0.12, maxZoom: 1.05 }} minZoom={0.3} maxZoom={1.7}
          deleteKeyCode={["Backspace", "Delete"]} edgesReconnectable={false}>
          <Background variant={BackgroundVariant.Dots} color="#cbd3e5" gap={20} size={1}/>
          <MiniMap pannable zoomable nodeStrokeWidth={3}/>
          <Controls showInteractive={false}/>
        </ReactFlow>
      </div>
    </div>
    <div className="aira-flow-inspector" aria-live="polite">
      {item ? <><div><strong>{item.label}</strong><p>{item.help}</p></div><div className="aira-flow-inspector-actions">
        {enabledTools.includes(item.id) ? <button type="button" onClick={() => disable(item.id)} disabled={item.id === "local_search"}><Check size={14}/> {item.id === "local_search" ? "Required" : "Enabled — disconnect"}</button>
          : <button type="button" onClick={() => enable(item.id)}><Plus size={14}/> Connect & enable</button>}
        {item.id !== "local_search" && <button type="button" className="aira-flow-remove" onClick={() => remove(item.id)}><Trash2 size={14}/> Remove node</button>}
      </div></> : <><strong>Select a tool node to configure its permission</strong><span>You can pan, zoom, arrange nodes and drag additional tools from the library.</span></>}
    </div>
  </div>;
}
