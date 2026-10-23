/**
 * The flow editor is a visual representation of agent TOOL PERMISSIONS,
 * not a deterministic workflow engine. The model selects permitted tools at runtime.
 * Keep this catalog in sync with aira.profiles.ALLOWED_TOOLS.
 */
export const AGENT_TOOL_CATALOG = [
  { id: "local_search", label: "Local PDF retrieval", help: "Required evidence search over indexed papers." },
  { id: "conversation_search", label: "Conversation memory", help: "Find earlier turns; never treated as scientific evidence." },
  { id: "web_search", label: "Web browsing", help: "Requires the per-chat Web permission." },
  { id: "academic_search", label: "Academic search", help: "Requires the per-chat Academic permission." },
  { id: "context_read", label: "Read working notes", help: "Inspect verified notes kept in conversation context." },
  { id: "context_edit", label: "Edit working notes", help: "Only after explicit user request and source verification." },
] as const;

export type AgentFlowTool = typeof AGENT_TOOL_CATALOG[number]["id"];
export type AgentFlowPosition = { x: number; y: number };
export type AgentFlowLayout = { positions: Record<string, AgentFlowPosition>; visibleTools: AgentFlowTool[] };

export const TOOL_NODE_PREFIX = "tool:";
export const toolNodeId = (tool: AgentFlowTool) => `${TOOL_NODE_PREFIX}${tool}`;
export const toolFromNodeId = (node: string): AgentFlowTool | null => {
  const value = node.startsWith(TOOL_NODE_PREFIX) ? node.slice(TOOL_NODE_PREFIX.length) : "";
  return AGENT_TOOL_CATALOG.some(tool => tool.id === value) ? value as AgentFlowTool : null;
};
export function normalizeAgentTools(tools: readonly string[]): AgentFlowTool[] {
  return AGENT_TOOL_CATALOG.filter(tool => tool.id === "local_search" || tools.includes(tool.id)).map(tool => tool.id);
}
export function enableAgentTool(tools: readonly string[], tool: AgentFlowTool): AgentFlowTool[] {
  return normalizeAgentTools([...tools, tool]);
}
export function disableAgentTool(tools: readonly string[], tool: AgentFlowTool): AgentFlowTool[] {
  return normalizeAgentTools(tools.filter(id => id !== tool));
}
export function validAgentToolConnection(source: string | null, target: string | null): boolean {
  return source === "agent" && target !== null && toolFromNodeId(target) !== null;
}
export function defaultAgentPositions(): Record<string, AgentFlowPosition> {
  return {
    question: { x: 35, y: 210 },
    agent: { x: 315, y: 210 },
    answer: { x: 920, y: 210 },
    ...Object.fromEntries(AGENT_TOOL_CATALOG.map((tool, index) => [toolNodeId(tool.id),
      { x: 645 + (index > 2 ? 20 : 0), y: -125 + index * 116 }])),
  };
}

export function readAgentFlowLayout(id: string): AgentFlowLayout | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(`aira.agent-flow-layout.${id}`);
    if (!raw) return null;
    const payload: unknown = JSON.parse(raw);
    if (typeof payload !== "object" || payload === null) return null;
    const candidate = payload as Record<string, unknown>;
    const positions: Record<string, AgentFlowPosition> = {};
    if (typeof candidate.positions === "object" && candidate.positions !== null) {
      for (const [key, pos] of Object.entries(candidate.positions)) {
        if (typeof pos === "object" && pos !== null && typeof (pos as AgentFlowPosition).x === "number"
          && typeof (pos as AgentFlowPosition).y === "number"
          && Number.isFinite((pos as AgentFlowPosition).x) && Number.isFinite((pos as AgentFlowPosition).y)) {
          positions[key] = { x: (pos as AgentFlowPosition).x, y: (pos as AgentFlowPosition).y };
        }
      }
    }
    return { positions, visibleTools: normalizeAgentTools(Array.isArray(candidate.visibleTools) ? candidate.visibleTools.filter((value): value is string => typeof value === "string") : []) };
  } catch { return null; }
}
export function writeAgentFlowLayout(id: string, layout: AgentFlowLayout): void {
  if (typeof window === "undefined") return;
  try { window.localStorage.setItem(`aira.agent-flow-layout.${id}`, JSON.stringify(layout)); } catch { /* Browsers with disabled storage can still edit tools. */ }
}
