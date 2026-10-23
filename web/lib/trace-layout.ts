import type { TraceRun, TraceSpan } from "./types";

export type TraceGraphNode = {
  key: string; spanId?: string; kind: TraceSpan["kind"] | "input" | "output";
  label: string; detail: string; x: number; y: number;
};
export type TraceGraphLink = { from: string; to: string; relation?: "parent" | "timeline" };
export type TraceGraphLayout = { width: number; height: number; nodes: TraceGraphNode[]; links: TraceGraphLink[] };

/**
 * Research execution diagram: the root research.chat span is the *trace*, not
 * an operation. Place recorded model/tools in chronological order so an LLM
 * invocation is never depicted as parallel solely because it shares a parent.
 * Dotted arrows indicate OBSERVED start-time order, not inferred causality.
 * Parent relationships are preserved separately for inspection and solid links.
 */
export function layoutResearchTrace(trace: TraceRun): TraceGraphLayout {
  const root = trace.spans.find(s => s.name === "research.chat" || s.name === "literature.survey");
  const operations = trace.spans
    .filter(s => s.id !== root?.id && s.kind !== "operation" && s.name !== "agent.chain" && !s.name.startsWith("diagnostics."))
    .sort((a, b) => a.start_us - b.start_us || a.id.localeCompare(b.id));
  const cardWidth = 215, cardHeight = 105, gap = 44;
  const x = (i: number) => 35 + i * (cardWidth + gap);
  const height = 330;
  const middle = (height - cardHeight) / 2;
  const nodes: TraceGraphNode[] = [{
    key: "input", spanId: root?.id, kind: "input", label: "User request", detail: "Click to inspect input", x: x(0), y: middle,
  }];
  for (const [index, span] of operations.entries()) {
    const kind = span.name.startsWith("web.") || span.name.startsWith("academic.") || span.name.startsWith("tool.") || span.name.startsWith("memory.") ? "tool" : span.kind;
    const details = [span.tool || span.model || span.strategy || "", span.evidence_count == null ? "" : `${span.evidence_count} results`, `${(span.duration_us / 1000000).toFixed(2)} s`, span.error ? "ERROR" : ""].filter(Boolean).join(" · ");
    nodes.push({key: span.id, spanId: span.id, kind, label: span.name.replace(/\./g, " "), detail: details, x: x(index + 1), y: middle});
  }
  nodes.push({key: "output", spanId: root?.id, kind: "output", label: "Assistant response", detail: "Click to inspect output", x: x(operations.length + 1), y: middle});

  const links: TraceGraphLink[] = [];
  const operationIds = new Set(operations.map(s => s.id));
  const spansById = new Map(trace.spans.map(s => [s.id, s]));
  function closestDisplayedParent(span: TraceSpan): string | null {
    let parent = span.parent_id;
    const visited = new Set<string>([span.id]);
    while (parent && !visited.has(parent)) {
      visited.add(parent);
      if (operationIds.has(parent)) return parent;
      parent = spansById.get(parent)?.parent_id || null;
    }
    return null;
  }
  for (let i = 0; i < nodes.length - 1; i++) {
    const next = operations[i]; // next node, when not output
    const isParent = next && closestDisplayedParent(next) === nodes[i].key;
    links.push({from: nodes[i].key, to: nodes[i + 1].key, relation: isParent ? "parent" : "timeline"});
  }
  for (const span of operations) {
    const parent = closestDisplayedParent(span);
    if (parent && nodes.find(n => n.key === parent)!.x < nodes.find(n => n.key === span.id)!.x &&
        !links.some(link => link.from === parent && link.to === span.id)) {
      links.push({from: parent, to: span.id, relation: "parent"});
    }
  }
  return {width: x(operations.length + 1) + cardWidth + 35, height, nodes, links};
}
