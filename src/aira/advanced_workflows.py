"""Advanced, bounded AIRA workflow interpreter.

Version-2 graphs support arbitrarily placed conditional nodes, nested loops and
repeated tools. "Arbitrary" means composable nodes and nesting, not unrestricted
execution: size, iteration, and step limits are mandatory. The legacy DAG path
is kept separate for compatibility with existing saved agents.
"""
from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from time import perf_counter
from typing import TYPE_CHECKING

from aira.observability import annotate_llm_response, callbacks, trace_span
from aira.progress import begin_workflow, record_operation, record_workflow_event

if TYPE_CHECKING:
    from aira.agent_workflows import WorkflowDefinition

SEARCH_TYPES = {"local_search", "web_search", "academic_search"}
CONTROL_TYPES = {"evidence_gate", "loop"}
ALLOWED_TYPES = SEARCH_TYPES | CONTROL_TYPES | {"question", "synthesize", "answer"}


def validate_advanced_graph(graph: WorkflowDefinition) -> None:
    """Reject unsafe/unreachable graphs *before* any tool or network call."""
    by_id = {n.id: n for n in graph.nodes}
    if len(by_id) != len(graph.nodes) or any(not n.id or n.type not in ALLOWED_TYPES for n in graph.nodes):
        raise ValueError("Every workflow node needs a unique ID and a known type")
    for kind in ("question", "synthesize", "answer"):
        if sum(n.type == kind for n in graph.nodes) != 1:
            raise ValueError(f"Advanced workflows require exactly one {kind} node")
    counts = defaultdict(int)
    edges = defaultdict(list)
    reverse = defaultdict(list)
    unique = set()
    for edge in graph.edges:
        if edge.source not in by_id or edge.target not in by_id or edge.source == edge.target:
            raise ValueError("Connections must join two different existing nodes")
        key = (edge.source, edge.target, edge.condition)
        if key in unique:
            raise ValueError("Duplicate workflow connection")
        unique.add(key)
        src, dst = by_id[edge.source].type, by_id[edge.target].type
        if dst == "question" or src == "answer" or (src == "synthesize" and dst != "answer"):
            raise ValueError("Question is the start; synthesis can only connect to answer")
        if src == "evidence_gate" and edge.condition not in {"enough", "insufficient"}:
            raise ValueError("Evidence-check edges must be labeled enough / insufficient")
        if src == "loop" and edge.condition not in {"repeat", "done"}:
            raise ValueError("Loop edges must be labeled repeat / done")
        if src not in CONTROL_TYPES and edge.condition is not None:
            raise ValueError("Only condition or loop nodes accept branch labels")
        edges[edge.source].append(edge)
        reverse[edge.target].append(edge)
        counts[edge.source] += 1
    start = next(n.id for n in graph.nodes if n.type == "question")
    finish = next(n.id for n in graph.nodes if n.type == "answer")
    synthesis = next(n.id for n in graph.nodes if n.type == "synthesize")
    if len(edges[synthesis]) != 1 or edges[synthesis][0].target != finish or reverse[finish] != edges[synthesis]:
        raise ValueError("Synthesis must be the only connection into Answer")
    if not edges[start] or reverse[start] or edges[finish]:
        raise ValueError("Question must have outgoing links and Answer must be terminal")
    for node in graph.nodes:
        outgoing = edges[node.id]
        if node.type in CONTROL_TYPES:
            expected = {"enough", "insufficient"} if node.type == "evidence_gate" else {"repeat", "done"}
            if len(outgoing) != 2 or {edge.condition for edge in outgoing} != expected:
                raise ValueError(f"{node.id} needs exactly one link for each branch: {sorted(expected)}")
        elif node.type not in {"answer", "synthesize"} and not outgoing:
            raise ValueError(f"{node.id} has no outgoing connection")
        if node.type in SEARCH_TYPES and node.query_template:
            probe = node.query_template.replace("{question}", "").replace("{iteration}", "")
            if "{" in probe or "}" in probe:
                raise ValueError("Search query supports only {question} and {iteration} placeholders")
    def reachable(origin, forwards):
        seen, queue = set(), [origin]
        while queue:
            cur = queue.pop()
            if cur in seen:
                continue
            seen.add(cur)
            queue.extend(x.target if forwards else x.source for x in (edges[cur] if forwards else reverse[cur]))
        return seen
    if reachable(start, True) != set(by_id) or reachable(finish, False) != set(by_id):
        raise ValueError("Every workflow node must be reachable from Question and lead to Answer")
    # Only a designated loop's 'repeat' edge is allowed to create a cycle.
    # Removing all repeat edges MUST yield a DAG.
    indegree = {node_id: 0 for node_id in by_id}
    for outgoing in edges.values():
        for edge in outgoing:
            if edge.condition != "repeat":
                indegree[edge.target] += 1
    queue = deque(node_id for node_id, value in indegree.items() if value == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for edge in edges[node]:
            if edge.condition == "repeat":
                continue
            indegree[edge.target] -= 1
            if indegree[edge.target] == 0:
                queue.append(edge.target)
    if visited != len(by_id):
        raise ValueError("Cycles must be explicitly controlled by Loop repeat branches")
    # Loop repeat paths must return to their OWN loop node, and done paths must
    # eventually exit it. This also detects meaningless unconnected loop edges.
    for node in graph.nodes:
        if node.type != "loop":
            continue
        repeat = next(x.target for x in edges[node.id] if x.condition == "repeat")
        done = next(x.target for x in edges[node.id] if x.condition == "done")
        if node.id not in reachable(repeat, True):
            raise ValueError(f"Loop {node.id} repeat branch must return to the loop")
        # Done branch must escape without visiting this loop again (otherwise it
        # merely uses the global step limit instead of terminating naturally).
        pending, seen = [done], set()
        exit_found = False
        while pending:
            cur = pending.pop()
            if cur == node.id or cur in seen:
                continue
            if cur == synthesis:
                exit_found = True
                break
            seen.add(cur)
            pending.extend(e.target for e in edges[cur] if e.condition != "repeat")
        if not exit_found:
            raise ValueError(f"Loop {node.id} done branch must lead to synthesis without returning to the loop")


def execute_advanced_workflow(wf: WorkflowDefinition, *, question: str, session_id: str, settings,
                              factory, retriever, retrieval, generation, profile, use_web: bool,
                              use_academic: bool, history: list[dict] | None = None) -> dict:
    """Execute a version-2 graph in bounded synchronized waves.

    Independent search nodes in a wave run concurrently. A merge is a barrier for
    that wave. Synthesis is executed *once* after all active branches settle.
    Decision nodes read accepted, deduplicated evidence from completed searches.
    """
    from aira.agent import (
        SYSTEM_PROMPT,
        _academic_context_results,
        _configured_academic_search,
        _diversified_web_search,
        _recent_chat_messages,
        _response_text,
        _web_sources,
        make_model,
    )
    by_id = {n.id: n for n in wf.nodes}
    edges = defaultdict(list)
    for edge in wf.edges:
        edges[edge.source].append(edge)
    selected_tools = {n.type for n in wf.nodes if n.type in SEARCH_TYPES}
    if not selected_tools.issubset(set(profile.get("enabled_tools") or [])):
        raise ValueError("Workflow requests research tools disabled in its agent profile")

    begin_workflow(wf.model_dump())
    start = perf_counter()
    steps = []
    sources: list[dict] = []
    evidence: list[str] = []
    seen_evidence: set[str] = set()
    iterations: dict[str, int] = defaultdict(int)
    web_used = academic_used = any_failed = False
    frontier = [next(n.id for n in wf.nodes if n.type == "question")]
    stage = 0
    invocations = 0
    synthesis_reached = False
    # Nested repeat controllers restart whenever their enclosing loop begins a
    # fresh iteration. Search each repeat body only until it returns to that
    # controller; this intentionally excludes enclosing loops.
    nested_loops: dict[str, set[str]] = {}
    for item in wf.nodes:
        if item.type != "loop":
            continue
        repeat_target = next(e.target for e in edges[item.id] if e.condition == "repeat")
        found: set[str] = set()
        pending = [repeat_target]
        seen: set[str] = set()
        while pending:
            active = pending.pop()
            if active == item.id or active in seen:
                continue
            seen.add(active)
            if by_id[active].type == "loop":
                found.add(active)
            pending.extend(edge.target for edge in edges[active])
        nested_loops[item.id] = found

    def annotate(node_id: str, status: str, started: float, *, source_count: int = 0,
                 branch: str | None = None, reason: str | None = None, span_id: str | None = None,
                 iteration: int | None = None):
        row = {"node": node_id, "status": status, "stage": stage,
               "start_ms": round((started-start)*1000),
               "latency_ms": round((perf_counter()-started)*1000), "sources": source_count}
        if branch is not None:
            row["branch"] = branch
        if reason is not None:
            row["reason"] = reason
        if span_id:
            row["span_id"] = span_id
        if iteration is not None:
            row["iteration"] = iteration
        steps.append(row)
        # Completed searches already publish from their worker thread, so parallel
        # finishes appear live instead of waiting for the entire wave.
        if by_id[node_id].type not in SEARCH_TYPES or status != "completed":
            record_workflow_event(node_id, status, stage=stage, sources=source_count,
                                  branch=branch, iteration=iteration, latency_ms=row["latency_ms"],
                                  span_id=span_id, reason=reason)

    def search(node_id: str, iteration: int) -> dict:
        node = by_id[node_id]
        started = perf_counter()
        record_workflow_event(node_id, "running", stage=stage, iteration=iteration)
        if node.type == "web_search" and not use_web or node.type == "academic_search" and not use_academic:
            return {"id": node_id, "status": "skipped", "reason": "Disabled in Research Chat settings", "started": started,
                    "rows": [], "texts": [], "span_id": None}
        # Only two literal placeholders supported, no eval / format execution.
        query = node.query_template.replace("{question}", question).replace("{iteration}", str(iteration))[:600]
        record_operation(f"workflow.{node.type}")
        with trace_span(f"workflow.{node.type}", **{"workflow.node": node_id,
                         "workflow.iteration": iteration}) as span:
            if node.type == "local_search":
                hits = retriever.search(query=query, options=retrieval)
                rows = [{"id": h["id"], "path": h.get("path", ""), "version": h.get("version"),
                         "page": h.get("page"), "score": h.get("score"), "source_kind": "pdf"} for h in hits]
                texts = [h.get("text", "")[:1500] for h in hits]
            elif node.type == "academic_search":
                hits = _academic_context_results(_configured_academic_search(settings, factory, query))[:8]
                rows = _web_sources(hits)
                texts = [h.get("excerpt", "")[:1400] for h in hits]
            else:
                hits = _diversified_web_search(settings, factory, query)[:8]
                rows = _web_sources(hits)
                texts = [h.get("excerpt", "")[:1200] for h in hits]
            if span is not None:
                span.set_attribute("workflow.sources", len(rows))
                span_id = f"{span.get_span_context().span_id:016x}" if span.is_recording() else None
            else:
                span_id = None
        # Mark independent searches finished immediately, even if other searches
        # in the same parallel wave are still running. Citation merge remains stable.
        record_workflow_event(node_id, "completed", stage=stage, sources=len(rows), iteration=iteration,
                              latency_ms=round((perf_counter()-started)*1000), span_id=span_id)
        return {"id": node_id, "status": "completed", "started": started,
                "rows": rows, "texts": texts, "span_id": span_id}

    while frontier:
        # Reject runaway workflows before calling external services again.
        if invocations + len(set(frontier)) > wf.max_steps:
            raise RuntimeError(f"Workflow exceeded its safety limit of {wf.max_steps} node executions")
        # Sorted tool IDs, rather than completion order, determine citation numbers.
        present = list(dict.fromkeys(frontier))
        invocations += len(present)
        next_nodes: list[str] = []
        def enqueue(node_id: str, condition: str | None = None):
            for edge in edges[node_id]:
                if condition is None and edge.condition is None or condition is not None and edge.condition == condition:
                    next_nodes.append(edge.target)  # noqa: B023 (used within this iteration)

        search_ids = sorted(n for n in present if by_id[n].type in SEARCH_TYPES)
        outcomes = {}
        if wf.parallel and len(search_ids) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(search_ids)), thread_name_prefix="aira-advanced") as pool:
                futures = {pool.submit(copy_context().run, search, node, max(iterations.values(), default=0)+1): node for node in search_ids}
                for future in as_completed(futures):
                    node_id = futures[future]
                    try:
                        outcomes[node_id] = future.result()
                    except Exception as exc:
                        outcomes[node_id] = {"id": node_id, "status": "failed", "started": perf_counter(),
                                             "reason": f"{type(exc).__name__}: research search failed", "rows": [], "texts": []}
        else:
            for node_id in search_ids:
                try:
                    outcomes[node_id] = search(node_id, max(iterations.values(), default=0)+1)
                except Exception as exc:
                    outcomes[node_id] = {"id": node_id, "status": "failed", "started": perf_counter(),
                                         "reason": f"{type(exc).__name__}: research search failed", "rows": [], "texts": []}
        for node_id in search_ids:
            result = outcomes[node_id]
            accepted = 0
            if result["status"] == "failed":
                any_failed = True
            for row, text in zip(result["rows"], result["texts"]):
                identifier = str(row.get("id", ""))
                signature = f"{identifier}:{row.get('page')}:{text[:80]}"
                if not text.strip() or signature in seen_evidence or len(sources) >= 18:
                    continue
                seen_evidence.add(signature)
                number = len(sources) + 1
                sources.append(row)
                evidence.append(f"[{number}] {row.get('path', '')} page={row.get('page')}:\n{text}")
                accepted += 1
            if result["status"] == "completed":
                web_used |= by_id[node_id].type == "web_search"
                academic_used |= by_id[node_id].type == "academic_search"
            annotate(node_id, result["status"], result["started"], source_count=accepted,
                     reason=result.get("reason"), span_id=result.get("span_id"))
            enqueue(node_id)
        for node_id in present:
            node = by_id[node_id]
            if node.type in SEARCH_TYPES:
                continue
            if node.type == "synthesize":
                synthesis_reached = True
                continue
            if node.type == "answer":
                continue
            started = perf_counter()
            record_workflow_event(node_id, "running", stage=stage)
            if node.type == "question":
                annotate(node_id, "completed", started)
                enqueue(node_id)
            elif node.type == "evidence_gate":
                branch = "enough" if len(sources) >= node.min_sources else "insufficient"
                record_operation("workflow.evidence_gate")
                with trace_span("workflow.evidence_gate", **{"workflow.node": node_id,
                        "workflow.branch": branch, "workflow.sources": len(sources)}) as span:
                    span_id = f"{span.get_span_context().span_id:016x}" if span is not None and span.is_recording() else None
                annotate(node_id, "completed", started, source_count=len(sources), branch=branch, span_id=span_id)
                enqueue(node_id, branch)
            elif node.type == "loop":
                branch = "done" if iterations[node_id] >= node.max_iterations or len(sources) >= node.min_sources else "repeat"
                record_operation("workflow.loop")
                if branch == "repeat":
                    iterations[node_id] += 1
                    # A nested loop's counter belongs to the current outer
                    # iteration, not to the entire research run.
                    for child in nested_loops.get(node_id, set()):
                        iterations[child] = 0
                with trace_span("workflow.loop", **{"workflow.node": node_id, "workflow.branch": branch,
                        "workflow.iteration": iterations[node_id]}) as span:
                    span_id = f"{span.get_span_context().span_id:016x}" if span is not None and span.is_recording() else None
                annotate(node_id, "completed", started, source_count=len(sources), branch=branch,
                         iteration=iterations[node_id], span_id=span_id)
                enqueue(node_id, branch)
        # Use wave barriers: convergence is processed once per wave, although
        # an explicit cycle can legitimately revisit a node in later waves.
        frontier = list(dict.fromkeys(next_nodes))
        stage += 1
    if not synthesis_reached:
        raise RuntimeError("Workflow terminated without reaching synthesis")
    if any_failed and not sources:
        raise RuntimeError("All attempted research searches failed; no evidence available")

    record_operation("workflow.synthesize")
    llm = make_model(settings, generation)
    custom = profile.get("system_instructions", "")
    system = SYSTEM_PROMPT + ("\nAGENT SPECIALIZATION:\n" + custom[:6000] if custom else "")
    content = "\n\n".join(evidence)[:16000] or "No evidence was retrieved."
    user_input = (f"QUESTION: {question}\n\nUNTRUSTED RETRIEVED EVIDENCE:\n{content}\n\n"
                  "Cite only the numbered evidence above. If evidence is missing, say so.")
    started = perf_counter()
    synth_id = next(n.id for n in wf.nodes if n.type == "synthesize")
    record_workflow_event(synth_id, "running", stage=stage)
    with trace_span("workflow.synthesize", **{"workflow.node": next(n.id for n in wf.nodes if n.type == 'synthesize')}) as span:
        response = llm.invoke([("system", system), *_recent_chat_messages(history), ("human", user_input)],
                              **({"config": {"callbacks": callbacks()}} if callbacks() else {}))
        annotate_llm_response(span, response)
        span_id = f"{span.get_span_context().span_id:016x}" if span is not None and span.is_recording() else None
    annotate(synth_id, "completed", started, source_count=len(sources), span_id=span_id)
    answer_id = next(n.id for n in wf.nodes if n.type == 'answer')
    record_workflow_event(answer_id, "running", stage=stage+1)
    annotate(answer_id, "completed", perf_counter(), source_count=len(sources))
    return {"answer": _response_text(response), "model": getattr(llm, "model", generation.model or settings.ollama_model),
            "session_id": session_id, "mode": "visual_workflow", "sources": sources,
            "web_used": web_used, "academic_used": academic_used, "web_pages_fetched": 0,
            "web_fetch_errors": [], "conversation_sources": [], "memory_route": "research",
            "workflow_steps": steps, "workflow_graph": wf.model_dump()}
