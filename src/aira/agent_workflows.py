"""Bounded, validated executable DAGs for visual AIRA research agents.

The optional evidence gate routes from initial parallel retrieval to either direct
synthesis or fallback searches. No arbitrary code, loops, or implicit permissions.
Simple (schema version 1) graphs remain valid and use the same saved schema.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select

from aira.db import AgentWorkflow, ChatTurn, utcnow
from aira.observability import annotate_llm_response, callbacks, trace_span
from aira.options import GenerationOptions, RetrievalOptions
from aira.progress import begin_workflow, record_operation, record_workflow_event

TOOL_NODES = {"local_search", "academic_search", "web_search"}
NODE_TYPES = TOOL_NODES | {"question", "evidence_gate", "synthesize", "answer"}
TOOL_ORDER = ("local_search", "academic_search", "web_search")


class WorkflowNode(BaseModel):
    id: str = Field(max_length=40)
    type: str
    x: float = Field(default=0, ge=-10000, le=10000)
    y: float = Field(default=0, ge=-10000, le=10000)
    min_sources: int = Field(default=1, ge=1, le=18)  # evidence gate / loop target
    max_iterations: int = Field(default=3, ge=1, le=10)  # loop body visit limit
    query_template: str = Field(default="{question}", max_length=400)  # optional repeated search query


class WorkflowEdge(BaseModel):
    source: str
    target: str
    condition: Literal["enough", "insufficient", "repeat", "done"] | None = None


class WorkflowDefinition(BaseModel):
    enabled: bool = True
    parallel: bool = True
    schema_version: Literal[1, 2] = 1
    max_steps: int = Field(default=96, ge=4, le=180)
    nodes: list[WorkflowNode] = Field(min_length=3, max_length=45)
    edges: list[WorkflowEdge] = Field(min_length=2, max_length=95)

    @model_validator(mode="after")
    def validate_graph(self):
        if self.schema_version == 2:
            from aira.advanced_workflows import validate_advanced_graph
            validate_advanced_graph(self)
            return self
        if len(self.nodes) > 7 or len(self.edges) > 12:
            raise ValueError("Legacy workflows support up to seven nodes; use the advanced graph editor")
        ids = [node.id for node in self.nodes]
        if len(set(ids)) != len(ids) or any(node.id != node.type or node.type not in NODE_TYPES for node in self.nodes):
            raise ValueError("Workflow must use unique, known node IDs and types")
        present = set(ids)
        if not {"question", "synthesize", "answer"}.issubset(present):
            raise ValueError("Question, synthesis and answer nodes are required")
        pairs = [(edge.source, edge.target) for edge in self.edges]
        if len(set(pairs)) != len(pairs) or any(not {a, b}.issubset(present) or a == b for a, b in pairs):
            raise ValueError("Workflow contains an invalid or repeated connection")
        if ("synthesize", "answer") not in pairs:
            raise ValueError("Synthesis must connect to answer")
        if any(edge.condition is not None and edge.source != "evidence_gate" for edge in self.edges):
            raise ValueError("Only evidence-gate connections can have conditions")
        tools = present & TOOL_NODES
        edge_map = {(edge.source, edge.target): edge.condition for edge in self.edges}
        if "evidence_gate" not in present:
            allowed = {("question", "synthesize"), ("synthesize", "answer")}
            for tool in tools:
                allowed.update({("question", tool), (tool, "synthesize")})
                if ("question", tool) not in pairs or (tool, "synthesize") not in pairs:
                    raise ValueError(f"{tool} requires Question → Tool → Synthesis")
            if any(pair not in allowed for pair in pairs):
                raise ValueError("Workflow contains an invalid connection")
            if not tools and ("question", "synthesize") not in pairs:
                raise ValueError("Question requires a path to synthesis")
        else:
            if ("question", "synthesize") in pairs:
                raise ValueError("A gated workflow cannot bypass its evidence check")
            primary = {tool for tool in tools if ("question", tool) in pairs}
            fallback = {tool for tool in tools if ("evidence_gate", tool) in pairs}
            if not primary or not fallback or primary & fallback or primary | fallback != tools:
                raise ValueError("Evidence gate requires initial and fallback search nodes")
            allowed = {("synthesize", "answer"), ("evidence_gate", "synthesize")}
            for tool in primary:
                allowed.update({("question", tool), (tool, "evidence_gate")})
                if (tool, "evidence_gate") not in pairs:
                    raise ValueError(f"{tool} requires a connection to the evidence check")
            for tool in fallback:
                allowed.update({("evidence_gate", tool), (tool, "synthesize")})
                if (tool, "synthesize") not in pairs:
                    raise ValueError(f"{tool} requires a connection to synthesis")
            if set(pairs) != allowed:
                raise ValueError("Evidence gate has missing or invalid branch connections")
            if edge_map[("evidence_gate", "synthesize")] != "enough" or any(
                edge_map[("evidence_gate", tool)] != "insufficient" for tool in fallback
            ):
                raise ValueError("Evidence gate connections must specify enough / insufficient conditions")
        return self


def starter_workflow() -> dict:
    return WorkflowDefinition(nodes=[
        WorkflowNode(id="question", type="question", x=20, y=170),
        WorkflowNode(id="local_search", type="local_search", x=310, y=170),
        WorkflowNode(id="synthesize", type="synthesize", x=620, y=170),
        WorkflowNode(id="answer", type="answer", x=930, y=170),
    ], edges=[
        WorkflowEdge(source="question", target="local_search"),
        WorkflowEdge(source="local_search", target="synthesize"),
        WorkflowEdge(source="synthesize", target="answer"),
    ]).model_dump()


def get_workflow(factory, profile_id: str) -> dict | None:
    with factory() as session:
        row = session.get(AgentWorkflow, profile_id)
        return row.definition if row else None


def save_workflow(factory, profile_id: str, definition: WorkflowDefinition, allowed_tools: list[str]) -> dict:
    selected = {n.type for n in definition.nodes if n.type in TOOL_NODES}
    if not selected.issubset(set(allowed_tools)):
        raise ValueError("Graph uses a tool not permitted by the agent profile")
    with factory.begin() as session:
        row = session.get(AgentWorkflow, profile_id)
        if row is None:
            row = AgentWorkflow(profile_id=profile_id, definition=definition.model_dump())
            session.add(row)
        else:
            row.definition = definition.model_dump()
            row.updated_at = utcnow()
    return definition.model_dump()


def list_workflow_runs(factory, profile_id: str, *, limit: int = 15) -> list[dict]:
    """Read past runs recorded with conversations; never poll Phoenix automatically."""
    results = []
    with factory() as session:
        turns = session.scalars(select(ChatTurn).where(ChatTurn.role == "assistant")
                                .order_by(ChatTurn.id.desc()).limit(600)).all()
        for turn in turns:
            extra = turn.extra or {}
            if extra.get("agent_profile_id") != profile_id or not extra.get("workflow_steps"):
                continue
            results.append({"id": turn.id, "conversation_id": turn.conversation_id,
                            "created_at": turn.created_at.isoformat(),
                            "trace_id": extra.get("trace_id"),
                            "steps": extra["workflow_steps"],
                            "workflow": extra.get("workflow_graph")})
            if len(results) >= limit:
                break
    return results


def get_workflow_run(factory, profile_id: str, turn_id: int) -> dict | None:
    """Resolve a saved run's Phoenix trace ID without exposing arbitrary queries."""
    with factory() as session:
        turn = session.get(ChatTurn, turn_id)
        metadata = (turn.extra or {}) if turn is not None else {}
        if turn is None or turn.role != "assistant" or metadata.get("agent_profile_id") != profile_id or not metadata.get("workflow_steps"):
            return None
        return {"trace_id": metadata.get("trace_id"), "steps": metadata["workflow_steps"]}


def execute_workflow(definition: dict, *, question: str, session_id: str, settings, factory,
                     retriever, retrieval: RetrievalOptions, generation: GenerationOptions,
                     profile: dict, use_web: bool, use_academic: bool, history: list[dict] | None = None) -> dict:
    """Run first-pass searches concurrently, optionally gate fallback, then synthesize.

    Evidence and citation numbering are merged in stable tool order, independent of
    which parallel search finishes first. Saved stages correlate to Phoenix span IDs.
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

    wf = WorkflowDefinition.model_validate(definition)
    if wf.schema_version == 2:
        from aira.advanced_workflows import execute_advanced_workflow
        return execute_advanced_workflow(wf, question=question, session_id=session_id,
            settings=settings, factory=factory, retriever=retriever,
            retrieval=retrieval, generation=generation, profile=profile,
            use_web=use_web, use_academic=use_academic, history=history)
    selected = {node.id for node in wf.nodes if node.id in TOOL_NODES}
    if not selected.issubset(set(profile.get("enabled_tools") or [])):
        raise ValueError("This workflow requests tools no longer allowed by its profile")
    gated = any(node.id == "evidence_gate" for node in wf.nodes)
    pairs = {(e.source, e.target) for e in wf.edges}
    primary = [tool for tool in TOOL_ORDER if tool in selected and ("question", tool) in pairs]
    fallback = [tool for tool in TOOL_ORDER if tool in selected and ("evidence_gate", tool) in pairs]
    begin_workflow(wf.model_dump())
    record_workflow_event("question", "running", stage=0)
    record_workflow_event("question", "completed", stage=0)
    start = perf_counter()
    steps: list[dict] = [{"node": "question", "status": "completed", "stage": 0,
                          "start_ms": 0, "latency_ms": 0, "sources": 0}]
    sources: list[dict] = []
    evidence: list[str] = []
    web_used = False
    academic_used = False
    any_error = False

    def _search(tool: str) -> dict:
        started_at = perf_counter()
        record_workflow_event(tool, "running", stage=1 if tool in primary else 3)
        if tool == "web_search" and not use_web or tool == "academic_search" and not use_academic:
            return {"tool": tool, "status": "skipped", "reason": "Per-chat permission is off",
                    "start_ms": round((started_at-start)*1000), "latency_ms": 0, "rows": [], "texts": []}
        record_operation(f"workflow.{tool}")
        with trace_span(f"workflow.{tool}", **{"workflow.node": tool}) as span:
            if tool == "local_search":
                hits = retriever.search(query=question, options=retrieval)
                rows = [{"id": hit["id"], "path": hit["path"], "version": hit.get("version"),
                         "page": hit.get("page"), "score": hit.get("score"), "source_kind": "pdf"} for hit in hits]
                texts = [hit.get("text", "")[:1500] for hit in hits]
            elif tool == "academic_search":
                payload = _configured_academic_search(settings, factory, question)
                hits = _academic_context_results(payload)[:8]
                rows = _web_sources(hits)
                texts = [hit.get("excerpt", "")[:1400] for hit in hits]
            else:
                hits = _diversified_web_search(settings, factory, question)[:8]
                rows = _web_sources(hits)
                texts = [hit.get("excerpt", "")[:1200] for hit in hits]
            span_id = (f"{span.get_span_context().span_id:016x}" if span is not None and span.is_recording() else None)
            if span is not None:
                span.set_attribute("workflow.sources", len(rows))
        record_workflow_event(tool, "completed", stage=1 if tool in primary else 3,
                              sources=len(rows), latency_ms=round((perf_counter()-started_at)*1000), span_id=span_id)
        return {"tool": tool, "status": "completed", "start_ms": round((started_at-start)*1000),
                "latency_ms": round((perf_counter()-started_at)*1000), "span_id": span_id,
                "rows": rows, "texts": texts}

    def _phase(tools: list[str], stage: int) -> None:
        nonlocal web_used, academic_used, any_error
        if not tools:
            return
        outcomes: dict[str, dict] = {}
        if wf.parallel and len(tools) > 1:
            with ThreadPoolExecutor(max_workers=min(len(tools), 3), thread_name_prefix="aira-workflow") as executor:
                futures = {executor.submit(copy_context().run, _search, tool): tool for tool in tools}
                for future in as_completed(futures):
                    tool = futures[future]
                    try:
                        outcomes[tool] = future.result()
                    except Exception as exc:
                        outcomes[tool] = {"tool": tool, "status": "failed", "reason": f"{type(exc).__name__}: search failed",
                                          "start_ms": round((perf_counter()-start)*1000),
                                          "latency_ms": 0, "rows": [], "texts": []}
        else:
            for tool in tools:
                try:
                    outcomes[tool] = _search(tool)
                except Exception as exc:
                    outcomes[tool] = {"tool": tool, "status": "failed", "reason": f"{type(exc).__name__}: search failed",
                                      "start_ms": round((perf_counter()-start)*1000),
                                      "latency_ms": 0, "rows": [], "texts": []}
        for tool in tools:  # preserve deterministic citations independent of thread completion order
            outcome = outcomes[tool]
            if outcome["status"] in {"failed", "skipped"}:
                record_workflow_event(tool, outcome["status"], stage=stage,
                                      latency_ms=outcome.get("latency_ms"), reason=outcome.get("reason"))
            if outcome["status"] == "failed":
                any_error = True
            if outcome["status"] == "completed":
                web_used |= tool == "web_search"
                academic_used |= tool == "academic_search"
            accepted = 0
            for row, content in zip(outcome["rows"], outcome["texts"]):
                if not content.strip() or len(evidence) >= 18:
                    continue
                number = len(sources) + 1
                sources.append(row)
                evidence.append(f"[{number}] {row.get('path', '')} page={row.get('page')}:\n{content}")
                accepted += 1
            steps.append({"node": tool, "status": outcome["status"], "stage": stage,
                          "start_ms": outcome["start_ms"], "latency_ms": outcome["latency_ms"],
                          "sources": accepted, **({"span_id": outcome["span_id"]} if outcome.get("span_id") else {}),
                          **({"reason": outcome["reason"]} if outcome.get("reason") else {})})

    _phase(primary, 1)
    synthesis_stage = 2
    if gated:
        threshold = next(node.min_sources for node in wf.nodes if node.id == "evidence_gate")
        enough = len(sources) >= threshold
        record_operation("workflow.evidence_gate")
        record_workflow_event("evidence_gate", "running", stage=2)
        with trace_span("workflow.evidence_gate", **{"workflow.node": "evidence_gate",
                "workflow.min_sources": threshold, "workflow.evidence_count": len(sources),
                "workflow.branch": "enough" if enough else "insufficient"}) as span:
            span_id = (f"{span.get_span_context().span_id:016x}" if span is not None and span.is_recording() else None)
        steps.append({"node": "evidence_gate", "status": "completed", "stage": 2,
                      "start_ms": round((perf_counter()-start)*1000), "latency_ms": 0,
                      "sources": len(sources), "branch": "enough" if enough else "insufficient",
                      **({"span_id": span_id} if span_id else {})})
        record_workflow_event("evidence_gate", "completed", stage=2,
                              branch="enough" if enough else "insufficient", sources=len(sources),
                              latency_ms=0, span_id=span_id)
        if enough:
            for tool in fallback:
                record_workflow_event(tool, "skipped", stage=3, reason="Evidence threshold met")
                steps.append({"node": tool, "status": "skipped", "reason": "Evidence threshold met",
                              "stage": 3, "start_ms": round((perf_counter()-start)*1000),
                              "latency_ms": 0, "sources": 0})
        else:
            _phase(fallback, 3)
        synthesis_stage = 4
    if any_error and not evidence:
        raise RuntimeError("Every attempted research search failed; no evidence available for synthesis")
    context = "\n\n".join(evidence)[:16000] or "No evidence was retrieved."
    model = make_model(settings, generation)
    custom = profile.get("system_instructions", "")
    system = SYSTEM_PROMPT + ("\nAGENT SPECIALIZATION:\n" + custom[:6000] if custom else "")
    user_input = f"QUESTION: {question}\n\nUNTRUSTED RETRIEVED EVIDENCE:\n{context}\n\nCite only the provided numbered sources; if evidence is missing, say so."
    started = perf_counter()
    record_operation("workflow.synthesize")
    record_workflow_event("synthesize", "running", stage=synthesis_stage)
    with trace_span("workflow.synthesize", **{"workflow.node": "synthesize", "llm.model_name": str(getattr(model, "model", "unknown"))}) as span:
        response = model.invoke([("system", system), *_recent_chat_messages(history), ("human", user_input)],
                                **({"config": {"callbacks": callbacks()}} if callbacks() else {}))
        annotate_llm_response(span, response)
        span_id = (f"{span.get_span_context().span_id:016x}" if span is not None and span.is_recording() else None)
    steps.append({"node": "synthesize", "status": "completed", "stage": synthesis_stage,
                  "start_ms": round((started-start)*1000),
                  "latency_ms": round((perf_counter()-started)*1000), "sources": len(sources),
                  **({"span_id": span_id} if span_id else {})})
    record_workflow_event("synthesize", "completed", stage=synthesis_stage,
                          sources=len(sources), latency_ms=round((perf_counter()-started)*1000), span_id=span_id)
    record_operation("workflow.answer")
    record_workflow_event("answer", "running", stage=synthesis_stage+1)
    record_workflow_event("answer", "completed", stage=synthesis_stage+1, sources=len(sources), latency_ms=0)
    steps.append({"node": "answer", "status": "completed", "stage": synthesis_stage + 1,
                  "start_ms": round((perf_counter()-start)*1000), "latency_ms": 0, "sources": len(sources)})
    return {"answer": _response_text(response), "model": getattr(model, "model", generation.model or settings.ollama_model),
            "session_id": session_id, "mode": "visual_workflow", "sources": sources,
            "web_used": web_used, "academic_used": academic_used, "web_pages_fetched": 0,
            "web_fetch_errors": [], "conversation_sources": [], "memory_route": "research",
            "workflow_steps": steps, "workflow_graph": wf.model_dump()}
