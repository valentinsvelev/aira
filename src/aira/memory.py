"""Two independent, provenance-preserving context stores.

Conversation memory is scoped to a single chat and NEVER promoted to verified
research evidence. Research evidence continues to use the existing paper retriever.
Routing is deterministic in RAG mode; the Agentic RAG model can also choose tools.
"""
from __future__ import annotations

import re
from typing import Literal

from sqlalchemy import func, or_, select

from aira.db import ChatConversation, ChatTurn
from aira.observability import trace_span

MemoryRoute = Literal["auto", "conversation", "research", "both", "neither"]

# Questions referring to earlier dialogue benefit from the immutable turn archive.
_CONVERSATION = re.compile(
    r"\b(we (?:said|decided|discussed|agreed|mentioned|concluded|talked)|"
    r"you (?:said|told|mentioned|recommended)|i (?:said|asked|mentioned)|"
    r"earlier|previous(?:ly)?|last time|yesterday|our (?:chat|discussion|conversation)|"
    r"remember|recall|what did (?:we|i|you)|which paper (?:did we|we)|"
    r"(?:discussed|mentioned) (?:before|earlier|yesterday))\b", re.I
)
_RESEARCH = re.compile(
    r"\b(papers?|study|studies|source|evidence|research|experiment|benchmark|"
    r"article|citation|pdf|arxiv|literature|authors?|method|quantization|"
    r"retriev\w*|qlora|lora|nf4|limitations?|performance|faster)\b", re.I
)
_SOCIAL = re.compile(r"^(?:hi|hello|hey|thanks|thank you|ok|okay|good morning)[!.\s]*$", re.I)
_TERMS = re.compile(r"[a-z0-9][a-z0-9_-]{1,}", re.I)
_STOP = frozenset({"what", "where", "which", "when", "how", "why", "does", "did", "the", "a",
                   "an", "and", "or", "it", "its", "we", "you", "our", "that", "this", "about",
                   "from", "with", "for", "into", "were", "was", "are", "have", "has", "been",
                   "could", "would", "should", "please", "earlier", "previously", "discussed",
                   "mentioned", "said", "paper", "papers", "conversation"})


def route_context(question: str, requested: MemoryRoute = "auto", history: list[dict] | None = None) -> str:
    """Pick an initial store without an extra inference call.

    This is a cheap baseline, NOT an LLM router. Agentic RAG can revise the
    decision by calling independent conversation and paper retrieval tools.
    """
    if requested != "auto":
        if requested not in ("conversation", "research", "both", "neither"):
            raise ValueError("Unknown memory route")
        return requested
    question = question.strip()
    if _SOCIAL.fullmatch(question):
        return "neither"
    convo = bool(_CONVERSATION.search(question))
    research = bool(_RESEARCH.search(question))
    # "the paper we discussed yesterday" needs both kinds of evidence.
    if convo:
        # Remembering a past decision ABOUT a paper does not imply a new
        # scientific retrieval. A comparison/source-evidence request does.
        if research and re.search(r"\b(compare|contrast|versus|vs\.?|against|"
                                  r"evidence|according to|source)\b", question, re.I):
            return "both"
        return "conversation"
    return "research" if research else "neither"  # General knowledge does not imply local PDFs.


def _keywords(query: str) -> list[str]:
    terms = [word.lower() for word in _TERMS.findall(query) if word.lower() not in _STOP]
    return list(dict.fromkeys(terms))[:8]


def _to_hit(turn: ChatTurn, *, score: float = 0.0) -> dict:
    return {"turn_id": turn.id, "conversation_id": turn.conversation_id,
            "role": turn.role, "text": turn.content, "created_at": turn.created_at.isoformat(),
            "score": round(float(score), 6), "source_kind": "conversation_turn"}


def search_conversation(factory, conversation_id: str, query: str, *, top_k: int = 5,
                        max_chars: int = 900) -> list[dict]:
    """Full-text search inside ONE chat; never searches other users' conversations.

    PostgreSQL uses full-text ranking; SQLite tests use portable indexed-ID plus
    escaped LIKE terms. Results are immutable chat turns, not verified paper facts.
    """
    if not 1 <= top_k <= 12:
        raise ValueError("top_k must be 1..12")
    words = _keywords(query)
    if not words:
        return []
    with trace_span("memory.conversation.search", top_k=top_k) as span:
        with factory() as db:
            if db.get(ChatConversation, conversation_id) is None:
                return []
            q = select(ChatTurn).where(ChatTurn.conversation_id == conversation_id)
            if db.bind.dialect.name == "postgresql":
                vector = func.to_tsvector("simple", ChatTurn.content)
                # OR semantics for conversational queries: find the old message
                # even if not every term from the current question is present.
                matches = [vector.op("@@")(func.plainto_tsquery("simple", w)) for w in words]
                rank = sum(func.ts_rank(vector, func.plainto_tsquery("simple", w)) for w in words)
                rows = db.execute(q.where(or_(*matches)).order_by(rank.desc(), ChatTurn.id.desc())
                                  .limit(top_k)).scalars().all()
                hits = [_to_hit(row) for row in rows]
            else:
                # LIKE wildcards are escaped; conversation_id is always mandatory.
                filters = [ChatTurn.content.ilike("%" + w.replace("%", r"\%").replace("_", r"\_") + "%", escape="\\")
                           for w in words]
                rows = db.scalars(q.where(or_(*filters)).order_by(ChatTurn.id.desc()).limit(250)).all()
                scored = [(sum(w in row.content.lower() for w in words), row) for row in rows]
                scored.sort(key=lambda pair: (-pair[0], -pair[1].id))
                hits = [_to_hit(row, score=score) for score, row in scored[:top_k]]
        if not hits:
            # A vague reference like "the paper we discussed yesterday" has no
            # literal overlap with a paper title. Surface recent turns as an
            # explicitly low-confidence fallback instead of inventing matches.
            with factory() as db:
                rows = db.scalars(select(ChatTurn).where(
                    ChatTurn.conversation_id == conversation_id)
                    .order_by(ChatTurn.id.desc()).limit(min(top_k, 4))).all()
                hits = [_to_hit(row, score=0.0) for row in rows]
        if span:
            span.set_attribute("memory.results_count", len(hits))
        return [{**hit, "text": hit["text"][:max_chars]} for hit in hits]


def read_conversation_turns(factory, conversation_id: str, turn_ids: list[int],
                            *, max_chars: int = 2400) -> list[dict]:
    """Read exact archived turns, preserving roles and enforcing chat isolation."""
    ids = list(dict.fromkeys(turn_ids))[:12]
    if not ids or any(not isinstance(t, int) or isinstance(t, bool) or t < 1 for t in ids):
        return []
    with factory() as db:
        rows = db.scalars(select(ChatTurn).where(ChatTurn.conversation_id == conversation_id,
                                                 ChatTurn.id.in_(ids))).all()
        lookup = {row.id: row for row in rows}
        return [{**_to_hit(lookup[turn_id]), "text": lookup[turn_id].content[:max_chars]}
                for turn_id in ids if turn_id in lookup]


def render_conversation_evidence(hits: list[dict], max_chars: int = 3200) -> str:
    """Label archived claims distinctly: a remembered statement is not a paper citation."""
    if not hits:
        return "(No matching earlier conversation turns found.)"
    blocks = [f"[conversation-turn:{hit['turn_id']}] {hit['role']} on {hit['created_at']}:\n"
              + hit['text'][:900] for hit in hits]
    return "\n\n".join(blocks)[:max_chars]
