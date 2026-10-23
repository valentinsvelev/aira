"""Durable, paginated conversation history shared by Next.js and FastAPI.

Messages are stored once in PostgreSQL (or SQLite during tests). Retrieval is
cursor-paginated; only a bounded window is reintroduced into the model prompt.
Existing evidence and context notes retain their independent source lineage.
"""
from sqlalchemy import select

from aira.chat_titles import fallback_title
from aira.db import ChatConversation, ChatTurn, utcnow


def serialize_conversation(chat: ChatConversation) -> dict:
    return {"id": chat.id, "title": chat.title, "created_at": chat.created_at.isoformat(),
            "updated_at": chat.updated_at.isoformat()}


def serialize_turn(turn: ChatTurn) -> dict:
    return {"id": turn.id, "conversation_id": turn.conversation_id, "role": turn.role,
            "content": turn.content, "mode": turn.mode, "model": turn.model,
            "sources": turn.sources or [], "metadata": turn.extra or {},
            "created_at": turn.created_at.isoformat()}


def list_conversations(factory, *, limit: int = 50, offset: int = 0) -> dict:
    with factory() as db:
        rows = db.scalars(select(ChatConversation)
                          .order_by(ChatConversation.updated_at.desc(), ChatConversation.id.desc())
                          .offset(offset).limit(limit + 1)).all()
        return {"conversations": [serialize_conversation(r) for r in rows[:limit]],
                "has_more": len(rows) > limit}


def create_conversation(factory, title: str = "New chat") -> dict:
    with factory.begin() as db:
        row = ChatConversation(title=title.strip() or "New chat")
        db.add(row)
        db.flush()
        result = serialize_conversation(row)
    return result


def get_conversation(factory, id: str) -> dict | None:
    with factory() as db:
        row = db.get(ChatConversation, id)
        return serialize_conversation(row) if row is not None else None


def rename_conversation(factory, id: str, title: str) -> dict | None:
    with factory.begin() as db:
        row = db.get(ChatConversation, id)
        if row is None:
            return None
        row.title = title.strip()
        row.updated_at = utcnow()
        db.flush()
        result = serialize_conversation(row)
    return result


def delete_conversation(factory, id: str) -> bool:
    """Delete conversation and messages, but preserve immutable evidence/audit records."""
    with factory.begin() as db:
        row = db.get(ChatConversation, id)
        if row is None:
            return False
        db.delete(row)
    return True


def get_messages(factory, conversation_id: str, *, limit: int = 50,
                 before: int | None = None) -> dict | None:
    with factory() as db:
        if db.get(ChatConversation, conversation_id) is None:
            return None
        query = select(ChatTurn).where(ChatTurn.conversation_id == conversation_id)
        if before is not None:
            query = query.where(ChatTurn.id < before)
        rows = db.scalars(query.order_by(ChatTurn.id.desc()).limit(limit + 1)).all()
        has_more = len(rows) > limit
        page = rows[:limit]
        return {"messages": [serialize_turn(r) for r in reversed(page)],
                "has_more": has_more,
                "next_cursor": page[-1].id if has_more and page else None}


def recent_history(factory, conversation_id: str, limit: int = 8) -> list[dict]:
    """Return a small tail, not the entire chat transcript, for each model call."""
    with factory() as db:
        return [{"role": row.role, "content": row.content} for row in reversed(
            db.scalars(select(ChatTurn)
                       .where(ChatTurn.conversation_id == conversation_id)
                       .order_by(ChatTurn.id.desc()).limit(limit)).all())]


def save_exchange(factory, conversation_id: str, question: str, result: dict,
                  mode: str, parameters: dict | None = None) -> None:
    """Persist both turns atomically; failure never leaves an orphaned user turn."""
    with factory.begin() as db:
        chat = db.get(ChatConversation, conversation_id)
        if chat is None:
            raise LookupError("Conversation does not exist")
        if chat.title == "New chat":
            chat.title = fallback_title(question)
        db.add(ChatTurn(conversation_id=conversation_id, role="user", content=question))
        # Keep citations and compact request metadata, not duplicate PDF chunks or prompts.
        sources = result.get("sources") or []
        db.add(ChatTurn(conversation_id=conversation_id, role="assistant",
                        content=result["answer"], mode=mode,
                        model=result.get("model"), sources=sources,
                        extra={"web_used": result.get("web_used", False),
                               "trace_id": result.get("trace_id"),
                               "web_pages_fetched": result.get("web_pages_fetched", 0),
                               "academic_used": result.get("academic_used", False),
                               "agent_profile_id": result.get("agent_profile_id", "default"),
                               "agent_profile_name": result.get("agent_profile_name", "Research assistant"),
                               "web_fetch_errors": result.get("web_fetch_errors") or [],
                               "parameters": parameters or {},
                               "memory_route": result.get("memory_route", "research"),
                               "conversation_sources": result.get("conversation_sources", []),
                               "workflow_steps": result.get("workflow_steps", [])[:182],
                               "workflow_graph": result.get("workflow_graph")}))
        chat.updated_at = utcnow()
