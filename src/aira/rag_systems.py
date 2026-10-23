"""Persisted RAG system configurations. No external service required."""
from sqlalchemy import select

from aira.db import RAGSystemProfile, utcnow
from aira.options import GenerationOptions, RetrievalOptions


def serialize(p):
    return {"id": p.id, "name": p.name, "description": p.description,
            "system_instructions": p.system_instructions,
            "generation": p.generation, "retrieval": RetrievalOptions.model_validate(p.retrieval).model_dump()}


def default_system():
    return {"id": "default", "name": "Standard RAG", "description": "Built-in evidence-grounded retrieval",
            "system_instructions": "", "retrieval": RetrievalOptions().model_dump(),
            "generation": GenerationOptions().model_dump(), "builtin": True}


def default_system_body():
    """Fields stored in the existing profile table; no database migration."""
    return {key: val for key, val in default_system().items() if key not in ("id", "builtin")}


def list_systems(factory):
    with factory() as db:
        custom_default = db.get(RAGSystemProfile, "default")
        built_in = {**serialize(custom_default), "builtin": True} if custom_default else default_system()
        others = db.scalars(select(RAGSystemProfile).where(RAGSystemProfile.id != "default")
                            .order_by(RAGSystemProfile.created_at))
        return [built_in, *(serialize(p) for p in others)]


def get_system(factory, profile_id):
    with factory() as db:
        p = db.get(RAGSystemProfile, profile_id)
        if p:
            data = serialize(p)
            return {**data, "builtin": True} if profile_id == "default" else data
        return default_system() if profile_id == "default" else None


def create_system(factory, data):
    with factory.begin() as db:
        p = RAGSystemProfile(**data)
        db.add(p); db.flush()
        return serialize(p)


def update_system(factory, profile_id, data):
    with factory.begin() as db:
        p = db.get(RAGSystemProfile, profile_id)
        if p is None and profile_id == "default":
            # A stable built-in override, same pattern as the default research agent.
            p = RAGSystemProfile(id="default", **default_system_body())
            db.add(p)
        if p is None: return None
        for key, value in data.items():
            setattr(p, key, value)
        p.updated_at = utcnow()
        db.flush()
        output = serialize(p)
        return {**output, "builtin": True} if profile_id == "default" else output


def delete_system(factory, profile_id):
    with factory.begin() as db:
        p = db.get(RAGSystemProfile, profile_id)
        if p is None: return False
        db.delete(p)
        return True
