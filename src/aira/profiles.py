"""Persisted agent presets with constrained tool access (not arbitrary executable skills)."""
from sqlalchemy import select

from aira.db import AgentProfile, AgentWorkflow, utcnow

ALLOWED_TOOLS = frozenset({"local_search", "conversation_search", "web_search", "academic_search", "context_read", "context_edit"})
DEFAULT_TOOLS = ["local_search", "conversation_search", "web_search", "academic_search", "context_read", "context_edit"]


def serialize(profile: AgentProfile) -> dict:
    return {"id": profile.id, "name": profile.name, "description": profile.description,
            "system_instructions": profile.system_instructions,
            "enabled_tools": list(profile.enabled_tools),
            "generation": profile.generation,
            "created_at": profile.created_at.isoformat(), "updated_at": profile.updated_at.isoformat()}


def list_profiles(factory) -> list[dict]:
    with factory() as db:
        return [serialize(p) for p in db.scalars(select(AgentProfile).where(AgentProfile.id != "default").order_by(AgentProfile.created_at, AgentProfile.id))]


def get_profile(factory, profile_id: str) -> dict | None:
    with factory() as db:
        found = db.get(AgentProfile, profile_id)
        return serialize(found) if found else None


def create_profile(factory, name: str, description: str, system_instructions: str,
                   enabled_tools: list[str], generation: dict | None = None) -> dict:
    with factory.begin() as db:
        p = AgentProfile(name=name.strip(), description=description.strip(),
                         system_instructions=system_instructions.strip(),
                         enabled_tools=list(enabled_tools), generation=generation)
        db.add(p)
        db.flush()
        return serialize(p)


def update_profile(factory, profile_id: str, **changes) -> dict | None:
    with factory.begin() as db:
        p = db.get(AgentProfile, profile_id)
        if p is None:
            return None
        for key, value in changes.items():
            if value is not None:
                setattr(p, key, value.strip() if isinstance(value, str)
                        else list(value) if key == "enabled_tools" else value)
        p.updated_at = utcnow()
        db.flush()
        return serialize(p)


def save_default_profile(factory, **changes) -> dict:
    """Persist a customization of the built-in agent without changing its stable ID.

    The original settings remain in effect until the user edits the profile.
    """
    with factory.begin() as db:
        profile = db.get(AgentProfile, "default")
        if profile is None:
            profile = AgentProfile(id="default", name="Research assistant",
                                   description="Built-in evidence-grounded researcher",
                                   system_instructions="", enabled_tools=list(DEFAULT_TOOLS))
            db.add(profile)
        for key, value in changes.items():
            if value is not None:
                setattr(profile, key, value.strip() if isinstance(value, str)
                        else list(value) if key == "enabled_tools" else value)
        profile.updated_at = utcnow()
        db.flush()
        return {**serialize(profile), "builtin": True}


def delete_profile(factory, profile_id: str) -> bool:
    with factory.begin() as db:
        p = db.get(AgentProfile, profile_id)
        if p is None:
            return False
        workflow = db.get(AgentWorkflow, profile_id)
        if workflow is not None:
            db.delete(workflow)
        db.delete(p)
        return True
