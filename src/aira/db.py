"""Transactional metadata and immutable evidence. Use Alembic before production migrations."""
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
    inspect,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return str(uuid4())


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    path: Mapped[str] = mapped_column(String(1000), nullable=False, unique=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    versions: Mapped[list["DocumentVersion"]] = relationship(back_populates="document")


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    document: Mapped[Document] = relationship(back_populates="versions")


class EvidenceChunk(Base):
    __tablename__ = "evidence_chunks"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_kind: Mapped[str] = mapped_column(String(24), nullable=False)  # pdf|text|web_snippet
    document_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_versions.id"), nullable=True, index=True
    )
    source_url: Mapped[str | None] = mapped_column(String(2000))
    page: Mapped[int | None] = mapped_column(Integer)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    detail: Mapped[str] = mapped_column(Text, default="", nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ResearchSession(Base):
    __tablename__ = "research_sessions"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ContextNote(Base):
    __tablename__ = "context_notes"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("research_sessions.id"), index=True)
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_id: Mapped[str] = mapped_column(ForeignKey("evidence_chunks.id"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ContextEvent(Base):
    __tablename__ = "context_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(String(100), index=True)
    operation: Mapped[str] = mapped_column(String(24))
    status: Mapped[str] = mapped_column(String(24))
    note_id: Mapped[str | None] = mapped_column(String(36))
    old_claim: Mapped[str | None] = mapped_column(Text)
    new_claim: Mapped[str | None] = mapped_column(Text)
    evidence_id: Mapped[str | None] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(Text, default="")
    previous_revision: Mapped[int] = mapped_column(Integer)
    new_revision: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)



class ChatConversation(Base):
    """One user-created research conversation. Its ID also scopes context notes."""
    __tablename__ = "chat_conversations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(120), nullable=False, default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    messages: Mapped[list["ChatTurn"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", passive_deletes=True)


class ChatTurn(Base):
    """Small, append-only turns. Store source metadata, never duplicate evidence text."""
    __tablename__ = "chat_turns"
    __table_args__ = (Index("ix_chat_turns_conversation_cursor", "conversation_id", "id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("chat_conversations.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(12), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str | None] = mapped_column(String(12))
    model: Mapped[str | None] = mapped_column(String(120))
    sources: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    extra: Mapped[dict] = mapped_column("metadata", JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    conversation: Mapped[ChatConversation] = relationship(back_populates="messages")


class SchemaMigration(Base):
    """One-time additive application data upgrades; never repeated on restarts."""
    __tablename__ = "app_schema_migrations"
    name: Mapped[str] = mapped_column(String(100), primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AgentProfile(Base):
    """Locally configured instruction presets. Tool access is enforced server-side."""
    __tablename__ = "agent_profiles"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    system_instructions: Mapped[str] = mapped_column(Text, nullable=False, default="")
    enabled_tools: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    generation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

class AgentWorkflow(Base):
    """Optional executable graph; separate from legacy agent profile and browser layout."""
    __tablename__ = "agent_workflows"
    profile_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    definition: Mapped[dict] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RAGSystemProfile(Base):
    """Saved local RAG configurations; safe additive table creation."""
    __tablename__ = "rag_system_profiles"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(String(500), default="", nullable=False)
    system_instructions: Mapped[str] = mapped_column(Text, default="", nullable=False)
    retrieval: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    generation: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def make_session_factory(database_url: str):
    if database_url.startswith("sqlite:///") and ":memory:" not in database_url:
        db_file = Path(database_url.replace("sqlite:///", "", 1))
        db_file.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False} if database_url.startswith("sqlite") else {},
        pool_pre_ping=True,
    )
    if database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def enable_foreign_keys(connection, record):
            connection.execute("PRAGMA foreign_keys=ON")
    from aira import literature  # noqa: F401 — register review tables before create_all
    # Opt-in deployment mode: migrations are applied explicitly before API startup.
    # Development and the existing test suite retain backwards-compatible auto bootstrap.
    if os.getenv("AIRA_REQUIRE_MIGRATIONS", "").lower() in {"1", "true", "yes"}:
        try:
            with engine.connect() as connection:
                revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
        except Exception as exc:
            engine.dispose()
            raise RuntimeError("Database has no Alembic revision; run 'make db-migrate' first") from exc
        if revision != "0001_aira_baseline":
            engine.dispose()
            raise RuntimeError(f"Database revision {revision!r} is not supported; run Alembic migrations")
        return sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)
    # Safe, additive migration for databases created before agent generation profiles.
    # Keep the existing table and profiles; do not drop or recreate user records.
    if "agent_profiles" in inspect(engine).get_table_names():
        columns = {column["name"] for column in inspect(engine).get_columns("agent_profiles")}
        if "generation" not in columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE agent_profiles ADD COLUMN generation JSON"))
    # Profiles created before conversation_search existed get it enabled
    # exactly once (the marker name below is stored in the database), then retain every subsequent user permission edit.
    with sessionmaker(bind=engine).begin() as db:
        if db.get(SchemaMigration, "v1.5_conversation_tool") is None:
            for profile in db.query(AgentProfile).all():
                if "conversation_search" not in (profile.enabled_tools or []):
                    profile.enabled_tools = [*(profile.enabled_tools or []), "conversation_search"]
            db.add(SchemaMigration(name="v1.5_conversation_tool"))
    if engine.dialect.name == "postgresql":
        # Make archived conversation searching efficient without copying chat text
        # into the scientific-paper vector index.
        with engine.begin() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chat_turns_fts ON chat_turns "
                              "USING GIN (to_tsvector('simple', content))"))
        from aira.vector_store import pgvector_schema
        pgvector_schema(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
