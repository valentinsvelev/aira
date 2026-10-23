"""Version-safe pgvector indexes for 384D MiniLM and 1024D GTE-large.

The existing 384D table stays intact. Model identity and index dimension are
bound together; embeddings of different dimensions are never mixed. This is
incremental: reindexing one model never touches the other one's rows.
"""
from __future__ import annotations

import json
from functools import lru_cache

from sqlalchemy import text

DIMENSIONS = 384  # Legacy integration-test compatibility
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
GTE_LARGE = "Alibaba-NLP/gte-large-en-v1.5"


def model_spec(name: str) -> tuple[str, int]:
    # Any other configured model name is stored in the 384-dimensional table.
    if name == GTE_LARGE:
        return "chunk_embeddings_1024", 1024
    return "chunk_embeddings", 384


@lru_cache(maxsize=2)
def load_embedding_model(name: str):
    from sentence_transformers import SentenceTransformer
    if name == GTE_LARGE:
        # Required by the model card. Only use for this known model;
        # the Hugging Face implementation runs custom model code.
        return SentenceTransformer(name, trust_remote_code=True)
    return SentenceTransformer(name)


def embed_texts(name: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = load_embedding_model(name)
    rows = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
    expected = model_spec(name)[1]
    if any(len(row) != expected for row in rows):
        raise ValueError(f"{name} must produce {expected}-dimensional embeddings")
    return rows


def pgvector_schema(engine):
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as con:
        con.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        for table, dim in (("chunk_embeddings", 384), ("chunk_embeddings_1024", 1024)):
            con.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                  chunk_id varchar(64) NOT NULL REFERENCES evidence_chunks(id),
                  model_name text NOT NULL,
                  embedding vector({dim}) NOT NULL,
                  created_at timestamptz NOT NULL DEFAULT now(),
                  PRIMARY KEY(chunk_id, model_name)
                )
            """))
            con.execute(text(f"""
                CREATE INDEX IF NOT EXISTS ix_{table}_hnsw_cosine
                ON {table} USING hnsw (embedding vector_cosine_ops)
            """))


class EmbeddingIndex:
    def __init__(self, session_factory, model_name: str):
        self.factory = session_factory
        self.model_name = model_name
        self.table, self.dimensions = model_spec(model_name)

    def _require_postgres(self):
        if self.factory.kw["bind"].dialect.name != "postgresql":
            raise RuntimeError("Dense retrieval requires PostgreSQL with pgvector")

    def stats(self) -> dict:
        if self.factory.kw["bind"].dialect.name != "postgresql":
            return {"enabled": False, "indexed": 0, "missing": None,
                    "model": self.model_name, "dimensions": self.dimensions}
        with self.factory() as db:
            indexed = db.execute(text(f"""
                SELECT count(*) FROM {self.table} WHERE model_name=:model
            """), {"model": self.model_name}).scalar_one()
            missing = db.execute(text(f"""
                SELECT count(*) FROM evidence_chunks c
                JOIN document_versions v ON v.id=c.document_version_id
                JOIN documents d ON d.id=v.document_id
                LEFT JOIN {self.table} e ON e.chunk_id=c.id AND e.model_name=:model
                WHERE d.active AND v.active AND e.chunk_id IS NULL
            """), {"model": self.model_name}).scalar_one()
        return {"enabled": True, "indexed": indexed, "missing": missing,
                "model": self.model_name, "dimensions": self.dimensions}

    def sync(self, batch_size: int = 32) -> dict:
        self._require_postgres()
        if not 1 <= batch_size <= 128:
            raise ValueError("batch_size must be 1..128")
        processed = 0
        while True:
            with self.factory() as db:
                rows = db.execute(text(f"""
                    SELECT c.id, c.text FROM evidence_chunks c
                    JOIN document_versions v ON v.id=c.document_version_id
                    JOIN documents d ON d.id=v.document_id
                    LEFT JOIN {self.table} e ON e.chunk_id=c.id AND e.model_name=:model
                    WHERE d.active AND v.active AND e.chunk_id IS NULL
                    ORDER BY c.id LIMIT :batch
                """), {"model": self.model_name, "batch": batch_size}).all()
            if not rows:
                break
            embeddings = embed_texts(self.model_name, [row.text for row in rows])
            with self.factory.begin() as db:
                for row, embedding in zip(rows, embeddings):
                    db.execute(text(f"""
                        INSERT INTO {self.table} (chunk_id, model_name, embedding)
                        VALUES (:id, :model, CAST(:embedding AS vector))
                        ON CONFLICT (chunk_id, model_name) DO NOTHING
                    """), {"id": row.id, "model": self.model_name,
                            "embedding": json.dumps(embedding)})
            processed += len(rows)
        return {"embedded": processed, **self.stats()}

    def dense_search(self, query: str, limit: int, min_similarity: float = 0.0) -> list[dict]:
        self._require_postgres()
        if self.stats()["indexed"] == 0:
            raise RuntimeError(f"No vectors indexed for {self.model_name}; run index-embeddings --model first")
        query_embedding = json.dumps(embed_texts(self.model_name, [query])[0])
        with self.factory() as db:
            rows = db.execute(text(f"""
                SELECT c.id, c.text, c.source_kind, c.page, d.path, v.version,
                       1 - (e.embedding <=> CAST(:query_vec AS vector)) AS similarity
                FROM {self.table} e
                JOIN evidence_chunks c ON c.id=e.chunk_id
                JOIN document_versions v ON v.id=c.document_version_id
                JOIN documents d ON d.id=v.document_id
                WHERE e.model_name=:model AND v.active AND d.active
                  AND 1 - (e.embedding <=> CAST(:query_vec AS vector)) >= :min_similarity
                ORDER BY e.embedding <=> CAST(:query_vec AS vector), c.id
                LIMIT :top_k
            """), {"query_vec": query_embedding, "model": self.model_name,
                    "top_k": limit, "min_similarity": min_similarity}).mappings().all()
        return [{**dict(r), "text": r["text"][:1800], "score": float(r["similarity"]),
                 "retrieval_method": "dense"} for r in rows]
