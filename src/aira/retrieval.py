"""LlamaIndex BM25 + PostgreSQL pgvector dense search + RRF hybrid search."""
from sqlalchemy import select

from aira.db import Document, DocumentVersion, EvidenceChunk
from aira.observability import annotate_retrieval, trace_span
from aira.options import RetrievalOptions
from aira.vector_store import EmbeddingIndex


class LocalRetriever:
    def __init__(self, session_factory, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
        self.session_factory = session_factory
        self.index = EmbeddingIndex(session_factory, embedding_model)
        self._signature = None
        self._bm25 = None
        self._active = {}

    def _refresh(self):
        with self.session_factory() as db:
            active_versions = tuple(db.scalars(select(DocumentVersion.id)
                .join(Document, DocumentVersion.document_id == Document.id)
                .where(Document.active.is_(True), DocumentVersion.active.is_(True))
                .order_by(DocumentVersion.id)).all())
            if active_versions == self._signature:
                return
            rows = db.execute(
                select(EvidenceChunk, Document.path, DocumentVersion.version)
                .join(DocumentVersion, EvidenceChunk.document_version_id == DocumentVersion.id)
                .join(Document, DocumentVersion.document_id == Document.id)
                .where(Document.active.is_(True), DocumentVersion.active.is_(True))
                .order_by(EvidenceChunk.id)
            ).all()
        self._signature = active_versions
        self._active = {chunk.id: {
            'id': chunk.id, 'text': chunk.text, 'source_kind': chunk.source_kind,
            'path': path, 'version': version, 'page': chunk.page,
        } for chunk, path, version in rows}
        if not rows:
            self._bm25 = None
            return
        try:
            from llama_index.core.schema import TextNode
            from llama_index.retrievers.bm25 import BM25Retriever
        except ImportError as exc:
            raise RuntimeError('Install project dependencies for LlamaIndex BM25 retrieval') from exc
        nodes = [TextNode(id_=record['id'], text=record['text'],
                          metadata={'path': record['path'], 'page': record['page']})
                 for record in self._active.values()]
        self._bm25 = BM25Retriever.from_defaults(
            nodes=nodes, similarity_top_k=min(100, len(nodes)))

    def _bm25_search(self, query: str, n: int) -> list[dict]:
        self._refresh()
        if self._bm25 is None:
            return []
        matches = self._bm25.retrieve(query)
        return [{**self._active[m.node.node_id], 'score': float(m.score or 0),
                 'text': self._active[m.node.node_id]['text'][:1800],
                 'retrieval_method': 'sparse'}
                for m in matches[:n] if m.node.node_id in self._active]

    def search(self, query: str, top_k: int = 5,
               options: RetrievalOptions | None = None) -> list[dict]:
        with trace_span("retrieval.search", strategy=(options.mode if options else "sparse"),
                        top_k=(options.top_k if options else top_k),
                        embedding_model=(options.embedding_model if options else "sparse"),
                        reranker=(options.reranker if options else "none")) as span:
            hits = self._search(query, top_k, options)
            annotate_retrieval(span, hits)
            if span is not None:
                from aira.observability import capture_content
                span.set_attribute("retrieval.strategy", options.mode if options else "sparse")
                span.set_attribute("retriever.top_k", options.top_k if options else top_k)
                span.set_attribute("retrieval.reranker", options.reranker if options else "none")
                span.set_attribute("retrieval.returned_count", len(hits))
                if capture_content():
                    span.set_attribute("input.value", query[:1800])
                    span.set_attribute("output.value", f"{len(hits)} ranked passages (full passages in retrieval.documents)")
            return hits

    def _search(self, query: str, top_k: int = 5,
               options: RetrievalOptions | None = None) -> list[dict]:
        if not query.strip():
            return []
        opts = options or RetrievalOptions(top_k=min(max(top_k, 1), 12))
        index = (self.index if opts.embedding_model == self.index.model_name
                 else EmbeddingIndex(self.session_factory, opts.embedding_model))
        if opts.mode == 'sparse':
            candidates = self._bm25_search(query, opts.bm25_candidates)
        elif opts.mode == 'dense':
            candidates = index.dense_search(query, opts.dense_candidates, opts.min_similarity)
        else:
            lexical = self._bm25_search(query, opts.bm25_candidates)
            semantic = index.dense_search(query, opts.dense_candidates, opts.min_similarity)
            # Reciprocal Rank Fusion avoids mixing incomparable raw BM25/cosine scores.
            combined = {}
            for stream in (lexical, semantic):
                for rank, hit in enumerate(stream, 1):
                    entry = combined.setdefault(hit['id'], {**hit, 'score': 0.0,
                                                            'retrieval_method': 'hybrid'})
                    entry['score'] += 1.0 / (opts.rrf_k + rank)
            candidates = sorted(combined.values(), key=lambda h: (-h['score'], h['id']))
        if opts.reranker == 'none':
            return candidates[:opts.top_k]
        from aira.reranking import rerank_hits
        return rerank_hits(query, candidates[:opts.rerank_candidates], opts.reranker, opts.top_k)
