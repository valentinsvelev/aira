"""Research Agent command-line workflow."""
import argparse
import json

from aira.agent import answer_with_local_context, run_agent
from aira.config import Settings, make_context_manager
from aira.conversations import recent_history
from aira.db import make_session_factory
from aira.ingestion import Ingestor
from aira.memory import search_conversation
from aira.options import GenerationOptions, RetrievalOptions
from aira.retrieval import LocalRetriever
from aira.semantic import SemanticChecker
from aira.vector_store import EmbeddingIndex


def main():
    parser = argparse.ArgumentParser(prog="aira")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Synchronize local paper directory")
    search = sub.add_parser("search", help="Sparse/dense/hybrid retrieval")
    web_search = sub.add_parser("web-search", help="Live web search, capture immutable snippet evidence")
    web_search.add_argument("query")
    web_search.add_argument("--top-k", type=int, default=4)
    academic = sub.add_parser("academic-search", help="Query arXiv/OpenAlex (free academic APIs)")
    academic.add_argument("query")
    academic.add_argument("--source", action="append", choices=["arxiv", "openalex"], help="May be specified twice")
    academic.add_argument("--top-k", type=int, default=5)
    academic.add_argument("--year-from", type=int)
    scholar = sub.add_parser("scholar-url", help="Create a browser link; does not scrape Google Scholar")
    scholar.add_argument("query")
    import_pdf = sub.add_parser("import-paper", help="User-approved arXiv/ACL direct PDF download")
    import_pdf.add_argument("url")
    index_cmd = sub.add_parser("index-embeddings", help="Backfill pgvector embeddings, restartable")
    index_cmd.add_argument("--model", default=None, choices=["sentence-transformers/all-MiniLM-L6-v2", "Alibaba-NLP/gte-large-en-v1.5"])
    status_cmd = sub.add_parser("retrieval-status", help="Show pgvector indexing progress")
    status_cmd.add_argument("--model", default=None, choices=["sentence-transformers/all-MiniLM-L6-v2", "Alibaba-NLP/gte-large-en-v1.5"])
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--embedding-model", choices=["sentence-transformers/all-MiniLM-L6-v2", "Alibaba-NLP/gte-large-en-v1.5"], default=None)
    search.add_argument("--mode", choices=["sparse", "dense", "hybrid"], default="sparse")
    memory_search = sub.add_parser("memory-search", help="Search older turns in one saved conversation")
    memory_search.add_argument("query")
    memory_search.add_argument("--session", required=True, help="Saved conversation UUID")
    memory_search.add_argument("--top-k", type=int, default=5)
    notes = sub.add_parser("notes", help="Inspect editable working memory")
    notes.add_argument("--session", default="default")
    edits = sub.add_parser("events", help="Inspect accepted/rejected context edits")
    edits.add_argument("--session", default="default")
    note = sub.add_parser("note", help="Propose a context note, verified against source evidence")
    note.add_argument("--claim", required=True)
    note.add_argument("--quote", required=True, help="Exact short excerpt from stored source")
    note.add_argument("--evidence-id", required=True)
    note.add_argument("--session", default="default")
    note.add_argument("--replace-note-id", default=None)
    note.add_argument("--expected-revision", type=int)
    compare = sub.add_parser("compare", help="Standalone local semantic cosine check")
    compare.add_argument("--original", required=True)
    compare.add_argument("--edited", required=True)
    answer = sub.add_parser("answer", help="Local LLM RAG (always passes notes and retrieved context)")
    answer.add_argument("question")
    answer.add_argument("--session", default="default")
    answer.add_argument("--web", action="store_true", help="Search web and read source webpages")
    answer.add_argument("--web-pages", type=int, choices=range(0,4), default=2)
    answer.add_argument("--academic", action="store_true", help="Include arXiv/OpenAlex abstracts")
    answer.add_argument("--embedding-model", choices=["sentence-transformers/all-MiniLM-L6-v2", "Alibaba-NLP/gte-large-en-v1.5"], default=None)
    answer.add_argument("--model", choices=["llama3.2:3b", "qwen3.5:4b", "granite4.1:3b"], default=None)
    answer.add_argument("--mode", choices=["sparse", "dense", "hybrid"], default=None)
    answer.add_argument("--top-k", type=int, default=5)
    answer.add_argument("--temperature", type=float, default=0.0)
    answer.add_argument("--num-predict", type=int, default=-1)
    ask = sub.add_parser("ask", help="LangGraph tool-calling agent with preloaded context")
    ask.add_argument("question")
    ask.add_argument("--session", default="default")
    ask.add_argument("--web", action="store_true", help="Enable search + browse agent tools")
    ask.add_argument("--web-pages", type=int, choices=range(0,4), default=2)
    ask.add_argument("--academic", action="store_true", help="Enable academic search and agent tool")
    ask.add_argument("--embedding-model", choices=["sentence-transformers/all-MiniLM-L6-v2", "Alibaba-NLP/gte-large-en-v1.5"], default=None)
    ask.add_argument("--model", choices=["llama3.2:3b", "qwen3.5:4b", "granite4.1:3b"], default=None)
    ask.add_argument("--mode", choices=["sparse", "dense", "hybrid"], default=None)
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--temperature", type=float, default=0.0)
    ask.add_argument("--num-predict", type=int, default=-1)
    for target in (search, answer, ask):
        target.add_argument("--reranker", choices=["none", "jinaai/jina-reranker-v3.5", "Alibaba-NLP/gte-reranker-modernbert-base"], default="none")
        target.add_argument("--rerank-candidates", type=int, default=20)
    for target in (answer, ask):
        target.add_argument("--memory-route", choices=["auto", "conversation", "research", "both", "neither"], default="auto")
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.command == "scholar-url":
        from aira.academic import scholar_url
        result = {"url": scholar_url(args.query), "mode": "open_in_your_browser"}
    elif args.command == "compare":
        checker = SemanticChecker(settings.embedding_model, settings.similarity_threshold)
        score = checker.compare(args.original, args.edited)
        result = {"cosine_similarity": round(score, 6),
                  "threshold": settings.similarity_threshold,
                  "above_threshold": score >= settings.similarity_threshold,
                  "warning": "Cosine similarity is NOT proof of factual entailment"}
    else:
        factory = make_session_factory(settings.database_url)
        context = make_context_manager(factory, settings)
        if args.command == "ingest":
            result = Ingestor(factory, settings.papers_dir).sync()
        elif args.command == "index-embeddings":
            result = EmbeddingIndex(factory, args.model or settings.embedding_model).sync()
        elif args.command == "retrieval-status":
            result = EmbeddingIndex(factory, args.model or settings.embedding_model).stats()
        elif args.command == "academic-search":
            from aira.academic import search_academic
            result = search_academic(factory, args.query, sources=args.source,
                                     top_k=args.top_k, year_from=args.year_from,
                                     openalex_api_key=settings.openalex_api_key)
        elif args.command == "import-paper":
            from aira.academic import import_approved_pdf
            result = import_approved_pdf(factory, settings.papers_dir, args.url)
        elif args.command == "web-search":
            from aira.websearch import search_web
            result = search_web(factory, args.query, provider=settings.web_search_provider,
                                api_key=settings.tavily_api_key, max_results=args.top_k)
        elif args.command == "search":
            result = LocalRetriever(factory, settings.embedding_model).search(
                args.query, options=RetrievalOptions(mode=args.mode, top_k=args.top_k,
                    embedding_model=args.embedding_model or settings.embedding_model,
                    reranker=args.reranker, rerank_candidates=args.rerank_candidates))
        elif args.command == "memory-search":
            result = {"results": search_conversation(factory, args.session, args.query, top_k=args.top_k)}
        elif args.command == "notes":
            result = context.notes(args.session)
        elif args.command == "events":
            result = context.events(args.session)
        elif args.command == "note":
            result = context.propose(args.session, args.claim, args.quote, args.evidence_id,
                                     args.replace_note_id, args.expected_revision)
        elif args.command == "answer":
            retrieval = RetrievalOptions(
                mode=args.mode or (settings.retrieval_mode if factory.kw['bind'].dialect.name == 'postgresql' else 'sparse'),
                top_k=args.top_k, embedding_model=args.embedding_model or settings.embedding_model,
                reranker=args.reranker, rerank_candidates=args.rerank_candidates)
            result = answer_with_local_context(args.question, args.session, settings,
                                               LocalRetriever(factory, settings.embedding_model), context,
                                               retrieval, GenerationOptions(model=args.model, temperature=args.temperature,
                                                                           num_predict=args.num_predict),
                                               use_web=args.web, session_factory=factory,
                                               use_academic=args.academic, web_fetch_pages=args.web_pages,
                                               history=recent_history(factory, args.session),
                                               memory_route=args.memory_route)
        elif args.command == "ask":
            retrieval = RetrievalOptions(
                mode=args.mode or (settings.retrieval_mode if factory.kw['bind'].dialect.name == 'postgresql' else 'sparse'),
                top_k=args.top_k, embedding_model=args.embedding_model or settings.embedding_model,
                reranker=args.reranker, rerank_candidates=args.rerank_candidates)
            result = run_agent(args.question, args.session, settings, factory,
                               LocalRetriever(factory, settings.embedding_model), context,
                               retrieval, GenerationOptions(model=args.model, temperature=args.temperature,
                                                            num_predict=args.num_predict), use_web=args.web,
                               use_academic=args.academic, web_fetch_pages=args.web_pages,
                               history=recent_history(factory, args.session),
                               memory_route=args.memory_route)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
