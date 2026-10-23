"""Human-directed academic discovery. No Google Scholar scraping or paid services.

Results are metadata/abstracts, NOT authenticated paper text. Full PDF downloads
require a separate, user-approved request, and only approved public hosts are
supported. This is intentionally narrower than a general-purpose URL fetcher.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote_plus, unquote, urlparse
from uuid import uuid4

import httpx

from aira.db import EvidenceChunk
from aira.ingestion import digest

ARXIV_API = "https://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"
USER_AGENT = "EvidenceResearchAgent/0.7 (personal local research tool; metadata discovery)"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
ALLOWED_SOURCES = {"arxiv", "openalex", "acl", "acm"}
MAX_PDF_BYTES = 20 * 1024 * 1024


class AcademicSearchError(RuntimeError):
    """A provider rejected a request or returned data we cannot parse."""


def scholar_url(query: str) -> str:
    """Construct a human-opened search URL; never scrape Scholar."""
    query = query.strip()
    if not 1 <= len(query) <= 500:
        raise ValueError("Scholar query must contain 1–500 characters")
    return "https://scholar.google.com/scholar?q=" + quote_plus(query)


def _validate_query(query: str, top_k: int, sources: list[str], year_from: int | None):
    if not 1 <= len(query.strip()) <= 500:
        raise ValueError("Academic query must contain 1–500 characters")
    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be 1–50")
    if not sources or set(sources) - ALLOWED_SOURCES:
        raise ValueError("sources must be a nonempty subset of arxiv, openalex, acl, acm")
    if year_from is not None and not 1900 <= year_from <= 2100:
        raise ValueError("year_from must be between 1900 and 2100")


def _arxiv_results(query: str, limit: int) -> list[dict]:
    # Explicitly use arXiv's documented Atom API rather than scraping HTML pages.
    stopwords = {"what", "which", "the", "and", "or", "in", "on", "for", "to", "from",
                 "are", "is", "how", "does", "latest", "recent", "new", "papers", "paper",
                 "research", "developments", "about", "compare", "with", "show", "find"}
    terms = [term for term in re.findall(r"[\w.-]+", query, flags=re.UNICODE)
             if term.casefold() not in stopwords][:4]
    if not terms:
        return []
    search_expression = " AND ".join(f"all:{term}" for term in terms)
    with httpx.Client(timeout=20.0, follow_redirects=False, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(ARXIV_API, params={"search_query": search_expression,
                                                  "start": 0, "max_results": limit * 3})
        response.raise_for_status()
        root = ET.fromstring(response.content)
    results = []
    for entry in root.findall("a:entry", ATOM):
        identifier = (entry.findtext("a:id", "", ATOM)).strip()
        arxiv_id = identifier.rsplit("/", 1)[-1].split("v")[0]
        if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})", arxiv_id, re.I):
            continue
        published = entry.findtext("a:published", "", ATOM)
        results.append({
            "provider": "arxiv", "provider_id": arxiv_id,
            "title": " ".join(entry.findtext("a:title", "", ATOM).split()),
            "authors": [e.findtext("a:name", "", ATOM) for e in entry.findall("a:author", ATOM)],
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "abstract": " ".join(entry.findtext("a:summary", "", ATOM).split())[:3000],
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "doi": next((x.text.strip().lower() for x in entry if x.tag.endswith('}doi') and x.text), None),
            "arxiv_id": arxiv_id,
        })
    return results


def _openalex_abstract(inverted: dict | None) -> str:
    if not isinstance(inverted, dict):
        return ""
    positions = []
    for word, indices in inverted.items():
        if isinstance(word, str) and isinstance(indices, list):
            positions.extend((position, word) for position in indices if isinstance(position, int) and 0 <= position <= 10000)
    positions.sort()
    return " ".join(word for _, word in positions)[:3000]


def _openalex_results(query: str, limit: int, year_from: int | None, api_key: str | None) -> list[dict]:
    params: dict = {"search": query, "per_page": limit}
    if year_from is not None:
        params["filter"] = f"from_publication_date:{year_from}-01-01"
    # Anonymous queries are free within the provider's current published limits.
    if api_key:
        params["api_key"] = api_key
    with httpx.Client(timeout=20.0, follow_redirects=False, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get(OPENALEX_API, params=params)
        response.raise_for_status()
        payload = response.json()
    results = []
    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue
        url = item.get("doi") or item.get("id")
        if not isinstance(url, str) or not url.startswith("https://"):
            continue
        # OpenAlex may offer publisher HTML pages instead of downloadable PDFs.
        locations = [item.get("best_oa_location") or {}, item.get("primary_location") or {}]
        pdf_url = next((loc.get("pdf_url") for loc in locations if isinstance(loc, dict) and
                        isinstance(loc.get("pdf_url"), str) and loc.get("pdf_url", "").startswith("https://")), None)
        results.append({
            "provider": "openalex", "provider_id": item.get("id", ""),
            "title": item.get("display_name") or item.get("title") or "Untitled",
            "authors": [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])
                        if isinstance(a, dict)],
            "year": item.get("publication_year"),
            "abstract": _openalex_abstract(item.get("abstract_inverted_index")),
            "url": url, "pdf_url": pdf_url,
            "doi": (item.get("doi") or "").removeprefix("https://doi.org/").lower() or None,
            "arxiv_id": None,
        })
    return results


def _crossref_results(query: str, limit: int, year_from: int | None, source: str) -> list[dict]:
    """Venue-specific Crossref DOI metadata; never scrape ACL/ACM websites.

    ACL papers generally use the 10.18653 DOI prefix; ACM uses 10.1145.
    These are metadata searches, not guaranteed exhaustive collection searches.
    """
    prefix = {"acl": "10.18653", "acm": "10.1145"}[source]
    filters = [f"prefix:{prefix}"]
    if year_from is not None:
        filters.append(f"from-pub-date:{year_from}-01-01")
    with httpx.Client(timeout=20.0, follow_redirects=False, headers={"User-Agent": USER_AGENT}) as client:
        response = client.get("https://api.crossref.org/works", params={
            "query.bibliographic": query, "filter": ",".join(filters), "rows": limit,
        })
        response.raise_for_status()
        payload = response.json()
    items = payload.get("message", {}).get("items", [])
    results = []
    for item in items:
        doi = str(item.get("DOI") or "").strip().lower()
        if not doi.startswith(prefix + "/"):
            continue
        title = (item.get("title") or [""])[0]
        if not isinstance(title, str) or not title.strip():
            continue
        issued = (item.get("published") or item.get("issued") or {}).get("date-parts", [[]])
        year = issued[0][0] if issued and issued[0] else None
        authors = [" ".join(filter(None, [a.get("given", ""), a.get("family", "")]))
                   for a in item.get("author", []) if isinstance(a, dict)]
        acl_identifier = doi.removeprefix("10.18653/v1/") if source == "acl" else None
        is_acl_paper = bool(acl_identifier and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,100}", acl_identifier))
        url = f"https://aclanthology.org/{acl_identifier}/" if is_acl_paper else "https://doi.org/" + doi
        # ACL offers a stable public PDF endpoint. ACM full text may be access controlled.
        pdf = f"https://aclanthology.org/{acl_identifier}.pdf" if is_acl_paper else None
        results.append({"provider": source, "provider_id": doi, "title": title.strip(),
                        "authors": authors, "year": year, "abstract": "", "url": url,
                        "pdf_url": pdf, "doi": doi, "arxiv_id": None})
    return results


def _capture_abstract(session_factory, paper: dict) -> str | None:
    abstract = paper.get("abstract", "").strip()
    if not abstract:
        return None
    # A metadata abstract is immutable captured text, NOT a verified full paper.
    url = paper["url"]
    evidence_id = digest(f"academic-abstract\n{url}\n{abstract}")
    with session_factory.begin() as db:
        if db.get(EvidenceChunk, evidence_id) is None:
            db.add(EvidenceChunk(id=evidence_id, source_kind="academic_abstract",
                                 source_url=url, text=abstract, text_sha256=digest(abstract), ordinal=0))
    return evidence_id


def search_academic(session_factory, query: str, *, sources: list[str] | None = None,
                    top_k: int = 5, year_from: int | None = None,
                    openalex_api_key: str | None = None) -> dict:
    """Search public academic APIs. Capture available abstracts as cited metadata evidence.

    A partial provider failure never silently replaces missing results with fabricated data.
    """
    sources = sources or ["arxiv", "openalex"]
    _validate_query(query, top_k, sources, year_from)
    results: list[dict] = []
    errors: dict[str, str] = {}
    per_provider: dict[str, list[dict]] = {}
    for source in dict.fromkeys(sources):
        try:
            if source == "arxiv":
                found = _arxiv_results(query, top_k)
                if year_from is not None:
                    found = [p for p in found if p["year"] is not None and p["year"] >= year_from]
            elif source == "openalex":
                found = _openalex_results(query, top_k, year_from, openalex_api_key)
            else:
                found = _crossref_results(query, top_k, year_from, source)
            per_provider[source] = found
        except (httpx.HTTPError, ET.ParseError, ValueError, TypeError, KeyError) as exc:
            errors[source] = f"{type(exc).__name__}: provider unavailable or invalid response"
    # Interleave providers before deduplication so a broad arXiv search cannot
    # crowd out ACL / ACM / OpenAlex records when top_k is small.
    for i in range(max((len(rows) for rows in per_provider.values()), default=0)):
        for source in dict.fromkeys(sources):
            rows = per_provider.get(source, [])
            if i < len(rows):
                results.append(rows[i])
    # Merge by any stable identifier available, falling back to normalized title.
    seen: list[dict] = []
    aliases: dict[str, int] = {}
    for paper in results:
        keys = ["title:" + re.sub(r"\W+", "", paper["title"].casefold())]
        if paper.get("doi"):
            keys.append("doi:" + paper["doi"].lower())
        if paper.get("arxiv_id"):
            keys.append("arxiv:" + paper["arxiv_id"].lower())
        match = next((aliases[k] for k in keys if k in aliases), None)
        if match is None:
            paper["found_in"] = [paper["provider"]]
            match = len(seen)
            seen.append(paper)
        else:
            original = seen[match]
            original["found_in"] = sorted(set(original["found_in"] + [paper["provider"]]))
            if not original.get("pdf_url"):
                original["pdf_url"] = paper.get("pdf_url")
            if not original.get("abstract"):
                original["abstract"] = paper.get("abstract", "")
            if not original.get("doi"):
                original["doi"] = paper.get("doi")
        for key in keys:
            aliases[key] = match
    ordered = seen[:top_k]
    timestamp = datetime.now(UTC).isoformat()
    for paper in ordered:
        paper["retrieved_at"] = timestamp
        paper["source_kind"] = "academic_metadata_unverified"
        paper["evidence_id"] = _capture_abstract(session_factory, paper)
    return {"results": ordered, "errors": errors, "query": query, "sources": sources}


def _approved_pdf_url(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port is not None
            or parsed.query or parsed.fragment or len(url) > 1000):
        raise ValueError("Only direct HTTPS academic PDF URLs without query strings are supported")
    host = (parsed.hostname or "").lower()
    path = unquote(parsed.path)
    if host in {"arxiv.org", "export.arxiv.org"}:
        match = re.fullmatch(r"/(?:pdf|abs)/(\d{4}\.\d{4,5})(?:\.pdf)?/?", path)
        if not match:
            raise ValueError("Expected an arXiv paper URL such as https://arxiv.org/abs/2106.09685")
        arxiv_id = match.group(1)
        return f"https://arxiv.org/pdf/{arxiv_id}", f"arxiv_{arxiv_id}.pdf"
    if host in {"aclanthology.org", "www.aclanthology.org"}:
        match = re.fullmatch(r"/([A-Za-z0-9][A-Za-z0-9._-]{2,100})(?:\.pdf)?/?", path)
        if not match:
            raise ValueError("Expected an ACL Anthology paper page or direct PDF URL")
        ident = match.group(1)
        return f"https://aclanthology.org/{ident}.pdf", f"acl_{ident}.pdf"
    raise ValueError("Import supports only direct arXiv or ACL Anthology PDFs. Download other PDFs manually and upload them.")


def import_approved_pdf(session_factory, papers_dir: Path, url: str) -> dict:
    """Explicitly approved download; refuses redirects and private/untrusted hosts."""
    approved, filename = _approved_pdf_url(url)
    root = papers_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / filename
    if destination.is_symlink():
        raise ValueError("Refusing to overwrite a symbolic link")
    temp = root / f".{uuid4().hex}.download"
    count = 0
    try:
        with httpx.Client(timeout=35, follow_redirects=False, headers={"User-Agent": USER_AGENT}) as client:
            current = approved
            for redirect_count in range(4):
                with client.stream("GET", current) as response:
                    if response.is_redirect:
                        from urllib.parse import urljoin
                        redirected = urljoin(current, response.headers.get("location", ""))
                        _, redirected_name = _approved_pdf_url(redirected)
                        if redirected_name != filename:
                            raise ValueError("PDF redirect changed the requested paper")
                        current = redirected
                        if redirect_count == 3:
                            raise ValueError("Too many PDF download redirects")
                        continue
                    response.raise_for_status()
                    if (response.headers.get("content-length", "").isdigit() and
                            int(response.headers["content-length"]) > MAX_PDF_BYTES):
                        raise ValueError("PDF exceeds 20 MiB")
                    with temp.open("xb") as output:
                        head = b""
                        for block in response.iter_bytes(65536):
                            count += len(block)
                            if count > MAX_PDF_BYTES:
                                raise ValueError("PDF exceeds 20 MiB")
                            if len(head) < 1024:
                                head += block[:1024 - len(head)]
                            output.write(block)
                    break
        if not head.startswith(b"%PDF-"):
            raise ValueError("Remote response is not a PDF")
        # Reuse existing versioned ingestion semantics and stable evidence IDs.
        from aira.ingestion import Ingestor
        temp.replace(destination)
        result = Ingestor(session_factory, root).ingest_file(destination)
        return {**result, "downloaded_from": approved}
    finally:
        temp.unlink(missing_ok=True)
