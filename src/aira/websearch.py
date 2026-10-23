"""Explicit, source-traceable web search. Search-result snippets are not verified page content.

Default provider `ddgs` requires no API key. It queries public search backends, which
may rate-limit or block requests. Tavily is an optional API-key-backed alternative.
Neither provider fetches arbitrary URLs supplied by an agent.
"""
from datetime import UTC, datetime
from urllib.parse import urlparse

import httpx

from aira.sources import capture_web_snippet


class WebSearchError(RuntimeError):
    """Expected user-facing web provider or configuration failure."""


def _ddgs_results(query: str, limit: int) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise WebSearchError('Install the free web-search dependency: pip install -e ".[web]"') from exc
    try:
        return list(DDGS(timeout=12).text(query, max_results=limit, backend="auto") or [])
    except Exception as exc:
        # Search providers can rate-limit or refuse automated requests.
        raise WebSearchError(f"Free web search unavailable ({type(exc).__name__}); retry later") from exc


def _tavily_results(query: str, limit: int, api_key: str | None) -> list[dict]:
    if not api_key:
        raise WebSearchError("TAVILY_API_KEY required when WEB_SEARCH_PROVIDER=tavily")
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"query": query, "max_results": limit, "include_raw_content": False},
            )
            response.raise_for_status()
            payload = response.json()
        return payload.get("results", [])
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise WebSearchError(f"Tavily web search unavailable ({type(exc).__name__})") from exc


def search_web(session_factory, query: str, *, provider: str = "ddgs",
               api_key: str | None = None, max_results: int = 4) -> list[dict]:
    """Return bounded search-result snippets with stable evidence IDs and provenance.

    Search itself is network-backed. Captured evidence is exactly the provider's snippet,
    never an assertion that the linked page or its claims were independently verified.
    """
    query = query.strip()
    if not query or len(query) > 2000:
        raise ValueError("Web query must contain 1–2000 characters")
    if not 1 <= max_results <= 8:
        raise ValueError("max_results must be between 1 and 8")
    if provider == "disabled":
        raise WebSearchError("Web search is disabled (WEB_SEARCH_PROVIDER=disabled)")
    if provider == "ddgs":
        raw = _ddgs_results(query, max_results)
    elif provider == "tavily":
        raw = _tavily_results(query, max_results, api_key)
    else:
        raise WebSearchError("WEB_SEARCH_PROVIDER must be ddgs, tavily, or disabled")

    now = datetime.now(UTC).isoformat()
    results = []
    seen_urls: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = str(item.get("href") or item.get("url") or "").strip()
        text = str(item.get("body") or item.get("content") or "").strip()[:3500]
        title = str(item.get("title") or "").strip()[:250]
        parsed = urlparse(url)
        if (parsed.scheme not in {"https", "http"} or not parsed.hostname
                or parsed.username or parsed.password or url in seen_urls
                or not text or len(url) > 2000):
            continue
        seen_urls.add(url)
        try:
            evidence_id = capture_web_snippet(session_factory, url, text)
        except ValueError:
            continue
        results.append({"title": title, "url": url, "excerpt": text[:1500],
                        "evidence_id": evidence_id, "source_kind": "web_snippet_unverified",
                        "provider": provider, "retrieved_at": now})
        if len(results) >= max_results:
            break
    return results
