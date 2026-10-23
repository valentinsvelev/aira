"""Bounded, evidence-capturing web-page reading for opt-in agent research.

This module is deliberately narrower than a general-purpose autonomous browser:
HTML/text only, public HTTP(S) destinations, small downloads and no JS execution.
It does not bypass paywalls, CAPTCHA, or site restrictions. Treat all fetched
content as UNTRUSTED source text, not agent instructions.
"""
from __future__ import annotations

import ipaddress
import socket
from datetime import UTC, datetime
from urllib.parse import urljoin, urlparse

import httpx

from aira.sources import capture_web_page

MAX_BYTES = 1_500_000
MAX_TEXT_CHARS = 15_000
MAX_REDIRECTS = 2


class WebPageError(RuntimeError):
    """Controlled page-fetch failure; callers should retain snippets as fallback."""


def validate_public_web_url(url: str, resolver=None) -> None:
    """Deny obvious SSRF targets (DNS rebinding is a residual risk; run API locally)."""
    parsed = urlparse(url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or len(url) > 2000):
        raise WebPageError("Only public HTTP(S) webpages are supported")
    try:
        port = parsed.port
    except ValueError as exc:
        raise WebPageError("Invalid URL port") from exc
    if port is not None and port not in {80, 443}:
        raise WebPageError("Nonstandard webpage ports are not allowed")
    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith((".local", ".internal", ".localhost")):
        raise WebPageError("Local hosts cannot be fetched")
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        resolve = resolver or socket.getaddrinfo
        try:
            addresses = [ipaddress.ip_address(item[4][0]) for item in
                         resolve(host, port or (443 if parsed.scheme == "https" else 80),
                                 type=socket.SOCK_STREAM)]
        except (socket.gaierror, OSError, ValueError) as exc:
            raise WebPageError("Could not resolve the public website") from exc
    if not addresses or not all(ip.is_global for ip in addresses):
        raise WebPageError("Private, loopback or special-purpose IP addresses are blocked")


def fetch_page_text(url: str, *, client=None, resolver=None) -> tuple[str, str]:
    """Fetch one public HTML page with manual redirect and response-size limits.

    Transport-level DNS rebinding remains a theoretical risk. Do not expose this
    unauthenticated local server publicly or provide it cloud network access.
    """
    try:
        import trafilatura
    except ImportError as exc:
        raise WebPageError('Install page extraction: pip install -e ".[web]"') from exc
    close_client = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(12.0, connect=5.0),
                                  follow_redirects=False, trust_env=False,
                                  headers={"User-Agent": "EvidenceResearchAgent/0.8 (personal research; respectful fetching)",
                                           "Accept": "text/html, text/plain"})
    try:
        current = url
        for redirect in range(MAX_REDIRECTS + 1):
            validate_public_web_url(current, resolver=resolver)
            with http.stream("GET", current, follow_redirects=False) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location", "")
                    if not location or redirect == MAX_REDIRECTS:
                        raise WebPageError("Page redirects too many times")
                    current = urljoin(current, location)
                    continue
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise WebPageError(f"Website returned HTTP {response.status_code}") from exc
                media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if media_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
                    raise WebPageError("Website did not return HTML or plain text")
                chunks: list[bytes] = []
                consumed = 0
                for chunk in response.iter_bytes():
                    consumed += len(chunk)
                    if consumed > MAX_BYTES:
                        raise WebPageError("Webpage exceeds the permitted download size")
                    chunks.append(chunk)
                content = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
                if media_type == "text/plain":
                    text = content
                else:
                    text = trafilatura.extract(content, include_comments=False, include_tables=True)
                if not text or len(text.strip()) < 80:
                    raise WebPageError("Page text could not be extracted")
                return current, text.strip()[:MAX_TEXT_CHARS]
        raise WebPageError("Too many redirects")
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise WebPageError(f"Page is unavailable ({type(exc).__name__})") from exc
    finally:
        if close_client:
            http.close()


def read_search_result(session_factory, result: dict, **kwargs) -> dict:
    """Read a previously discovered search result and capture its actual page text."""
    original_url = result["url"]
    final_url, text = fetch_page_text(original_url, **kwargs)
    evidence_id = capture_web_page(session_factory, final_url, text)
    return {"title": result.get("title", final_url), "url": final_url,
            "excerpt": text[:5000], "evidence_id": evidence_id,
            "source_kind": "web_page_unverified", "provider": "page_fetch",
            "retrieved_at": datetime.now(UTC).isoformat(),
            "chars_captured": len(text)}


def read_top_results(session_factory, results: list[dict], max_pages: int = 2,
                     *, fetcher=read_search_result) -> tuple[list[dict], list[dict]]:
    """Best-effort retrieval: never conceal failed fetches or fabricate webpage text."""
    if not 0 <= max_pages <= 3:
        raise ValueError("max_pages must be between 0 and 3")
    pages, failures = [], []
    # Search engines often rank inaccessible/paywalled pages first. Give at most
    # two extra candidates a chance, while bounding outbound requests.
    for item in results[:min(len(results), max_pages + 2)]:
        if len(pages) >= max_pages:
            break
        try:
            pages.append(fetcher(session_factory, item))
        except (WebPageError, ValueError) as exc:
            failures.append({"url": item.get("url", ""), "reason": str(exc)})
    return pages, failures
