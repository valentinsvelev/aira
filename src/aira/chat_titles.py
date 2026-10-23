"""Generate short conversation titles locally with an optional tiny Ollama model.

Titles are metadata only: no PDF chunks, retrieved evidence or entire chat history
are sent to the title model. Failures fall back to a deterministic short title.
"""
from __future__ import annotations

import logging
import os
import re

import httpx
from sqlalchemy import update

from aira.db import ChatConversation, utcnow

LOG = logging.getLogger(__name__)


def fallback_title(question: str, max_chars: int = 60) -> str:
    text = re.sub(r"\s+", " ", question).strip()
    if not text:
        return "New chat"
    if len(text) <= max_chars:
        return text
    prefix = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,.;:!?-")
    return (prefix or text[:max_chars]).strip() + "…"


def clean_title(raw: str, max_chars: int = 60) -> str:
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip().strip('"\'`')
    text = re.sub(r"^(?:title\s*:\s*|#+\s*)", "", text, flags=re.IGNORECASE)
    text = text.split("\n", 1)[0].strip()
    # Prefer a label, never store multi-sentence rambling or explanatory notes.
    text = re.split(r"\s+(?:Explanation|Reasoning):", text, maxsplit=1)[0].strip()
    return text[:max_chars].strip(" ,.;:-") if text else ""


def generate_title(question: str, answer: str, *, model: str | None = None,
                   base_url: str | None = None, timeout: float = 12.0) -> str:
    """Use a deliberately small local Ollama model. No hosted API or paid key."""
    model = model or os.getenv("OLLAMA_TITLE_MODEL", "qwen3.5:0.8b")
    base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
    # Keep the input short and avoid leaking research contents through title metadata.
    prompt = (
        "Create a short, specific chat title (3–7 words, at most 60 characters) "
        "based only on the question and answer below. Return just the title, "
        "without quotation marks, markdown, or any explanation.\n\n"
        f"Question: {question[:650]}\n"
        f"Answer: {answer[:700]}"
    )
    try:
        response = httpx.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "stream": False,
                "think": False,
                "keep_alive": "5m",
                "messages": [
                    {"role": "system", "content": "You are a concise title generator. Return only a short title."},
                    {"role": "user", "content": prompt},
                ],
                "options": {"temperature": 0.2, "num_predict": 48, "num_ctx": 2048},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        title = clean_title(str(response.json().get("message", {}).get("content", "")))
        return title or fallback_title(question)
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
        LOG.info("Short title generation failed (%s); using fallback", type(exc).__name__)
        return fallback_title(question)


def update_auto_title(factory, conversation_id: str, expected_title: str,
                      question: str, answer: str, *, model: str | None = None,
                      base_url: str | None = None) -> str:
    """CAS update: never overwrite a title the user has renamed in the meantime."""
    title = generate_title(question, answer, model=model, base_url=base_url)
    with factory.begin() as db:
        db.execute(
            update(ChatConversation)
            .where(ChatConversation.id == conversation_id,
                   ChatConversation.title == expected_title)
            .values(title=title, updated_at=utcnow())
        )
    return title
