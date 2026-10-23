"""Automatic chat titles from a small local model, without a live Ollama server."""
import httpx

from aira.chat_titles import (
    clean_title,
    fallback_title,
    generate_title,
    update_auto_title,
)
from aira.conversations import (
    create_conversation,
    get_conversation,
    rename_conversation,
    save_exchange,
)
from aira.db import make_session_factory


def test_title_model_is_small_and_no_research_corpus_sent(monkeypatch):
    requests = []
    def fake_post(url, *, json, timeout):
        requests.append((url, json, timeout))
        return httpx.Response(200, json={"message": {"content": "Quantization Memory Tradeoffs"}},
                              request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx, "post", fake_post)
    title = generate_title("How does QLoRA work?", "It uses NF4.", base_url="http://ollama:11434")
    assert title == "Quantization Memory Tradeoffs"
    assert requests[0][1]["model"] == "qwen3.5:0.8b"
    assert requests[0][1]["think"] is False
    assert requests[0][1]["options"]["num_predict"] <= 60
    assert "How does QLoRA work?" in str(requests[0][1])


def test_short_title_fallback_and_cleaning(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ConnectError("off")))
    title = generate_title("Please explain the role of NormalFloat quantization in long fine-tuning runs", "answer")
    assert title and len(title) <= 61
    assert clean_title('<think>private</think> "QLoRA vs LoRA"') == "QLoRA vs LoRA"
    assert fallback_title("   ") == "New chat"


def test_auto_title_saves_and_respects_manual_rename(tmp_path, monkeypatch):
    factory = make_session_factory(f"sqlite:///{tmp_path / 'titles.db'}")
    row = create_conversation(factory)
    question = "Compare LoRA and QLoRA memory usage"
    save_exchange(factory, row["id"], question, {"answer":"answer"}, "answer")
    provisional = get_conversation(factory, row["id"])["title"]
    monkeypatch.setattr("aira.chat_titles.generate_title", lambda *args, **kwargs: "LoRA and QLoRA Compared")
    update_auto_title(factory, row["id"], provisional, question, "answer")
    assert get_conversation(factory, row["id"])["title"] == "LoRA and QLoRA Compared"
    rename_conversation(factory, row["id"], "Custom user title")
    update_auto_title(factory, row["id"], "LoRA and QLoRA Compared", question, "answer")
    assert get_conversation(factory, row["id"])["title"] == "Custom user title"
