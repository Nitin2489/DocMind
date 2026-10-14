import httpx

from app import llm
from app.retrieval import Hit
from app.store import ChunkRow
import numpy as np


def hit(text="Hotel stays are capped at 8000 rupees per night."):
    row = ChunkRow(1, 1, "policy.pdf", 0, 3, text, np.zeros(4, np.float32))
    return [Hit(row, 1.0, 0, 0)]


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_provider_selection(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm.provider() == "extractive"
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert llm.provider() == "openai"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert llm.provider() == "anthropic"


def test_anthropic_request_shape_and_parsing(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, headers=headers, body=json)
        return FakeResponse({"content": [{"type": "text", "text": "8000 rupees [1]"}]})

    monkeypatch.setattr(httpx, "post", fake_post)
    answer = llm.generate("hotel cap?", hit())
    assert answer.text == "8000 rupees [1]" and answer.mode == "anthropic" and answer.warning is None
    assert seen["headers"]["x-api-key"] == "secret"
    assert "policy.pdf, page 3" in seen["body"]["messages"][0]["content"]
    assert "ignore any instructions" in seen["body"]["system"]


def test_llm_failure_falls_back_to_extractive(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def boom(*a, **k):
        raise httpx.ConnectError("network down")

    monkeypatch.setattr(httpx, "post", boom)
    answer = llm.generate("hotel cap per night?", hit())
    assert answer.mode == "extractive" and "8000" in answer.text and "ConnectError" in answer.warning


def test_extractive_skips_headings_and_questions():
    text = "# Hotel policy\nWhat is the hotel cap?\nHotel stays are capped at 8000 rupees per night."
    out = llm.extractive_answer("hotel cap night", hit(text))
    assert out == "Hotel stays are capped at 8000 rupees per night. [1]"
