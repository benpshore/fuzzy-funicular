import json
import types

import httpx
import pytest

from funicular import llm
from funicular.summarize import ask, summarize


class FakeProvider(llm.Provider):
    """Deterministic provider: returns JSON shaped by the system prompt it receives."""

    name = "fake"
    model = "fake-1"

    def __init__(self):
        self.calls = []

    def complete(self, system, user, *, max_tokens=4096, json_mode=False):
        self.calls.append((system[:20], user[:80]))
        if system.startswith("You extract"):
            page = user.split("page ")[1].split(")")[0] if "page " in user else "?"
            body = {
                "findings": [{"claim": f"finding from page {page}", "page": page, "quote": "q"}],
                "numbers": [{"what": "HR", "value": "0.84", "page": page}],
                "methods": "cohort",
                "limitations": ["confounding"],
                "population": "MND",
            }
            return llm.Reply(json.dumps(body), self.name, self.model, 10, 5)
        if system.startswith("You write"):
            card = {
                "one_line": "Riluzole was associated with lower mortality.",
                "study_type": "cohort",
                "question": "q",
                "population": "MND",
                "sample_size": "2418",
                "intervention": "riluzole",
                "comparator": "none",
                "primary_outcomes": ["mortality"],
                "key_findings": [{"finding": "HR 0.84", "pages": "1"}],
                "effect_sizes": [
                    {"measure": "HR", "value": "0.84", "ci": "0.76-0.93", "pages": "1"}
                ],
                "limitations": ["confounding"],
                "funding_conflicts": "not reported",
                "confidence": "moderate: registry",
                "keywords": ["riluzole", "MND"],
            }
            return llm.Reply(
                "```json\n" + json.dumps(card) + "\n```", self.name, self.model, 20, 10
            )
        return llm.Reply(
            f"Answer citing [{user.split('[')[1].split(']')[0]}]" if "[" in user else "no",
            self.name,
            self.model,
            5,
            5,
        )


def test_parse_json_reply_variants():
    assert llm.parse_json_reply('{"a": 1}') == {"a": 1}
    assert llm.parse_json_reply('Sure:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json_reply('text {"a": {"b": 2}} more') == {"a": {"b": 2}}
    assert llm.parse_json_reply("nothing") is None and llm.parse_json_reply("") is None


def test_openai_compat_provider_and_retries():
    seen = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        assert req.headers["authorization"] == "Bearer ghp_x"
        body = json.loads(req.content)
        assert body["model"] == "openai/gpt-4.1" and body["messages"][0]["role"] == "system"
        if seen["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            json={
                "model": "openai/gpt-4.1",
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    p = llm.OpenAICompatProvider(
        "github",
        llm.GITHUB_MODELS_URL,
        "ghp_x",
        "openai/gpt-4.1",
        transport=httpx.MockTransport(handler),
    )
    r = p.complete("sys", "user", json_mode=True)
    assert r.text == "hi" and r.input_tokens == 3 and seen["n"] == 2 and not r.refused
    bad = llm.OpenAICompatProvider(
        "openai",
        "https://api.openai.com/v1",
        "k",
        "m",
        transport=httpx.MockTransport(lambda r: httpx.Response(401)),
    )
    with pytest.raises(llm.LLMUnavailable):
        bad.complete("s", "u")


def test_anthropic_provider_builds_the_right_request(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    prov = llm.AnthropicProvider(model="claude-opus-5", effort="low")
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            content=[
                types.SimpleNamespace(type="thinking", thinking=""),
                types.SimpleNamespace(type="text", text="ok"),
            ],
            stop_reason="end_turn",
            model="claude-opus-5",
            usage=types.SimpleNamespace(input_tokens=7, output_tokens=2),
        )

    monkeypatch.setattr(prov, "_create", fake_create)
    r = prov.complete("system prompt", "hello", max_tokens=100)
    assert r.text == "ok" and r.input_tokens == 7 and not r.refused
    assert captured["model"] == "claude-opus-5" and captured["max_tokens"] == 100
    assert captured["thinking"] == {"type": "adaptive"} and captured["output_config"] == {
        "effort": "low"
    }
    assert (
        captured["fallbacks"] == "default"
        and "server-side-fallback-2026-07-01" in captured["betas"]
    )
    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    # refusal surfaces as a flag, never as an exception or an empty success
    monkeypatch.setattr(
        prov,
        "_create",
        lambda **k: types.SimpleNamespace(
            content=[], stop_reason="refusal", model="claude-opus-5", usage=None
        ),
    )
    assert prov.complete("s", "u").refused


def test_get_provider_selection(monkeypatch):
    for k in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE",
        "OPENAI_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "FUNICULAR_LLM_PROVIDER",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(llm, "_ant_profile_exists", lambda: False)
    with pytest.raises(llm.LLMUnavailable):
        llm.get_provider("auto")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_abc")
    p = llm.get_provider("auto")
    assert p.name == "github" and p.model == llm.DEFAULT_GITHUB_MODEL
    monkeypatch.setenv("OPENAI_API_KEY", "sk")
    assert llm.get_provider("auto").name == "openai"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    assert llm.get_provider("auto").name == "anthropic"
    assert llm.get_provider("ollama").base_url.startswith("http://127.0.0.1:11434")
    monkeypatch.setenv("FUNICULAR_LLM_PROVIDER", "none")
    with pytest.raises(llm.LLMUnavailable):
        llm.get_provider()
    names = [d["name"] for d in llm.describe_providers()]
    assert names[:3] == ["anthropic", "openai", "github"]


def test_summarize_map_reduce_and_sampling():
    prov = FakeProvider()
    text = "\f".join(f"Page {i} text. " * 200 for i in range(1, 4))
    s = summarize(text, prov, title="T")
    assert s.card["one_line"].startswith("Riluzole") and s.chunks_used == s.chunks_total >= 3
    assert not s.sampled and s.input_tokens > 0
    assert all(n.get("page") for n in s.notes)
    big = "\f".join(f"Page {i} " + ("word " * 800) for i in range(1, 60))
    s2 = summarize(big, prov, title="Big", budget_tokens=4000)
    assert s2.sampled and s2.chunks_used < s2.chunks_total and "coverage_note" in s2.card
    with pytest.raises(ValueError):
        summarize("   ", prov)


def test_ask_grounds_and_cites():
    prov = FakeProvider()
    a = ask("what?", [{"doc_id": "d1", "title": "T", "page": 3, "text": "the passage"}], prov)
    assert "[d1 p.3]" in a.text and a.citations[0]["page"] == 3
    empty = ask("what?", [], prov)
    assert "No passages" in empty.text and empty.citations == []
