"""LLM providers for summarisation and Ask.

* Anthropic — the official ``anthropic`` SDK. Default model ``claude-opus-5`` with adaptive
  thinking, configurable ``effort``, and server-side refusal fallbacks enabled.
* OpenAI-compatible — plain HTTP to any ``/v1/chat/completions`` endpoint: OpenAI itself,
  GitHub Models (``ghp_…`` token), Ollama, LM Studio, or another local server.

Configuration (environment / .env):
  FUNICULAR_LLM_PROVIDER   anthropic | openai | github | ollama | lmstudio | none  (auto-detect)
  ANTHROPIC_API_KEY        (or an `ant auth login` profile)  FUNICULAR_ANTHROPIC_MODEL
  OPENAI_API_KEY / OPENAI_BASE_URL / FUNICULAR_OPENAI_MODEL
  GITHUB_TOKEN (ghp_…) for GitHub Models; FUNICULAR_GITHUB_MODEL
  FUNICULAR_LLM_EFFORT     low | medium | high | xhigh | max   (Anthropic only; default medium)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_OPENAI_MODEL = "gpt-4.1"
DEFAULT_GITHUB_MODEL = "openai/gpt-4.1"
DEFAULT_OLLAMA_MODEL = "llama3.1"
GITHUB_MODELS_URL = "https://models.github.ai/inference"
OLLAMA_URL = "http://127.0.0.1:11434/v1"
LMSTUDIO_URL = "http://127.0.0.1:1234/v1"


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class Reply:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    refused: bool = False
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "raw"}


class Provider:
    name = "base"
    model = ""

    def complete(
        self, system: str, user: str, *, max_tokens: int = 4096, json_mode: bool = False
    ) -> Reply:
        raise NotImplementedError


# --------------------------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------------------------
class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str | None = None, effort: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise LLMUnavailable("install the llm extra: uv sync --extra llm") from exc
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=3)
        self.model = model or os.environ.get("FUNICULAR_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.effort = effort or os.environ.get("FUNICULAR_LLM_EFFORT", "medium")

    def _create(self, **kwargs: Any) -> Any:
        """One seam for tests to stub."""
        return self.client.beta.messages.create(**kwargs)

    def complete(
        self, system: str, user: str, *, max_tokens: int = 4096, json_mode: bool = False
    ) -> Reply:
        t0 = time.monotonic()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": self.effort},
            # policy refusals are re-run on a fallback model server-side
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }
        if self.model.startswith(("claude-opus", "claude-sonnet", "claude-fable", "claude-mythos")):
            kwargs["thinking"] = {"type": "adaptive"}
        a = self._anthropic
        try:
            resp = self._create(**kwargs)
        except a.RateLimitError as exc:
            raise LLMUnavailable(f"Anthropic rate limit: {exc}") from exc
        except a.AuthenticationError as exc:
            raise LLMUnavailable(
                "Anthropic API key missing or invalid (ANTHROPIC_API_KEY or `ant auth login`)"
            ) from exc
        except a.APIStatusError as exc:
            raise LLMUnavailable(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except a.APIConnectionError as exc:
            raise LLMUnavailable(f"cannot reach the Anthropic API: {exc}") from exc
        refused = getattr(resp, "stop_reason", None) == "refusal"
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        usage = getattr(resp, "usage", None)
        return Reply(
            text=text,
            provider=self.name,
            model=getattr(resp, "model", self.model),
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            seconds=time.monotonic() - t0,
            refused=refused,
        )


# --------------------------------------------------------------------------------------------
# OpenAI-compatible (OpenAI, GitHub Models, Ollama, LM Studio, ...)
# --------------------------------------------------------------------------------------------
class OpenAICompatProvider(Provider):
    def __init__(
        self, name: str, base_url: str, api_key: str, model: str, transport: Any = None
    ) -> None:
        import httpx

        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=600, transport=transport
        )

    def complete(
        self, system: str, user: str, *, max_tokens: int = 4096, json_mode: bool = False
    ) -> Reply:
        t0 = time.monotonic()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                r = self.client.post("/chat/completions", json=body)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(float(r.headers.get("Retry-After", "2") or 2), 15))
                last_exc = RuntimeError(f"{self.name} HTTP {r.status_code}")
                continue
            if r.status_code == 401:
                raise LLMUnavailable(f"{self.name}: authentication failed (check the token)")
            if r.status_code >= 400:
                raise LLMUnavailable(f"{self.name}: HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            text = ((choice.get("message") or {}).get("content")) or ""
            usage = data.get("usage") or {}
            return Reply(
                text=text,
                provider=self.name,
                model=data.get("model", self.model),
                input_tokens=usage.get("prompt_tokens", 0) or 0,
                output_tokens=usage.get("completion_tokens", 0) or 0,
                seconds=time.monotonic() - t0,
                refused=choice.get("finish_reason") == "content_filter",
                raw=data,
            )
        raise LLMUnavailable(f"{self.name}: {last_exc}")


# --------------------------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------------------------
def describe_providers() -> list[dict[str, Any]]:
    """What is configured, for the doctor and the UI. Never reveals key material."""
    out = []
    try:
        import anthropic  # noqa: F401

        sdk = True
    except Exception:  # noqa: BLE001
        sdk = False
    out.append(
        {
            "name": "anthropic",
            "configured": bool(
                os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            )
            and sdk,
            "model": os.environ.get("FUNICULAR_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
            "note": "" if sdk else "uv sync --extra llm",
        }
    )
    out.append(
        {
            "name": "openai",
            "configured": bool(os.environ.get("OPENAI_API_KEY")),
            "model": os.environ.get("FUNICULAR_OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            "note": "",
        }
    )
    gh = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    out.append(
        {
            "name": "github",
            "configured": gh.startswith(("ghp_", "github_pat_", "gho_")),
            "model": os.environ.get("FUNICULAR_GITHUB_MODEL", DEFAULT_GITHUB_MODEL),
            "note": "GitHub Models",
        }
    )
    out.append(
        {
            "name": "ollama",
            "configured": True,
            "model": os.environ.get("FUNICULAR_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            "note": "local, used only when selected",
        }
    )
    out.append(
        {
            "name": "lmstudio",
            "configured": True,
            "model": os.environ.get("FUNICULAR_LMSTUDIO_MODEL", "local-model"),
            "note": "local, used only when selected",
        }
    )
    return out


def get_provider(name: str | None = None, *, transport: Any = None) -> Provider:
    """Return the requested provider, or the first configured one when name is auto/None."""
    name = (name or os.environ.get("FUNICULAR_LLM_PROVIDER") or "auto").lower()
    if name == "none":
        raise LLMUnavailable("LLM disabled (FUNICULAR_LLM_PROVIDER=none)")
    if name == "auto":
        for candidate in ("anthropic", "openai", "github"):
            try:
                return get_provider(candidate, transport=transport)
            except LLMUnavailable:
                continue
        raise LLMUnavailable(
            "no LLM configured: set ANTHROPIC_API_KEY (or run `ant auth login`), OPENAI_API_KEY, "
            "or GITHUB_TOKEN; or FUNICULAR_LLM_PROVIDER=ollama|lmstudio for a local model"
        )
    if name == "anthropic":
        if not (
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or os.environ.get("ANTHROPIC_PROFILE")
            or _ant_profile_exists()
        ):
            raise LLMUnavailable("Anthropic not configured")
        return AnthropicProvider()
    if name == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise LLMUnavailable("OPENAI_API_KEY not set")
        return OpenAICompatProvider(
            "openai",
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            key,
            os.environ.get("FUNICULAR_OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            transport,
        )
    if name == "github":
        key = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
        if not key:
            raise LLMUnavailable("GITHUB_TOKEN (ghp_…) not set")
        return OpenAICompatProvider(
            "github",
            os.environ.get("FUNICULAR_GITHUB_MODELS_URL", GITHUB_MODELS_URL),
            key,
            os.environ.get("FUNICULAR_GITHUB_MODEL", DEFAULT_GITHUB_MODEL),
            transport,
        )
    if name == "ollama":
        return OpenAICompatProvider(
            "ollama",
            os.environ.get("OLLAMA_BASE_URL", OLLAMA_URL),
            "",
            os.environ.get("FUNICULAR_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            transport,
        )
    if name == "lmstudio":
        return OpenAICompatProvider(
            "lmstudio",
            os.environ.get("LMSTUDIO_BASE_URL", LMSTUDIO_URL),
            "",
            os.environ.get("FUNICULAR_LMSTUDIO_MODEL", "local-model"),
            transport,
        )
    raise LLMUnavailable(f"unknown provider {name!r}")


def _ant_profile_exists() -> bool:
    from pathlib import Path

    return (Path.home() / ".config" / "anthropic").is_dir()


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def parse_json_reply(text: str) -> dict | None:
    """Tolerant JSON extraction: fenced block, bare object, or the first {...} span."""
    if not text:
        return None
    m = _JSON_FENCE.search(text)
    candidates = [m.group(1)] if m else []
    s = text.strip()
    if s.startswith("{"):
        candidates.append(s)
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        candidates.append(text[i : j + 1])
    for c in candidates:
        try:
            v = json.loads(c)
            if isinstance(v, dict):
                return v
        except ValueError:
            continue
    return None


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)
