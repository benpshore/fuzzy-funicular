"""Structured, evidence-linked summaries (the Consensus-style card) and grounded Q&A.

Summaries are map-reduce over the document's chunks: every chunk yields a small JSON of
findings with the page they come from; the reduce step writes the final card and keeps the
page references. Nothing is truncated silently: if a document is too long for the budget,
the chunk stage samples evenly and says so in the card.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMUnavailable, Provider, Reply, approx_tokens, parse_json_reply
from .search import Chunk, chunk_text

log = logging.getLogger(__name__)

SUMMARY_FIELDS = [
    "one_line",
    "study_type",
    "question",
    "population",
    "sample_size",
    "intervention",
    "comparator",
    "primary_outcomes",
    "key_findings",
    "effect_sizes",
    "limitations",
    "funding_conflicts",
    "confidence",
    "keywords",
]

MAP_SYSTEM = (
    "You extract evidence from one section of a scholarly document for a clinician-scientist. "
    "Return only JSON with keys: findings (list of {claim, page, quote}), numbers (list of "
    "{what, value, page}), methods (short string or null), limitations (list of strings), "
    "population (string or null). Quote verbatim, short. Use the page number given in the "
    "section header. If the section has no substantive content, return empty lists."
)
REDUCE_SYSTEM = (
    "You write a rigorous evidence card from extracted notes on one scholarly document. "
    "Return only JSON with keys: one_line, study_type, question, population, sample_size, "
    "intervention, comparator, primary_outcomes (list), key_findings (list of {finding, "
    "pages}), effect_sizes (list of {measure, value, ci, pages}), limitations (list), "
    "funding_conflicts, confidence (low|moderate|high with one sentence why), keywords (list). "
    "Report absolute effects when present, not only relative ones. Say 'not reported' rather "
    "than guessing. Never invent numbers; every number must appear in the notes."
)
ASK_SYSTEM = (
    "Answer the question using only the provided passages from the user's own documents. "
    "Cite passages as [doc-id p.N] after each claim. If the passages do not answer the "
    "question, say so plainly. Be concise and precise; keep clinical terminology exact."
)


@dataclass
class Summary:
    card: dict[str, Any]
    provider: str
    model: str
    chunks_used: int
    chunks_total: int
    sampled: bool
    input_tokens: int
    output_tokens: int
    seconds: float
    notes: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _chunks_for(text: str, budget_tokens: int) -> tuple[list[Chunk], bool]:
    chunks = chunk_text(text, target=2400, overlap=120)
    total = sum(approx_tokens(c.text) for c in chunks)
    if total <= budget_tokens or not chunks:
        return chunks, False
    keep = max(4, int(len(chunks) * budget_tokens / total))
    step = len(chunks) / keep
    picked = [chunks[int(i * step)] for i in range(keep)]
    return picked, True


def summarize(
    text: str,
    provider: Provider,
    *,
    title: str = "",
    budget_tokens: int = 60_000,
    progress=None,
) -> Summary:
    t0 = time.monotonic()
    chunks, sampled = _chunks_for(text, budget_tokens)
    if not chunks:
        raise ValueError("nothing to summarise")
    notes: list[dict] = []
    tin = tout = 0
    for i, ch in enumerate(chunks):
        if progress:
            progress("summarize", i, len(chunks) + 1)
        header = (
            f"Document: {title or 'untitled'}\n"
            f"Section {i + 1}/{len(chunks)} (page {ch.page or '?'})\n\n"
        )
        reply = provider.complete(MAP_SYSTEM, header + ch.text, max_tokens=1500, json_mode=True)
        tin += reply.input_tokens
        tout += reply.output_tokens
        if reply.refused:
            notes.append({"page": ch.page, "error": "refused"})
            continue
        parsed = parse_json_reply(reply.text) or {"findings": [], "raw": reply.text[:500]}
        parsed["page"] = ch.page
        notes.append(parsed)
    if progress:
        progress("summarize", len(chunks), len(chunks) + 1)
    reduce_input = (
        f"Document: {title or 'untitled'}\n"
        + (
            "NOTE: the document was longer than the budget; sections were sampled evenly.\n"
            if sampled
            else ""
        )
        + "Extracted notes (JSON per section):\n"
        + json.dumps(notes, ensure_ascii=False)[:200_000]
    )
    reply: Reply = provider.complete(REDUCE_SYSTEM, reduce_input, max_tokens=4000, json_mode=True)
    tin += reply.input_tokens
    tout += reply.output_tokens
    card = parse_json_reply(reply.text) or {
        "one_line": reply.text.strip()[:500],
        "parse_error": True,
    }
    for k in SUMMARY_FIELDS:
        card.setdefault(k, None)
    if sampled:
        card["coverage_note"] = (
            f"summary based on {len(chunks)} sampled sections of a longer document"
        )
    if progress:
        progress("summarize", len(chunks) + 1, len(chunks) + 1)
    return Summary(
        card,
        reply.provider,
        reply.model,
        len(chunks),
        len(chunk_text(text, target=2400, overlap=120)),
        sampled,
        tin,
        tout,
        time.monotonic() - t0,
        notes,
    )


@dataclass
class Answer:
    text: str
    citations: list[dict]
    provider: str
    model: str
    seconds: float
    refused: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def ask(
    question: str, passages: list[dict], provider: Provider, *, max_passages: int = 12
) -> Answer:
    """passages: [{doc_id, title, page, text}] from the search index."""
    t0 = time.monotonic()
    if not passages:
        return Answer(
            "No passages matched the question in the library.",
            [],
            provider.name,
            provider.model,
            time.monotonic() - t0,
        )
    used = passages[:max_passages]
    ctx = "\n\n".join(
        f"[{p['doc_id']} p.{p.get('page') or '?'}] {p.get('title', '')}\n{p['text']}" for p in used
    )
    reply = provider.complete(
        ASK_SYSTEM, f"Passages:\n{ctx}\n\nQuestion: {question}", max_tokens=2500
    )
    cites = [
        {"doc_id": p["doc_id"], "page": p.get("page"), "title": p.get("title", "")} for p in used
    ]
    return Answer(
        reply.text, cites, reply.provider, reply.model, time.monotonic() - t0, reply.refused
    )


def provider_or_error(name: str | None = None) -> Provider:
    try:
        from .llm import get_provider

        return get_provider(name)
    except LLMUnavailable:
        raise
