"""MCP server exposing the library to Claude Desktop, ChatGPT and other MCP clients.

Run with `funicular mcp` (stdio, the transport desktop apps spawn) or `funicular mcp --http PORT`
(streamable HTTP on loopback with a bearer token, for clients that connect over HTTP).

Tools are read-only except `ask`, which sends passages to the configured LLM provider.
"""

from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
from typing import Any

from .config import Settings
from .search import SearchIndex
from .web.store import Store

D_1 = (
    "Full-text, jargon-tolerant and semantic search over the library. Returns passages wit"
    "h doc_id and page."
)
D_2 = (
    "Extracted text of a document: view = layout | markdown | text. Long documents are ret"
    "urned in full unless max_chars is set."
)
D_3 = "Evidence card (structured summary) for a document, if one has been written."
D_4 = "Verified bibliographic record (DOI, authors, journal, citations, open-access link)."
D_5 = "Grounded answer from the library using the configured LLM provider; cites [doc_id p.N]."


def build_server(settings: Settings | None = None):
    from mcp.server.mcpserver import MCPServer

    settings = settings or Settings.from_env()
    store = Store(settings.data_dir / "funicular.sqlite3")
    index = SearchIndex(settings.data_dir / "search.sqlite3", embed=settings.embeddings)
    server = MCPServer(
        name="funicular",
        instructions=(
            "Private document library on this Mac. Use search to find passages, get_document to "
            "read extracted text, get_summary for the evidence card, get_scholar for the verified "
            "bibliographic record. Cite passages as [doc_id p.N]."
        ),
    )

    @server.tool(description=D_1)
    def search(query: str, limit: int = 10, mode: str = "auto") -> list[dict[str, Any]]:
        hits, info = index.search(query, limit=max(1, min(limit, 50)), mode=mode)
        return [h.to_dict() for h in hits]

    @server.tool(description="List documents (newest first). Optional tag filter.")
    def list_documents(limit: int = 50, tag: str | None = None) -> list[dict[str, Any]]:
        return [d.to_public() for d in store.list(limit=max(1, min(limit, 500)), tag=tag)]

    @server.tool(description=D_2)
    def get_document(doc_id: str, view: str = "markdown", max_chars: int = 0) -> dict[str, Any]:
        doc = store.get(doc_id)
        if not doc:
            return {"error": "not found"}
        key = view if view in ("layout", "markdown", "text") else "markdown"
        for k in (key, "markdown", "text", "layout"):
            if k in doc.outputs:
                text = (doc.dir / doc.outputs[k]).read_text(encoding="utf-8", errors="replace")
                if max_chars and len(text) > max_chars:
                    text = text[:max_chars] + f"\n\n[truncated at {max_chars} of {len(text)} chars]"
                return {
                    "doc_id": doc.id,
                    "title": doc.title,
                    "view": k,
                    "text": text,
                    "tags": doc.tags,
                }
        return {"doc_id": doc.id, "title": doc.title, "error": "no text extracted yet"}

    @server.tool(description=D_3)
    def get_summary(doc_id: str) -> dict[str, Any]:
        return _json_output(store, doc_id, "summary.json")

    @server.tool(description=D_4)
    def get_scholar(doc_id: str) -> dict[str, Any]:
        return _json_output(store, doc_id, "scholar.json")

    @server.tool(description="Extracted reference list with per-entry resolution, if available.")
    def get_references(doc_id: str) -> dict[str, Any]:
        return _json_output(store, doc_id, "references.json")

    @server.tool(description=D_5)
    def ask(
        question: str, doc_id: str | None = None, provider: str | None = None
    ) -> dict[str, Any]:
        from .llm import LLMUnavailable, get_provider
        from .summarize import ask as _ask

        hits, _ = index.search(question, limit=12, doc_ids=[doc_id] if doc_id else None)
        passages = []
        with index._conn() as c:  # noqa: SLF001
            for h in hits:
                row = c.execute(
                    "SELECT text FROM chunks WHERE doc_id=? AND idx=?", (h.doc_id, h.idx)
                ).fetchone()
                if row:
                    passages.append(
                        {"doc_id": h.doc_id, "title": h.title, "page": h.page, "text": row["text"]}
                    )
        try:
            prov = get_provider(provider)
        except LLMUnavailable as exc:
            return {"error": str(exc), "passages": passages}
        return _ask(question, passages, prov).to_dict()

    server._funicular = {"store": store, "index": index}  # noqa: SLF001 - for tests
    return server


def _json_output(store: Store, doc_id: str, name: str) -> dict[str, Any]:
    doc = store.get(doc_id)
    if not doc:
        return {"error": "not found"}
    p: Path = doc.dir / name
    if not p.is_file():
        return {"doc_id": doc_id, "available": False}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}


def run(transport: str = "stdio", port: int = 8788) -> None:
    server = build_server()
    if transport == "stdio":
        server.run("stdio")
        return
    token = os.environ.get("FUNICULAR_MCP_TOKEN")
    if not token:
        raise SystemExit(
            "set FUNICULAR_MCP_TOKEN for the HTTP transport (clients send it as a Bearer token)"
        )
    app = server.streamable_http_app(host="127.0.0.1")
    import uvicorn
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    class Bearer(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            supplied = request.headers.get("authorization", "")
            if not hmac.compare_digest(supplied.encode(), f"Bearer {token}".encode()):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await call_next(request)

    app.add_middleware(Bearer)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
