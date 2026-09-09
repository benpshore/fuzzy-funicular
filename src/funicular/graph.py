"""Literature graph: citation neighbourhood of the library's papers (about 40 nodes), with
similarity edges from embeddings and a force-directed layout computed server-side.

Sources: Semantic Scholar citations/references (primary), Crossref reference DOIs, OpenAlex when
an API key is configured. The layout runs on torch (MPS on Apple Silicon) when available, else
numpy; both are exact, the device only changes speed.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .scholar.clients import Fetcher
from .scholar.model import Work

log = logging.getLogger(__name__)


@dataclass
class Node:
    id: str  # doi, or s2:<paperId> when no DOI
    title: str
    year: int | None = None
    cited_by: int | None = None
    in_library: bool = False
    doc_id: str | None = None
    doi: str | None = None
    s2_id: str | None = None
    x: float = 0.0
    y: float = 0.0
    degree: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Edge:
    source: str
    target: str
    kind: str  # cites | similar
    weight: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Graph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    seeds: list[str] = field(default_factory=list)
    device: str = "numpy"
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "seeds": self.seeds,
            "device": self.device,
            "seconds": round(self.seconds, 2),
            "notes": self.notes,
        }


def _node_id(w: Work) -> str:
    return w.doi or (f"s2:{w.s2_id}" if w.s2_id else f"t:{w.title[:60].lower()}")


def _s2_neighbours(fetcher: Fetcher, ident: str, kind: str, limit: int) -> list[Work]:
    """kind = citations | references via the Semantic Scholar graph API."""
    import os
    from urllib.parse import quote

    from .scholar.clients import parse_s2

    headers = {}
    key = os.environ.get("FUNICULAR_S2_KEY")
    if key:
        headers["x-api-key"] = key
    data = fetcher.get(
        f"https://api.semanticscholar.org/graph/v1/paper/{quote(ident, safe=':/')}/{kind}",
        params={"fields": "title,year,externalIds,citationCount,venue,authors", "limit": limit},
        headers=headers,
    )
    out: list[Work] = []
    wrapper = "citingPaper" if kind == "citations" else "citedPaper"
    for row in (data or {}).get("data") or []:
        p = row.get(wrapper) or {}
        if p.get("paperId") or p.get("title"):
            out.append(parse_s2(p))
    return out


def build_graph(
    seeds: list[dict[str, Any]],
    fetcher: Fetcher,
    *,
    max_nodes: int = 40,
    per_seed: int = 12,
    embedder=None,
    similarity_threshold: float = 0.6,
) -> Graph:
    """seeds: [{doc_id, doi, s2_id, title, year}] for library documents with a resolved record."""
    t0 = time.monotonic()
    g = Graph()
    nodes: dict[str, Node] = {}
    edges: set[tuple[str, str, str]] = set()

    def add(w: Work, *, in_library: bool = False, doc_id: str | None = None) -> str | None:
        nid = _node_id(w)
        if nid.startswith("t:") and not w.title:
            return None
        if nid not in nodes:
            if len(nodes) >= max_nodes and not in_library:
                return None
            nodes[nid] = Node(
                nid, w.title or nid, w.year, w.cited_by, in_library, doc_id, w.doi, w.s2_id
            )
        elif in_library:
            nodes[nid].in_library = True
            nodes[nid].doc_id = doc_id
        return nid

    seed_ids: list[str] = []
    for s in seeds:
        w = Work(
            doi=s.get("doi"),
            s2_id=s.get("s2_id"),
            title=s.get("title", ""),
            year=s.get("year"),
            cited_by=s.get("cited_by"),
        )
        nid = add(w, in_library=True, doc_id=s.get("doc_id"))
        if nid:
            seed_ids.append(nid)
    g.seeds = seed_ids

    for s, nid in zip(seeds, seed_ids, strict=False):
        ident = f"DOI:{s['doi']}" if s.get("doi") else (s.get("s2_id") or "")
        if not ident:
            continue
        budget_each = max(2, per_seed // 2)
        try:
            refs = _s2_neighbours(fetcher, ident, "references", budget_each)
            cites = _s2_neighbours(fetcher, ident, "citations", budget_each)
        except Exception as exc:  # noqa: BLE001
            g.notes.append(f"{nid}: neighbour lookup failed ({exc})")
            continue
        if not refs and s.get("doi"):
            # publisher elided references on S2: fall back to Crossref's reference list
            cw = fetcher.crossref_work(s["doi"])
            if cw:
                for doi in cw.referenced_dois[:budget_each]:
                    refs.append(Work(doi=doi, title=doi))
        for w in refs:
            tid = add(w)
            if tid:
                edges.add((nid, tid, "cites"))
        for w in cites:
            sid = add(w)
            if sid:
                edges.add((sid, nid, "cites"))

    # Fill in titles for DOI-only reference nodes (cheap: Crossref is cached)
    for node in list(nodes.values()):
        if node.doi and node.title == node.doi:
            cw = fetcher.crossref_work(node.doi)
            if cw:
                node.title, node.year, node.cited_by = cw.title or node.title, cw.year, cw.cited_by

    # similarity edges between nodes with titles
    if embedder is not None and len(nodes) >= 3:
        ids = [n for n, nd in nodes.items() if nd.title and not nd.title.startswith("10.")]
        try:
            vecs = embedder.encode([nodes[i].title for i in ids])
            sims = vecs @ vecs.T
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    if sims[a, b] >= similarity_threshold:
                        edges.add((ids[a], ids[b], "similar"))
        except Exception as exc:  # noqa: BLE001
            g.notes.append(f"similarity edges skipped: {exc}")

    g.nodes = list(nodes.values())
    g.edges = [Edge(a, b, k, 1.0 if k == "cites" else 0.5) for a, b, k in sorted(edges)]
    deg: dict[str, int] = {}
    for e in g.edges:
        deg[e.source] = deg.get(e.source, 0) + 1
        deg[e.target] = deg.get(e.target, 0) + 1
    for n in g.nodes:
        n.degree = deg.get(n.id, 0)
    g.device = layout(g)
    g.seconds = time.monotonic() - t0
    return g


def layout(g: Graph, iterations: int = 300, seed: int = 7) -> str:
    """Fruchterman-Reingold layout into a 1000×1000 box. Uses torch on MPS/CUDA when present."""
    n = len(g.nodes)
    if n == 0:
        return "numpy"
    index = {node.id: i for i, node in enumerate(g.nodes)}
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    adj = np.zeros((n, n), dtype=np.float32)
    for e in g.edges:
        i, j = index[e.source], index[e.target]
        adj[i, j] = adj[j, i] = max(adj[i, j], e.weight)
    device = "numpy"
    try:
        import torch

        from .resources import preferred_torch_device

        dev = preferred_torch_device()
        if dev in ("mps", "cuda"):
            device = f"torch:{dev}"
            pos = _fr_torch(torch, torch.device(dev), pos, adj, iterations)
    except Exception as exc:  # noqa: BLE001 - torch optional
        log.debug("torch layout unavailable: %s", exc)
    if device == "numpy":
        pos = _fr_numpy(pos, adj, iterations)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    for node in g.nodes:
        i = index[node.id]
        node.x = float(60 + 880 * (pos[i, 0] - lo[0]) / span[0])
        node.y = float(60 + 880 * (pos[i, 1] - lo[1]) / span[1])
    return device


def _fr_numpy(pos: np.ndarray, adj: np.ndarray, iterations: int) -> np.ndarray:
    n = len(pos)
    k = math.sqrt(4.0 / n)
    t = 0.5
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2) + 1e-6
        rep = (k * k / dist)[:, :, None] * delta / dist[:, :, None]
        att = (adj * dist * dist / k)[:, :, None] * delta / dist[:, :, None]
        disp = rep.sum(axis=1) - att.sum(axis=1)
        length = np.linalg.norm(disp, axis=1)[:, None] + 1e-6
        pos = pos + disp / length * np.minimum(length, t)
        t *= 0.985
    return pos


def _fr_torch(torch, device, pos: np.ndarray, adj: np.ndarray, iterations: int) -> np.ndarray:
    p = torch.tensor(pos, device=device)
    a = torch.tensor(adj, device=device)
    n = p.shape[0]
    k = math.sqrt(4.0 / n)
    t = 0.5
    for _ in range(iterations):
        delta = p[:, None, :] - p[None, :, :]
        dist = torch.linalg.norm(delta, dim=2) + 1e-6
        rep = (k * k / dist)[:, :, None] * delta / dist[:, :, None]
        att = (a * dist * dist / k)[:, :, None] * delta / dist[:, :, None]
        disp = rep.sum(dim=1) - att.sum(dim=1)
        length = torch.linalg.norm(disp, dim=1)[:, None] + 1e-6
        p = p + disp / length * torch.minimum(length, torch.tensor(t, device=device))
        t *= 0.985
    return p.cpu().numpy()
