import json

import httpx
import numpy as np

from funicular import graph as gmod
from funicular.scholar import clients

SEED_DOI = "10.1000/seed"


def handler(req: httpx.Request) -> httpx.Response:
    u = req.url
    if u.host == "api.semanticscholar.org" and u.path.endswith("/references"):
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "citedPaper": {
                            "paperId": "r1",
                            "title": "Reference One",
                            "year": 2010,
                            "citationCount": 50,
                            "externalIds": {"DOI": "10.1000/r1"},
                        }
                    },
                    {
                        "citedPaper": {
                            "paperId": "r2",
                            "title": "Reference Two",
                            "year": 2012,
                            "externalIds": {},
                        }
                    },
                ]
            },
        )
    if u.host == "api.semanticscholar.org" and u.path.endswith("/citations"):
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "citingPaper": {
                            "paperId": "c1",
                            "title": "Citing One",
                            "year": 2024,
                            "citationCount": 2,
                            "externalIds": {"DOI": "10.1000/c1"},
                        }
                    }
                ]
            },
        )
    if u.host == "api.crossref.org":
        doi = u.path.split("/works/")[-1]
        return httpx.Response(
            200,
            json={
                "message": {
                    "DOI": doi,
                    "title": [f"Title of {doi}"],
                    "issued": {"date-parts": [[2005]]},
                    "is-referenced-by-count": 9,
                }
            },
        )
    return httpx.Response(404)


class FakeEmbedder:
    dim = 4

    def encode(self, texts):
        vecs = []
        for t in texts:
            v = np.array(
                [1.0 if "Reference" in t else 0.0, 1.0 if "Citing" in t else 0.0, 0.1, 0.1]
            )
            vecs.append(v / np.linalg.norm(v))
        return np.vstack(vecs)


def test_build_graph_from_seed():
    f = clients.Fetcher(transport=httpx.MockTransport(handler))
    f.MIN_INTERVAL = {}
    seeds = [{"doc_id": "d1", "doi": SEED_DOI, "title": "Seed paper", "year": 2020, "cited_by": 3}]
    g = gmod.build_graph(seeds, f, max_nodes=40, embedder=FakeEmbedder())
    ids = {n.id for n in g.nodes}
    assert SEED_DOI in ids and "10.1000/r1" in ids and "s2:r2" in ids and "10.1000/c1" in ids
    seed = next(n for n in g.nodes if n.id == SEED_DOI)
    assert seed.in_library and seed.doc_id == "d1" and seed.degree == 3
    kinds = {(e.source, e.target, e.kind) for e in g.edges}
    assert (SEED_DOI, "10.1000/r1", "cites") in kinds and ("10.1000/c1", SEED_DOI, "cites") in kinds
    assert any(k == "similar" for _, _, k in kinds)  # Reference One ~ Reference Two
    assert all(60 <= n.x <= 940 and 60 <= n.y <= 940 for n in g.nodes)
    d = g.to_dict()
    assert json.dumps(d) and d["seeds"] == [SEED_DOI]
    f.close()


def test_max_nodes_and_crossref_fallback():
    def h(req: httpx.Request) -> httpx.Response:
        if req.url.host == "api.semanticscholar.org":
            return httpx.Response(200, json={"data": []})  # references elided
        return handler(req)

    f = clients.Fetcher(transport=httpx.MockTransport(h))
    f.MIN_INTERVAL = {}

    # Crossref record for the seed carries reference DOIs
    def h2(req):
        if req.url.host == "api.crossref.org" and req.url.path.endswith(SEED_DOI):
            return httpx.Response(
                200,
                json={
                    "message": {
                        "DOI": SEED_DOI,
                        "title": ["Seed"],
                        "reference": [
                            {"DOI": "10.1000/x1"},
                            {"DOI": "10.1000/x2"},
                            {"DOI": "10.1000/x3"},
                        ],
                    }
                },
            )
        return h(req)

    f2 = clients.Fetcher(transport=httpx.MockTransport(h2))
    f2.MIN_INTERVAL = {}
    g = gmod.build_graph(
        [{"doc_id": "d", "doi": SEED_DOI, "title": "Seed"}], f2, max_nodes=3, per_seed=6
    )
    assert len(g.nodes) == 3  # seed + 2 (cap)
    titles = {n.id: n.title for n in g.nodes}
    assert titles["10.1000/x1"].startswith("Title of")
    f.close()
    f2.close()


def test_layout_is_deterministic_and_bounded():
    g = gmod.Graph(
        nodes=[gmod.Node(str(i), f"n{i}") for i in range(6)],
        edges=[gmod.Edge(str(i), str((i + 1) % 6), "cites") for i in range(6)],
    )
    dev = gmod.layout(g)
    p1 = [(n.x, n.y) for n in g.nodes]
    gmod.layout(g)
    assert p1 == [(n.x, n.y) for n in g.nodes]
    assert dev in ("numpy", "torch:mps", "torch:cuda")
    assert gmod.layout(gmod.Graph()) == "numpy"
