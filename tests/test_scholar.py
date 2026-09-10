from pathlib import Path

import httpx
import pymupdf
import pytest

from funicular.scholar import clients, ids, metadata, refs, rename
from funicular.scholar.model import Author, Work

DOI = "10.1038/nrdp.2017.71"
CROSSREF = {
    "message": {
        "DOI": DOI,
        "title": ["Amyotrophic lateral sclerosis"],
        "container-title": ["Nature Reviews Disease Primers"],
        "author": [
            {"given": "Orla", "family": "Hardiman"},
            {"given": "Ammar", "family": "Al-Chalabi"},
            {"given": "Adriano", "family": "Chio"},
        ],
        "issued": {"date-parts": [[2017, 10, 5]]},
        "volume": "3",
        "page": "17071",
        "type": "journal-article",
        "publisher": "Springer",
        "URL": "https://doi.org/" + DOI,
        "reference-count": 200,
        "is-referenced-by-count": 1400,
        "reference": [{"DOI": "10.1000/ref1"}, {"key": "x"}],
    }
}
CROSSREF_QUERY = {
    "message": {
        "items": [
            CROSSREF["message"],
            {
                "DOI": "10.9/other",
                "title": ["Something else entirely"],
                "issued": {"date-parts": [[2001]]},
            },
        ]
    }
}
OPENALEX = {
    "id": "https://openalex.org/W4211041983",
    "doi": "https://doi.org/" + DOI,
    "title": "Amyotrophic lateral sclerosis",
    "publication_year": 2017,
    "cited_by_count": 1441,
    "referenced_works": ["https://openalex.org/W1", "https://openalex.org/W2"],
    "primary_location": {
        "source": {"display_name": "Nature Reviews Disease Primers"},
        "landing_page_url": "https://x",
    },
    "open_access": {"oa_url": "https://oa.example/als.pdf"},
    "authorships": [{"author": {"display_name": "Orla Hardiman"}}],
    "biblio": {"volume": "3", "first_page": "17071"},
    "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/28980624"},
    "type": "article",
}
S2 = {
    "paperId": "f19ad",
    "title": "Amyotrophic lateral sclerosis",
    "year": 2017,
    "citationCount": 988,
    "referenceCount": 210,
    "externalIds": {"DOI": DOI, "PubMed": "28980624"},
    "venue": "Nature Reviews Disease Primers",
    "authors": [{"name": "Orla Hardiman"}],
    "openAccessPdf": {"url": "https://s2.example/als.pdf"},
    "url": "https://s2/x",
}
PUBMED_SEARCH = {"esearchresult": {"idlist": ["28980624"]}}
PUBMED_SUMMARY = {
    "result": {
        "uids": ["28980624"],
        "28980624": {
            "uid": "28980624",
            "title": "Amyotrophic lateral sclerosis.",
            "source": "Nat Rev Dis Primers",
            "fulljournalname": "Nature reviews. Disease primers",
            "pubdate": "2017 Oct 5",
            "volume": "3",
            "pages": "17071",
            "authors": [{"name": "Hardiman O"}, {"name": "Chio A"}],
            "articleids": [{"idtype": "doi", "value": DOI}, {"idtype": "pmc", "value": "PMC9999"}],
        },
    }
}
EPMC = {
    "resultList": {
        "result": [
            {
                "pmid": "28980624",
                "doi": DOI,
                "title": "Amyotrophic lateral sclerosis.",
                "journalTitle": "Nat Rev Dis Primers",
                "pubYear": "2017",
                "authorString": "Hardiman O, Chio A.",
                "citedByCount": 1200,
            }
        ]
    }
}
SCITE = {
    "total": 1517,
    "supporting": 13,
    "contradicting": 1,
    "mentioning": 1461,
    "unclassified": 42,
    "citingPublications": 1690,
}
UNPAYWALL = {
    "best_oa_location": {
        "url_for_pdf": "https://oa.example/als.pdf",
        "url": "https://oa.example/als",
    }
}
ARXIV = """<?xml version='1.0'?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/1706.03762v7</id><published>2017-06-12T17:57:34Z</published>
<title>Attention Is All You Need</title>
<summary>The dominant sequence transduction models...</summary>
<author><name>Ashish Vaswani</name></author><author><name>Noam Shazeer</name></author>
<link title="pdf" href="http://arxiv.org/pdf/1706.03762v7" rel="related" type="application/pdf"/>
<arxiv:journal_ref>NeurIPS 2017</arxiv:journal_ref></entry></feed>"""


def _handler(request: httpx.Request) -> httpx.Response:
    u = request.url
    host, path = u.host, u.path
    calls.append(f"{host}{path}?{u.query.decode()}")
    if host == "api.crossref.org" and path == f"/works/{DOI}":
        return httpx.Response(200, json=CROSSREF)
    if host == "api.crossref.org" and path == "/works":
        return httpx.Response(200, json=CROSSREF_QUERY)
    if host == "api.crossref.org":
        return httpx.Response(404, json={"status": "error"})
    if host == "api.openalex.org" and path.startswith("/works/doi:"):
        return httpx.Response(200, json=OPENALEX)
    if host == "api.openalex.org":
        return httpx.Response(200, json={"results": [OPENALEX]})
    if host == "api.semanticscholar.org" and "/paper/search" in path:
        return httpx.Response(200, json={"data": [S2]})
    if host == "api.semanticscholar.org":
        return httpx.Response(200, json=S2)
    if host == "eutils.ncbi.nlm.nih.gov" and "esearch" in path:
        return httpx.Response(200, json=PUBMED_SEARCH)
    if host == "eutils.ncbi.nlm.nih.gov":
        pmid = u.params.get("id") or "28980624"
        rec = dict(PUBMED_SUMMARY["result"]["28980624"], uid=pmid)
        return httpx.Response(200, json={"result": {"uids": [pmid], pmid: rec}})
    if host == "www.ebi.ac.uk":
        return httpx.Response(200, json=EPMC)
    if host == "api.scite.ai":
        return httpx.Response(200, json=SCITE)
    if host == "api.unpaywall.org":
        return httpx.Response(200, json=UNPAYWALL)
    if host == "export.arxiv.org":
        return httpx.Response(200, text=ARXIV)
    return httpx.Response(500)


calls: list[str] = []


@pytest.fixture
def fetcher(tmp_path):
    calls.clear()
    f = clients.Fetcher(
        cache=clients.Cache(tmp_path / "cache.sqlite3"), transport=httpx.MockTransport(_handler)
    )
    f.MIN_INTERVAL = {}  # no pacing in tests
    yield f
    f.close()


# ---------------------------------------------------------------- ids
def test_extract_ids():
    text = (
        "Nature Reviews 2017. https://doi.org/10.1038/nrdp.2017.71. PMID: 28980624 PMC1234567 "
        "arXiv:1706.03762v2 ISBN 978-0-306-40615-7\nReferences\n1. Foo. doi:10.1000/ref.one."
    )
    r = ids.extract_ids(text)
    assert r.doi == DOI and r.all_dois == [DOI, "10.1000/ref.one"]
    assert r.arxiv == "1706.03762v2" and r.pmid == "28980624" and r.pmcid == "PMC1234567"
    assert r.isbn == "9780306406157"
    assert ids.normalize_doi("https://doi.org/10.1000/ABC.Received") == "10.1000/abc"
    assert ids.isbn_valid("0-306-40615-2") == "0306406152" and ids.isbn_valid("0306406153") is None
    assert not ids.extract_ids("nothing here").any


# ---------------------------------------------------------------- parsers
def test_parsers():
    w = clients.parse_crossref(CROSSREF["message"])
    assert (
        w.doi == DOI
        and w.year == 2017
        and w.first_author == "Hardiman"
        and w.referenced_dois == ["10.1000/ref1"]
    )
    o = clients.parse_openalex(OPENALEX)
    assert (
        o.openalex_id == "W4211041983" and o.oa_url and o.pmid == "28980624" and o.pages == "17071"
    )
    s = clients.parse_s2(S2)
    assert s.s2_id == "f19ad" and s.doi == DOI and s.cited_by == 988
    p = clients.parse_pubmed(PUBMED_SUMMARY["result"]["28980624"])
    assert p.pmid == "28980624" and p.doi == DOI and p.pmcid == "PMC9999" and p.year == 2017
    e = clients.parse_europepmc(EPMC["resultList"]["result"][0])
    assert e.pmid == "28980624" and e.authors[0].family == "Hardiman"
    a = clients.parse_arxiv(ARXIV)
    assert (
        a.arxiv == "1706.03762v7"
        and a.title == "Attention Is All You Need"
        and a.oa_url.endswith("v7")
    )
    assert a.authors[0].family == "Vaswani" and a.journal == "NeurIPS 2017"
    assert clients.parse_arxiv("<bad") is None


def test_fetcher_caches_and_handles_404(fetcher, tmp_path):
    w = fetcher.crossref_work(DOI)
    assert w and w.title.startswith("Amyotrophic")
    n = len(calls)
    fetcher.crossref_work(DOI)
    assert len(calls) == n  # served from cache
    assert fetcher.crossref_work("10.9999/missing") is None
    assert fetcher.scite_tallies(DOI)["supporting"] == 13
    assert fetcher.unpaywall(DOI).endswith(".pdf")
    assert fetcher.pubmed_by_doi(DOI).pmid == "28980624"
    assert fetcher.europepmc("DOI:" + DOI).pmcid is None


# ---------------------------------------------------------------- metadata
def _pdf_with(tmp_path: Path, title: str, body: str, meta: dict | None = None) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 80), title, fontsize=18)
    y = 120
    for line in body.splitlines():
        page.insert_text((50, y), line, fontsize=10)
        y += 14
    if meta:
        doc.set_metadata(meta)
    p = tmp_path / "paper.pdf"
    doc.save(p)
    doc.close()
    return p


def test_extract_and_resolve_by_doi(fetcher, tmp_path):
    p = _pdf_with(
        tmp_path,
        "Amyotrophic lateral sclerosis",
        "Orla Hardiman, Ammar Al-Chalabi, Adriano Chio\nNature Reviews Disease Primers 2017\n"
        f"doi:{DOI}",
        {"title": "Microsoft Word - draft.docx", "author": "Hardiman"},
    )
    ex = metadata.extract_from_pdf(p)
    assert ex.ids.doi == DOI and ex.pdf_title == "" and "Amyotrophic" in ex.guessed_title
    res = metadata.resolve(ex, fetcher)
    assert res.verified and res.confidence >= 0.8
    w = res.work
    assert w.doi == DOI and w.journal.startswith("Nature") and w.cited_by == 1400
    assert "openalex" in w.sources and "semanticscholar" in w.sources and "pubmed" in w.sources
    assert w.oa_url and w.scite["supporting"] == 13 and w.openalex_id and w.pmcid == "PMC9999"
    assert any("DOI printed" in e for e in res.evidence)


def test_resolve_by_title_when_no_doi(fetcher, tmp_path):
    p = _pdf_with(tmp_path, "Amyotrophic lateral sclerosis", "Hardiman O et al. 2017\nA primer.")
    ex = metadata.extract_from_pdf(p)
    assert ex.ids.doi is None
    res = metadata.resolve(ex, fetcher)
    assert res.work and res.work.doi == DOI and res.verified
    assert any("title search" in e for e in res.evidence)


def test_resolution_is_rejected_when_page_disagrees(fetcher, tmp_path):
    p = _pdf_with(tmp_path, "Mediterranean cooking for beginners", f"Recipes. doi:{DOI}")
    res = metadata.resolve(metadata.extract_from_pdf(p), fetcher)
    assert res.work is not None and not res.verified
    assert any("NOT confirmed" in e for e in res.evidence)


def test_arxiv_route(fetcher, tmp_path):
    p = _pdf_with(
        tmp_path,
        "Attention Is All You Need",
        "Ashish Vaswani, Noam Shazeer\narXiv:1706.03762v7 2017",
    )
    res = metadata.resolve(metadata.extract_from_pdf(p), fetcher, sources=("arxiv",))
    assert res.work and res.work.arxiv == "1706.03762v7" and res.verified


# ---------------------------------------------------------------- references
REFTEXT = """Body text discussing things.

References

1. Hardiman O, Al-Chalabi A, Chio A. Amyotrophic lateral sclerosis. Nat Rev Dis Primers.
2017;3:17071. doi:10.1038/nrdp.2017.71
2. Miller RG, Mitchell JD, Moore DH. Riluzole for amyotrophic lateral sclerosis (ALS)/motor
neuron disease (MND). Cochrane Database Syst Rev. 2012;(3):CD001447. PMID: 22419278
3. Smith J. A completely unknown manuscript about nothing in particular. Journal of Obscurity.
1999;1:1-2.
"""


def test_extract_references_numbered():
    out = refs.extract_references(REFTEXT)
    assert [r.n for r in out] == [1, 2, 3]
    assert out[0].doi == DOI and out[0].year == 2017 and out[0].first_author == "Hardiman"
    assert out[1].pmid == "22419278" and "Riluzole" in out[1].raw
    assert out[2].title_guess.startswith("A completely unknown")


def test_extract_references_author_year():
    text = """Bibliography

Hardiman, O., Al-Chalabi, A. (2017). Amyotrophic lateral sclerosis. Nature Reviews Disease
Primers, 3, 17071.
Miller, R. G., Mitchell, J. D. (2012). Riluzole for ALS. Cochrane Database, 3.
"""
    out = refs.extract_references(text)
    assert len(out) == 2 and out[0].year == 2017 and out[1].first_author == "Miller"
    assert refs.extract_references("no references here") == []


def test_resolve_references(fetcher):
    out = refs.extract_references(REFTEXT)
    r0 = refs.resolve_reference(out[0], fetcher)
    assert r0.method == "doi" and r0.confidence > 0.95
    r1 = refs.resolve_reference(out[1], fetcher)
    assert r1.method == "pmid" and r1.work.pmid == "22419278"
    r2 = refs.resolve_reference(out[2], fetcher)
    assert r2.method == "unresolved" and r2.ref.raw.startswith("3. Smith")
    rep = refs.ReferenceReport([r0, r1, r2])
    assert rep.resolved == 2 and rep.to_dict()["total"] == 3


# ---------------------------------------------------------------- rename
def test_rename_plan(tmp_path):
    w = Work(
        doi=DOI,
        title="Amyotrophic lateral sclerosis: a review/overview?",
        year=2017,
        authors=[Author("Hardiman", "Orla"), Author("Chio"), Author("Shaw")],
        journal="Nat Rev",
    )
    assert (
        rename.build_name(w)
        == "Hardiman et al. - 2017 - Amyotrophic lateral sclerosis a reviewoverview"
    )
    w2 = Work(title="X", authors=[Author("A"), Author("B")])
    assert rename.build_name(w2) == "A and B - n.d. - X"
    src = tmp_path / "1234.pdf"
    src.write_bytes(b"%PDF")
    plan = rename.plan_rename(src, w)
    assert plan.changed and plan.target.name.endswith(".pdf") and plan.target.parent == tmp_path
    (tmp_path / plan.target.name).write_bytes(b"%PDF")  # collision
    plan2 = rename.plan_rename(src, w)
    assert plan2.target.name.endswith(" (2).pdf")
    rename.apply_rename(plan2)
    assert plan2.target.exists() and not src.exists()
    assert rename.build_name(w, "{year}-{first_author}-{journal}") == "2017-Hardiman-Nat Rev"
    assert rename.build_name(w, "{nope}") == rename.build_name(w)


def test_fetcher_treats_non_json_200_as_miss(tmp_path):
    def handler(request):
        return httpx.Response(200, text="<html>gateway hiccup</html>")

    f = clients.Fetcher(
        cache=clients.Cache(tmp_path / "c.sqlite3"), transport=httpx.MockTransport(handler)
    )
    f.MIN_INTERVAL = {}
    try:
        assert f.crossref_work(DOI) is None
    finally:
        f.close()


def test_cache_closes_connections(tmp_path):
    import psutil

    cache = clients.Cache(tmp_path / "c.sqlite3")
    me = psutil.Process()
    base = me.num_fds()
    for i in range(50):
        cache.put(f"k{i}", {"i": i})
        assert cache.get(f"k{i}") == {"i": i}
    assert me.num_fds() - base < 5
