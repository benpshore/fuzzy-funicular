"""HTTP clients for the metadata sources, with a persistent cache, per-host rate limiting,
retries, polite headers, and pure parsers (the parsers are what the tests pin down).

Keys / etiquette (all optional, via environment):
  FUNICULAR_CONTACT_EMAIL    mailto for Crossref polite pool, OpenAlex, Unpaywall
  FUNICULAR_OPENALEX_KEY     OpenAlex API key (their free daily budget is small)
  FUNICULAR_S2_KEY           Semantic Scholar API key (higher rate limits)
  FUNICULAR_NCBI_KEY         NCBI E-utilities key
  FUNICULAR_SCITE_KEY        Scite API token (tallies are public; more needs a token)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from .model import Author, Work, clean_title

log = logging.getLogger(__name__)
UA = "fuzzy-funicular/0.1 (https://github.com/benpshore/fuzzy-funicular)"


def contact_email() -> str:
    return os.environ.get("FUNICULAR_CONTACT_EMAIL", "") or "funicular@example.invalid"


class Cache:
    def __init__(self, path: Path | None, ttl: float = 7 * 86400) -> None:
        self.path = path
        self.ttl = ttl
        self._lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as c:
                c.execute(
                    "CREATE TABLE IF NOT EXISTS http_cache (k TEXT PRIMARY KEY, t REAL, v TEXT)"
                )

    def get(self, key: str) -> Any | None:
        if not self.path:
            return None
        with self._lock, sqlite3.connect(self.path) as c:
            row = c.execute("SELECT t, v FROM http_cache WHERE k=?", (key,)).fetchone()
        if not row or time.time() - row[0] > self.ttl:
            return None
        return json.loads(row[1])

    def put(self, key: str, value: Any) -> None:
        if not self.path:
            return
        with self._lock, sqlite3.connect(self.path) as c:
            c.execute(
                "INSERT OR REPLACE INTO http_cache (k, t, v) VALUES (?,?,?)",
                (key, time.time(), json.dumps(value)),
            )


class Fetcher:
    """One client for every source. `transport` lets tests inject httpx.MockTransport."""

    MIN_INTERVAL = {
        "api.crossref.org": 0.1,
        "api.openalex.org": 0.15,
        "api.semanticscholar.org": 1.0,  # 1 rps unauthenticated
        "eutils.ncbi.nlm.nih.gov": 0.35,
        "www.ebi.ac.uk": 0.2,
        "api.scite.ai": 0.5,
        "api.unpaywall.org": 0.2,
        "export.arxiv.org": 3.0,
    }

    def __init__(
        self,
        cache: Cache | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.cache = cache or Cache(None)
        headers = {"User-Agent": f"{UA} mailto:{contact_email()}", "Accept": "application/json"}
        self.client = httpx.Client(
            headers=headers, timeout=timeout, transport=transport, follow_redirects=True
        )
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def close(self) -> None:
        self.client.close()

    def get(
        self,
        url: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
        cache_key: str | None = None,
        json_body: bool = True,
    ) -> Any | None:
        key = cache_key or (url + "?" + json.dumps(params or {}, sort_keys=True))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        host = httpx.URL(url).host
        for attempt in range(3):
            with self._lock:
                wait = self.MIN_INTERVAL.get(host, 0.2) - (
                    time.monotonic() - self._last.get(host, 0)
                )
                if wait > 0:
                    time.sleep(wait)
                self._last[host] = time.monotonic()
            try:
                r = self.client.get(url, params=params, headers=headers)
            except httpx.HTTPError as exc:
                log.info("%s: %s", url, exc)
                time.sleep(0.5 * (attempt + 1))
                continue
            if r.status_code == 404:
                self.cache.put(key, {"__missing__": True})
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                retry = float(r.headers.get("Retry-After", "0") or 0)
                time.sleep(min(max(retry, 1.0 * (attempt + 1)), 10))
                continue
            if r.status_code >= 400:
                log.info("%s -> %s", url, r.status_code)
                return None
            data = r.json() if json_body else r.text
            self.cache.put(key, data)
            return data
        return None

    # ---------------------------------------------------------------- sources
    def crossref_work(self, doi: str) -> Work | None:
        data = self.get(f"https://api.crossref.org/works/{quote(doi, safe='/')}")
        if not data or "__missing__" in data:
            return None
        return parse_crossref(data.get("message", {}))

    def crossref_query(self, bibliographic: str, rows: int = 5) -> list[Work]:
        data = self.get(
            "https://api.crossref.org/works",
            params={
                "query.bibliographic": bibliographic[:500],
                "rows": rows,
                "mailto": contact_email(),
            },
        )
        items = ((data or {}).get("message") or {}).get("items") or []
        return [parse_crossref(i) for i in items]

    def openalex_work(self, doi: str) -> Work | None:
        params = {
            "mailto": contact_email(),
            "select": "id,doi,title,publication_year,cited_by_count,referenced_works,"
            "primary_location,open_access,authorships,biblio,ids,type",
        }
        key = os.environ.get("FUNICULAR_OPENALEX_KEY")
        if key:
            params["api_key"] = key
        data = self.get(f"https://api.openalex.org/works/doi:{quote(doi, safe='/')}", params=params)
        if not data or "__missing__" in data:
            return None
        return parse_openalex(data)

    def openalex_search(self, title: str, rows: int = 5) -> list[Work]:
        params = {
            "search": title[:300],
            "per-page": rows,
            "mailto": contact_email(),
            "select": "id,doi,title,publication_year,cited_by_count,primary_location,"
            "open_access,authorships,biblio,ids,type",
        }
        key = os.environ.get("FUNICULAR_OPENALEX_KEY")
        if key:
            params["api_key"] = key
        data = self.get("https://api.openalex.org/works", params=params)
        return [parse_openalex(w) for w in ((data or {}).get("results") or [])]

    def s2_paper(self, ident: str) -> Work | None:
        """ident: DOI:..., ARXIV:..., PMID:..., or an S2 paper id."""
        headers = {}
        key = os.environ.get("FUNICULAR_S2_KEY")
        if key:
            headers["x-api-key"] = key
        fields = (
            "title,year,authors,venue,externalIds,citationCount,referenceCount,abstract,"
            "openAccessPdf,url"
        )
        data = self.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{quote(ident, safe=':/')}",
            params={"fields": fields},
            headers=headers,
        )
        if not data or "__missing__" in data:
            return None
        return parse_s2(data)

    def s2_search(self, title: str, rows: int = 5) -> list[Work]:
        headers = {}
        key = os.environ.get("FUNICULAR_S2_KEY")
        if key:
            headers["x-api-key"] = key
        fields = "title,year,authors,venue,externalIds,citationCount,referenceCount,url"
        data = self.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": title[:300], "limit": rows, "fields": fields},
            headers=headers,
        )
        return [parse_s2(p) for p in ((data or {}).get("data") or [])]

    def pubmed_by_doi(self, doi: str) -> Work | None:
        params = {"db": "pubmed", "term": f"{doi}[doi]", "retmode": "json"}
        key = os.environ.get("FUNICULAR_NCBI_KEY")
        if key:
            params["api_key"] = key
        data = self.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params)
        ids = ((data or {}).get("esearchresult") or {}).get("idlist") or []
        if not ids:
            return None
        return self.pubmed_summary(ids[0])

    def pubmed_summary(self, pmid: str) -> Work | None:
        params = {"db": "pubmed", "id": pmid, "retmode": "json"}
        key = os.environ.get("FUNICULAR_NCBI_KEY")
        if key:
            params["api_key"] = key
        data = self.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", params=params
        )
        result = (data or {}).get("result") or {}
        rec = result.get(pmid)
        return parse_pubmed(rec) if rec else None

    def europepmc(self, query: str) -> Work | None:
        data = self.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "format": "json", "resultType": "lite", "pageSize": 1},
        )
        results = (((data or {}).get("resultList") or {}).get("result")) or []
        return parse_europepmc(results[0]) if results else None

    def scite_tallies(self, doi: str) -> dict | None:
        headers = {}
        key = os.environ.get("FUNICULAR_SCITE_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data = self.get(f"https://api.scite.ai/tallies/{quote(doi, safe='/')}", headers=headers)
        if not data or "__missing__" in data:
            return None
        return {
            k: data.get(k)
            for k in (
                "total",
                "supporting",
                "contradicting",
                "mentioning",
                "unclassified",
                "citingPublications",
            )
        }

    def unpaywall(self, doi: str) -> str | None:
        data = self.get(
            f"https://api.unpaywall.org/v2/{quote(doi, safe='/')}",
            params={"email": contact_email()},
        )
        if not data or "__missing__" in data:
            return None
        loc = data.get("best_oa_location") or {}
        return loc.get("url_for_pdf") or loc.get("url")

    def arxiv(self, arxiv_id: str) -> Work | None:
        text = self.get(
            "https://export.arxiv.org/api/query", params={"id_list": arxiv_id}, json_body=False
        )
        if not text:
            return None
        return parse_arxiv(text)


# ------------------------------------------------------------------------------------------
# parsers (pure)
# ------------------------------------------------------------------------------------------
def _year(parts: Any) -> int | None:
    try:
        return int(parts["date-parts"][0][0])
    except KeyError, IndexError, TypeError, ValueError:
        return None


def parse_crossref(m: dict) -> Work:
    w = Work(source="crossref")
    w.doi = (m.get("DOI") or "").lower() or None
    w.title = clean_title((m.get("title") or [""])[0]) if m.get("title") else ""
    w.journal = (m.get("container-title") or [""])[0] if m.get("container-title") else ""
    w.authors = [
        Author(a.get("family", "") or a.get("name", ""), a.get("given", ""))
        for a in m.get("author") or []
    ]
    w.year = (
        _year(m.get("issued") or {})
        or _year(m.get("published-print") or {})
        or _year(m.get("created") or {})
    )
    w.volume = str(m.get("volume") or "")
    w.issue = str(m.get("issue") or "")
    w.pages = str(m.get("page") or "")
    w.publisher = m.get("publisher") or ""
    w.type = m.get("type") or ""
    w.url = m.get("URL") or ""
    w.references_count = m.get("reference-count")
    w.cited_by = m.get("is-referenced-by-count")
    w.abstract = clean_title(m.get("abstract") or "")
    w.referenced_dois = [r["DOI"].lower() for r in m.get("reference") or [] if r.get("DOI")]
    w.sources = ["crossref"]
    return w


def parse_openalex(d: dict) -> Work:
    w = Work(source="openalex")
    w.openalex_id = (d.get("id") or "").rsplit("/", 1)[-1] or None
    w.doi = ((d.get("doi") or "").replace("https://doi.org/", "").lower()) or None
    w.title = clean_title(d.get("title") or d.get("display_name") or "")
    w.year = d.get("publication_year")
    w.cited_by = d.get("cited_by_count")
    w.type = d.get("type") or ""
    loc = d.get("primary_location") or {}
    w.journal = ((loc.get("source") or {}).get("display_name")) or ""
    w.url = loc.get("landing_page_url") or ""
    oa = d.get("open_access") or {}
    w.oa_url = oa.get("oa_url")
    bib = d.get("biblio") or {}
    w.volume, w.issue = str(bib.get("volume") or ""), str(bib.get("issue") or "")
    if bib.get("first_page"):
        w.pages = str(bib["first_page"]) + (f"-{bib['last_page']}" if bib.get("last_page") else "")
    for a in d.get("authorships") or []:
        name = (a.get("author") or {}).get("display_name") or ""
        if name:
            parts = name.split()
            w.authors.append(Author(parts[-1], " ".join(parts[:-1])))
    ids = d.get("ids") or {}
    w.pmid = (ids.get("pmid") or "").rsplit("/", 1)[-1] or None
    w.referenced_dois = []
    w.sources = ["openalex"]
    w.referenced_works = [r.rsplit("/", 1)[-1] for r in d.get("referenced_works") or []]  # type: ignore[attr-defined]
    return w


def parse_s2(p: dict) -> Work:
    w = Work(source="semanticscholar")
    w.s2_id = p.get("paperId")
    ext = p.get("externalIds") or {}
    w.doi = (ext.get("DOI") or "").lower() or None
    w.pmid = str(ext.get("PubMed")) if ext.get("PubMed") else None
    w.arxiv = ext.get("ArXiv")
    w.title = clean_title(p.get("title") or "")
    w.year = p.get("year")
    w.journal = p.get("venue") or ""
    w.cited_by = p.get("citationCount")
    w.references_count = p.get("referenceCount")
    w.abstract = p.get("abstract") or ""
    w.url = p.get("url") or ""
    oa = p.get("openAccessPdf") or {}
    w.oa_url = oa.get("url")
    for a in p.get("authors") or []:
        name = a.get("name") or ""
        if name:
            parts = name.split()
            w.authors.append(Author(parts[-1], " ".join(parts[:-1])))
    w.sources = ["semanticscholar"]
    return w


def parse_pubmed(rec: dict) -> Work:
    w = Work(source="pubmed")
    w.pmid = str(rec.get("uid") or "") or None
    w.title = clean_title(rec.get("title") or "").rstrip(".")
    w.journal = rec.get("fulljournalname") or rec.get("source") or ""
    w.volume, w.issue, w.pages = (
        str(rec.get("volume") or ""),
        str(rec.get("issue") or ""),
        str(rec.get("pages") or ""),
    )
    pd = rec.get("pubdate") or ""
    w.year = int(pd[:4]) if pd[:4].isdigit() else None
    for a in rec.get("authors") or []:
        name = a.get("name") or ""
        if name:
            parts = name.split(" ", 1)
            w.authors.append(Author(parts[0], parts[1] if len(parts) > 1 else ""))
    for aid in rec.get("articleids") or []:
        if aid.get("idtype") == "doi":
            w.doi = aid.get("value", "").lower() or None
        if aid.get("idtype") == "pmc":
            w.pmcid = aid.get("value")
    w.sources = ["pubmed"]
    return w


def parse_europepmc(r: dict) -> Work:
    w = Work(source="europepmc")
    w.pmid = r.get("pmid")
    w.pmcid = r.get("pmcid")
    w.doi = (r.get("doi") or "").lower() or None
    w.title = clean_title(r.get("title") or "").rstrip(".")
    w.journal = r.get("journalTitle") or ""
    w.year = int(r["pubYear"]) if str(r.get("pubYear", "")).isdigit() else None
    for name in (r.get("authorString") or "").split(","):
        name = name.strip().rstrip(".")
        if name:
            parts = name.split(" ", 1)
            w.authors.append(Author(parts[0], parts[1] if len(parts) > 1 else ""))
    w.cited_by = r.get("citedByCount")
    w.sources = ["europepmc"]
    return w


def parse_arxiv(atom: str) -> Work | None:
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    try:
        root = ET.fromstring(atom)  # noqa: S314 - arXiv's own feed; defused not needed for trusted host
    except ET.ParseError:
        return None
    entry = root.find("a:entry", ns)
    if entry is None:
        return None
    w = Work(source="arxiv")
    w.title = clean_title(entry.findtext("a:title", default="", namespaces=ns))
    w.abstract = clean_title(entry.findtext("a:summary", default="", namespaces=ns))
    aid = entry.findtext("a:id", default="", namespaces=ns)
    w.arxiv = aid.rsplit("/abs/", 1)[-1] or None
    w.url = aid
    pub = entry.findtext("a:published", default="", namespaces=ns)
    w.year = int(pub[:4]) if pub[:4].isdigit() else None
    doi = entry.findtext("arxiv:doi", default="", namespaces=ns)
    w.doi = doi.lower() or None
    w.journal = entry.findtext("arxiv:journal_ref", default="", namespaces=ns) or "arXiv"
    for a in entry.findall("a:author", ns):
        name = a.findtext("a:name", default="", namespaces=ns)
        if name:
            parts = name.split()
            w.authors.append(Author(parts[-1], " ".join(parts[:-1])))
    for link in entry.findall("a:link", ns):
        if link.get("title") == "pdf":
            w.oa_url = link.get("href")
    w.sources = ["arxiv"]
    return w
