"""Scholarly feeds: RSS/Atom (journals, bioRxiv/medRxiv), arXiv queries, PubMed queries.

Entries are stored once; "staging" an entry finds an open-access PDF (arXiv, Unpaywall,
OpenAlex, Europe PMC) and imports it into the library so extraction, indexing, metadata and
(when an LLM is configured) the evidence card are ready before you open it. Polling is manual
or on an in-process interval while the server runs — no cron, no external scheduler.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx

from .scholar.clients import Fetcher
from .scholar.ids import DOI_RE, DOI_URL_RE, normalize_doi

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS feeds (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,          -- rss | arxiv | pubmed
    name TEXT NOT NULL,
    query TEXT NOT NULL,         -- URL for rss, search expression for arxiv/pubmed
    added_at REAL NOT NULL,
    last_polled REAL,
    last_status TEXT NOT NULL DEFAULT '',
    auto_stage INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY,
    feed_id INTEGER NOT NULL REFERENCES feeds(id) ON DELETE CASCADE,
    guid TEXT NOT NULL,
    title TEXT NOT NULL,
    link TEXT NOT NULL DEFAULT '',
    doi TEXT,
    arxiv TEXT,
    pmid TEXT,
    published TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    seen_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'new',   -- new | staged | imported | skipped | failed
    doc_id TEXT,
    note TEXT NOT NULL DEFAULT '',
    UNIQUE (feed_id, guid)
);
"""


@dataclass
class Entry:
    guid: str
    title: str
    link: str = ""
    doi: str | None = None
    arxiv: str | None = None
    pmid: str | None = None
    published: str = ""
    summary: str = ""


class FeedStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")  # poll thread and requests both write
            conn.executescript(SCHEMA)
        finally:
            conn.close()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add(self, kind: str, name: str, query: str, *, auto_stage: bool = False) -> int:
        if kind not in ("rss", "arxiv", "pubmed"):
            raise ValueError("kind must be rss, arxiv or pubmed")
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO feeds (kind, name, query, added_at, auto_stage) VALUES (?,?,?,?,?)",
                (kind, name[:120], query[:2000], time.time(), int(auto_stage)),
            )
            return int(cur.lastrowid)

    def remove(self, feed_id: int) -> None:
        with self._conn() as c:
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("DELETE FROM entries WHERE feed_id=?", (feed_id,))
            c.execute("DELETE FROM feeds WHERE id=?", (feed_id,))

    def list(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT f.*, (SELECT COUNT(*) FROM entries e WHERE e.feed_id=f.id) AS entries,"
                " (SELECT COUNT(*) FROM entries e WHERE e.feed_id=f.id AND e.status='new') AS new"
                " FROM feeds f ORDER BY f.added_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get(self, feed_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM feeds WHERE id=?", (feed_id,)).fetchone()
        return dict(row) if row else None

    def upsert_entries(self, feed_id: int, entries: list[Entry]) -> int:
        added = 0
        with self._conn() as c:
            for e in entries:
                cur = c.execute(
                    "INSERT OR IGNORE INTO entries (feed_id, guid, title, link, doi, arxiv, pmid,"
                    " published, summary, seen_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        feed_id,
                        e.guid,
                        e.title[:500],
                        e.link[:1000],
                        e.doi,
                        e.arxiv,
                        e.pmid,
                        e.published[:40],
                        e.summary[:4000],
                        time.time(),
                    ),
                )
                added += cur.rowcount
            c.execute(
                "UPDATE feeds SET last_polled=?, last_status=? WHERE id=?",
                (time.time(), f"{len(entries)} entries, {added} new", feed_id),
            )
        return added

    def entries(
        self, feed_id: int | None = None, *, status: str | None = None, limit: int = 200
    ) -> list[dict]:
        sql = "SELECT e.*, f.name AS feed_name FROM entries e JOIN feeds f ON f.id=e.feed_id"
        where, args = [], []
        if feed_id is not None:
            where.append("e.feed_id=?")
            args.append(feed_id)
        if status:
            where.append("e.status=?")
            args.append(status)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY e.seen_at DESC, e.id DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, args)]

    def entry(self, entry_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
        return dict(row) if row else None

    def set_status(
        self, entry_id: int, status: str, *, doc_id: str | None = None, note: str = ""
    ) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE entries SET status=?, doc_id=COALESCE(?, doc_id), note=? WHERE id=?",
                (status, doc_id, note[:500], entry_id),
            )

    def mark_failed(self, feed_id: int, msg: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE feeds SET last_polled=?, last_status=? WHERE id=?",
                (time.time(), ("error: " + msg)[:200], feed_id),
            )


# --------------------------------------------------------------------------------------------
# polling
# --------------------------------------------------------------------------------------------
def _ids_from(text: str) -> tuple[str | None, str | None]:
    m = DOI_URL_RE.search(text) or DOI_RE.search(text)
    doi = normalize_doi(m.group(1)) if m else None
    a = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?", text, re.I)
    arxiv = (a.group(1) + (a.group(2) or "")) if a else None
    return doi, arxiv


def parse_feed_text(text: str) -> list[Entry]:
    import feedparser

    parsed = feedparser.parse(text)
    out: list[Entry] = []
    for e in parsed.entries:
        link = e.get("link", "") or ""
        blob = " ".join(
            str(e.get(k, ""))
            for k in ("id", "link", "summary", "title", "dc_identifier", "prism_doi")
        )
        doi, arxiv = _ids_from(blob)
        guid = (
            e.get("id")
            or link
            or hashlib.sha1((e.get("title", "") + link).encode(), usedforsecurity=False).hexdigest()
        )
        published = e.get("published", "") or e.get("updated", "") or ""
        summary = re.sub(r"<[^>]+>", " ", e.get("summary", "") or "")
        summary = re.sub(r"\s+", " ", summary).strip()
        out.append(
            Entry(
                guid=str(guid),
                title=(e.get("title", "") or "").strip(),
                link=link,
                doi=doi,
                arxiv=arxiv,
                published=str(published),
                summary=summary,
            )
        )
    return out


def arxiv_query_url(query: str, max_results: int = 50) -> str:
    return (
        f"https://export.arxiv.org/api/query?search_query={quote(query)}&sortBy=submittedDate"
        f"&sortOrder=descending&max_results={max_results}"
    )


class Poller:
    def __init__(
        self, store: FeedStore, fetcher: Fetcher, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.store = store
        self.fetcher = fetcher
        self.http = httpx.Client(
            timeout=30,
            follow_redirects=True,
            transport=transport,
            headers={"User-Agent": "fuzzy-funicular/0.1 feeds"},
        )

    def close(self) -> None:
        self.http.close()

    def poll(self, feed_id: int) -> int:
        feed = self.store.get(feed_id)
        if not feed:
            raise KeyError(feed_id)
        try:
            if feed["kind"] == "rss":
                entries = parse_feed_text(self._get_text(feed["query"]))
            elif feed["kind"] == "arxiv":
                entries = parse_feed_text(self._get_text(arxiv_query_url(feed["query"])))
            else:
                entries = self._pubmed(feed["query"])
        except Exception as exc:  # noqa: BLE001 - recorded on the feed, never raised into the UI
            self.store.mark_failed(feed_id, str(exc))
            return 0
        return self.store.upsert_entries(feed_id, entries)

    def poll_all(self) -> dict[int, int]:
        return {f["id"]: self.poll(f["id"]) for f in self.store.list()}

    def _get_text(self, url: str) -> str:
        r = self.http.get(url)
        r.raise_for_status()
        return r.text

    def _pubmed(self, query: str, retmax: int = 50) -> list[Entry]:
        data = self.fetcher.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": retmax,
                "sort": "date",
            },
            cache_key=f"pubmed-feed:{query}:{int(time.time() // 3600)}",
        )
        ids = ((data or {}).get("esearchresult") or {}).get("idlist") or []
        out: list[Entry] = []
        for pmid in ids:
            w = self.fetcher.pubmed_summary(pmid)
            if not w:
                continue
            out.append(
                Entry(
                    guid=f"pmid:{pmid}",
                    title=w.title,
                    link=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    doi=w.doi,
                    pmid=pmid,
                    published=str(w.year or ""),
                    summary=w.journal,
                )
            )
        return out

    # ---------------------------------------------------------------- staging
    def find_pdf_url(self, entry: dict) -> str | None:
        if entry.get("arxiv"):
            return f"https://arxiv.org/pdf/{entry['arxiv']}"
        doi = entry.get("doi")
        if doi:
            for fn in (
                self.fetcher.unpaywall,
                lambda d: (self.fetcher.openalex_work(d) or _NoOA()).oa_url,
            ):
                try:
                    url = fn(doi)
                except Exception as exc:  # noqa: BLE001
                    log.debug("oa lookup failed: %s", exc)
                    url = None
                if url:
                    return url
        if entry.get("pmid"):
            w = self.fetcher.europepmc(f"EXT_ID:{entry['pmid']} AND SRC:MED")
            if w and w.pmcid:
                return f"https://www.ebi.ac.uk/europepmc/webservices/rest/{w.pmcid}/fullTextXML"
        return None

    def download_pdf(self, url: str, dest: Path, *, max_bytes: int = 200 * 2**20) -> Path:
        try:
            with self.http.stream("GET", url) as r:
                r.raise_for_status()
                size = 0
                with dest.open("wb") as fh:
                    for chunk in r.iter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError("PDF larger than the staging limit")
                        fh.write(chunk)
            with dest.open("rb") as fh:
                head = fh.read(5)
            if not head.startswith(b"%PDF"):
                raise ValueError("URL did not return a PDF")
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
        return dest


class _NoOA:
    oa_url = None
