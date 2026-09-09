"""Search index: chunked full text (FTS5), trigram index for scientific jargon, phonetic
expansion for garbled queries, and optional semantic vectors. Results from every route are
fused with reciprocal-rank fusion so a good hit from any of them floats to the top.

Lives in its own SQLite file next to the document store.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import unicodedata
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .embeddings import Embedder, from_blob, get_embedder, to_blob
from .textnorm import normalize, phonetic_key, tokens

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    doc_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    page INTEGER,
    text TEXT NOT NULL,
    norm TEXT NOT NULL,
    PRIMARY KEY (doc_id, idx)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    doc_id UNINDEXED, idx UNINDEXED, norm,
    tokenize = "unicode61 remove_diacritics 2 tokenchars '-_'"
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_tri USING fts5(
    doc_id UNINDEXED, idx UNINDEXED, norm, tokenize = "trigram"
);
CREATE TABLE IF NOT EXISTS vocab (
    term TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    n INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS vocab_key ON vocab(key);
CREATE TABLE IF NOT EXISTS vectors (
    doc_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,
    PRIMARY KEY (doc_id, idx)
);
CREATE TABLE IF NOT EXISTS docs (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    chunks INTEGER NOT NULL DEFAULT 0,
    embedded INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
);
"""

_TOKEN = re.compile(r"[\w\-']+", re.UNICODE)


# --------------------------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------------------------
@dataclass
class Chunk:
    idx: int
    text: str
    page: int | None = None


def chunk_text(text: str, *, target: int = 1100, overlap: int = 150) -> list[Chunk]:
    """Paragraph-aware chunks of ~target chars. Form feeds mark page breaks (as our
    extractors emit them) and set the chunk's page number."""
    chunks: list[Chunk] = []
    page = 1 if "\f" in text else None
    buf = ""
    idx = 0
    parts: list[tuple[str, int | None]] = []
    for pno, page_text in enumerate(text.split("\f"), start=1):
        for para in re.split(r"\n\s*\n", page_text):
            p = para.strip()
            if p:
                parts.append((p, pno if page is not None else None))
    cur_page: int | None = None
    for para, pno in parts:
        if cur_page is None:
            cur_page = pno
        if buf and pno != cur_page:
            # never let a chunk span a page boundary: page attribution stays exact
            chunks.append(Chunk(idx, buf.strip(), cur_page))
            idx += 1
            buf = para
            cur_page = pno
            continue
        if buf and len(buf) + len(para) + 2 > target:
            chunks.append(Chunk(idx, buf.strip(), cur_page))
            idx += 1
            tail = buf[-overlap:] if overlap else ""
            buf = (tail + "\n" + para) if tail else para
            cur_page = pno
        else:
            buf = (buf + "\n\n" + para) if buf else para
        while len(buf) > target * 2:  # a single enormous paragraph
            chunks.append(Chunk(idx, buf[: target * 2].strip(), cur_page))
            idx += 1
            buf = buf[target * 2 - overlap :]
    if buf.strip():
        chunks.append(Chunk(idx, buf.strip(), cur_page))
    return chunks


# --------------------------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------------------------
@dataclass
class Hit:
    doc_id: str
    idx: int
    page: int | None
    score: float
    snippet: str
    routes: list[str] = field(default_factory=list)
    title: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QueryInfo:
    normalized: str
    terms: list[str]
    expansions: dict[str, list[str]]
    routes: list[str]
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


class SearchIndex:
    def __init__(self, path: Path, *, embed: str = "auto") -> None:
        self.path = path
        self.embed_pref = embed
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
        finally:
            conn.close()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN")
            yield conn
            if conn.in_transaction:
                conn.execute("COMMIT")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def embedder(self) -> Embedder | None:
        return get_embedder(self.embed_pref)

    # ------------------------------------------------------------------ write
    def index_document(self, doc_id: str, title: str, text: str, *, embed: bool = True) -> int:
        chunks = chunk_text(text)
        with self._conn() as c:
            self._delete(c, doc_id)
            for ch in chunks:
                norm = normalize(ch.text)
                c.execute(
                    "INSERT INTO chunks (doc_id, idx, page, text, norm) VALUES (?,?,?,?,?)",
                    (doc_id, ch.idx, ch.page, ch.text, norm),
                )
                c.execute(
                    "INSERT INTO chunks_fts (doc_id, idx, norm) VALUES (?,?,?)",
                    (doc_id, ch.idx, norm),
                )
                c.execute(
                    "INSERT INTO chunks_tri (doc_id, idx, norm) VALUES (?,?,?)",
                    (doc_id, ch.idx, norm),
                )
            self._update_vocab(c, title + "\n" + text)
            c.execute(
                "INSERT OR REPLACE INTO docs (doc_id, title, chunks, embedded, updated_at)"
                " VALUES (?,?,?,0,?)",
                (doc_id, title, len(chunks), time.time()),
            )
        if embed and chunks:
            self.embed_document(doc_id, chunks, title)
        return len(chunks)

    def embed_document(
        self, doc_id: str, chunks: list[Chunk] | None = None, title: str = ""
    ) -> bool:
        emb = self.embedder()
        if emb is None:
            return False
        if chunks is None:
            with self._conn() as c:
                rows = c.execute(
                    "SELECT idx, page, text FROM chunks WHERE doc_id=? ORDER BY idx", (doc_id,)
                ).fetchall()
            chunks = [Chunk(r["idx"], r["text"], r["page"]) for r in rows]
        texts = [(title + "\n" + ch.text) if title else ch.text for ch in chunks]
        vecs = emb.encode(texts)
        with self._conn() as c:
            c.execute("DELETE FROM vectors WHERE doc_id=?", (doc_id,))
            for ch, v in zip(chunks, vecs, strict=True):
                c.execute(
                    "INSERT INTO vectors (doc_id, idx, model, dim, vec) VALUES (?,?,?,?,?)",
                    (doc_id, ch.idx, emb.name, emb.dim, to_blob(v)),
                )
            c.execute("UPDATE docs SET embedded=1 WHERE doc_id=?", (doc_id,))
        return True

    def remove_document(self, doc_id: str) -> None:
        with self._conn() as c:
            self._delete(c, doc_id)
            c.execute("DELETE FROM docs WHERE doc_id=?", (doc_id,))

    def _delete(self, c: sqlite3.Connection, doc_id: str) -> None:
        for table in ("chunks", "chunks_fts", "chunks_tri", "vectors"):
            c.execute(f"DELETE FROM {table} WHERE doc_id=?", (doc_id,))  # noqa: S608

    def _update_vocab(self, c: sqlite3.Connection, text: str) -> None:
        counts: dict[str, int] = defaultdict(int)
        for t in tokens(text):
            if len(t) >= 3 and not t.isdigit():
                counts[t] += 1
        for term, n in counts.items():
            key = phonetic_key(term)
            if not key:
                continue
            c.execute(
                "INSERT INTO vocab (term, key, n) VALUES (?,?,?)"
                " ON CONFLICT(term) DO UPDATE SET n = n + excluded.n",
                (term, key, n),
            )

    def stats(self) -> dict:
        with self._conn() as c:
            docs = c.execute(
                "SELECT COUNT(*) AS n, SUM(chunks) AS ch, SUM(embedded) AS e FROM docs"
            ).fetchone()
            vocab = c.execute("SELECT COUNT(*) AS n FROM vocab").fetchone()["n"]
            model = c.execute("SELECT model FROM vectors LIMIT 1").fetchone()
        return {
            "documents": docs["n"] or 0,
            "chunks": docs["ch"] or 0,
            "embedded_documents": docs["e"] or 0,
            "vocabulary": vocab,
            "embedding_model": model["model"] if model else None,
        }

    # ------------------------------------------------------------------ query
    def expand_terms(self, terms: list[str]) -> dict[str, list[str]]:
        """For each query term absent from the vocabulary, propose known terms that sound the
        same (metaphone) — this is what rescues dictated or misspelt drug names."""
        out: dict[str, list[str]] = {}
        with self._conn() as c:
            for t in terms:
                if len(t) < 4 or t.isdigit():
                    continue
                known = c.execute("SELECT 1 FROM vocab WHERE term=?", (t,)).fetchone()
                if known:
                    continue
                key = phonetic_key(t)
                if not key:
                    continue
                rows = c.execute(
                    "SELECT term, n FROM vocab WHERE key=? ORDER BY n DESC LIMIT 6", (key,)
                ).fetchall()
                cands = [r["term"] for r in rows if abs(len(r["term"]) - len(t)) <= 3]
                if cands:
                    out[t] = cands[:4]
        return out

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        mode: str = "auto",
        doc_ids: list[str] | None = None,
    ) -> tuple[list[Hit], QueryInfo]:
        t0 = time.monotonic()
        q = normalize(query)
        terms = [t for t in tokens(q) if t]
        expansions = self.expand_terms(terms) if terms else {}
        routes: list[str] = []
        ranked: dict[tuple[str, int], dict] = {}

        def add(route: str, ordered: list[tuple[str, int]]) -> None:
            if not ordered:
                return
            routes.append(route)
            for rank, key in enumerate(ordered, start=1):
                entry = ranked.setdefault(key, {"score": 0.0, "routes": []})
                entry["score"] += 1.0 / (60 + rank)  # reciprocal rank fusion
                entry["routes"].append(route)

        if terms and mode in ("auto", "keyword", "all"):
            add("keyword", self._fts(terms, expansions, limit * 3, doc_ids))
            if any(len(t) >= 5 for t in terms):
                add("jargon", self._trigram(terms, limit * 2, doc_ids))
        if q and mode in ("auto", "semantic", "all"):
            sem = self._semantic(query, limit * 2, doc_ids)
            add("semantic", sem)

        keys = sorted(ranked, key=lambda k: -ranked[k]["score"])[:limit]
        hits = self._materialise(keys, ranked, terms, expansions)
        info = QueryInfo(q, terms, expansions, sorted(set(routes)), round(time.monotonic() - t0, 3))
        return hits, info

    def _fts(self, terms, expansions, limit, doc_ids) -> list[tuple[str, int]]:
        groups = []
        for t in terms:
            alts = [t] + expansions.get(t, [])
            quoted = " OR ".join('"' + a.replace('"', '""') + '"' for a in alts)
            groups.append(f"({quoted})")
        match = " AND ".join(groups)
        # relax to OR when the strict query finds nothing (partial matches still help)
        for expr in (match, " OR ".join(groups)):
            rows = self._match("chunks_fts", expr, limit, doc_ids)
            if rows:
                return rows
        return []

    def _trigram(self, terms, limit, doc_ids) -> list[tuple[str, int]]:
        long_terms = [t for t in terms if len(t) >= 5]
        expr = " AND ".join('"' + t.replace('"', '""') + '"' for t in long_terms)
        return self._match("chunks_tri", expr, limit, doc_ids)

    def _match(self, table, expr, limit, doc_ids) -> list[tuple[str, int]]:
        if not expr:
            return []
        sql = f"SELECT doc_id, idx FROM {table} WHERE {table} MATCH ?"  # noqa: S608
        args: list = [expr]
        if doc_ids:
            sql += " AND doc_id IN (" + ",".join("?" * len(doc_ids)) + ")"
            args += doc_ids
        sql += f" ORDER BY bm25({table}) LIMIT ?"
        args.append(limit)
        try:
            with self._conn() as c:
                return [(r["doc_id"], r["idx"]) for r in c.execute(sql, args)]
        except sqlite3.OperationalError as exc:
            log.debug("fts query failed (%s): %s", expr, exc)
            return []

    def _semantic(self, query, limit, doc_ids) -> list[tuple[str, int]]:
        emb = self.embedder()
        if emb is None:
            return []
        with self._conn() as c:
            sql = "SELECT doc_id, idx, dim, vec FROM vectors WHERE model=?"
            args: list = [emb.name]
            if doc_ids:
                sql += " AND doc_id IN (" + ",".join("?" * len(doc_ids)) + ")"
                args += doc_ids
            rows = c.execute(sql, args).fetchall()
        if not rows:
            return []
        qv = emb.encode([query])[0]
        mat = np.vstack([from_blob(r["vec"], r["dim"]) for r in rows])
        sims = mat @ qv
        order = np.argsort(-sims)[:limit]
        return [(rows[i]["doc_id"], rows[i]["idx"]) for i in order if sims[i] > 0.15]

    def _materialise(self, keys, ranked, terms, expansions) -> list[Hit]:
        if not keys:
            return []
        hits: list[Hit] = []
        want = set(terms) | {a for alts in expansions.values() for a in alts}
        with self._conn() as c:
            titles = {r["doc_id"]: r["title"] for r in c.execute("SELECT doc_id, title FROM docs")}
            for doc_id, idx in keys:
                row = c.execute(
                    "SELECT page, text, norm FROM chunks WHERE doc_id=? AND idx=?", (doc_id, idx)
                ).fetchone()
                if not row:
                    continue
                hits.append(
                    Hit(
                        doc_id=doc_id,
                        idx=idx,
                        page=row["page"],
                        score=round(ranked[(doc_id, idx)]["score"], 5),
                        snippet=make_snippet(row["text"], want),
                        routes=sorted(set(ranked[(doc_id, idx)]["routes"])),
                        title=titles.get(doc_id, ""),
                    )
                )
        return hits


def _fold_keep_length(text: str) -> str:
    """Per-character fold (diacritics, case, apostrophes) that keeps every index aligned with
    the original text, so matches found here can be bracketed in the original."""
    out = []
    for ch in text:
        base = unicodedata.normalize("NFKD", ch)
        base = base[0] if base else ch
        base = base.translate(_FOLD_TABLE).lower()
        out.append(base if len(base) == 1 else ch.lower()[:1] or ch)
    return "".join(out)


_FOLD_TABLE = str.maketrans({"’": "'", "‘": "'", "‐": "-", "‑": "-", "–": "-", "—": "-"})


def make_snippet(text: str, want: set[str], width: int = 240) -> str:
    """Window around the first matched term (diacritic/case-insensitive), with [brackets]."""
    folded = _fold_keep_length(text)
    spans: list[tuple[int, int]] = []
    for w in sorted((w for w in want if w), key=len, reverse=True):
        i = folded.find(w)
        if i >= 0 and not any(a <= i < b for a, b in spans):
            spans.append((i, i + len(w)))
    if not spans:
        return text[:width].replace("\n", " ") + ("…" if len(text) > width else "")
    first = min(a for a, _ in spans)
    start = max(0, first - width // 3)
    end = min(len(text), start + width)
    pieces = []
    cursor = start
    for a, b in sorted(spans):
        if a < start or b > end:
            continue
        pieces.append(text[cursor:a])
        pieces.append("[" + text[a:b] + "]")
        cursor = b
    pieces.append(text[cursor:end])
    window = "".join(pieces).replace("\n", " ")
    return ("…" if start > 0 else "") + window + ("…" if end < len(text) else "")


def dumps(o) -> str:
    return json.dumps(o, ensure_ascii=False)
