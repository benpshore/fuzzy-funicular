"""SQLite store: documents, tags, full-text search (FTS5), sessions, OAuth state.

One connection per call, WAL mode, foreign keys on. All queries are parameterised. The FTS
query is built from quoted tokens so user input can never reach the MATCH grammar raw.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    original_name TEXT NOT NULL,
    kind TEXT NOT NULL,
    size INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    status TEXT NOT NULL,           -- queued | running | done | failed
    stage TEXT NOT NULL DEFAULT '',
    progress REAL NOT NULL DEFAULT 0,
    needs_ocr INTEGER NOT NULL DEFAULT 0,
    ocr_used INTEGER NOT NULL DEFAULT 0,
    pages INTEGER NOT NULL DEFAULT 0,
    preview TEXT NOT NULL DEFAULT '',
    warnings TEXT NOT NULL DEFAULT '[]',
    outputs TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'upload',
    dir TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_created ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS documents_sha ON documents(sha256);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE
);
CREATE TABLE IF NOT EXISTS doc_tags (
    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (doc_id, tag_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS doc_fts USING fts5(
    doc_id UNINDEXED, title, body, tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS sessions (
    id_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    login TEXT NOT NULL,
    csrf TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_seen REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS oauth_states (
    state TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    next_path TEXT NOT NULL DEFAULT '/'
);
CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,             -- archive | folder
    source TEXT NOT NULL,
    status TEXT NOT NULL,           -- queued | running | done | failed
    total INTEGER NOT NULL DEFAULT 0,
    done INTEGER NOT NULL DEFAULT 0,
    imported INTEGER NOT NULL DEFAULT 0,
    skipped INTEGER NOT NULL DEFAULT 0,
    errors TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    ts REAL NOT NULL,
    event TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT ''
);
"""


@dataclass
class Document:
    id: str
    title: str
    original_name: str
    kind: str
    size: int
    sha256: str
    created_at: float
    updated_at: float
    status: str
    stage: str
    progress: float
    needs_ocr: bool
    ocr_used: bool
    pages: int
    preview: str
    warnings: list[str]
    outputs: dict[str, str]
    error: str
    source: str
    dir: Path
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: sqlite3.Row, tags: list[str] | None = None) -> Document:
        return cls(
            id=row["id"],
            title=row["title"],
            original_name=row["original_name"],
            kind=row["kind"],
            size=row["size"],
            sha256=row["sha256"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            status=row["status"],
            stage=row["stage"],
            progress=row["progress"],
            needs_ocr=bool(row["needs_ocr"]),
            ocr_used=bool(row["ocr_used"]),
            pages=row["pages"],
            preview=row["preview"],
            warnings=json.loads(row["warnings"] or "[]"),
            outputs=json.loads(row["outputs"] or "{}"),
            error=row["error"],
            source=row["source"],
            dir=Path(row["dir"]),
            tags=tags or [],
        )

    def to_public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "original_name": self.original_name,
            "kind": self.kind,
            "size": self.size,
            "created_at": self.created_at,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "needs_ocr": self.needs_ocr,
            "ocr_used": self.ocr_used,
            "pages": self.pages,
            "warnings": self.warnings,
            "outputs": sorted(self.outputs),
            "error": self.error,
            "tags": self.tags,
        }


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # executescript manages its own transaction; keep it outside the BEGIN/COMMIT wrapper.
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
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
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

    # ---------------------------------------------------------------- documents
    def create_document(
        self,
        *,
        title: str,
        original_name: str,
        kind: str,
        size: int,
        sha256: str,
        dir: Path,
        source: str = "upload",
    ) -> Document:
        doc_id = secrets.token_urlsafe(12)
        now = time.time()
        with self._conn() as c:
            c.execute(
                "INSERT INTO documents (id,title,original_name,kind,size,sha256,created_at,"
                "updated_at,status,dir,source) VALUES (?,?,?,?,?,?,?,?,'queued',?,?)",
                (doc_id, title, original_name, kind, size, sha256, now, now, str(dir), source),
            )
        doc = self.get(doc_id)
        assert doc is not None
        return doc

    def set_dir(self, doc_id: str, directory: Path) -> None:
        with self._conn() as c:
            c.execute("UPDATE documents SET dir=? WHERE id=?", (str(directory), doc_id))

    def get(self, doc_id: str) -> Document | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            if not row:
                return None
            tags = [
                r["name"]
                for r in c.execute(
                    "SELECT t.name FROM tags t JOIN doc_tags d ON d.tag_id=t.id "
                    "WHERE d.doc_id=? ORDER BY t.name",
                    (doc_id,),
                )
            ]
        return Document.from_row(row, tags)

    def find_by_sha(self, sha256: str) -> Document | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM documents WHERE sha256=? ORDER BY created_at DESC LIMIT 1", (sha256,)
            ).fetchone()
        return Document.from_row(row) if row else None

    def list(
        self, *, limit: int = 200, status: str | None = None, tag: str | None = None
    ) -> list[Document]:
        sql = "SELECT d.* FROM documents d"
        args: list[Any] = []
        where = []
        if tag:
            sql += " JOIN doc_tags dt ON dt.doc_id=d.id JOIN tags t ON t.id=dt.tag_id"
            where.append("t.name=?")
            args.append(tag)
        if status:
            where.append("d.status=?")
            args.append(status)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY d.created_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            rows = c.execute(sql, args).fetchall()
            docs = [Document.from_row(r) for r in rows]
            self._attach_tags(c, docs)
        return docs

    def _attach_tags(self, c: sqlite3.Connection, docs: list[Document]) -> None:
        if not docs:
            return
        ids = [d.id for d in docs]
        placeholders = ",".join("?" * len(ids))  # only "?" characters, values are bound below
        by_doc: dict[str, list[str]] = {}
        head = "SELECT dt.doc_id, t.name FROM doc_tags dt JOIN tags t ON t.id=dt.tag_id"
        sql = f"{head} WHERE dt.doc_id IN ({placeholders}) ORDER BY t.name"  # noqa: S608
        for r in c.execute(sql, ids):
            by_doc.setdefault(r["doc_id"], []).append(r["name"])
        for d in docs:
            d.tags = by_doc.get(d.id, [])

    def update_progress(
        self, doc_id: str, stage: str, progress: float, status: str = "running"
    ) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE documents SET stage=?, progress=?, status=?, updated_at=? WHERE id=?",
                (stage, progress, status, time.time(), doc_id),
            )

    def finish(
        self,
        doc_id: str,
        *,
        title: str | None,
        pages: int,
        needs_ocr: bool,
        ocr_used: bool,
        preview: str,
        warnings: list[str],
        outputs: dict[str, str],
        body_text: str,
    ) -> None:
        with self._conn() as c:
            if title:
                c.execute("UPDATE documents SET title=? WHERE id=?", (title, doc_id))
            c.execute(
                "UPDATE documents SET status='done', stage='done', progress=100, pages=?,"
                " needs_ocr=?, ocr_used=?, preview=?, warnings=?, outputs=?, error='',"
                " updated_at=? WHERE id=?",
                (
                    pages,
                    int(needs_ocr),
                    int(ocr_used),
                    preview,
                    json.dumps(warnings),
                    json.dumps(outputs),
                    time.time(),
                    doc_id,
                ),
            )
            row = c.execute("SELECT title FROM documents WHERE id=?", (doc_id,)).fetchone()
            c.execute("DELETE FROM doc_fts WHERE doc_id=?", (doc_id,))
            c.execute(
                "INSERT INTO doc_fts (doc_id, title, body) VALUES (?,?,?)",
                (doc_id, row["title"], body_text[:2_000_000]),
            )

    def set_outputs(self, doc_id: str, outputs: dict[str, str]) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE documents SET outputs=?, updated_at=? WHERE id=?",
                (json.dumps(outputs), time.time(), doc_id),
            )

    def fail(self, doc_id: str, error: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE documents SET status='failed', stage='failed', error=?, updated_at=?"
                " WHERE id=?",
                (error[:2000], time.time(), doc_id),
            )

    def requeue(self, doc_id: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE documents SET status='queued', stage='', progress=0, error='', updated_at=?"
                " WHERE id=?",
                (time.time(), doc_id),
            )

    def delete(self, doc_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM doc_fts WHERE doc_id=?", (doc_id,))
            c.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    def rename(self, doc_id: str, title: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE documents SET title=?, updated_at=? WHERE id=?",
                (title, time.time(), doc_id),
            )
            c.execute("UPDATE doc_fts SET title=? WHERE doc_id=?", (title, doc_id))

    # ---------------------------------------------------------------- tags
    def set_tags(self, doc_id: str, names: list[str]) -> list[str]:
        clean = []
        for n in names:
            n = re.sub(r"\s+", " ", n).strip()[:40]
            if n and n.lower() not in {c.lower() for c in clean}:
                clean.append(n)
        with self._conn() as c:
            c.execute("DELETE FROM doc_tags WHERE doc_id=?", (doc_id,))
            for n in clean:
                c.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (n,))
                tid = c.execute("SELECT id FROM tags WHERE name=?", (n,)).fetchone()["id"]
                c.execute(
                    "INSERT OR IGNORE INTO doc_tags (doc_id, tag_id) VALUES (?,?)", (doc_id, tid)
                )
            c.execute("DELETE FROM tags WHERE id NOT IN (SELECT tag_id FROM doc_tags)")
        return clean

    def all_tags(self) -> list[tuple[str, int]]:
        with self._conn() as c:
            return [
                (r["name"], r["n"])
                for r in c.execute(
                    "SELECT t.name, COUNT(dt.doc_id) AS n FROM tags t "
                    "LEFT JOIN doc_tags dt ON dt.tag_id=t.id GROUP BY t.id ORDER BY t.name"
                )
            ]

    # ---------------------------------------------------------------- search
    def search(self, query: str, *, limit: int = 50) -> list[tuple[Document, str]]:
        match = fts_query(query)
        if not match:
            return []
        with self._conn() as c:
            rows = c.execute(
                "SELECT d.*, snippet(doc_fts, 2, '[', ']', '…', 18) AS snip "
                "FROM doc_fts f JOIN documents d ON d.id = f.doc_id "
                "WHERE doc_fts MATCH ? ORDER BY bm25(doc_fts, 5.0, 1.0) LIMIT ?",
                (match, limit),
            ).fetchall()
            docs = [Document.from_row(r) for r in rows]
            self._attach_tags(c, docs)
        return [(d, r["snip"]) for d, r in zip(docs, rows, strict=True)]

    # ---------------------------------------------------------------- sessions
    def create_session(self, user_id: int, login: str) -> tuple[str, str]:
        sid = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)
        now = time.time()
        with self._conn() as c:
            c.execute(
                "INSERT INTO sessions (id_hash,user_id,login,csrf,created_at,last_seen)"
                " VALUES (?,?,?,?,?,?)",
                (_hash(sid), user_id, login, csrf, now, now),
            )
        return sid, csrf

    def get_session(self, sid: str, *, idle_seconds: int, absolute_seconds: int) -> dict | None:
        now = time.time()
        with self._conn() as c:
            row = c.execute("SELECT * FROM sessions WHERE id_hash=?", (_hash(sid),)).fetchone()
            if not row:
                return None
            if now - row["last_seen"] > idle_seconds or now - row["created_at"] > absolute_seconds:
                c.execute("DELETE FROM sessions WHERE id_hash=?", (_hash(sid),))
                return None
            if now - row["last_seen"] > 60:
                c.execute("UPDATE sessions SET last_seen=? WHERE id_hash=?", (now, _hash(sid)))
        return dict(row)

    def delete_session(self, sid: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM sessions WHERE id_hash=?", (_hash(sid),))

    def purge_expired(self, *, idle_seconds: int, absolute_seconds: int) -> None:
        now = time.time()
        with self._conn() as c:
            c.execute(
                "DELETE FROM sessions WHERE last_seen < ? OR created_at < ?",
                (now - idle_seconds, now - absolute_seconds),
            )
            c.execute("DELETE FROM oauth_states WHERE created_at < ?", (now - 600,))

    # ---------------------------------------------------------------- oauth state
    def create_state(self, next_path: str) -> str:
        state = secrets.token_urlsafe(32)
        with self._conn() as c:
            c.execute(
                "INSERT INTO oauth_states (state, created_at, next_path) VALUES (?,?,?)",
                (state, time.time(), next_path),
            )
        return state

    def consume_state(self, state: str, *, max_age: int = 600) -> str | None:
        """Single use: returns next_path if valid and deletes it."""
        with self._conn() as c:
            row = c.execute("SELECT * FROM oauth_states WHERE state=?", (state,)).fetchone()
            if row:
                c.execute("DELETE FROM oauth_states WHERE state=?", (state,))
            if not row or time.time() - row["created_at"] > max_age:
                return None
            return row["next_path"]

    # ---------------------------------------------------------------- batches
    def create_batch(self, kind: str, source: str) -> str:
        bid = secrets.token_urlsafe(9)
        now = time.time()
        with self._conn() as c:
            c.execute(
                "INSERT INTO batches (id,kind,source,status,created_at,updated_at)"
                " VALUES (?,?,?,'queued',?,?)",
                (bid, kind, source, now, now),
            )
        return bid

    def update_batch(self, bid: str, **fields: Any) -> None:
        if not fields:
            return
        cols = []
        args: list[Any] = []
        for k, v in fields.items():
            if k not in ("status", "total", "done", "imported", "skipped", "errors"):
                raise ValueError(k)
            cols.append(f"{k}=?")
            args.append(json.dumps(v) if k == "errors" else v)
        args += [time.time(), bid]
        with self._conn() as c:
            c.execute(f"UPDATE batches SET {', '.join(cols)}, updated_at=? WHERE id=?", args)  # noqa: S608

    def get_batch(self, bid: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM batches WHERE id=?", (bid,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["errors"] = json.loads(d["errors"] or "[]")
        return d

    def list_batches(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM batches ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["errors"] = json.loads(d["errors"] or "[]")
            out.append(d)
        return out

    def add_tags(self, doc_id: str, names: list[str]) -> None:
        doc = self.get(doc_id)
        if doc is not None:
            self.set_tags(doc_id, list(doc.tags) + list(names))

    # ---------------------------------------------------------------- audit
    def audit(self, event: str, detail: str = "") -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO audit (ts, event, detail) VALUES (?,?,?)",
                (time.time(), event, detail[:500]),
            )

    def recent_audit(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            return [
                dict(r) for r in c.execute("SELECT * FROM audit ORDER BY ts DESC LIMIT ?", (limit,))
            ]


def _hash(sid: str) -> str:
    return hashlib.sha256(sid.encode()).hexdigest()


_TOKEN = re.compile(r"[\w\-']+", re.UNICODE)


def fts_query(user_query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression: quoted tokens, prefix on the last one."""
    tokens = _TOKEN.findall(user_query or "")[:12]
    if not tokens:
        return ""
    parts = ['"' + t.replace('"', '""') + '"' for t in tokens]
    parts[-1] = parts[-1] + "*"
    return " AND ".join(parts)
