"""Bring a file into the library: verify type, hash, copy into the per-document directory,
register it, and queue extraction. Shared by uploads and the iCloud inbox watcher."""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from ..config import ExtractSettings, Settings
from ..sniff import Kind, sniff
from .jobs import JobManager
from .store import Document, Store

ACCEPTED = {
    Kind.PDF,
    Kind.IMAGE,
    Kind.AUDIO,
    Kind.VIDEO,
    Kind.OFFICE,
    Kind.HTML,
    Kind.TEXT,
    Kind.EPUB,
}


class ImportError_(ValueError):
    pass


@dataclass
class Imported:
    doc: Document
    duplicate_of: Document | None = None


_NAME = re.compile(r"[^\w.\- ()\[\]&,+']+", re.UNICODE)


def display_name(name: str) -> str:
    base = Path(name).name
    base = _NAME.sub("_", base).strip(" .")
    return base[:180] or "document"


def import_stream(
    settings: Settings,
    store: Store,
    jobs: JobManager,
    stream: BinaryIO,
    filename: str,
    *,
    source: str = "upload",
    extract: ExtractSettings | None = None,
) -> Imported:
    """Copy a stream into a fresh document directory with a size cap, then register it."""
    limit = settings.max_upload_mb * 1024 * 1024
    docs_root = settings.data_dir / "docs"
    docs_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = settings.data_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix.lower()[:12]
    ext = ext if re.fullmatch(r"\.[a-z0-9]{1,10}", ext) else ""
    tmp = tmp_dir / (hashlib.sha1(filename.encode(), usedforsecurity=False).hexdigest() + ext)
    h = hashlib.sha256()
    size = 0
    with tmp.open("wb") as out:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                out.close()
                tmp.unlink(missing_ok=True)
                raise ImportError_(f"file exceeds the {settings.max_upload_mb} MB limit")
            h.update(chunk)
            out.write(chunk)
    if size == 0:
        tmp.unlink(missing_ok=True)
        raise ImportError_("empty file")
    return _register(settings, store, jobs, tmp, filename, h.hexdigest(), size, source, extract)


def import_path(
    settings: Settings,
    store: Store,
    jobs: JobManager,
    path: Path,
    *,
    source: str = "inbox",
    move: bool = True,
    extract: ExtractSettings | None = None,
) -> Imported:
    limit = settings.max_upload_mb * 1024 * 1024
    size = path.stat().st_size
    if size == 0:
        raise ImportError_("empty file")
    if size > limit:
        raise ImportError_(f"file exceeds the {settings.max_upload_mb} MB limit")
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    if not move:
        tmp_dir = settings.data_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / (h.hexdigest()[:16] + path.suffix.lower())
        shutil.copy2(path, tmp)
        path = tmp
    return _register(
        settings,
        store,
        jobs,
        path,
        path.name,
        h.hexdigest(),
        size,
        source,
        extract,
        discard_rejects=not move,  # a user's inbox file is theirs; only our temp copies go
    )


def _register(
    settings: Settings,
    store: Store,
    jobs: JobManager,
    staged: Path,
    filename: str,
    sha256: str,
    size: int,
    source: str,
    extract: ExtractSettings | None,
    *,
    discard_rejects: bool = True,
) -> Imported:
    sn = sniff(staged)
    if sn.kind not in ACCEPTED:
        if discard_rejects:
            staged.unlink(missing_ok=True)
        raise ImportError_(f"unsupported or unrecognised file type ({sn.ext or 'no extension'})")
    dup = store.find_by_sha(sha256)
    name = display_name(filename)
    ext = "." + sn.ext if sn.ext else _ext_for_kind(sn.kind)
    doc = store.create_document(
        title=name,
        original_name=name,
        kind=sn.kind.value,
        size=size,
        sha256=sha256,
        dir=settings.data_dir / "docs" / "pending",
        source=source,
    )
    doc_dir = settings.data_dir / "docs" / doc.id
    doc_dir.mkdir(parents=True, exist_ok=True)
    dest = doc_dir / f"original{ext}"
    shutil.move(str(staged), dest)
    store.set_dir(doc.id, doc_dir)
    doc = store.get(doc.id)
    assert doc is not None
    jobs.submit(doc.id, extract)
    return Imported(doc=doc, duplicate_of=dup)


def _ext_for_kind(kind: Kind) -> str:
    return {Kind.PDF: ".pdf", Kind.IMAGE: ".png", Kind.HTML: ".html", Kind.TEXT: ".txt"}.get(
        kind, ".bin"
    )
