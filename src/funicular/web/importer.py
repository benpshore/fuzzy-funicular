"""Bring a file into the library: verify type, hash, copy into the per-document directory,
register it, and queue extraction. Shared by uploads and the iCloud inbox watcher."""

from __future__ import annotations

import hashlib
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from ..archives import (
    ArchiveError,
    Limits,
    extract_archive,
    is_archive,
    iter_compatible,
    remove_tree,
    tempdir,
)
from ..config import ExtractSettings, Settings
from ..icloud import ensure_local
from ..resources import check_cancel
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


@dataclass
class BatchResult:
    total: int = 0
    imported: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)


_NAME = re.compile(r"[^\w.\- ()\[\]&,+']+", re.UNICODE)


def display_name(name: str) -> str:
    base = Path(name).name
    base = _NAME.sub("_", base).strip(" .")
    return base[:180] or "document"


def stage_stream(settings: Settings, stream: BinaryIO, filename: str) -> tuple[Path, str, int]:
    """Copy a stream into the staging area with a size cap. Returns (path, sha256, size)."""
    limit = settings.max_upload_mb * 1024 * 1024
    docs_root = settings.data_dir / "docs"
    docs_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = settings.data_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix.lower()[:12]
    ext = ext if re.fullmatch(r"\.[a-z0-9]{1,10}", ext) else ""
    stem = hashlib.sha1(f"{filename}{time.time_ns()}".encode(), usedforsecurity=False).hexdigest()
    tmp = tmp_dir / (stem + ext)
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
    return tmp, h.hexdigest(), size


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
    """Stage a stream and register it as one document (archives: see stage_stream + batch)."""
    tmp, sha, size = stage_stream(settings, stream, filename)
    return _register(settings, store, jobs, tmp, filename, sha, size, source, extract)


def import_path(
    settings: Settings,
    store: Store,
    jobs: JobManager,
    path: Path,
    *,
    source: str = "inbox",
    move: bool = True,
    extract: ExtractSettings | None = None,
    tags: list[str] | None = None,
    skip_duplicates: bool = False,
    filename: str | None = None,
) -> Imported:
    limit = settings.max_upload_mb * 1024 * 1024
    path = ensure_local(path, timeout=settings.icloud_wait_seconds).path
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
    if skip_duplicates:
        dup = store.find_by_sha(h.hexdigest())
        if dup is not None:
            if not move:
                path.unlink(missing_ok=True) if path.parent == settings.data_dir / "tmp" else None
            return Imported(doc=dup, duplicate_of=dup)
    return _register(
        settings,
        store,
        jobs,
        path,
        filename or path.name,
        h.hexdigest(),
        size,
        source,
        extract,
        discard_rejects=not move,  # a user's inbox file is theirs; only our temp copies go
        tags=tags,
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
    tags: list[str] | None = None,
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
    if tags:
        store.set_tags(doc.id, tags)
    doc = store.get(doc.id)
    assert doc is not None
    jobs.submit(doc.id, extract)
    return Imported(doc=doc, duplicate_of=dup)


def _ext_for_kind(kind: Kind) -> str:
    return {Kind.PDF: ".pdf", Kind.IMAGE: ".png", Kind.HTML: ".html", Kind.TEXT: ".txt"}.get(
        kind, ".bin"
    )


# --------------------------------------------------------------------------------------------
# archives and folders (run inside a batch job; see JobManager.submit_batch)
# --------------------------------------------------------------------------------------------
def _folder_tags(prefix: str, relpath: str) -> list[str]:
    tags = [prefix]
    parent = str(Path(relpath).parent)
    if parent not in ("", "."):
        tags.append("folder:" + parent.replace("\\", "/")[:80])
    return tags


def import_archive(
    settings: Settings,
    store: Store,
    jobs: JobManager,
    archive: Path,
    *,
    source: str,
    extract: ExtractSettings | None,
    progress=None,
    result: BatchResult | None = None,
    delete_after: bool = False,
    label: str | None = None,
) -> BatchResult:
    """Unpack an archive (nested ones too) and import every compatible file inside."""
    result = result or BatchResult()
    work = tempdir(settings.data_dir / "tmp", prefix="archive-")
    limits = Limits(max_total_bytes=settings.max_upload_mb * 2**20 * 4)
    try:
        report = extract_archive(archive, work, limits)
        result.skipped += len(report.skipped)
        result.errors.extend(report.skipped[:50])
        files = report.files
        result.total += len(files)
        label = "archive:" + Path(label or archive.name).name[:80]
        for n, f in enumerate(files):
            check_cancel()
            if progress:
                progress(n, len(files))
            sn = sniff(f.path)
            if sn.kind not in ACCEPTED:
                result.skipped += 1
                continue
            try:
                imp = import_path(
                    settings,
                    store,
                    jobs,
                    f.path,
                    source=source,
                    move=True,
                    extract=extract,
                    tags=_folder_tags(label, f.relpath),
                    skip_duplicates=True,
                )
                if imp.duplicate_of is not None and imp.doc.id == imp.duplicate_of.id:
                    result.skipped += 1
                else:
                    result.imported += 1
                    result.doc_ids.append(imp.doc.id)
            except ImportError_ as exc:
                result.errors.append(f"{f.relpath}: {exc}")
        if progress:
            progress(len(files), len(files))
    except ArchiveError as exc:
        result.errors.append(str(exc))
    finally:
        remove_tree(work)
        if delete_after:
            archive.unlink(missing_ok=True)
    return result


def import_folder(
    settings: Settings,
    store: Store,
    jobs: JobManager,
    folder: Path,
    *,
    recursive: bool = True,
    source: str,
    extract: ExtractSettings | None,
    progress=None,
) -> BatchResult:
    """Import every compatible file (and archive) under a folder without moving the originals."""
    result = BatchResult()
    found = list(iter_compatible(folder, recursive=recursive, max_files=settings.max_batch_files))
    result.total = len(found)
    label = "folder:" + folder.name[:80]
    for n, f in enumerate(found):
        check_cancel()
        if progress:
            progress(n, len(found))
        try:
            if f.kind is Kind.ARCHIVE and is_archive(f.path):
                import_archive(
                    settings, store, jobs, f.path, source=source, extract=extract, result=result
                )
                continue
            imp = import_path(
                settings,
                store,
                jobs,
                f.path,
                source=source,
                move=False,
                extract=extract,
                tags=_folder_tags(label, f.relpath),
                skip_duplicates=True,
            )
            if imp.duplicate_of is not None and imp.doc.id == imp.duplicate_of.id:
                result.skipped += 1
            else:
                result.imported += 1
                result.doc_ids.append(imp.doc.id)
        except (ImportError_, OSError, TimeoutError) as exc:
            result.errors.append(f"{f.relpath}: {exc}")
    if progress:
        progress(len(found), len(found))
    return result
