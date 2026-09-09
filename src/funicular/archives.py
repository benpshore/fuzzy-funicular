"""Archives in and out: safe extraction of zip / tar(.gz|.bz2|.xz) with bomb limits and nested
archives, discovery of compatible files inside folders, and streaming tar.gz export.

Safety rules for extraction (all enforced here, not left to the stdlib defaults):
* members are written only below ``dest``; absolute paths, ``..`` and symlinks are skipped
* total uncompressed bytes, member count, per-member ratio and nesting depth are capped
* every member is copied in chunks with a running byte budget, so a lying header cannot
  make us write more than the budget
"""

from __future__ import annotations

import gzip
import io
import logging
import os
import shutil
import tarfile
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .sniff import Kind, sniff

log = logging.getLogger(__name__)

ARCHIVE_EXTS = (".zip", ".tar", ".tgz", ".tar.gz", ".tbz2", ".tar.bz2", ".txz", ".tar.xz")
SKIP_DIRS = {"__MACOSX", ".git", "node_modules", ".Trash", ".Spotlight-V100", ".fseventsd"}
SKIP_FILES = {".DS_Store", "Thumbs.db", "desktop.ini", "Icon\r"}
BUNDLE_SUFFIXES = (
    ".app",
    ".photoslibrary",
    ".framework",
    ".bundle",
    ".xcodeproj",
    ".numbers",
    ".pages",
    ".key",
)  # opaque macOS packages: never descend
COMPATIBLE_KINDS = {
    Kind.PDF,
    Kind.IMAGE,
    Kind.AUDIO,
    Kind.VIDEO,
    Kind.OFFICE,
    Kind.HTML,
    Kind.TEXT,
    Kind.EPUB,
}


class ArchiveError(ValueError):
    pass


@dataclass
class Limits:
    max_total_bytes: int = 20 * 2**30
    max_files: int = 20_000
    max_depth: int = 3
    max_member_bytes: int = 8 * 2**30
    max_ratio: float = 400.0  # uncompressed / compressed per member (bomb heuristic)


@dataclass
class Extracted:
    path: Path
    relpath: str  # path inside the archive, nested archives joined with "/"
    size: int


@dataclass
class ExtractReport:
    files: list[Extracted] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    total_bytes: int = 0
    nested: int = 0


def archive_kind(path: Path) -> str | None:
    """'zip' | 'tar' | None, decided by magic bytes (extension is only a hint)."""
    try:
        with path.open("rb") as fh:
            head = fh.read(512)
            fh.seek(257)
            tar_magic = fh.read(8)
    except OSError:
        return None
    if head[:4] == b"PK\x03\x04":
        return "zip"
    if head[:2] == b"\x1f\x8b" or head[:3] == b"BZh" or head[:6] == b"\xfd7zXZ\x00":
        # compressed stream: assume tar inside if it opens as tar, else a single gzip file
        try:
            with tarfile.open(path, "r:*") as tf:
                tf.next()
            return "tar"
        except tarfile.TarError, EOFError, OSError:
            return "gzip" if head[:2] == b"\x1f\x8b" else None
    if tar_magic.startswith(b"ustar"):
        return "tar"
    if path.suffix.lower() == ".tar":
        try:
            with tarfile.open(path, "r:"):
                return "tar"
        except tarfile.TarError, OSError:
            return None
    return None


def is_archive(path: Path) -> bool:
    return archive_kind(path) in ("zip", "tar", "gzip")


# --------------------------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------------------------
def _safe_relpath(name: str) -> str | None:
    name = name.replace("\\", "/")
    if not name or name.startswith("/") or name.startswith("//"):
        return None
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    if any(part in SKIP_DIRS for part in parts[:-1]) or parts[-1] in SKIP_FILES:
        return None
    if parts[-1].startswith("._"):  # AppleDouble resource forks
        return None
    return "/".join(parts)


def _copy_capped(src: io.BufferedIOBase, dst: Path, budget: int, member_cap: int) -> int:
    written = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as out:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > member_cap:
                out.close()
                dst.unlink(missing_ok=True)
                raise ArchiveError(f"member exceeds {member_cap} bytes: {dst.name}")
            if written > budget:
                out.close()
                dst.unlink(missing_ok=True)
                raise ArchiveError("archive exceeds the total extraction budget")
            out.write(chunk)
    return written


def extract_archive(
    path: Path,
    dest: Path,
    limits: Limits | None = None,
    *,
    depth: int = 0,
    prefix: str = "",
    report: ExtractReport | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ExtractReport:
    limits = limits or Limits()
    report = report or ExtractReport()
    kind = archive_kind(path)
    if kind is None:
        raise ArchiveError(f"not a supported archive: {path.name}")
    dest.mkdir(parents=True, exist_ok=True)
    if kind == "gzip":
        # single gzipped file, e.g. paper.pdf.gz
        name = path.name[:-3] if path.name.lower().endswith(".gz") else path.name + ".out"
        target = dest / name
        with gzip.open(path, "rb") as gz:
            size = _copy_capped(
                gz, target, limits.max_total_bytes - report.total_bytes, limits.max_member_bytes
            )
        report.total_bytes += size
        report.files.append(Extracted(target, prefix + name, size))
        return report

    members: list[tuple[str, int, int, Callable[[], io.BufferedIOBase]]] = []
    if kind == "zip":
        zf = zipfile.ZipFile(path)
        for info in zf.infolist():
            if info.is_dir():
                continue
            # symlinks in zips carry the S_IFLNK bit in the high 16 bits of external_attr
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                report.skipped.append(f"symlink: {info.filename}")
                continue
            rel = _safe_relpath(info.filename)
            if rel is None:
                report.skipped.append(f"unsafe or ignored: {info.filename}")
                continue
            if (
                info.compress_size
                and info.file_size / max(info.compress_size, 1) > limits.max_ratio
            ):
                report.skipped.append(f"suspicious compression ratio: {info.filename}")
                continue
            members.append((rel, info.file_size, info.compress_size, lambda i=info: zf.open(i)))  # type: ignore[return-value]
        closer = zf.close
    else:
        tf = tarfile.open(path, "r:*")
        for info in tf:
            if not info.isreg():
                if info.issym() or info.islnk():
                    report.skipped.append(f"link: {info.name}")
                continue
            rel = _safe_relpath(info.name)
            if rel is None:
                report.skipped.append(f"unsafe or ignored: {info.name}")
                continue
            members.append((rel, info.size, info.size, lambda i=info: tf.extractfile(i)))  # type: ignore[return-value]
        closer = tf.close

    try:
        total = len(members)
        for n, (rel, size, _csize, opener) in enumerate(members):
            if len(report.files) >= limits.max_files:
                report.skipped.append("file count limit reached")
                break
            if size > limits.max_member_bytes:
                report.skipped.append(f"too large: {rel}")
                continue
            target = dest / rel
            src = opener()
            if src is None:
                continue
            with src:
                written = _copy_capped(
                    src,
                    target,
                    limits.max_total_bytes - report.total_bytes,
                    limits.max_member_bytes,
                )
            report.total_bytes += written
            if progress:
                progress(n + 1, total)
            if is_archive(target) and not _looks_like_document(target):
                if depth + 1 < limits.max_depth:
                    report.nested += 1
                    sub = target.with_name(target.name + ".d")
                    extract_archive(
                        target,
                        sub,
                        limits,
                        depth=depth + 1,
                        prefix=prefix + rel + "/",
                        report=report,
                    )
                    target.unlink(missing_ok=True)
                else:
                    report.skipped.append(f"nested archive too deep: {rel}")
                continue
            report.files.append(Extracted(target, prefix + rel, written))
    finally:
        closer()
    return report


def _looks_like_document(path: Path) -> bool:
    """Office files and EPUBs are zip containers we must NOT unpack."""
    return sniff(path).kind in (Kind.OFFICE, Kind.EPUB)


# --------------------------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------------------------
@dataclass
class Found:
    path: Path
    kind: Kind
    size: int
    relpath: str


def iter_compatible(
    root: Path,
    *,
    recursive: bool = True,
    max_files: int = 50_000,
    include_archives: bool = True,
) -> Iterator[Found]:
    """Yield compatible files under root (archives included when asked), skipping junk,
    hidden entries and macOS bundles. Never follows symlinked directories."""
    root = root.resolve()
    count = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        d = Path(dirpath)
        dirnames[:] = sorted(
            n
            for n in dirnames
            if not n.startswith(".")
            and n not in SKIP_DIRS
            and not n.endswith(BUNDLE_SUFFIXES)
            and not (d / n).is_symlink()
        )
        if not recursive:
            dirnames[:] = []
        for name in sorted(filenames):
            if name.startswith(".") or name in SKIP_FILES or name.startswith("~$"):
                continue
            p = d / name
            if p.is_symlink() or not p.is_file():
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size == 0:
                continue
            kind = sniff(p).kind
            if kind in COMPATIBLE_KINDS:
                pass
            elif include_archives and kind is Kind.ARCHIVE and is_archive(p):
                pass
            else:
                continue
            yield Found(p, kind, size, str(p.relative_to(root)))
            count += 1
            if count >= max_files:
                return


def count_compatible(root: Path, *, limit: int = 2000) -> dict[str, int]:
    """Shallow-ish count for folder listings: how many documents / archives sit here."""
    docs = archives = 0
    for f in iter_compatible(root, recursive=True, max_files=limit):
        if f.kind is Kind.ARCHIVE:
            archives += 1
        else:
            docs += 1
    return {"documents": docs, "archives": archives, "capped": docs + archives >= limit}


# --------------------------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------------------------
def build_tar_gz(items: Iterable[tuple[Path, str]], out: Path) -> Path:
    """Write a tar.gz of (file, arcname) pairs to out. Spooled to disk so any size streams."""
    with tarfile.open(out, "w:gz", compresslevel=6) as tf:
        for path, arcname in items:
            if path.is_file():
                tf.add(path, arcname=arcname, recursive=False)
    return out


def tempdir(base: Path, prefix: str = "extract-") -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=base))


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
