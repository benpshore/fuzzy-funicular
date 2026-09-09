"""Content sniffing. We never trust the extension alone: uploads are checked against magic
bytes so a renamed executable or HTML file cannot masquerade as a PDF."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class Kind(StrEnum):
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    OFFICE = "office"  # docx/pptx/xlsx and friends -> docling
    HTML = "html"
    TEXT = "text"  # md/txt/csv
    EPUB = "epub"
    ARCHIVE = "archive"  # zip / tar / gzip container that is not an Office or EPUB document
    UNKNOWN = "unknown"


IMAGE_EXT = {"jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp", "heic", "heif", "gif"}
AUDIO_EXT = {"wav", "mp3", "m4a", "aac", "ogg", "flac", "aiff", "aif", "caf", "opus"}
VIDEO_EXT = {"mp4", "mov", "mkv", "webm", "avi", "m4v"}
OFFICE_EXT = {"docx", "dotx", "docm", "pptx", "potx", "ppsx", "xlsx", "xlsm", "odt", "ods", "odp"}
HTML_EXT = {"html", "htm", "xhtml"}
ARCHIVE_EXT = {"zip", "tar", "tgz", "gz", "bz2", "tbz2", "xz", "txz"}
TEXT_EXT = {"md", "markdown", "txt", "text", "csv", "adoc", "asciidoc", "tex", "vtt"}


@dataclass(frozen=True)
class Sniffed:
    kind: Kind
    ext: str  # normalised, lowercase, no dot
    mismatch: bool = False  # extension said one thing, magic bytes another


_ZIP_PATH: str | None = None  # set by sniff() so _zip_kind can look inside the container


def _zip_kind(ext: str) -> Kind:
    """Office and EPUB files are zips; decide by the container's own manifest, not the name."""
    if ext in ("docx", "dotx", "docm", "pptx", "potx", "ppsx", "xlsx", "xlsm"):
        return Kind.OFFICE
    if ext == "epub":
        return Kind.EPUB
    if _ZIP_PATH:
        import zipfile

        try:
            with zipfile.ZipFile(_ZIP_PATH) as zf:
                names = zf.namelist()[:64]
                if "[Content_Types].xml" in names:
                    return Kind.OFFICE
                if "mimetype" in names:
                    mt = zf.read("mimetype")[:64]
                    if mt.startswith(b"application/epub"):
                        return Kind.EPUB
                    if mt.startswith(b"application/vnd.oasis.opendocument"):
                        return Kind.OFFICE
        except zipfile.BadZipFile, OSError, KeyError:
            pass
    return Kind.ARCHIVE


def _magic_kind(head: bytes, ext: str) -> Kind | None:
    if head.startswith(b"%PDF-"):
        return Kind.PDF
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return Kind.IMAGE
    if head[:3] == b"\xff\xd8\xff":
        return Kind.IMAGE
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return Kind.IMAGE
    if head[:2] == b"BM":
        return Kind.IMAGE
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return Kind.IMAGE
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return Kind.IMAGE
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return Kind.AUDIO
    if head[:4] == b"fLaC" or head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3"):
        return Kind.AUDIO
    if head[:4] == b"OggS":
        return Kind.AUDIO
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return Kind.AUDIO
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"mif1", b"msf1", b"heif"):
            return Kind.IMAGE
        if brand in (b"M4A ", b"M4B ", b"M4P "):
            return Kind.AUDIO
        # isom/mp42/qt: could be audio-only m4a or video; let the extension decide.
        return Kind.AUDIO if ext in AUDIO_EXT else Kind.VIDEO
    if head[:4] == b"PK\x03\x04":
        return _zip_kind(ext)
    if head[:2] == b"\x1f\x8b" or head[:3] == b"BZh" or head[:6] == b"\xfd7zXZ\x00":
        return Kind.ARCHIVE
    if head[:4] == b"\x1a\x45\xdf\xa3":  # Matroska / WebM
        return Kind.VIDEO
    lowered = head[:512].lstrip().lower()
    if lowered.startswith((b"<!doctype html", b"<html", b"<?xml")) or b"<html" in lowered:
        return Kind.HTML
    return None


def sniff(path: Path) -> Sniffed:
    global _ZIP_PATH
    ext = path.suffix.lower().lstrip(".")
    if ext == "gz" and path.name.lower().endswith(".tar.gz"):
        ext = "tgz"
    try:
        with path.open("rb") as fh:
            head = fh.read(1024)
            fh.seek(257)
            tar_magic = fh.read(5)
    except OSError:
        head = b""
        tar_magic = b""
    _ZIP_PATH = str(path)
    try:
        by_magic = _magic_kind(head, ext)
    finally:
        _ZIP_PATH = None
    if by_magic is None and tar_magic == b"ustar":
        by_magic = Kind.ARCHIVE
    by_ext = _ext_kind(ext)
    if by_magic is None:
        # Text formats have no magic; accept the extension if the bytes look like text.
        if by_ext in (Kind.TEXT, Kind.HTML) and _looks_textual(head):
            return Sniffed(by_ext, ext)
        return Sniffed(
            by_ext if by_ext != Kind.UNKNOWN and _looks_textual(head) else Kind.UNKNOWN, ext
        )
    mismatch = by_ext not in (Kind.UNKNOWN, by_magic)
    return Sniffed(by_magic, ext, mismatch=mismatch)


def _ext_kind(ext: str) -> Kind:
    if ext == "pdf":
        return Kind.PDF
    if ext in IMAGE_EXT:
        return Kind.IMAGE
    if ext in AUDIO_EXT:
        return Kind.AUDIO
    if ext in VIDEO_EXT:
        return Kind.VIDEO
    if ext in OFFICE_EXT:
        return Kind.OFFICE
    if ext in HTML_EXT:
        return Kind.HTML
    if ext in TEXT_EXT:
        return Kind.TEXT
    if ext == "epub":
        return Kind.EPUB
    if ext in ARCHIVE_EXT:
        return Kind.ARCHIVE
    return Kind.UNKNOWN


def _looks_textual(head: bytes) -> bool:
    if not head:
        return True
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False
