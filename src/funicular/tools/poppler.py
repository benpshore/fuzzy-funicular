"""poppler-utils: pdftotext -layout, pdfinfo, pdftoppm, pdffonts.

Everything here shells out with an explicit argv (never a shell string) and a hard timeout,
so a hostile or malformed PDF cannot hang the ingest worker or inject arguments.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..resources import run as _run_limited
from .binaries import MissingBinaryError, find_binary

BREW_HINT = "Install with: brew install poppler   (Linux: apt-get install poppler-utils)"
DEFAULT_TIMEOUT = 600  # seconds; large scanned books can legitimately take minutes


def _require(name: str) -> str:
    path = find_binary(name, env_override=f"FUNICULAR_{name.upper()}")
    if not path:
        raise MissingBinaryError(name, BREW_HINT)
    return path


def available() -> bool:
    return find_binary("pdftotext", "FUNICULAR_PDFTOTEXT") is not None


def version() -> str | None:
    path = find_binary("pdftotext", "FUNICULAR_PDFTOTEXT")
    if not path:
        return None
    out = subprocess.run([path, "-v"], capture_output=True, text=True, timeout=30)
    # pdftotext prints its banner on stderr
    m = re.search(r"pdftotext version (\S+)", out.stderr + out.stdout)
    return m.group(1) if m else "unknown"


def _run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[bytes]:
    return _run_limited(argv, timeout=timeout)


def pdftotext_layout(
    pdf: Path,
    *,
    first: int | None = None,
    last: int | None = None,
    user_password: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Return the physical-layout text of *pdf* (pdftotext -layout), pages separated by \\f.

    -layout keeps columns, tables and indentation exactly where they sit on the page, which is
    the representation Ben asked for by default. Reading order across columns is NOT
    linearised here; see pymupdf4llm for that.
    """
    # No -nopgbrk: the form feed between pages is our page marker.
    argv = [_require("pdftotext"), "-layout", "-enc", "UTF-8"]
    if first is not None:
        argv += ["-f", str(first)]
    if last is not None:
        argv += ["-l", str(last)]
    if user_password:
        argv += ["-upw", user_password]
    argv += [str(pdf), "-"]
    proc = _run(argv, timeout)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise PopplerError(f"pdftotext exited {proc.returncode}: {err[:500]}")
    return proc.stdout.decode("utf-8", "replace")


def pdftotext_raw(pdf: Path, *, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Content-stream order text (pdftotext -raw). Useful as a cross-check for garbled fonts."""
    argv = [_require("pdftotext"), "-raw", "-enc", "UTF-8", str(pdf), "-"]
    proc = _run(argv, timeout)
    if proc.returncode != 0:
        raise PopplerError(proc.stderr.decode("utf-8", "replace")[:500])
    return proc.stdout.decode("utf-8", "replace")


class PopplerError(RuntimeError):
    pass


@dataclass
class PdfInfo:
    pages: int = 0
    title: str | None = None
    author: str | None = None
    creator: str | None = None
    producer: str | None = None
    encrypted: bool = False
    tagged: bool = False
    page_size: str | None = None
    pdf_version: str | None = None
    raw: dict[str, str] = field(default_factory=dict)


def pdfinfo(pdf: Path, *, timeout: int = 120) -> PdfInfo:
    argv = [_require("pdfinfo"), str(pdf)]
    proc = _run(argv, timeout)
    text = proc.stdout.decode("utf-8", "replace")
    if proc.returncode != 0 and not text.strip():
        raise PopplerError(proc.stderr.decode("utf-8", "replace")[:500])
    raw: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            raw[k.strip()] = v.strip()
    info = PdfInfo(raw=raw)
    info.pages = int(raw.get("Pages", "0") or 0)
    info.title = raw.get("Title") or None
    info.author = raw.get("Author") or None
    info.creator = raw.get("Creator") or None
    info.producer = raw.get("Producer") or None
    info.encrypted = not raw.get("Encrypted", "no").lower().startswith("no")
    info.tagged = raw.get("Tagged", "no").lower() == "yes"
    info.page_size = raw.get("Page size") or None
    info.pdf_version = raw.get("PDF version") or None
    return info


def pdffonts(pdf: Path, *, timeout: int = 120) -> list[dict[str, str]]:
    """Font table. A page whose fonts all lack a ToUnicode map will extract as garbage;
    the pipeline uses this to decide when OCR is the more faithful route."""
    argv = [_require("pdffonts"), str(pdf)]
    proc = _run(argv, timeout)
    lines = proc.stdout.decode("utf-8", "replace").splitlines()
    fonts: list[dict[str, str]] = []
    if len(lines) < 2:
        return fonts
    # Header:  name  type  encoding  emb sub uni object ID
    header = lines[0]
    cols = [m.start() for m in re.finditer(r"\S+", header)]
    names = header.split()
    for line in lines[2:]:
        if not line.strip():
            continue
        row: dict[str, str] = {}
        for i, (start, name) in enumerate(zip(cols, names, strict=False)):
            end = cols[i + 1] if i + 1 < len(cols) else len(line)
            row[name] = line[start:end].strip()
        fonts.append(row)
    return fonts


def pdftoppm_png(
    pdf: Path,
    out_prefix: Path,
    *,
    dpi: int = 200,
    first: int | None = None,
    last: int | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[Path]:
    """Rasterise pages to PNG files named <out_prefix>-<n>.png. Returns them in page order."""
    argv = [_require("pdftoppm"), "-png", "-r", str(dpi)]
    if first is not None:
        argv += ["-f", str(first)]
    if last is not None:
        argv += ["-l", str(last)]
    argv += [str(pdf), str(out_prefix)]
    proc = _run(argv, timeout)
    if proc.returncode != 0:
        raise PopplerError(proc.stderr.decode("utf-8", "replace")[:500])
    files = sorted(
        out_prefix.parent.glob(out_prefix.name + "-*.png"),
        key=lambda p: int(re.search(r"-(\d+)\.png$", p.name).group(1)),  # type: ignore[union-attr]
    )
    return files
