"""Ghostscript: repair/normalise damaged PDFs and rasterise as a last resort.

Ghostscript is deliberately used with -dSAFER and -dNOPAUSE -dBATCH, never with -dNOSAFER,
and never on a shell string. Output always goes to a caller-supplied path.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ..resources import run as _run_limited
from .binaries import MissingBinaryError, find_binary

BREW_HINT = "Install with: brew install ghostscript   (Linux: apt-get install ghostscript)"
DEFAULT_TIMEOUT = 900


class GhostscriptError(RuntimeError):
    pass


def _gs() -> str:
    path = find_binary("gs", "FUNICULAR_GS")
    if not path:
        raise MissingBinaryError("gs", BREW_HINT)
    return path


def available() -> bool:
    return find_binary("gs", "FUNICULAR_GS") is not None


def version() -> str | None:
    path = find_binary("gs", "FUNICULAR_GS")
    if not path:
        return None
    out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30)
    return out.stdout.strip() or None


def _base_argv() -> list[str]:
    return [_gs(), "-dSAFER", "-dNOPAUSE", "-dBATCH", "-dQUIET"]


def repair_pdf(src: Path, dst: Path, *, timeout: int = DEFAULT_TIMEOUT) -> Path:
    """Re-write *src* through the pdfwrite device.

    This rebuilds the xref table, re-embeds fonts and drops many of the structural faults that
    make poppler or MuPDF refuse a file. Text content is preserved; it is not rasterised.
    """
    argv = _base_argv() + [
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.7",
        "-dPDFSETTINGS=/prepress",
        "-dDetectDuplicateImages=true",
        f"-sOutputFile={dst}",
        str(src),
    ]
    proc = _run_limited(argv, timeout=timeout)
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        err = proc.stderr.decode("utf-8", "replace") + proc.stdout.decode("utf-8", "replace")
        raise GhostscriptError(f"gs pdfwrite failed ({proc.returncode}): {err[:500]}")
    return dst


def rasterize_png(
    src: Path,
    out_prefix: Path,
    *,
    dpi: int = 200,
    first: int | None = None,
    last: int | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[Path]:
    """Rasterise with the png16m device to <out_prefix>-<n>.png; returns files in page order."""
    argv = _base_argv() + [
        "-sDEVICE=png16m",
        f"-r{dpi}",
        "-dTextAlphaBits=4",
        "-dGraphicsAlphaBits=4",
    ]
    if first is not None:
        argv.append(f"-dFirstPage={first}")
    if last is not None:
        argv.append(f"-dLastPage={last}")
    argv += [f"-sOutputFile={out_prefix}-%d.png", str(src)]
    proc = _run_limited(argv, timeout=timeout)
    if proc.returncode != 0:
        raise GhostscriptError(proc.stderr.decode("utf-8", "replace")[:500])
    files = sorted(
        out_prefix.parent.glob(out_prefix.name + "-*.png"),
        key=lambda p: int(re.search(r"-(\d+)\.png$", p.name).group(1)),  # type: ignore[union-attr]
    )
    return files


def text_only_pdf(src: Path, dst: Path, *, timeout: int = DEFAULT_TIMEOUT) -> Path:
    """Write a copy with raster images and vector art filtered out (-dFILTERIMAGE/-dFILTERVECTOR).

    Handy for difficult scholarly PDFs where a full-page background image confuses layout
    analysis, and for confirming that a page's text layer is real rather than an OCR overlay.
    """
    argv = _base_argv() + [
        "-sDEVICE=pdfwrite",
        "-dFILTERIMAGE",
        "-dFILTERVECTOR",
        f"-sOutputFile={dst}",
        str(src),
    ]
    proc = _run_limited(argv, timeout=timeout)
    if proc.returncode != 0 or not dst.exists():
        raise GhostscriptError(proc.stderr.decode("utf-8", "replace")[:500])
    return dst
