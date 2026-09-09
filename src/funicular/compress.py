"""PDF compression with a single 0-100 "strength" knob and a dry-run preview.

Two engines, chosen automatically:

* PyMuPDF ``rewrite_images`` — re-encodes raster images (downsample + JPEG quality), keeps
  text, fonts, links and forms untouched. Fast; the default for born-digital PDFs.
* Ghostscript ``pdfwrite`` presets (/screen … /prepress) — also re-writes fonts and content
  streams; better for scans and PDFs full of vector junk; slower.

The preview compresses a sample of pages with the same settings and scales the ratio to the
whole document, so the UI can show "≈ 3.1 MB → 0.9 MB, about 12 s" before anything is written.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

from .resources import check_cancel
from .tools import ghostscript


@dataclass(frozen=True)
class Level:
    strength: int  # 0..100
    dpi: int  # target dpi for raster images
    quality: int  # JPEG quality
    gs_preset: str  # /screen /ebook /printer /prepress
    label: str


def level_for(strength: int) -> Level:
    s = max(0, min(int(strength), 100))
    if s >= 85:
        return Level(s, 72, 40, "/screen", "Smallest (72 dpi, screen)")
    if s >= 65:
        return Level(s, 110, 55, "/ebook", "Small (110 dpi, e-book)")
    if s >= 40:
        return Level(s, 150, 70, "/ebook", "Balanced (150 dpi)")
    if s >= 15:
        return Level(s, 220, 82, "/printer", "Light (220 dpi, print)")
    return Level(s, 300, 90, "/prepress", "Lossless-ish (300 dpi, prepress)")


PRESETS = {"smallest": 90, "small": 70, "balanced": 50, "light": 25, "archive": 5}


@dataclass
class Plan:
    engine: str  # pymupdf | ghostscript
    level: Level
    pages: int
    original_bytes: int
    estimated_bytes: int
    estimated_seconds: float
    sample_pages: int
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["level"] = asdict(self.level)
        d["ratio"] = round(self.estimated_bytes / max(self.original_bytes, 1), 3)
        return d


@dataclass
class Result:
    engine: str
    level: Level
    original_bytes: int
    output_bytes: int
    seconds: float
    output: Path

    def to_dict(self) -> dict:
        d = asdict(self)
        d["level"] = asdict(self.level)
        d["output"] = str(self.output)
        d["ratio"] = round(self.output_bytes / max(self.original_bytes, 1), 3)
        return d


def _choose_engine(doc: pymupdf.Document, engine: str) -> str:
    if engine in ("pymupdf", "ghostscript"):
        return engine if engine != "ghostscript" or ghostscript.available() else "pymupdf"
    # auto: scans (image-only pages) and font-heavy PDFs benefit from pdfwrite; else PyMuPDF
    n = min(doc.page_count, 5)
    image_only = 0
    for i in range(n):
        page = doc[i]
        if len("".join(page.get_text("text").split())) < 25 and page.get_images():
            image_only += 1
    if image_only * 2 > n and ghostscript.available():
        return "ghostscript"
    return "pymupdf"


def _pymupdf_compress(src: Path, dst: Path, level: Level, pages: list[int] | None) -> None:
    with pymupdf.open(src) as doc:
        if pages is not None:
            doc.select(pages)
        doc.rewrite_images(
            dpi_threshold=level.dpi + 20,
            dpi_target=level.dpi,
            quality=level.quality,
            lossy=True,
            lossless=True,
            bitonal=level.strength >= 65,
            set_to_gray=False,
        )
        doc.subset_fonts()
        doc.save(
            dst,
            garbage=4,
            deflate=True,
            deflate_images=True,
            deflate_fonts=True,
            use_objstms=True,
            clean=True,
        )


def _gs_compress(src: Path, dst: Path, level: Level, pages: list[int] | None, timeout: int) -> None:
    from .resources import run

    argv = [
        ghostscript._gs(),  # noqa: SLF001 - same package, deliberate
        "-dSAFER",
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.7",
        f"-dPDFSETTINGS={level.gs_preset}",
        "-dDetectDuplicateImages=true",
        "-dDownsampleColorImages=true",
        f"-dColorImageResolution={level.dpi}",
        "-dDownsampleGrayImages=true",
        f"-dGrayImageResolution={level.dpi}",
        "-dDownsampleMonoImages=true",
        f"-dMonoImageResolution={max(level.dpi * 2, 150)}",
        f"-dJPEGQ={level.quality}",
    ]
    if pages:
        argv += [f"-dFirstPage={pages[0] + 1}", f"-dLastPage={pages[-1] + 1}"]
    argv += [f"-sOutputFile={dst}", str(src)]
    proc = run(argv, timeout=timeout)
    if proc.returncode != 0 or not dst.exists():
        raise ghostscript.GhostscriptError(proc.stderr.decode("utf-8", "replace")[:400])


def preview(src: Path, strength: int, *, engine: str = "auto", sample: int = 4) -> Plan:
    """Dry run on a page sample; nothing is written next to the source."""
    level = level_for(strength)
    size = src.stat().st_size
    with pymupdf.open(src) as doc:
        n = doc.page_count
        chosen = _choose_engine(doc, engine)
    if n == 0:
        return Plan(chosen, level, 0, size, size, 0.0, 0, "empty document")
    k = min(sample, n)
    step = max(1, n // k)
    pages = list(range(0, n, step))[:k]
    with tempfile.TemporaryDirectory(prefix="funicular-cmp-") as td:
        out = Path(td) / "sample.pdf"
        sample_src = Path(td) / "src-sample.pdf"
        # Measure against a sample extracted the same way so overheads cancel out.
        with pymupdf.open(src) as doc:
            doc.select(pages)
            # garbage=1 keeps duplicate image objects, so the sample's size per page matches
            # the original's and the ratio transfers to the whole document.
            doc.save(sample_src, garbage=1, deflate=True)
        t0 = time.monotonic()
        if chosen == "ghostscript":
            _gs_compress(sample_src, out, level, None, timeout=600)
        else:
            _pymupdf_compress(sample_src, out, level, None)
        dt = time.monotonic() - t0
        out_bytes = out.stat().st_size
    # Extrapolate the compressed output per sampled page to the whole document; this is
    # independent of how the original happened to be encoded.
    est_bytes = min(size, int(out_bytes * n / k))
    ratio = est_bytes / max(size, 1)
    est_seconds = round(dt / k * n * 1.15 + 0.5, 1)
    note = ""
    if ratio > 0.92:
        note = "little to gain: images are already small or the file is mostly text"
    return Plan(chosen, level, n, size, est_bytes, est_seconds, k, note)


def compress(
    src: Path, dst: Path, strength: int, *, engine: str = "auto", timeout: int = 1800
) -> Result:
    level = level_for(strength)
    size = src.stat().st_size
    with pymupdf.open(src) as doc:
        chosen = _choose_engine(doc, engine)
    t0 = time.monotonic()
    check_cancel()
    tmp = dst.with_suffix(".tmp.pdf")
    if chosen == "ghostscript":
        _gs_compress(src, tmp, level, None, timeout)
    else:
        _pymupdf_compress(src, tmp, level, None)
    # Never hand back something bigger than the input.
    if tmp.stat().st_size >= size:
        shutil.copy2(src, dst)
        tmp.unlink(missing_ok=True)
        chosen += " (no gain; original kept)"
    else:
        tmp.replace(dst)
    return Result(chosen, level, size, dst.stat().st_size, time.monotonic() - t0, dst)
