"""PDF extraction: poppler `pdftotext -layout` for the physical layout copy, pymupdf4llm
(with the pymupdf-layout model) for reading-order Markdown, PyMuPDF for a plain text copy,
pypdfium2 as an independent second reader, Ghostscript to repair files the others reject,
and the OCR hook for scans.

Order of operations for one file:

1. Detect (PyMuPDF): per-page text / image coverage -> which pages are scans, garbled, etc.
2. Repair (Ghostscript pdfwrite) if PyMuPDF or poppler could not open the file.
3. OCR (optional): OCR the flagged pages *into* a copy of the PDF as a real text layer, then
   run every downstream engine on that copy. This is what makes OCR'd scans come out with
   pdftotext -layout fidelity and keeps a searchable PDF as a by-product.
4. Layout text (poppler), Markdown (pymupdf4llm), plain text (PyMuPDF sorted), cross-checked
   against pypdfium2's character count so a silent engine failure is reported, not hidden.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from .config import ExtractSettings
from .detect import DocSignal, analyse_pdf
from .ocr import OcrUnavailableError, make_ocr_function, select_backend
from .ocr.layout import Positioned, layout_text
from .tools import ghostscript, poppler
from .tools.binaries import MissingBinaryError

log = logging.getLogger(__name__)

Progress = Callable[[str, int, int], None]


def _noop(_stage: str, _done: int, _total: int) -> None:
    pass


@dataclass
class PdfExtraction:
    source: Path
    layout_text: str = ""
    markdown: str = ""
    plain_text: str = ""
    signal: DocSignal = field(default_factory=DocSignal)
    info: poppler.PdfInfo | None = None
    fonts_total: int = 0
    fonts_without_unicode: int = 0
    repaired: bool = False
    ocr_pages: list[int] = field(default_factory=list)
    ocr_backend: str | None = None
    ocr_pdf: Path | None = None  # searchable copy with the OCR text layer, if OCR ran
    engines: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    pdfium_chars: int | None = None
    garbage_ratio: float = 0.0

    @property
    def page_count(self) -> int:
        if self.info and self.info.pages:
            return self.info.pages
        return self.signal.page_count

    def stats(self) -> dict:
        return {
            "pages": self.page_count,
            "layout_chars": len(self.layout_text),
            "markdown_chars": len(self.markdown),
            "plain_chars": len(self.plain_text),
            "pdfium_chars": self.pdfium_chars,
            "garbage_ratio": self.garbage_ratio,
            "fonts_total": self.fonts_total,
            "fonts_without_unicode": self.fonts_without_unicode,
            "repaired": self.repaired,
            "ocr_pages": self.ocr_pages,
            "ocr_backend": self.ocr_backend,
            "engines": self.engines,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------------------------
# Fallback physical layout from PyMuPDF word boxes (used when poppler is missing)
# --------------------------------------------------------------------------------------------
def physical_layout_from_page(page: pymupdf.Page) -> str:
    words = page.get_text("words")  # x0, y0, x1, y1, word, block, line, wordno
    groups: dict[tuple[int, int], list[tuple]] = {}
    for w in words:
        groups.setdefault((w[5], w[6]), []).append(w)
    items: list[Positioned] = []
    for key in sorted(groups):
        ws = sorted(groups[key], key=lambda w: w[0])
        items.append(
            Positioned(
                text=" ".join(w[4] for w in ws),
                x0=min(w[0] for w in ws),
                y0=min(w[1] for w in ws),
                x1=max(w[2] for w in ws),
                y1=max(w[3] for w in ws),
            )
        )
    return layout_text(items)


# --------------------------------------------------------------------------------------------
# pypdfium2: independent second reader for cross-checking and rendering
# --------------------------------------------------------------------------------------------
def pdfium_char_count(path: Path) -> int:
    import pypdfium2 as pdfium

    total = 0
    pdf = pdfium.PdfDocument(str(path))
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            tp = page.get_textpage()
            try:
                total += len("".join(tp.get_text_bounded().split()))
            finally:
                tp.close()
                page.close()
    finally:
        pdf.close()
    return total


def pdfium_text(path: Path) -> str:
    """Plain text via pdfium (used when MuPDF cannot open a file that pdfium can)."""
    import pypdfium2 as pdfium

    parts: list[str] = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            tp = page.get_textpage()
            try:
                parts.append(tp.get_text_bounded())
            finally:
                tp.close()
                page.close()
    finally:
        pdf.close()
    return "\f".join(parts)


def render_page_png(path: Path, page_index: int, *, width: int = 480) -> bytes:
    """Render one page to PNG bytes with pdfium (thumbnails for the web UI)."""
    import io

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        page = pdf[page_index]
        try:
            scale = width / max(page.get_width(), 1)
            bitmap = page.render(scale=scale)
            img = bitmap.to_pil()
        finally:
            page.close()
    finally:
        pdf.close()
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# --------------------------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------------------------
def extract_pdf(
    path: Path,
    settings: ExtractSettings | None = None,
    *,
    progress: Progress = _noop,
    workdir: Path | None = None,
) -> PdfExtraction:
    settings = settings or ExtractSettings()
    result = PdfExtraction(source=path)
    tmp = tempfile.TemporaryDirectory(prefix="funicular-pdf-") if workdir is None else None
    work = Path(tmp.name) if tmp else workdir
    assert work is not None
    try:
        return _extract_pdf(path, settings, result, progress, work)
    finally:
        if tmp:
            # The OCR'd PDF must outlive the temp dir; the caller copies it out in
            # pipeline.ingest before this point, so cleanup here is safe.
            tmp.cleanup()


def _extract_pdf(
    path: Path, settings: ExtractSettings, result: PdfExtraction, progress: Progress, work: Path
) -> PdfExtraction:
    progress("detect", 0, 1)
    signal = analyse_pdf(path, settings)
    active = path

    # ---- repair ---------------------------------------------------------------------------
    poppler_ok = poppler.available()
    if poppler_ok:
        try:
            result.info = poppler.pdfinfo(active)
        except (poppler.PopplerError, MissingBinaryError) as exc:
            result.warnings.append(f"pdfinfo: {exc}")
            result.info = None
    needs_repair = signal.error is not None and signal.error.startswith("open failed")
    if poppler_ok and result.info is None:
        needs_repair = True
    if needs_repair and settings.repair_with_ghostscript and ghostscript.available():
        repaired = work / (path.stem + ".repaired.pdf")
        try:
            ghostscript.repair_pdf(path, repaired, timeout=settings.timeout_seconds)
            active = repaired
            result.repaired = True
            result.engines["ghostscript"] = "repaired"
            signal = analyse_pdf(active, settings)
            if poppler_ok:
                try:
                    result.info = poppler.pdfinfo(active)
                except poppler.PopplerError as exc:
                    result.warnings.append(f"pdfinfo after repair: {exc}")
        except ghostscript.GhostscriptError as exc:
            result.warnings.append(f"ghostscript repair failed: {exc}")
    elif needs_repair:
        result.warnings.append("file could not be opened and Ghostscript repair is unavailable")
    result.signal = signal
    progress("detect", 1, 1)

    if signal.error and signal.error.startswith(("open failed", "encrypted")):
        result.warnings.append(signal.error)
        # Last resort: pdfium may still read it.
        try:
            result.plain_text = pdfium_text(active)
            result.engines["pypdfium2"] = "plain text (fallback)"
        except Exception as exc:
            result.warnings.append(f"pypdfium2: {exc}")
        return result

    # ---- fonts ----------------------------------------------------------------------------
    if poppler_ok:
        try:
            fonts = poppler.pdffonts(active)
            result.fonts_total = len(fonts)
            result.fonts_without_unicode = sum(1 for f in fonts if f.get("uni", "").lower() == "no")
            # Base-14 fonts legitimately lack ToUnicode; only warn when the text is
            # actually garbled (checked after extraction, see _garbage_ratio below).
        except (poppler.PopplerError, MissingBinaryError) as exc:
            result.warnings.append(f"pdffonts: {exc}")

    # ---- OCR ------------------------------------------------------------------------------
    ocr_targets: list[int] = []
    if settings.ocr == "auto":
        ocr_targets = signal.needs_ocr_pages
    elif settings.ocr == "force":
        ocr_targets = [p.number for p in signal.pages]
    if settings.ocr == "off" and signal.needs_ocr_pages:
        result.warnings.append(
            f"{len(signal.needs_ocr_pages)} page(s) look like scans and OCR is off: "
            f"{_summ(signal.needs_ocr_pages)}"
        )
    if ocr_targets:
        try:
            backend = select_backend(settings)
        except OcrUnavailableError as exc:
            result.warnings.append(str(exc))
            backend = None
        if backend is not None:
            ocr_function = make_ocr_function(backend)
            ocr_pdf = work / (path.stem + ".ocr.pdf")
            doc = pymupdf.open(active)
            try:
                total = len(ocr_targets)
                for i, pno in enumerate(ocr_targets):
                    progress("ocr", i, total)
                    page = doc[pno - 1]
                    page.remove_rotation()
                    try:
                        ocr_function(page, dpi=settings.ocr_dpi, keep_ocr_text=False)
                        result.ocr_pages.append(pno)
                    except Exception as exc:  # one bad page must not sink the document
                        result.warnings.append(f"ocr page {pno}: {exc}")
                progress("ocr", total, total)
                doc.save(ocr_pdf, garbage=3, deflate=True)
            finally:
                doc.close()
            if result.ocr_pages:
                active = ocr_pdf
                result.ocr_pdf = ocr_pdf
                result.ocr_backend = backend.name
                result.engines["ocr"] = f"{backend.name} on {len(result.ocr_pages)} page(s)"
                result.signal = analyse_pdf(active, settings)

    # ---- layout text (poppler) --------------------------------------------------------------
    progress("layout", 0, 1)
    if poppler_ok:
        try:
            result.layout_text = poppler.pdftotext_layout(active, timeout=settings.timeout_seconds)
            result.engines["layout"] = "pdftotext -layout"
        except (poppler.PopplerError, MissingBinaryError) as exc:
            result.warnings.append(f"pdftotext: {exc}")
    if not result.layout_text:
        try:
            with pymupdf.open(active) as doc:
                result.layout_text = "\f".join(physical_layout_from_page(p) for p in doc)
            result.engines["layout"] = "pymupdf words -> grid (poppler unavailable)"
        except Exception as exc:
            result.warnings.append(f"layout fallback: {exc}")
    progress("layout", 1, 1)

    # ---- markdown (pymupdf4llm + pymupdf-layout) ----------------------------------------------
    try:
        result.markdown = _markdown(active, settings, progress)
        result.engines["markdown"] = "pymupdf4llm" + (
            " (layout model)" if _layout_model_active() else " (legacy)"
        )
    except Exception as exc:
        result.warnings.append(f"pymupdf4llm: {exc}")

    # ---- plain text (PyMuPDF sorted) -----------------------------------------------------------
    progress("text", 0, 1)
    try:
        with pymupdf.open(active) as doc:
            result.plain_text = "\f".join(p.get_text("text", sort=True) for p in doc)
        result.engines["text"] = "pymupdf sort=True"
    except Exception as exc:
        result.warnings.append(f"pymupdf text: {exc}")
    progress("text", 1, 1)

    # ---- garble check ------------------------------------------------------------------------
    ratio = _garbage_ratio(result.plain_text or result.layout_text)
    result.garbage_ratio = round(ratio, 4)
    if ratio > 0.05:
        hint = (
            " (no font has a ToUnicode map)"
            if (result.fonts_total and result.fonts_without_unicode == result.fonts_total)
            else ""
        )
        result.warnings.append(
            f"{ratio:.0%} of extracted characters are unmappable{hint}; consider --ocr force"
        )

    # ---- cross-check with pdfium --------------------------------------------------------------
    try:
        result.pdfium_chars = pdfium_char_count(active)
        result.engines["pypdfium2"] = "cross-check"
        ours = len("".join(result.layout_text.split()))
        if result.pdfium_chars and ours < 0.7 * result.pdfium_chars:
            result.warnings.append(
                f"layout text has {ours} chars but pdfium sees {result.pdfium_chars}; "
                "an engine may have dropped content"
            )
    except Exception as exc:
        result.warnings.append(f"pypdfium2: {exc}")
    return result


def _layout_model_active() -> bool:
    try:
        import pymupdf4llm

        return bool(getattr(pymupdf4llm, "_use_layout", False))
    except Exception:
        return False


def _markdown(path: Path, settings: ExtractSettings, progress: Progress) -> str:
    import pymupdf4llm

    with pymupdf.open(path) as doc:
        n = doc.page_count
        chunk = 8
        parts: list[str] = []
        for start in range(0, n, chunk):
            progress("markdown", start, n)
            pages = list(range(start, min(start + chunk, n)))
            kwargs: dict = {"pages": pages, "show_progress": False}
            if _layout_model_active():
                # OCR (if any) already happened into the PDF; never let pymupdf4llm re-OCR.
                kwargs.update(use_ocr=False, force_ocr=False, page_separators=True)
            parts.append(pymupdf4llm.to_markdown(doc, **kwargs))
        progress("markdown", n, n)
    return "\n".join(parts)


def _garbage_ratio(text: str) -> float:
    """Share of non-space characters that are U+FFFD, private-use, or C0/C1 controls."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    bad = 0
    for c in chars:
        o = ord(c)
        if c == "\ufffd" or 0xE000 <= o <= 0xF8FF or o < 0x20 or 0x7F <= o < 0xA0:
            bad += 1
    return bad / len(chars)


def _summ(pages: list[int]) -> str:
    if len(pages) <= 8:
        return ", ".join(map(str, pages))
    return ", ".join(map(str, pages[:8])) + f", … (+{len(pages) - 8})"
