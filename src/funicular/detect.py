"""Scan / photo detection.

A page is a *scan* when it carries essentially no extractable text and a raster image covers
most of it. A page is *text-over-scan* when it has both (a prior OCR pass already left an
invisible text layer). Image files are photos by definition; the question there is only whether
they contain text worth OCRing, which we cannot know without OCR, so they are flagged
``needs_ocr`` whenever OCR is off.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

from .config import ExtractSettings

REPLACEMENT = "�"


@dataclass
class PageSignal:
    number: int  # 1-based
    chars: int
    words: int
    image_coverage: float  # 0..1 fraction of page area under raster images
    image_count: int
    drawing_count: int
    replacement_chars: int
    has_text: bool
    is_scan: bool
    text_over_scan: bool
    garbled: bool  # text exists but is mostly unmappable glyphs
    needs_ocr: bool
    width_pt: float = 0.0
    height_pt: float = 0.0
    rotation: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocSignal:
    pages: list[PageSignal] = field(default_factory=list)
    error: str | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def scan_pages(self) -> list[int]:
        return [p.number for p in self.pages if p.is_scan]

    @property
    def needs_ocr_pages(self) -> list[int]:
        return [p.number for p in self.pages if p.needs_ocr]

    @property
    def is_scanned_document(self) -> bool:
        """True when a clear majority of pages are image-only."""
        if not self.pages:
            return False
        return len(self.scan_pages) * 2 > len(self.pages)

    @property
    def has_any_text(self) -> bool:
        return any(p.has_text for p in self.pages)

    def to_dict(self) -> dict:
        return {
            "page_count": self.page_count,
            "scan_pages": self.scan_pages,
            "needs_ocr_pages": self.needs_ocr_pages,
            "is_scanned_document": self.is_scanned_document,
            "has_any_text": self.has_any_text,
            "error": self.error,
            "pages": [p.to_dict() for p in self.pages],
        }


def _image_coverage(page: pymupdf.Page) -> tuple[float, int]:
    """Union-free estimate: sum of clipped image bboxes over page area, capped at 1."""
    area = max(page.rect.get_area(), 1.0)
    covered = 0.0
    count = 0
    try:
        infos = page.get_image_info()
    except Exception:
        infos = []
    for info in infos:
        count += 1
        r = pymupdf.Rect(info["bbox"]) & page.rect
        if not r.is_empty:
            covered += r.get_area()
    # Also count full-page image masks drawn via Form XObjects that get_image_info misses:
    return min(covered / area, 1.0), count


def analyse_page(page: pymupdf.Page, settings: ExtractSettings) -> PageSignal:
    text = page.get_text("text") or ""
    stripped = "".join(text.split())
    chars = len(stripped)
    words = len(text.split())
    repl = stripped.count(REPLACEMENT)
    coverage, image_count = _image_coverage(page)
    try:
        drawing_count = len(page.get_drawings())
    except Exception:
        drawing_count = 0
    has_text = chars >= settings.min_chars_per_page
    garbled = has_text and repl / max(chars, 1) > 0.2
    is_scan = (not has_text) and coverage >= settings.scan_image_coverage
    text_over_scan = has_text and coverage >= 0.8 and not garbled
    needs_ocr = is_scan or garbled
    return PageSignal(
        number=page.number + 1,
        chars=chars,
        words=words,
        image_coverage=round(coverage, 4),
        image_count=image_count,
        drawing_count=drawing_count,
        replacement_chars=repl,
        has_text=has_text,
        is_scan=is_scan,
        text_over_scan=text_over_scan,
        garbled=garbled,
        needs_ocr=needs_ocr,
        width_pt=round(page.rect.width, 2),
        height_pt=round(page.rect.height, 2),
        rotation=page.rotation,
    )


def analyse_pdf(path: Path, settings: ExtractSettings | None = None) -> DocSignal:
    settings = settings or ExtractSettings()
    sig = DocSignal()
    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # corrupt or encrypted
        sig.error = f"open failed: {exc}"
        return sig
    with doc:
        if doc.needs_pass:
            sig.error = "encrypted: password required"
            return sig
        for page in doc:
            try:
                sig.pages.append(analyse_page(page, settings))
            except Exception as exc:
                sig.pages.append(
                    PageSignal(
                        number=page.number + 1,
                        chars=0,
                        words=0,
                        image_coverage=0.0,
                        image_count=0,
                        drawing_count=0,
                        replacement_chars=0,
                        has_text=False,
                        is_scan=False,
                        text_over_scan=False,
                        garbled=False,
                        needs_ocr=True,
                    )
                )
                sig.error = f"page {page.number + 1}: {exc}"
    return sig
