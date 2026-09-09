"""Route one input (file or URL) to the right engine and write the outputs.

Outputs next to each other in ``out_dir``:

* ``<stem>.layout.txt``  physical layout text (pdftotext -layout; OCR-reconstructed for images)
* ``<stem>.md``          reading-order Markdown (pymupdf4llm or docling)
* ``<stem>.txt``         plain text
* ``<stem>.json``        report: engines used, per-page scan signals, warnings, stats
* ``<stem>.ocr.pdf``     searchable PDF, only when OCR added a text layer
"""

from __future__ import annotations

import html
import json
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from . import __version__
from .config import ExtractSettings
from .sniff import Kind, sniff

Progress = Callable[[str, int, int], None]


def _noop(_stage: str, _done: int, _total: int) -> None:
    pass


@dataclass
class IngestResult:
    source: str
    kind: str
    out_dir: Path
    outputs: dict[str, Path] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    needs_ocr: bool = False
    seconds: float = 0.0
    title: str | None = None
    text_preview: str = ""

    def to_dict(self) -> dict:
        return {
            "funicular": __version__,
            "source": self.source,
            "kind": self.kind,
            "outputs": {k: str(v) for k, v in self.outputs.items()},
            "stats": self.stats,
            "warnings": self.warnings,
            "needs_ocr": self.needs_ocr,
            "seconds": round(self.seconds, 3),
            "title": self.title,
        }


_SAFE_STEM = re.compile(r"[^A-Za-z0-9._ -]+")


def safe_stem(name: str, limit: int = 120) -> str:
    stem = Path(name).stem
    stem = _SAFE_STEM.sub("_", stem).strip(" ._") or "document"
    return stem[:limit]


def is_url(s: str) -> bool:
    try:
        u = urlparse(s)
    except ValueError:
        return False
    return u.scheme in ("http", "https") and bool(u.netloc)


def ingest(
    source: str | Path,
    out_dir: Path,
    settings: ExtractSettings | None = None,
    *,
    progress: Progress = _noop,
    stem: str | None = None,
) -> IngestResult:
    settings = settings or ExtractSettings()
    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(source, str) and is_url(source):
        res = _ingest_url(source, out_dir, settings, progress, stem)
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        res = _ingest_file(path, out_dir, settings, progress, stem)
    res.seconds = time.monotonic() - t0
    if settings.write_json:
        report = out_dir / f"{stem or safe_stem(res.source)}.json"
        report.write_text(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
        res.outputs["report"] = report
    return res


def _write(out_dir: Path, stem: str, suffix: str, text: str) -> Path:
    p = out_dir / f"{stem}{suffix}"
    p.write_text(text, encoding="utf-8")
    return p


def _preview(text: str, n: int = 600) -> str:
    return re.sub(r"\s+", " ", text).strip()[:n]


def _ingest_file(
    path: Path, out_dir: Path, settings: ExtractSettings, progress: Progress, stem: str | None
) -> IngestResult:
    sn = sniff(path)
    stem = stem or safe_stem(path.name)
    res = IngestResult(source=str(path), kind=sn.kind.value, out_dir=out_dir)
    if sn.mismatch:
        res.warnings.append(f"extension .{sn.ext} does not match content ({sn.kind.value})")

    if sn.kind is Kind.PDF:
        return _ingest_pdf(path, out_dir, settings, progress, stem, res)
    if sn.kind is Kind.IMAGE:
        return _ingest_image(path, out_dir, settings, progress, stem, res)
    if sn.kind is Kind.TEXT:
        text = path.read_text(encoding="utf-8", errors="replace")
        if sn.ext in ("md", "markdown"):
            res.outputs["markdown"] = _write(out_dir, stem, ".md", text)
        res.outputs["text"] = _write(out_dir, stem, ".txt", text)
        res.stats = {"chars": len(text)}
        res.text_preview = _preview(text)
        return res
    if sn.kind is Kind.HTML:
        return _ingest_html(path, out_dir, settings, progress, stem, res)
    if sn.kind in (Kind.OFFICE, Kind.AUDIO, Kind.VIDEO, Kind.EPUB):
        return _ingest_docling(str(path), out_dir, settings, progress, stem, res)
    res.warnings.append(f"unsupported file type: .{sn.ext}")
    return res


def _ingest_pdf(path, out_dir, settings, progress, stem, res) -> IngestResult:
    from .pdf import extract_pdf

    with tempfile.TemporaryDirectory(prefix="funicular-work-") as td:
        ex = extract_pdf(path, settings, progress=progress, workdir=Path(td))
        if ex.ocr_pdf and ex.ocr_pdf.exists():
            dst = out_dir / f"{stem}.ocr.pdf"
            shutil.copy2(ex.ocr_pdf, dst)
            res.outputs["ocr_pdf"] = dst
    if settings.write_layout_text and ex.layout_text:
        res.outputs["layout"] = _write(out_dir, stem, ".layout.txt", ex.layout_text)
    if settings.write_markdown and ex.markdown:
        res.outputs["markdown"] = _write(out_dir, stem, ".md", ex.markdown)
    if settings.write_plain_text and ex.plain_text:
        res.outputs["text"] = _write(out_dir, stem, ".txt", ex.plain_text)
    res.stats = ex.stats()
    res.stats["signal"] = ex.signal.to_dict()
    res.warnings.extend(ex.warnings)
    res.needs_ocr = settings.ocr == "off" and bool(ex.signal.needs_ocr_pages)
    res.title = (ex.info.title if ex.info else None) or _guess_title(ex.markdown or ex.plain_text)
    res.text_preview = _preview(ex.plain_text or ex.layout_text)
    return res


def _ingest_image(path, out_dir, settings, progress, stem, res) -> IngestResult:
    res.needs_ocr = settings.ocr == "off"
    res.stats = {"photo": True}
    if settings.ocr == "off":
        res.warnings.append("image input and OCR is off; nothing to extract (use --ocr auto)")
        return res
    from .ocr import OcrUnavailableError, ocr_image

    progress("ocr", 0, 1)
    try:
        ocr = ocr_image(path, settings)
    except OcrUnavailableError as exc:
        res.warnings.append(str(exc))
        return res
    progress("ocr", 1, 1)
    res.outputs["layout"] = _write(out_dir, stem, ".layout.txt", ocr.layout)
    res.outputs["text"] = _write(out_dir, stem, ".txt", ocr.text + "\n")
    res.stats.update(
        {
            "ocr_backend": ocr.backend,
            "lines": len(ocr.lines),
            "mean_confidence": round(ocr.mean_confidence, 3),
            "width": ocr.width,
            "height": ocr.height,
        }
    )
    res.text_preview = _preview(ocr.text)
    res.title = _guess_title(ocr.text)
    return res


def _ingest_html(path, out_dir, settings, progress, stem, res) -> IngestResult:
    from . import docling_backend

    if docling_backend.available():
        return _ingest_docling(str(path), out_dir, settings, progress, stem, res)
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = strip_html(raw)
    res.outputs["text"] = _write(out_dir, stem, ".txt", text)
    res.stats = {"chars": len(text), "engine": "html.parser fallback (install docling)"}
    res.warnings.append("docling not installed; HTML reduced to plain text")
    res.text_preview = _preview(text)
    res.title = _html_title(raw)
    return res


def _ingest_docling(src, out_dir, settings, progress, stem, res) -> IngestResult:
    from . import docling_backend

    try:
        d = docling_backend.convert(src, settings, progress=progress)
    except docling_backend.DoclingUnavailableError as exc:
        res.warnings.append(str(exc))
        return res
    res.outputs["markdown"] = _write(out_dir, stem, ".md", d.markdown)
    res.outputs["text"] = _write(out_dir, stem, ".txt", d.text)
    res.stats = {
        "engine": f"docling {docling_backend.docling_version()}",
        "input_format": d.input_format,
        "pages": d.pages,
        "chars": len(d.text),
    }
    res.warnings.extend(d.warnings)
    res.text_preview = _preview(d.text)
    res.title = _guess_title(d.markdown)
    return res


def _ingest_url(url, out_dir, settings, progress, stem) -> IngestResult:
    stem = stem or safe_stem(urlparse(url).path.rsplit("/", 1)[-1] or urlparse(url).netloc)
    res = IngestResult(source=url, kind="url", out_dir=out_dir)
    return _ingest_docling(url, out_dir, settings, progress, stem, res)


# --------------------------------------------------------------------------------------------
# tiny helpers
# --------------------------------------------------------------------------------------------
def _guess_title(text: str) -> str | None:
    for line in text.splitlines():
        s = line.strip().lstrip("#").strip()
        if 4 <= len(s) <= 200 and not s.startswith(("|", "-", "*", "!")):
            return s
    return None


def _html_title(raw: str) -> str | None:
    m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
    return html.unescape(m.group(1)).strip() if m else None


def strip_html(raw: str) -> str:
    from html.parser import HTMLParser

    class _P(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.parts: list[str] = []
            self._skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "noscript"):
                self._skip += 1
            elif tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr", "section"):
                self.parts.append("\n")

        def handle_endtag(self, tag):
            if tag in ("script", "style", "noscript") and self._skip:
                self._skip -= 1

        def handle_data(self, data):
            if not self._skip:
                self.parts.append(data)

    p = _P()
    p.feed(raw)
    text = html.unescape("".join(p.parts))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
