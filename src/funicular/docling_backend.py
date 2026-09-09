"""docling bridge: audio (ASR), Office documents, HTML/web articles, EPUB, video, and images.

docling is an optional extra (`uv sync --extra docling`). Everything here imports it lazily so
the core PDF path never pays for torch at import time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import ExtractSettings

log = logging.getLogger(__name__)

Progress = Callable[[str, int, int], None]


class DoclingUnavailableError(RuntimeError):
    pass


@dataclass
class DoclingResult:
    markdown: str
    text: str
    input_format: str
    pages: int = 0
    meta: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def available() -> bool:
    try:
        import docling  # noqa: F401
    except Exception:
        return False
    return True


def docling_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("docling")
    except Exception:
        return None


def _asr_options(name: str):
    from docling.datamodel import asr_model_specs
    from docling.datamodel.asr_model_specs import AsrModelType

    key = name.strip().lower()
    valid = {m.value for m in AsrModelType}
    if key not in valid:
        raise ValueError(f"unknown ASR model {name!r}; valid: {sorted(valid)}")
    return getattr(asr_model_specs, key.upper())


def build_converter(settings: ExtractSettings):
    """A DocumentConverter with our defaults: ASR model from settings, macOS Vision OCR when the
    caller turned OCR on and we are on a Mac, otherwise no OCR (docling's default OCR engines
    would silently download models)."""
    import sys

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AsrPipelineOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import (
        AudioFormatOption,
        DocumentConverter,
        ImageFormatOption,
        PdfFormatOption,
    )

    asr = AsrPipelineOptions(asr_options=_asr_options(settings.asr_model))

    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = settings.ocr != "off"
    if pdf_opts.do_ocr and sys.platform == "darwin":
        try:
            from docling.datamodel.pipeline_options import OcrMacOptions

            pdf_opts.ocr_options = OcrMacOptions(lang=list(settings.ocr_languages))
        except Exception as exc:  # pragma: no cover - darwin only
            log.warning("OcrMacOptions unavailable, using docling default OCR: %s", exc)

    format_options = {
        InputFormat.AUDIO: AudioFormatOption(pipeline_options=asr),
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=pdf_opts),
    }
    return DocumentConverter(format_options=format_options)


def convert(
    source: str | Path,
    settings: ExtractSettings | None = None,
    *,
    progress: Progress | None = None,
) -> DoclingResult:
    """Convert a local file or an http(s) URL with docling and return Markdown + plain text."""
    settings = settings or ExtractSettings()
    if not available():
        raise DoclingUnavailableError(
            "docling is not installed. Run: uv sync --extra docling   (audio, docx, html, epub)"
        )
    from docling.datamodel.base_models import ConversionStatus

    if progress:
        progress("docling", 0, 1)
    converter = build_converter(settings)
    src = str(source)
    result = converter.convert(src, raises_on_error=False)
    warnings: list[str] = []
    if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
        errs = "; ".join(str(e.error_message) for e in result.errors) or str(result.status)
        raise RuntimeError(f"docling conversion failed: {errs}")
    if result.status == ConversionStatus.PARTIAL_SUCCESS:
        warnings.extend(str(e.error_message) for e in result.errors)
    doc = result.document
    md = doc.export_to_markdown()
    text = doc.export_to_text()
    fmt = str(getattr(result.input, "format", "") or "")
    pages = 0
    try:
        pages = len(doc.pages) if getattr(doc, "pages", None) else 0
    except Exception:
        pages = 0
    meta = {"origin": getattr(getattr(doc, "origin", None), "filename", None)}
    if progress:
        progress("docling", 1, 1)
    return DoclingResult(
        markdown=md, text=text, input_format=fmt, pages=pages, meta=meta, warnings=warnings
    )
