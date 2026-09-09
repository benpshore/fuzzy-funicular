"""OCR backends. Off by default; see ExtractSettings.ocr.

All backends produce the same primitive, an ``OcrLine`` with pixel-space bounding boxes, so the
layout reconstruction (``layout_text``) and the pymupdf4llm hook (``make_ocr_function``) are
backend-independent. The Apple Vision backend is the intended production path on macOS.
"""

from .backends import (
    OcrLine,
    OcrResult,
    OcrUnavailableError,
    describe_backends,
    make_ocr_function,
    ocr_image,
    select_backend,
)
from .layout import layout_text

__all__ = [
    "OcrLine",
    "OcrResult",
    "OcrUnavailableError",
    "describe_backends",
    "layout_text",
    "make_ocr_function",
    "ocr_image",
    "select_backend",
]
