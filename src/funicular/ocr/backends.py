"""OCR backends: Apple Vision (ocrmac), a macOCR-style CLI, and tesseract."""

from __future__ import annotations

import csv
import io
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from ..config import ExtractSettings
from ..tools.binaries import find_binary
from .layout import Positioned, layout_text

log = logging.getLogger(__name__)


class OcrUnavailableError(RuntimeError):
    pass


@dataclass
class OcrLine(Positioned):
    confidence: float = 0.0


@dataclass
class OcrResult:
    backend: str
    width: int
    height: int
    lines: list[OcrLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def layout(self) -> str:
        return layout_text(list(self.lines))

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return sum(line.confidence for line in self.lines) / len(self.lines)


class Backend:
    name = "base"

    def recognize(self, image: Image.Image) -> list[OcrLine]:  # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------------------------
# Apple Vision via the `ocrmac` package (pyobjc). macOS only.
# --------------------------------------------------------------------------------------------
class MacVisionBackend(Backend):
    name = "macocr"

    def __init__(self, languages: list[str], recognition_level: str = "accurate") -> None:
        try:
            from ocrmac import ocrmac as _ocrmac  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - darwin only
            raise OcrUnavailableError(
                "ocrmac is not installed. Run: uv sync --extra ocr  (macOS only)"
            ) from exc
        self._ocrmac = _ocrmac
        self.languages = languages
        self.recognition_level = recognition_level

    def recognize(self, image: Image.Image) -> list[OcrLine]:  # pragma: no cover - darwin only
        image = image.convert("RGB")
        kwargs: dict[str, Any] = {
            "recognition_level": self.recognition_level,
            "language_preference": self.languages or None,
        }
        try:
            ocr = self._ocrmac.OCR(image, **kwargs)
            raw = ocr.recognize(px=True)
        except ValueError:
            # Unsupported language preference on this macOS release: let Vision pick.
            kwargs["language_preference"] = None
            ocr = self._ocrmac.OCR(image, **kwargs)
            raw = ocr.recognize(px=True)
        lines: list[OcrLine] = []
        for text, conf, (x0, y0, x1, y1) in raw:
            lines.append(OcrLine(text=text, x0=x0, y0=y0, x1=x1, y1=y1, confidence=float(conf)))
        return lines


# --------------------------------------------------------------------------------------------
# A macOCR-style command line tool: invoked as `<bin> <image-path>`, prints text on stdout.
# Point FUNICULAR_MACOCR_BIN at it (e.g. the `ocr` binary from schappim/macOCR).
# --------------------------------------------------------------------------------------------
class MacOcrCliBackend(Backend):
    name = "macocr-cli"

    def __init__(self, binary: str) -> None:
        self.binary = binary

    def recognize(self, image: Image.Image) -> list[OcrLine]:
        with tempfile.TemporaryDirectory(prefix="funicular-ocr-") as td:
            p = Path(td) / "page.png"
            image.convert("RGB").save(p, format="PNG")
            proc = subprocess.run(
                [self.binary, str(p)], capture_output=True, timeout=600, check=False
            )
        if proc.returncode != 0:
            raise OcrUnavailableError(
                f"{self.binary} exited {proc.returncode}: "
                f"{proc.stderr.decode('utf-8', 'replace')[:300]}"
            )
        text = proc.stdout.decode("utf-8", "replace")
        # No geometry from a plain CLI: synthesise a top-to-bottom stack so layout_text works.
        lines: list[OcrLine] = []
        h = max(image.height / max(len(text.splitlines()), 1), 1.0)
        for i, raw in enumerate(text.splitlines()):
            if raw.strip():
                lines.append(
                    OcrLine(
                        text=raw, x0=0, y0=i * h, x1=image.width, y1=(i + 1) * h, confidence=1.0
                    )
                )
        return lines


# --------------------------------------------------------------------------------------------
# tesseract CLI (Linux fallback and cross-check). TSV output gives word boxes we merge to lines.
# --------------------------------------------------------------------------------------------
_TESS_LANG = {
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "nl": "nld",
    "la": "lat",
    "el": "ell",
    "ru": "rus",
    "ja": "jpn",
    "zh": "chi_sim",
    "ko": "kor",
}


def _tess_langs(languages: list[str]) -> str:
    codes = []
    for lang in languages:
        base = lang.split("-")[0].lower()
        codes.append(_TESS_LANG.get(base, base))
    return "+".join(dict.fromkeys(codes)) or "eng"


class TesseractBackend(Backend):
    name = "tesseract"

    def __init__(self, binary: str, languages: list[str], psm: int = 3) -> None:
        self.binary = binary
        self.lang = _tess_langs(languages)
        self.psm = psm

    def recognize(self, image: Image.Image) -> list[OcrLine]:
        with tempfile.TemporaryDirectory(prefix="funicular-ocr-") as td:
            p = Path(td) / "page.png"
            image.convert("RGB").save(p, format="PNG")
            argv = [
                self.binary,
                str(p),
                "-",
                "--psm",
                str(self.psm),
                "-l",
                self.lang,
                "-c",
                "preserve_interword_spaces=1",
                "tsv",
            ]
            proc = subprocess.run(argv, capture_output=True, timeout=600, check=False)
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", "replace")
            if "Failed loading language" in err:
                # Fall back to English rather than fail the whole document.
                if self.lang != "eng":
                    self.lang = "eng"
                    return self.recognize(image)
            raise OcrUnavailableError(f"tesseract failed: {err[:300]}")
        return _tsv_to_lines(proc.stdout.decode("utf-8", "replace"))


def _tsv_to_lines(tsv: str) -> list[OcrLine]:
    reader = csv.DictReader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)
    groups: dict[tuple[int, int, int, int], list[dict[str, str]]] = {}
    for row in reader:
        if row.get("level") != "5":
            continue
        word = (row.get("text") or "").strip()
        if not word:
            continue
        key = (
            int(row["page_num"]),
            int(row["block_num"]),
            int(row["par_num"]),
            int(row["line_num"]),
        )
        groups.setdefault(key, []).append(row)
    lines: list[OcrLine] = []
    for key in sorted(groups):
        words = sorted(groups[key], key=lambda r: int(r["left"]))
        x0 = min(int(w["left"]) for w in words)
        y0 = min(int(w["top"]) for w in words)
        x1 = max(int(w["left"]) + int(w["width"]) for w in words)
        y1 = max(int(w["top"]) + int(w["height"]) for w in words)
        confs = [float(w["conf"]) for w in words if float(w["conf"]) >= 0]
        conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
        lines.append(
            OcrLine(
                text=" ".join(w["text"] for w in words), x0=x0, y0=y0, x1=x1, y1=y1, confidence=conf
            )
        )
    return lines


# --------------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------------
def _is_darwin() -> bool:
    return sys.platform == "darwin"


def _ocrmac_importable() -> bool:
    if not _is_darwin():
        return False
    try:
        import ocrmac  # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        return False
    return True


def _macocr_cli() -> str | None:
    override = os.environ.get("FUNICULAR_MACOCR_BIN")
    if override and Path(override).expanduser().is_file():
        return str(Path(override).expanduser())
    for name in ("macocr", "ocr"):
        # `ocr` is too generic to trust from PATH unless it's in a Homebrew prefix.
        p = find_binary(name)
        if p and (name == "macocr" or p.startswith(("/opt/homebrew", "/usr/local"))):
            return p
    return None


def describe_backends() -> list[dict[str, Any]]:
    tess = find_binary("tesseract", "FUNICULAR_TESSERACT")
    return [
        {
            "name": "macocr",
            "description": "Apple Vision (VNRecognizeTextRequest) via ocrmac",
            "available": _ocrmac_importable(),
            "note": "" if _is_darwin() else "macOS only",
        },
        {
            "name": "macocr-cli",
            "description": "macOCR-style CLI (FUNICULAR_MACOCR_BIN)",
            "available": _macocr_cli() is not None,
            "note": _macocr_cli() or "",
        },
        {
            "name": "tesseract",
            "description": "tesseract CLI",
            "available": tess is not None,
            "note": tess or "brew install tesseract",
        },
    ]


def select_backend(settings: ExtractSettings) -> Backend:
    choice = settings.ocr_backend
    langs = settings.ocr_languages
    if choice in ("auto", "macocr") and _ocrmac_importable():
        return MacVisionBackend(langs)
    if choice == "macocr":
        raise OcrUnavailableError(
            "macocr backend requested but ocrmac is unavailable "
            f"(platform={platform.system()}). On macOS run: uv sync --extra ocr"
        )
    if choice in ("auto", "macocr-cli"):
        cli = _macocr_cli()
        if cli:
            return MacOcrCliBackend(cli)
        if choice == "macocr-cli":
            raise OcrUnavailableError("macocr-cli requested but FUNICULAR_MACOCR_BIN is not set")
    tess = find_binary("tesseract", "FUNICULAR_TESSERACT")
    if tess:
        return TesseractBackend(tess, langs)
    raise OcrUnavailableError(
        "No OCR backend available. On macOS: uv sync --extra ocr (Apple Vision). "
        "Anywhere: brew/apt install tesseract."
    )


# --------------------------------------------------------------------------------------------
# Entry points used by the pipeline
# --------------------------------------------------------------------------------------------
def load_image(path: Path) -> Image.Image:
    """Open an image; HEIC/HEIF are converted with `sips` on macOS (no extra dependency)."""
    ext = path.suffix.lower()
    if ext in {".heic", ".heif"}:
        sips = shutil.which("sips")
        if not sips:
            raise OcrUnavailableError("HEIC input needs macOS `sips` (or convert to JPEG first)")
        with tempfile.TemporaryDirectory(prefix="funicular-heic-") as td:
            out = Path(td) / "converted.png"
            subprocess.run(
                [sips, "-s", "format", "png", str(path), "--out", str(out)],
                capture_output=True,
                timeout=120,
                check=True,
            )
            img = Image.open(out)
            img.load()
            return img
    img = Image.open(path)
    img.load()
    return img


def ocr_image(path: Path, settings: ExtractSettings, backend: Backend | None = None) -> OcrResult:
    backend = backend or select_backend(settings)
    img = load_image(path)
    # Respect EXIF orientation for phone photos.
    try:
        from PIL import ImageOps

        img = ImageOps.exif_transpose(img) or img
    except Exception as exc:  # malformed EXIF is common on phone photos; keep going
        log.debug("exif transpose skipped: %s", exc)
    lines = backend.recognize(img)
    return OcrResult(backend=backend.name, width=img.width, height=img.height, lines=lines)


def make_ocr_function(backend: Backend) -> Callable[..., None]:
    """Adapt a backend to pymupdf4llm's ``ocr_function(page, dpi=, language=, keep_ocr_text=)``.

    pymupdf4llm renders the page *without* its legible text, hands us the pixels, and writes
    whatever we return back into the page as a real (invisible-font-agnostic) text layer.
    """
    import numpy as np
    from pymupdf4llm.ocr.exec_ocr_interface import exec_ocr_full

    def full_ocr(img: np.ndarray) -> list[tuple[list[list[float]], str, float]]:
        pil = Image.fromarray(img)
        out = []
        for line in backend.recognize(pil):
            box = [[line.x0, line.y0], [line.x1, line.y0], [line.x1, line.y1], [line.x0, line.y1]]
            out.append((box, line.text, line.confidence))
        return out

    def ocr_function(
        page: Any, dpi: int = 150, language: str | None = None, keep_ocr_text: bool = False
    ) -> None:
        exec_ocr_full(page, full_ocr, dpi=dpi, language=language, keep_ocr_text=keep_ocr_text)

    return ocr_function
