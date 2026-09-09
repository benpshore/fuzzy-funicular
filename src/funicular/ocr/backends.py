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
from ..resources import run as _run_limited
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
# Apple Vision (VNRecognizeTextRequest). macOS only.
#
# Two ways in, same output: the `ocrmac` package when it is installed, otherwise a direct call
# through pyobjc's Vision framework (pyobjc-framework-Vision ships cp314 universal2 wheels).
# Vision reports boxes normalised to the image with the origin at the bottom-left; we convert
# to top-left pixel boxes so the layout reconstruction is shared with every other backend.
# --------------------------------------------------------------------------------------------
class MacVisionBackend(Backend):
    name = "macocr"

    def __init__(self, languages: list[str], recognition_level: str = "accurate") -> None:
        self.languages = [lang for lang in languages if lang]
        self.recognition_level = recognition_level
        self._ocrmac = None
        self._vision = None
        try:
            from ocrmac import ocrmac as _ocrmac  # type: ignore[import-not-found]

            self._ocrmac = _ocrmac
        except Exception:  # noqa: BLE001 - fall through to the direct framework path
            try:
                import Vision as _vision  # type: ignore[import-not-found]

                self._vision = _vision
            except Exception as exc:
                raise OcrUnavailableError(
                    "Apple Vision is unavailable: install the ocr extra on macOS "
                    "(uv sync --extra ocr)"
                ) from exc

    # -- entry -----------------------------------------------------------------------------
    def recognize(self, image: Image.Image) -> list[OcrLine]:
        image = image.convert("RGB")
        if self._ocrmac is not None:
            return self._via_ocrmac(image)
        return self._via_vision(image)

    # -- ocrmac ----------------------------------------------------------------------------
    def _via_ocrmac(self, image: Image.Image) -> list[OcrLine]:
        kwargs: dict[str, Any] = {
            "recognition_level": self.recognition_level,
            "language_preference": self.languages or None,
        }
        try:
            raw = self._ocrmac.OCR(image, **kwargs).recognize(px=True)
        except ValueError:
            # This macOS release does not support one of the requested languages: let Vision
            # choose. (ocrmac validates against supportedRecognitionLanguages.)
            kwargs["language_preference"] = None
            raw = self._ocrmac.OCR(image, **kwargs).recognize(px=True)
        lines: list[OcrLine] = []
        for text, conf, (x0, y0, x1, y1) in raw:
            lines.append(
                OcrLine(
                    text=str(text),
                    x0=float(x0),
                    y0=float(y0),
                    x1=float(x1),
                    y1=float(y1),
                    confidence=float(conf),
                )
            )
        return lines

    # -- direct pyobjc ---------------------------------------------------------------------
    def _via_vision(self, image: Image.Image) -> list[OcrLine]:
        vision = self._vision
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = buf.getvalue()
        req = vision.VNRecognizeTextRequest.alloc().init()
        # 0 = accurate, 1 = fast (VNRequestTextRecognitionLevel)
        req.setRecognitionLevel_(1 if self.recognition_level == "fast" else 0)
        if hasattr(req, "setUsesLanguageCorrection_"):
            req.setUsesLanguageCorrection_(True)
        if self.languages:
            supported: list[str] = []
            try:
                res = req.supportedRecognitionLanguagesAndReturnError_(None)
                supported = list(res[0] if isinstance(res, tuple) else res or [])
            except Exception as exc:  # noqa: BLE001
                log.debug("supportedRecognitionLanguages failed: %s", exc)
            wanted = [lang for lang in self.languages if not supported or lang in supported]
            if wanted:
                req.setRecognitionLanguages_(wanted)
        elif hasattr(req, "setAutomaticallyDetectsLanguage_"):
            req.setAutomaticallyDetectsLanguage_(True)
        handler = vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
        ret = handler.performRequests_error_([req], None)
        ok, err = ret if isinstance(ret, tuple) else (bool(ret), None)
        if not ok or err is not None:
            raise OcrUnavailableError(f"Vision request failed: {err}")
        return vision_results_to_lines(req.results() or [], image.width, image.height)


def vision_results_to_lines(results: Any, width: int, height: int) -> list[OcrLine]:
    """Convert VNRecognizedTextObservation objects to top-left pixel OcrLines."""
    lines: list[OcrLine] = []
    for obs in results:
        text = ""
        try:
            cands = obs.topCandidates_(1)
            if cands:
                text = str(cands[0].string())
        except Exception:  # noqa: BLE001 - older observation classes expose .text()
            text = ""
        if not text and hasattr(obs, "text"):
            text = str(obs.text())
        if not text.strip():
            continue
        bbox = obs.boundingBox()
        x, y = float(bbox.origin.x), float(bbox.origin.y)
        w, h = float(bbox.size.width), float(bbox.size.height)
        x0 = x * width
        x1 = (x + w) * width
        y1 = (1.0 - y) * height
        y0 = y1 - h * height
        conf = float(obs.confidence()) if hasattr(obs, "confidence") else 0.0
        lines.append(OcrLine(text=text, x0=x0, y0=y0, x1=x1, y1=y1, confidence=conf))
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
            proc = _run_limited([self.binary, str(p)], timeout=600)
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
            proc = _run_limited(argv, timeout=600)
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
    """True when Apple Vision can be reached: via ocrmac or pyobjc's Vision framework."""
    if not _is_darwin():
        return False
    for mod in ("ocrmac", "Vision"):
        try:
            __import__(mod)
            return True
        except Exception as exc:  # noqa: BLE001
            log.debug("%s not importable: %s", mod, exc)
    return False


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
            "description": "Apple Vision (VNRecognizeTextRequest) via ocrmac or pyobjc",
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
