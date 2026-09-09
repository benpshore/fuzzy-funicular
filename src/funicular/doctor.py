"""`funicular doctor`: report every engine and binary, versions, and what is missing."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .ocr import describe_backends
from .tools import ghostscript, poppler
from .tools.binaries import find_binary


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""


def _pkg(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def run_checks() -> list[Check]:
    checks: list[Check] = []
    py = platform.python_version()
    base = Path(sys.base_prefix).resolve()
    managed = "uv/python" in base.as_posix()
    checks.append(
        Check(
            "python",
            py.startswith("3.14") and managed,
            f"{py} from {base}" + (" (uv-managed)" if managed else " (NOT uv-managed)"),
            "uv python install 3.14 && uv sync",
        )
    )
    for pkg, fix in (
        ("pymupdf", "uv sync"),
        ("pymupdf4llm", "uv sync"),
        ("pymupdf-layout", "uv sync"),
        ("pypdfium2", "uv sync"),
        ("pillow", "uv sync"),
    ):
        v = _pkg(pkg)
        checks.append(Check(pkg, v is not None, v or "missing", fix))
    for pkg, fix in (
        ("docling", "uv sync --extra docling"),
        ("mlx-whisper", "uv sync --extra docling (Apple Silicon only)"),
        ("ocrmac", "uv sync --extra ocr (macOS only)"),
        ("fastapi", "uv sync --extra web"),
    ):
        v = _pkg(pkg)
        optional = pkg in ("mlx-whisper", "ocrmac") and sys.platform != "darwin"
        checks.append(
            Check(
                pkg,
                v is not None or optional,
                v or ("n/a off macOS" if optional else "missing (optional)"),
                fix,
            )
        )

    pv = poppler.version()
    checks.append(Check("poppler pdftotext", pv is not None, pv or "missing", poppler.BREW_HINT))
    for b in ("pdfinfo", "pdftoppm", "pdffonts"):
        p = find_binary(b)
        checks.append(Check(f"poppler {b}", p is not None, p or "missing", poppler.BREW_HINT))
    gv = ghostscript.version()
    checks.append(Check("ghostscript", gv is not None, gv or "missing", ghostscript.BREW_HINT))
    ff = find_binary("ffmpeg")
    checks.append(
        Check(
            "ffmpeg (audio decode for ASR)",
            ff is not None,
            ff or "missing (optional)",
            "brew install ffmpeg",
        )
    )
    for b in describe_backends():
        checks.append(
            Check(
                f"ocr {b['name']}",
                bool(b["available"]) or b["name"] != "tesseract" and sys.platform != "darwin",
                (b["note"] or ("available" if b["available"] else "missing"))
                + f" — {b['description']}",
                "",
            )
        )
    return checks
