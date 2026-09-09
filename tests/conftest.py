from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

import make_fixtures  # noqa: E402


def _has(name: str) -> bool:
    return shutil.which(name) is not None


HAS_POPPLER = _has("pdftotext") and _has("pdfinfo")
HAS_GS = _has("gs")
HAS_TESSERACT = _has("tesseract")
HAS_ESPEAK = _has("espeak-ng") or _has("espeak")
try:
    import docling  # noqa: F401

    HAS_DOCLING = True
except Exception:
    HAS_DOCLING = False

needs_poppler = pytest.mark.skipif(not HAS_POPPLER, reason="poppler-utils not installed")
needs_gs = pytest.mark.skipif(not HAS_GS, reason="ghostscript not installed")
needs_tesseract = pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract not installed")
needs_docling = pytest.mark.skipif(not HAS_DOCLING, reason="docling extra not installed")
needs_asr = pytest.mark.skipif(
    not (HAS_DOCLING and os.environ.get("FUNICULAR_TEST_ASR") == "1"),
    reason="set FUNICULAR_TEST_ASR=1 to run the Whisper smoke test (downloads ~72 MB once)",
)


@pytest.fixture(scope="session")
def fixtures(tmp_path_factory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("fixtures")
    return make_fixtures.build(out)


@pytest.fixture
def out_dir(tmp_path) -> Path:
    d = tmp_path / "out"
    d.mkdir()
    return d
