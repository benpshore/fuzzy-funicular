import pytest

from conftest import needs_tesseract
from funicular.config import ExtractSettings
from funicular.ocr import OcrUnavailableError, ocr_image, select_backend
from funicular.ocr.backends import TesseractBackend, _tess_langs, _tsv_to_lines


def test_tsv_parsing_groups_words_into_lines():
    tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t10\t10\t40\t12\t95\tHello\n"
        "5\t1\t1\t1\t1\t2\t55\t10\t40\t12\t85\tworld\n"
        "5\t1\t1\t1\t2\t1\t10\t30\t60\t12\t90\tSecond\n"
        "5\t1\t1\t1\t2\t2\t80\t30\t10\t12\t-1\t\n"
    )
    lines = _tsv_to_lines(tsv)
    assert [ln.text for ln in lines] == ["Hello world", "Second"]
    assert lines[0].x0 == 10 and lines[0].x1 == 95
    assert abs(lines[0].confidence - 0.90) < 1e-6


def test_language_mapping():
    assert _tess_langs(["en-US"]) == "eng"
    assert _tess_langs(["de-DE", "en-GB", "de-AT"]) == "deu+eng"
    assert _tess_langs([]) == "eng"


def test_select_backend_errors_when_nothing_available(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.delenv("FUNICULAR_MACOCR_BIN", raising=False)
    monkeypatch.delenv("FUNICULAR_TESSERACT", raising=False)
    import funicular.ocr.backends as b

    monkeypatch.setattr(b, "_ocrmac_importable", lambda: False)
    monkeypatch.setattr(b, "_EXTRA_DIRS", (), raising=False)
    monkeypatch.setattr(b, "find_binary", lambda *_a, **_k: None)
    with pytest.raises(OcrUnavailableError):
        select_backend(ExtractSettings(ocr="auto"))
    with pytest.raises(OcrUnavailableError):
        select_backend(ExtractSettings(ocr="auto", ocr_backend="macocr"))


def test_macocr_cli_backend_uses_configured_binary(tmp_path, monkeypatch, fixtures):
    script = tmp_path / "fake-macocr"
    script.write_text("#!/bin/sh\nprintf 'line one\\nline two\\n'\n")
    script.chmod(0o755)
    monkeypatch.setenv("FUNICULAR_MACOCR_BIN", str(script))
    import funicular.ocr.backends as b

    monkeypatch.setattr(b, "_ocrmac_importable", lambda: False)
    backend = select_backend(ExtractSettings(ocr="auto", ocr_backend="macocr-cli"))
    assert backend.name == "macocr-cli"
    res = ocr_image(fixtures["photo"], ExtractSettings(ocr="auto"), backend)
    assert res.text.splitlines() == ["line one", "line two"]
    assert "line two" in res.layout


@needs_tesseract
def test_tesseract_reads_the_photo(fixtures):
    res = ocr_image(fixtures["photo"], ExtractSettings(ocr="auto", ocr_backend="tesseract"))
    assert res.backend == "tesseract"
    assert res.mean_confidence > 0.7
    assert "Riluzole" in res.text
    assert "Introduction" in res.layout
    # The photo is skewed 1.5°, so rows from the two columns do not share a baseline; what
    # must hold is that right-column text is placed well to the right in the layout copy.
    line = next(ln for ln in res.layout.splitlines() if "Statistical analysis" in ln)
    assert len(line) - len(line.lstrip()) > 30


@needs_tesseract
def test_tesseract_language_fallback(fixtures):
    backend = TesseractBackend("tesseract", ["xx-XX"])
    from funicular.ocr.backends import load_image

    lines = backend.recognize(load_image(fixtures["photo"]))
    assert backend.lang == "eng"
    assert lines
