import pymupdf
import pytest

from conftest import needs_gs, needs_poppler, needs_tesseract
from funicular import pdf as pdfmod
from funicular.config import ExtractSettings
from funicular.pdf import extract_pdf, physical_layout_from_page


@needs_poppler
def test_scholarly_layout_markdown_and_text(fixtures):
    stages: list[str] = []
    ex = extract_pdf(fixtures["scholarly"], progress=lambda s, d, t: stages.append(s))
    assert ex.page_count == 2
    assert not ex.repaired and ex.ocr_pages == []
    assert {"detect", "layout", "markdown", "text"} <= set(stages)
    # layout copy: physical columns preserved
    assert "1 Introduction" in ex.layout_text and "2.3 Statistical analysis" in ex.layout_text
    # markdown: logical reading order — the whole left column precedes the right column
    md = " ".join(ex.markdown.split())  # collapse the fixture's double-spaced headings
    assert md.index("2.2 Exposure definition") < md.index("2.3 Statistical analysis")
    assert md.index("2.3 Statistical analysis") < md.index("3 Results")
    assert "|" in md and "Unexposed" in md  # table detected
    assert "# " in md  # a heading was recognised
    # plain text exists and engines reported
    assert "Riluzole" in ex.plain_text
    assert ex.engines["layout"] == "pdftotext -layout"
    assert ex.engines["markdown"].startswith("pymupdf4llm")
    assert ex.pdfium_chars and ex.pdfium_chars > 1500
    assert ex.garbage_ratio < 0.01
    assert not [w for w in ex.warnings if "unmappable" in w]


def test_scan_with_ocr_off_is_reported_not_ocrd(fixtures):
    ex = extract_pdf(fixtures["scan"], ExtractSettings(ocr="off"))
    assert ex.signal.is_scanned_document
    assert ex.ocr_pages == [] and ex.ocr_pdf is None
    assert any("OCR is off" in w for w in ex.warnings)
    assert len("".join(ex.layout_text.split())) < 20


@needs_tesseract
@needs_poppler
def test_scan_with_ocr_auto_gets_a_text_layer(fixtures, tmp_path):
    ex = extract_pdf(fixtures["scan"], ExtractSettings(ocr="auto"), workdir=tmp_path)
    assert ex.ocr_pages == [1, 2]
    assert ex.ocr_backend == "tesseract"
    assert ex.ocr_pdf is not None and ex.ocr_pdf.exists()
    with pymupdf.open(ex.ocr_pdf) as doc:
        assert "Riluzole" in doc[0].get_text()
    assert "Riluzole" in ex.layout_text
    assert "Riluzole" in ex.markdown
    # After OCR the detector no longer flags the pages.
    assert ex.signal.needs_ocr_pages == []


@needs_tesseract
def test_mixed_pdf_ocrs_only_the_image_page(fixtures, tmp_path):
    ex = extract_pdf(fixtures["mixed"], ExtractSettings(ocr="auto"), workdir=tmp_path)
    assert ex.ocr_pages == [2]


def test_ocr_unavailable_is_a_warning_not_a_crash(fixtures, monkeypatch):
    from funicular.ocr import OcrUnavailableError

    def boom(_settings):
        raise OcrUnavailableError("nope")

    monkeypatch.setattr(pdfmod, "select_backend", boom)
    ex = extract_pdf(fixtures["scan"], ExtractSettings(ocr="auto"))
    assert ex.ocr_pages == []
    assert any("nope" in w for w in ex.warnings)


@needs_gs
def test_repair_path_is_taken_when_open_fails(fixtures, monkeypatch, tmp_path):
    from funicular.detect import DocSignal

    real = pdfmod.analyse_pdf
    calls = {"n": 0}

    def flaky(path, settings=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return DocSignal(error="open failed: simulated")
        return real(path, settings)

    monkeypatch.setattr(pdfmod, "analyse_pdf", flaky)
    ex = extract_pdf(fixtures["broken"], workdir=tmp_path)
    assert ex.repaired
    assert ex.engines.get("ghostscript") == "repaired"
    assert ex.page_count == 2
    assert "Riluzole" in (ex.layout_text or ex.plain_text)


def test_layout_fallback_without_poppler(fixtures, monkeypatch):
    monkeypatch.setattr(pdfmod.poppler, "available", lambda: False)
    ex = extract_pdf(fixtures["scholarly"])
    assert ex.engines["layout"].startswith("pymupdf words")
    line = next(ln for ln in ex.layout_text.splitlines() if "Introduction" in ln)
    assert "Statistical" in line  # both columns on one physical line


def test_physical_layout_from_page(fixtures):
    with pymupdf.open(fixtures["scholarly"]) as doc:
        text = physical_layout_from_page(doc[0])
    assert "Riluzole" in text
    assert "Unexposed" in text


def test_pdfium_helpers(fixtures):
    assert pdfmod.pdfium_char_count(fixtures["scholarly"]) > 1500
    assert "Riluzole" in pdfmod.pdfium_text(fixtures["scholarly"])
    png = pdfmod.render_page_png(fixtures["scholarly"], 0, width=120)
    assert png.startswith(b"\x89PNG")


def test_encrypted_pdf_is_reported(fixtures, tmp_path):
    enc = tmp_path / "enc.pdf"
    with pymupdf.open(fixtures["scholarly"]) as doc:
        doc.save(enc, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="o")
    ex = extract_pdf(enc, ExtractSettings(repair_with_ghostscript=False))
    assert any("encrypted" in w for w in ex.warnings)


def test_garbage_ratio():
    assert pdfmod._garbage_ratio("") == 0.0
    assert pdfmod._garbage_ratio("clean text") == 0.0
    assert pdfmod._garbage_ratio("���ab") == pytest.approx(0.6)
