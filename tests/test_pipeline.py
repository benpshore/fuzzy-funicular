import json

import pytest

from conftest import needs_asr, needs_docling, needs_poppler, needs_tesseract
from funicular.config import ExtractSettings
from funicular.pipeline import ingest, is_url, safe_stem, strip_html


def test_safe_stem_and_is_url():
    assert safe_stem("../../etc/passwd.pdf") == "passwd"
    assert safe_stem("Weird name (final) v2.docx") == "Weird name _final_ v2"
    assert safe_stem("....") == "document"
    assert len(safe_stem("x" * 500)) == 120
    assert is_url("https://example.org/a.html") and not is_url("file.pdf")
    assert not is_url("ftp://x")


@needs_poppler
def test_pdf_writes_all_outputs(fixtures, out_dir):
    r = ingest(fixtures["scholarly"], out_dir)
    assert set(r.outputs) == {"layout", "markdown", "text", "report"}
    assert (out_dir / "scholarly.layout.txt").exists()
    report = json.loads((out_dir / "scholarly.json").read_text())
    assert report["kind"] == "pdf" and report["stats"]["pages"] == 2
    assert report["title"].startswith("Riluzole")
    assert not r.needs_ocr
    assert r.text_preview


def test_scan_ocr_off_marks_needs_ocr(fixtures, out_dir):
    r = ingest(fixtures["scan"], out_dir)
    assert r.needs_ocr
    assert "ocr_pdf" not in r.outputs


@needs_tesseract
@needs_poppler
def test_scan_ocr_auto_writes_searchable_pdf(fixtures, out_dir):
    r = ingest(fixtures["scan"], out_dir, ExtractSettings(ocr="auto"))
    assert (out_dir / "scholarly-scan.ocr.pdf").exists()
    assert not r.needs_ocr
    assert "Riluzole" in (out_dir / "scholarly-scan.layout.txt").read_text()


def test_image_with_ocr_off(fixtures, out_dir):
    r = ingest(fixtures["photo"], out_dir)
    assert r.kind == "image" and r.needs_ocr
    assert set(r.outputs) == {"report"}


@needs_tesseract
def test_image_with_ocr(fixtures, out_dir):
    r = ingest(fixtures["photo"], out_dir, ExtractSettings(ocr="auto"))
    assert {"layout", "text"} <= set(r.outputs)
    assert r.stats["ocr_backend"] == "tesseract"
    assert "Riluzole" in r.outputs["text"].read_text()


def test_markdown_passthrough(fixtures, out_dir):
    r = ingest(fixtures["md"], out_dir)
    assert r.kind == "text"
    assert (out_dir / "notes.md").read_text().startswith("# Clinic notes")


def test_strip_html_fallback():
    text = strip_html(
        "<html><head><script>x()</script><style>p{}</style></head>"
        "<body><h1>Title</h1><p>One &amp; two</p></body></html>"
    )
    assert "x()" not in text and "p{}" not in text
    assert "Title" in text and "One & two" in text


def test_html_without_docling_uses_fallback(fixtures, out_dir, monkeypatch):
    from funicular import docling_backend

    monkeypatch.setattr(docling_backend, "available", lambda: False)
    r = ingest(fixtures["html"], out_dir)
    assert "text" in r.outputs and "markdown" not in r.outputs
    assert r.title == "Why registries matter for rare disease"
    assert any("docling not installed" in w for w in r.warnings)


def test_office_without_docling_is_a_clear_warning(fixtures, out_dir, monkeypatch):
    if "docx" not in fixtures:
        pytest.skip("python-docx not available to build the fixture")
    from funicular import docling_backend

    monkeypatch.setattr(docling_backend, "available", lambda: False)
    r = ingest(fixtures["docx"], out_dir)
    assert r.outputs.keys() == {"report"}
    assert any("uv sync --extra docling" in w for w in r.warnings)


@needs_docling
def test_html_via_docling(fixtures, out_dir):
    r = ingest(fixtures["html"], out_dir)
    md = r.outputs["markdown"].read_text()
    assert md.startswith("# Why registries matter")
    assert "console.log" not in md and "Home" not in md  # nav/script dropped
    assert "- Linkage quality is everything." in md


@needs_docling
def test_docx_via_docling(fixtures, out_dir):
    r = ingest(fixtures["docx"], out_dir)
    md = r.outputs["markdown"].read_text()
    assert "Multidisciplinary team summary" in md
    assert "ALSFRS-R" in md and "38" in md
    assert r.stats["input_format"].endswith("DOCX")


@needs_asr
def test_audio_via_docling_asr(fixtures, out_dir):
    if "wav" not in fixtures:
        pytest.skip("espeak-ng not available to synthesise speech")
    r = ingest(fixtures["wav"], out_dir, ExtractSettings(asr_model="whisper_tiny"))
    text = r.outputs["text"].read_text().lower()
    assert "patient" in text and "clinic" in text


def test_missing_file(out_dir):
    with pytest.raises(FileNotFoundError):
        ingest("does-not-exist.pdf", out_dir)


def test_unknown_type(out_dir, tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"\x00\x01\x02\x03")
    r = ingest(p, out_dir)
    assert r.kind == "unknown" and any("unsupported" in w for w in r.warnings)
