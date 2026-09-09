import pytest

from conftest import needs_gs, needs_poppler
from funicular.tools import ghostscript, poppler
from funicular.tools.binaries import find_binary


def test_find_binary_env_override(tmp_path, monkeypatch):
    fake = tmp_path / "pdftotext"
    fake.write_text("#!/bin/sh\necho hi\n")
    fake.chmod(0o755)
    monkeypatch.setenv("FUNICULAR_PDFTOTEXT", str(fake))
    assert find_binary("pdftotext", "FUNICULAR_PDFTOTEXT") == str(fake)


def test_find_binary_missing(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    assert find_binary("definitely-not-a-binary-xyz") is None


@needs_poppler
def test_pdftotext_layout_keeps_columns_side_by_side(fixtures):
    text = poppler.pdftotext_layout(fixtures["scholarly"])
    pages = text.split("\f")
    assert len([p for p in pages if p.strip()]) == 2
    # A line that carries both the left-column heading and right-column text.
    line = next(ln for ln in pages[0].splitlines() if "1 Introduction" in ln)
    assert "2.3 Statistical analysis" in line
    assert "24–36" in text  # en dash survives via the embedded font
    assert "Chiò" in text


@needs_poppler
def test_pdfinfo_and_pdffonts(fixtures):
    info = poppler.pdfinfo(fixtures["scholarly"])
    assert info.pages == 2
    assert info.title and "Riluzole" in info.title
    assert not info.encrypted
    fonts = poppler.pdffonts(fixtures["scholarly"])
    assert any(f["uni"] == "yes" for f in fonts)
    assert any(f["uni"] == "no" for f in fonts)


@needs_poppler
def test_pdftoppm_renders_pages(fixtures, tmp_path):
    files = poppler.pdftoppm_png(fixtures["scholarly"], tmp_path / "pg", dpi=50)
    assert [f.name for f in files] == ["pg-1.png", "pg-2.png"]


@needs_poppler
def test_pdftotext_error_on_garbage(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"not a pdf")
    with pytest.raises(poppler.PopplerError):
        poppler.pdftotext_layout(p)


@needs_gs
def test_ghostscript_repair_rebuilds_truncated_pdf(fixtures, tmp_path):
    out = ghostscript.repair_pdf(fixtures["broken"], tmp_path / "fixed.pdf")
    assert out.exists() and out.stat().st_size > 1000
    import pymupdf

    with pymupdf.open(out) as doc:
        assert doc.page_count == 2
        assert "Riluzole" in doc[0].get_text()


@needs_gs
def test_ghostscript_rasterize_and_text_only(fixtures, tmp_path):
    files = ghostscript.rasterize_png(fixtures["scholarly"], tmp_path / "r", dpi=40, last=1)
    assert len(files) == 1 and files[0].stat().st_size > 0
    stripped = ghostscript.text_only_pdf(fixtures["mixed"], tmp_path / "textonly.pdf")
    import pymupdf

    with pymupdf.open(stripped) as doc:
        assert doc.page_count == 2
        assert "Riluzole" in doc[0].get_text()
        assert not doc[1].get_images()  # the scanned page lost its raster image


@needs_gs
def test_ghostscript_error_surfaces(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"garbage")
    with pytest.raises(ghostscript.GhostscriptError):
        ghostscript.repair_pdf(p, tmp_path / "out.pdf")


def test_versions_do_not_crash():
    poppler.version()
    ghostscript.version()
