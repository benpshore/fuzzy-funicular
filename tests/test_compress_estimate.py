import pymupdf
import pytest

from conftest import needs_gs
from funicular import compress as cmp
from funicular.config import ExtractSettings
from funicular.estimate import BASELINE, Timings, estimate
from funicular.sniff import Kind


@pytest.fixture
def image_heavy_pdf(tmp_path, fixtures):
    """A PDF with a big colour photo per page, so downsampling has something to bite."""
    from PIL import Image

    doc = pymupdf.open()
    for i in range(3):
        img = Image.effect_noise((2400, 3200), 40 + i * 10).convert("RGB")
        p = tmp_path / f"photo{i}.png"
        img.save(p)
        page = doc.new_page(width=595, height=842)
        page.insert_image(pymupdf.Rect(40, 40, 555, 800), filename=str(p))
        page.insert_text((50, 30), "caption text on the page", fontsize=10)
    out = tmp_path / "heavy.pdf"
    doc.save(out)
    doc.close()
    return out


def test_levels_are_monotonic():
    dpis = [cmp.level_for(s).dpi for s in (0, 20, 50, 70, 90, 100)]
    assert dpis == sorted(dpis, reverse=True)
    assert cmp.level_for(-5).strength == 0 and cmp.level_for(500).strength == 100


def test_preview_then_compress_pymupdf(image_heavy_pdf, tmp_path):
    plan = cmp.preview(image_heavy_pdf, 80, engine="pymupdf")
    assert plan.engine == "pymupdf" and plan.pages == 3 and plan.sample_pages == 3
    assert plan.estimated_bytes < plan.original_bytes * 0.7
    assert plan.estimated_seconds > 0
    res = cmp.compress(image_heavy_pdf, tmp_path / "out.pdf", 80, engine="pymupdf")
    assert res.output_bytes < res.original_bytes * 0.7
    # estimate should be in the right ballpark (within a factor of two)
    assert 0.5 < res.output_bytes / plan.estimated_bytes < 2.0
    with pymupdf.open(res.output) as d:
        assert d.page_count == 3 and "caption" in d[0].get_text()


def test_compress_never_grows_the_file(fixtures, tmp_path):
    res = cmp.compress(fixtures["scholarly"], tmp_path / "o.pdf", 90, engine="pymupdf")
    assert res.output_bytes <= res.original_bytes


@needs_gs
def test_ghostscript_engine_and_auto_choice(fixtures, image_heavy_pdf, tmp_path):
    res = cmp.compress(fixtures["scan"], tmp_path / "gs.pdf", 85, engine="ghostscript")
    assert res.engine.startswith("ghostscript") and res.output_bytes <= res.original_bytes
    plan = cmp.preview(fixtures["scan"], 85)  # scan -> auto picks ghostscript
    assert plan.engine == "ghostscript"
    plan = cmp.preview(fixtures["scholarly"], 50)  # text PDF -> pymupdf
    assert plan.engine == "pymupdf"


def test_estimator_scales_with_pages_and_ocr():
    e1 = estimate(Kind.PDF, pages=10)
    e2 = estimate(Kind.PDF, pages=100)
    assert e2.seconds > e1.seconds * 5
    e3 = estimate(
        Kind.PDF,
        pages=10,
        ocr_pages=10,
        settings=ExtractSettings(ocr="auto", ocr_backend="tesseract"),
    )
    assert e3.seconds > e1.seconds and "ocr.tesseract" in e3.stages
    e4 = estimate(Kind.PDF, pages=10, settings=ExtractSettings(ocr="force", ocr_backend="macocr"))
    assert "ocr.macocr" in e4.stages
    audio = estimate(Kind.AUDIO, audio_minutes=10)
    audio_m1 = estimate(Kind.AUDIO, audio_minutes=10, apple_silicon=True)
    assert audio_m1.seconds < audio.seconds
    assert estimate(Kind.IMAGE).note.startswith("OCR off")
    assert estimate("unknown").unit == "n/a"


def test_timings_blend_recorded_samples(tmp_path):
    t = Timings(tmp_path / "timings.json")
    assert t.per_unit("layout") == BASELINE["layout"][0]
    for _ in range(5):
        t.record("layout", seconds=10.0, units=10)  # 1 s/page, far above baseline
    assert t.per_unit("layout") > 0.5
    t2 = Timings(tmp_path / "timings.json")  # persisted
    assert t2.per_unit("layout") == t.per_unit("layout")
    e = estimate(Kind.PDF, pages=10, timings=t)
    assert e.stages["layout"] > 5
