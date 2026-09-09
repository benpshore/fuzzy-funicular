from funicular.config import ExtractSettings
from funicular.detect import analyse_pdf


def test_text_pdf_has_no_scan_pages(fixtures):
    sig = analyse_pdf(fixtures["scholarly"])
    assert sig.page_count == 2
    assert sig.scan_pages == []
    assert sig.needs_ocr_pages == []
    assert sig.has_any_text and not sig.is_scanned_document
    assert all(p.chars > 200 for p in sig.pages)


def test_scan_pdf_is_flagged(fixtures):
    sig = analyse_pdf(fixtures["scan"])
    assert sig.scan_pages == [1, 2]
    assert sig.is_scanned_document
    assert not sig.has_any_text
    assert all(p.image_coverage > 0.9 for p in sig.pages)


def test_mixed_pdf_flags_only_the_image_page(fixtures):
    sig = analyse_pdf(fixtures["mixed"])
    assert sig.scan_pages == [2]
    assert sig.needs_ocr_pages == [2]
    assert not sig.is_scanned_document


def test_thresholds_are_configurable(fixtures):
    strict = ExtractSettings(min_chars_per_page=100_000)
    sig = analyse_pdf(fixtures["scholarly"], strict)
    # With an absurd char threshold every page is "text-less", but none has image coverage,
    # so none is a scan and none needs OCR.
    assert not sig.has_any_text
    assert sig.scan_pages == []


def test_unreadable_file_reports_error(tmp_path):
    p = tmp_path / "junk.pdf"
    p.write_bytes(b"%PDF-1.7\n" + b"\x00" * 100)
    sig = analyse_pdf(p)
    assert sig.error is not None
    assert sig.page_count == 0
