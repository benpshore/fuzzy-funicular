from pathlib import Path

from funicular.sniff import Kind, sniff


def test_pdf_by_magic(fixtures):
    s = sniff(fixtures["scholarly"])
    assert s.kind is Kind.PDF and not s.mismatch


def test_renamed_pdf_is_detected_and_flagged(fixtures, tmp_path):
    fake = tmp_path / "looks-like.docx"
    fake.write_bytes(fixtures["scholarly"].read_bytes())
    s = sniff(fake)
    assert s.kind is Kind.PDF and s.mismatch


def test_image_audio_office_html_text(fixtures):
    assert sniff(fixtures["photo"]).kind is Kind.IMAGE
    assert sniff(fixtures["html"]).kind is Kind.HTML
    assert sniff(fixtures["md"]).kind is Kind.TEXT
    if "docx" in fixtures:
        assert sniff(fixtures["docx"]).kind is Kind.OFFICE
    if "wav" in fixtures:
        assert sniff(fixtures["wav"]).kind is Kind.AUDIO


def test_binary_garbage_is_unknown(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_bytes(b"\x00\x01\x02binary")
    assert sniff(p).kind is Kind.UNKNOWN


def test_heic_brand():
    from funicular.sniff import _magic_kind

    head = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic"
    assert _magic_kind(head, "heic") is Kind.IMAGE
    assert _magic_kind(b"\x00\x00\x00\x18ftypM4A \x00\x00\x00\x00", "m4a") is Kind.AUDIO
    assert _magic_kind(b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00", "mp4") is Kind.VIDEO
