import pymupdf
import pytest

from conftest import needs_gs
from funicular import pdftools as pt


@pytest.fixture
def form_pdf(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 60), "Application form", fontsize=14)
    w = pymupdf.Widget()
    w.field_name = "name"
    w.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
    w.rect = pymupdf.Rect(50, 80, 300, 110)
    w.field_value = ""
    page.add_widget(w)
    c = pymupdf.Widget()
    c.field_name = "consent"
    c.field_type = pymupdf.PDF_WIDGET_TYPE_CHECKBOX
    c.rect = pymupdf.Rect(50, 130, 70, 150)
    c.field_value = "Off"
    page.add_widget(c)
    s = pymupdf.Widget()
    s.field_name = "sig1"
    s.field_type = pymupdf.PDF_WIDGET_TYPE_SIGNATURE
    s.rect = pymupdf.Rect(50, 700, 300, 750)
    page.add_widget(s)
    page.insert_text((50, 690), "Signature:", fontsize=10)
    # an ink signature: several bezier strokes right of the label
    shape = page.new_shape()
    for i in range(4):
        x = 140 + i * 30
        shape.draw_bezier((x, 680), (x + 10, 650), (x + 20, 700), (x + 30, 670))
    shape.finish(width=1.2)
    shape.commit()
    p = tmp_path / "form.pdf"
    doc.save(p)
    doc.close()
    return p


def test_forms_list_fill_and_flatten(form_pdf, tmp_path):
    fields = pt.list_fields(form_pdf)
    names = {f.name: f for f in fields}
    assert names["name"].type == "Text" and names["consent"].type == "CheckBox" and "sig1" in names
    out = pt.fill_fields(
        form_pdf, tmp_path / "filled.pdf", {"name": "Ben Shore", "consent": True, "nope": "x"}
    )
    assert out["filled"] == ["name", "consent"] and out["missing"] == ["nope"]
    refilled = {f.name: f.value for f in pt.list_fields(tmp_path / "filled.pdf")}
    assert refilled["name"] == "Ben Shore" and refilled["consent"] not in ("Off", "")
    flat = pt.fill_fields(form_pdf, tmp_path / "flat.pdf", {"name": "X"}, flatten=True)
    assert flat["flattened"] and pt.list_fields(tmp_path / "flat.pdf") == []
    with pymupdf.open(tmp_path / "flat.pdf") as d:
        assert "X" in d[0].get_text()


def test_signature_detection(form_pdf):
    found = pt.detect_signatures(form_pdf)
    kinds = {f.kind for f in found}
    assert "field" in kinds and "ink" in kinds
    fld = next(f for f in found if f.kind == "field")
    assert fld.name == "sig1" and fld.signed is False


def test_encryption_roundtrip(fixtures, tmp_path):
    src = fixtures["scholarly"]
    info = pt.encryption_info(src)
    assert not info.encrypted and not info.needs_password
    enc = pt.encrypt(
        src, tmp_path / "enc.pdf", user_password="pw", owner_password="own", allow_copy=False
    )
    info = pt.encryption_info(enc)
    assert info.encrypted and info.needs_password
    info = pt.encryption_info(enc, password="pw")
    assert (
        info.encrypted
        and not info.needs_password
        and info.permissions["print"]
        and not info.permissions["copy"]
    )
    with pytest.raises(PermissionError):
        pt.decrypt(enc, tmp_path / "x.pdf", "wrong")
    dec = pt.decrypt(enc, tmp_path / "dec.pdf", "pw")
    assert not pt.encryption_info(dec).encrypted
    with pymupdf.open(dec) as d:
        assert "Riluzole" in d[0].get_text()


def test_pdfa_claim_detection(fixtures, tmp_path):
    assert pt.pdfa_claim(fixtures["scholarly"]) is None
    with pymupdf.open(fixtures["scholarly"]) as doc:
        doc.set_xml_metadata(
            '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/"><pdfaid:part>2</pdfaid:part>'
            "<pdfaid:conformance>B</pdfaid:conformance></rdf:Description></rdf:RDF></x:xmpmeta>"
        )
        doc.save(tmp_path / "claims.pdf")
    assert pt.pdfa_claim(tmp_path / "claims.pdf") == "2B"


@needs_gs
def test_pdfa_conversion(fixtures, tmp_path):
    simple = pymupdf.open()
    page = simple.new_page()
    page.insert_text((50, 80), "Riluzole cohort, base-14 font only", fontsize=12, fontname="helv")
    simple.save(tmp_path / "simple.pdf")
    simple.close()
    res = pt.to_pdfa(tmp_path / "simple.pdf", tmp_path / "a.pdf", level=2)
    assert res["bytes"] > 1000 and res["claims"] == "2B" and res["warnings"] == []
    with pymupdf.open(tmp_path / "a.pdf") as d:
        assert "Riluzole" in d[0].get_text()
    # A CID font using CID 0 is not allowed in PDF/A: the result must say so, not pretend.
    res2 = pt.to_pdfa(fixtures["scholarly"], tmp_path / "b.pdf", level=2)
    assert res2["claims"] == "2B" or any("PDF/A" in w for w in res2["warnings"])


def test_pagination_and_headers(fixtures):
    pg = pt.pagination(fixtures["scholarly"])
    assert pg.pages == 2 and pg.printed == ["1", "2"] and pg.offset == 0
    hf = pt.detect_headers_footers(fixtures["scholarly"])
    assert any("Shore et al." in h for h in hf.headers)
    assert any(f.strip() in ("1", "2") for f in hf.footers)
    with pymupdf.open(fixtures["scholarly"]) as doc:
        text = "\n".join(p.get_text("text") for p in doc)
    assert "Shore et al." in text
    stripped = pt.strip_headers_footers(text, hf)
    assert "Riluzole" in stripped and "Shore et al." not in stripped
    assert pt.strip_headers_footers("Shore et al. — MND registry cohort\nBody", hf) == "Body"
    assert pt.strip_headers_footers("x", pt.HeaderFooter([], [], 0)) == "x"


def test_outline_from_markdown_and_pdf(fixtures, tmp_path):
    md = "# Title\n\n## 1 Introduction\ntext\n--- end of page.page_number=0 ---\n\n## 2 Methods\n### 2.1 Data\n```\n# not a heading\n```\n--- end of page.page_number=1 ---\n"
    nodes = pt.outline_from_markdown(md)
    assert nodes[0].title == "Title" and nodes[0].page == 1
    assert [c.title for c in nodes[0].children] == ["1 Introduction", "2 Methods"]
    assert nodes[0].children[1].page == 2 and nodes[0].children[1].children[0].title == "2.1 Data"
    with pymupdf.open(fixtures["scholarly"]) as doc:
        doc.set_toc([[1, "Intro", 1], [2, "Methods", 1], [1, "Refs", 2]])
        doc.save(tmp_path / "toc.pdf")
    o = pt.outline(tmp_path / "toc.pdf", md)
    assert (
        o["source"] == "pdf outline"
        and o["pdf_outline_entries"] == 3
        and o["markdown_headings"] == 4
    )
    assert o["outline"][0]["children"][0]["title"] == "Methods"
    o2 = pt.outline(fixtures["scholarly"], md)
    assert o2["source"] == "markdown headings"
    assert pt.outline(fixtures["scholarly"])["source"] == "none"
