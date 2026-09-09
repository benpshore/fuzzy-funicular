"""Deterministic test fixtures, generated with PyMuPDF so the suite has no binary blobs.

Run directly to (re)build into tests/fixtures/generated/, or let conftest build them lazily.
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import pymupdf

HERE = Path(__file__).parent
OUT = HERE / "generated"

TITLE = "Riluzole Exposure and Survival in Motor Neuron Disease: A Registry Cohort"
AUTHORS = "B. Shore¹, A. N. Other², C. Example¹"
ABSTRACT = (
    "Background. Population registries capture treatment exposure that trials cannot. "
    "Methods. We linked prescribing records to a national MND registry (n = 2,418) and fitted "
    "Cox models with time-varying exposure. Results. Adjusted HR 0.84 (95% CI 0.76–0.93). "
    "Conclusions. Effects were concentrated in the first 12 months after diagnosis."
)
COL1 = [
    "1  Introduction",
    "Motor neuron disease (MND) is a progressive neurodegenerative condition with a median "
    "survival of 24–36 months from symptom onset.¹ Riluzole remains the only widely licensed "
    "disease-modifying agent, with a modest survival benefit in trials.²",
    "Trial cohorts under-represent older patients, those with bulbar onset, and those with "
    "respiratory compromise at baseline.³ Registry data address this gap at the cost of "
    "confounding by indication, which we model explicitly (§2.3).",
    "2  Methods",
    "2.1  Data sources. Prescribing records were linked deterministically on health number "
    "and date of birth; 98.2% of registry entries matched.",
    "2.2  Exposure definition. Exposure began at first dispensing and ended 30 days after "
    "the last supply ran out.",
]
COL2 = [
    "2.3  Statistical analysis. Time-varying Cox models were adjusted for age, sex, site of "
    "onset, diagnostic delay, ALSFRS-R at diagnosis, and forced vital capacity.",
    "3  Results",
    "Of 2,418 patients, 1,904 (78.7%) received riluzole. Median follow-up was 19.4 months. "
    "Table 1 summarises baseline characteristics by exposure group.",
    "Table 1. Baseline characteristics",
    "TABLE",
    "4  Discussion",
    "The effect estimate is consistent with the pooled trial estimate (HR 0.85) despite a "
    "very different case mix, which argues against a large indication bias.⁴",
]
FOOTNOTES = [
    "¹ Department of Neurology, Example University Hospital.",
    "² Institute of Population Health, Example University.",
]
REFS = [
    "1. Hardiman O, et al. Amyotrophic lateral sclerosis. Nat Rev Dis Primers. 2017;3:17071.",
    "2. Miller RG, et al. Riluzole for ALS/MND. Cochrane Database Syst Rev. 2012;(3):CD001447.",
    "3. Chiò A, et al. Prognostic factors in ALS: a population-based study. 2009.",
    "4. Andrews JA, et al. Real-world evidence of riluzole effectiveness. 2020.",
]
TABLE_ROWS = [
    ("Characteristic", "Exposed", "Unexposed"),
    ("n", "1,904", "514"),
    ("Age, median (IQR)", "66 (58–73)", "72 (64–79)"),
    ("Bulbar onset, n (%)", "571 (30.0)", "201 (39.1)"),
    ("FVC % predicted", "84 (71–96)", "76 (61–90)"),
]


# MuPDF ships Droid Sans Fallback ("cjk"): a real embedded TrueType with a ToUnicode map, so
# en dashes, superscripts and accented names survive extraction on every platform.
_FONT = pymupdf.Font("cjk")
_FONTNAME = "F0"


def _two_column_page(doc: pymupdf.Document, page_no: int) -> None:
    page = doc.new_page(width=595, height=842)  # A4
    page.insert_font(fontname=_FONTNAME, fontbuffer=_FONT.buffer)
    y = 60.0
    if page_no == 0:
        page.insert_textbox(pymupdf.Rect(50, y, 545, y + 50), TITLE, fontsize=15, fontname="hebo")
        y += 52
        page.insert_textbox(pymupdf.Rect(50, y, 545, y + 16), AUTHORS, fontsize=10, fontname="heit")
        y += 26
        page.insert_textbox(
            pymupdf.Rect(70, y, 525, y + 70), ABSTRACT, fontsize=8.5, fontname=_FONTNAME, align=3
        )
        y += 86
    left = pymupdf.Rect(50, y, 290, 760)
    right = pymupdf.Rect(305, y, 545, 760)
    _fill_column(page, left, COL1 if page_no == 0 else REFS_COL(page_no))
    _fill_column(page, right, COL2 if page_no == 0 else [])
    # running header / footer
    page.insert_text((50, 30), "Shore et al. — MND registry cohort", fontsize=7, fontname="heit")
    page.insert_text((520, 815), str(page_no + 1), fontsize=8, fontname="helv")
    fy = 770
    for fn in FOOTNOTES if page_no == 0 else []:
        page.insert_text((50, fy), fn, fontsize=7, fontname=_FONTNAME)
        fy += 10


def REFS_COL(page_no: int) -> list[str]:
    return ["References"] + REFS


def _fill_column(page: pymupdf.Page, rect: pymupdf.Rect, paragraphs: list[str]) -> None:
    y = rect.y0
    for para in paragraphs:
        if para == "TABLE":
            y = _table(page, rect.x0, y, rect.width)
            continue
        heading = para[:1].isdigit() and "  " in para[:5] or para == "References"
        fontsize = 10 if heading else 9
        fontname = "hebo" if heading else _FONTNAME
        box = pymupdf.Rect(rect.x0, y, rect.x1, rect.y1)
        rc = page.insert_textbox(box, para, fontsize=fontsize, fontname=fontname, align=0)
        used = box.height - rc if rc >= 0 else box.height
        y += used + 6
        if y > rect.y1 - 20:
            break


def _table(page: pymupdf.Page, x: float, y: float, width: float) -> float:
    colw = [width * 0.46, width * 0.27, width * 0.27]
    rowh = 13
    for r, row in enumerate(TABLE_ROWS):
        cx = x
        for c, cell in enumerate(row):
            page.insert_text(
                (cx + 2, y + rowh * (r + 1) - 3),
                cell,
                fontsize=7.5,
                fontname="hebo" if r == 0 else _FONTNAME,
            )
            cx += colw[c]
        page.draw_line((x, y + rowh * (r + 1)), (x + width, y + rowh * (r + 1)), width=0.4)
    page.draw_line((x, y), (x + width, y), width=0.8)
    return y + rowh * (len(TABLE_ROWS) + 1)


def scholarly_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    _two_column_page(doc, 0)
    _two_column_page(doc, 1)
    doc.set_metadata({"title": TITLE, "author": "B. Shore"})
    doc.subset_fonts()
    doc.save(path, garbage=4, deflate=True)
    doc.close()
    return path


def scan_pdf(src: Path, path: Path, dpi: int = 200) -> Path:
    """Image-only PDF: every page rasterised and re-embedded, no text layer."""
    with pymupdf.open(src) as doc:
        out = pymupdf.open()
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=pix.tobytes("png"))
        out.save(path, garbage=4, deflate=True)
        out.close()
    return path


def mixed_pdf(src: Path, path: Path) -> Path:
    """Page 1 keeps its text layer, page 2 is image-only."""
    with pymupdf.open(src) as doc:
        out = pymupdf.open()
        out.insert_pdf(doc, from_page=0, to_page=0)
        pix = doc[1].get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
        new = out.new_page(width=doc[1].rect.width, height=doc[1].rect.height)
        new.insert_image(new.rect, stream=pix.tobytes("png"))
        out.save(path, garbage=4, deflate=True)
        out.close()
    return path


def photo_jpg(src: Path, path: Path) -> Path:
    """A 'phone photo' of page 1: rendered, slightly rotated, JPEG-compressed."""
    from PIL import Image

    with pymupdf.open(src) as doc:
        pix = doc[0].get_pixmap(dpi=150)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    img = img.rotate(1.5, expand=True, fillcolor=(235, 235, 230))
    img.save(path, format="JPEG", quality=70)
    return path


def broken_pdf(src: Path, path: Path) -> Path:
    """Truncate the xref/trailer so a strict parser must reconstruct it."""
    data = src.read_bytes()
    cut = data.rfind(b"xref")
    if cut == -1:
        cut = len(data) - 200
    path.write_bytes(data[:cut])
    return path


def article_html(path: Path) -> Path:
    path.write_text(
        """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Why registries matter for rare disease</title>
<style>body{font-family:sans-serif}</style><script>console.log('x')</script></head>
<body><nav><a href="/">Home</a></nav>
<article><h1>Why registries matter for rare disease</h1>
<p class="byline">By B. Shore · 4 min read</p>
<p>Randomised trials answer narrow questions in narrow populations. For motor neuron disease
that means most patients we actually treat were never eligible.</p>
<h2>What a registry can and cannot do</h2>
<p>A registry can show us exposure, dose, adherence and outcome across the whole population.
It cannot remove confounding by indication; it can only make it visible.</p>
<ul><li>Linkage quality is everything.</li>
<li>Report absolute risks, not just hazard ratios.</li></ul>
</article><footer>© Example</footer></body></html>
""",
        encoding="utf-8",
    )
    return path


def notes_md(path: Path) -> Path:
    path.write_text(
        "# Clinic notes\n\n- Reviewed FVC trend\n- Discussed NIV timing\n\n"
        "| Date | FVC % |\n|---|---|\n| 2026-06 | 84 |\n| 2026-09 | 76 |\n",
        encoding="utf-8",
    )
    return path


def sample_docx(path: Path) -> Path | None:
    try:
        import docx  # python-docx, ships with docling
    except ImportError:
        return None
    d = docx.Document()
    d.add_heading("Multidisciplinary team summary", level=1)
    d.add_paragraph("Patient reviewed in the MND clinic. Riluzole continued at 50 mg twice daily.")
    d.add_heading("Plan", level=2)
    for item in ("Repeat spirometry in 3 months", "Refer to respiratory physiology"):
        d.add_paragraph(item, style="List Bullet")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text = "Measure"
    t.cell(0, 1).text = "Value"
    t.cell(1, 0).text = "ALSFRS-R"
    t.cell(1, 1).text = "38"
    d.save(path)
    return path


def speech_wav(path: Path) -> Path | None:
    """Synthetic speech via espeak-ng when present (used for the ASR smoke test)."""
    import subprocess

    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        return None
    text = (
        "This is a test recording for the funicular document pipeline. "
        "The patient was reviewed in clinic and riluzole was continued."
    )
    subprocess.run([exe, "-s", "150", "-w", str(path), text], check=True, capture_output=True)
    return path


def build(out: Path = OUT) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    files["scholarly"] = scholarly_pdf(out / "scholarly.pdf")
    files["scan"] = scan_pdf(files["scholarly"], out / "scholarly-scan.pdf")
    files["mixed"] = mixed_pdf(files["scholarly"], out / "scholarly-mixed.pdf")
    files["photo"] = photo_jpg(files["scholarly"], out / "scholarly-photo.jpg")
    files["broken"] = broken_pdf(files["scholarly"], out / "scholarly-broken.pdf")
    files["html"] = article_html(out / "article.html")
    files["md"] = notes_md(out / "notes.md")
    docx_path = sample_docx(out / "summary.docx")
    if docx_path:
        files["docx"] = docx_path
    wav = speech_wav(out / "speech.wav")
    if wav:
        files["wav"] = wav
    return files


if __name__ == "__main__":
    for k, v in build().items():
        print(f"{k:10s} {v}  {v.stat().st_size:,} bytes")
