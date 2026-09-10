"""PDF utilities beyond text extraction:

* encryption — open with a password, write decrypted or AES-256 encrypted copies
* PDF/A — detect the claimed conformance (XMP pdfaid) and convert with Ghostscript
* signatures — signature form fields and ink-signature heuristics
* forms — list and fill AcroForm fields, optionally flatten
* pagination — page labels vs physical page numbers, printed folio detection
* header/footer — repeated lines in the top/bottom bands across pages
* outline — heading hierarchy from the PDF outline and from layout Markdown headings
"""

from __future__ import annotations

import re
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

from .textnorm import normalize
from .tools import ghostscript


# ------------------------------------------------------------------------------------------
# encryption
# ------------------------------------------------------------------------------------------
@dataclass
class EncryptionInfo:
    encrypted: bool
    needs_password: bool
    method: str = ""
    permissions: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def encryption_info(path: Path, password: str | None = None) -> EncryptionInfo:
    with pymupdf.open(path) as doc:
        if doc.needs_pass:
            ok = bool(password) and doc.authenticate(password) > 0
            if not ok:
                return EncryptionInfo(True, True)
        perms = doc.permissions
        enc = doc.is_encrypted or bool(password)
        return EncryptionInfo(
            enc,
            False,
            (doc.metadata.get("encryption") or "") if enc else "",
            {
                "print": bool(perms & pymupdf.PDF_PERM_PRINT),
                "copy": bool(perms & pymupdf.PDF_PERM_COPY),
                "modify": bool(perms & pymupdf.PDF_PERM_MODIFY),
                "annotate": bool(perms & pymupdf.PDF_PERM_ANNOTATE),
            },
        )


def open_with_password(path: Path, password: str | None) -> pymupdf.Document:
    doc = pymupdf.open(path)
    if doc.needs_pass:
        if not password or doc.authenticate(password) == 0:
            doc.close()
            raise PermissionError("password required or incorrect")
    return doc


def decrypt(path: Path, dest: Path, password: str | None) -> Path:
    doc = open_with_password(path, password)
    with doc:
        doc.save(dest, encryption=pymupdf.PDF_ENCRYPT_NONE, garbage=3, deflate=True)
    return dest


def encrypt(
    path: Path,
    dest: Path,
    *,
    user_password: str,
    owner_password: str | None = None,
    allow_print: bool = True,
    allow_copy: bool = False,
    password: str | None = None,
) -> Path:
    perms = pymupdf.PDF_PERM_ACCESSIBILITY
    if allow_print:
        perms |= pymupdf.PDF_PERM_PRINT
    if allow_copy:
        perms |= pymupdf.PDF_PERM_COPY
    doc = open_with_password(path, password)
    with doc:
        doc.save(
            dest,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            user_pw=user_password,
            owner_pw=owner_password or user_password,
            permissions=perms,
            garbage=3,
            deflate=True,
        )
    return dest


# ------------------------------------------------------------------------------------------
# PDF/A
# ------------------------------------------------------------------------------------------
_PDFA_PART = re.compile(r"pdfaid:part(?:=['\"]|>)(\d)", re.I)
_PDFA_CONF = re.compile(r"pdfaid:conformance(?:=['\"]|>)([A-Za-z])", re.I)


def pdfa_claim(path: Path) -> str | None:
    """'2B' etc. when the XMP metadata claims PDF/A conformance (a claim, not a validation)."""
    with pymupdf.open(path) as doc:
        try:
            xmp = doc.get_xml_metadata() or ""
        except Exception:  # noqa: BLE001
            xmp = ""
    m = _PDFA_PART.search(xmp)
    if not m:
        return None
    c = _PDFA_CONF.search(xmp)
    return m.group(1) + (c.group(1).upper() if c else "")


def _pdfa_def(tmp: Path) -> Path | None:
    """Ghostscript's PDFA_def.ps with the ICC path pointed at its bundled sRGB profile."""
    gs = ghostscript._gs()  # noqa: SLF001
    lib = None
    try:
        from .resources import run

        proc = run([gs, "-h"], timeout=20)
        for line in proc.stdout.decode("utf-8", "replace").splitlines():
            for part in line.split(":"):
                part = part.strip()
                if part.endswith("/lib") and (Path(part) / "PDFA_def.ps").is_file():
                    lib = Path(part)
    except Exception:  # noqa: BLE001
        lib = None
    if lib is None:
        for cand in Path("/").glob("**/ghostscript/*/lib/PDFA_def.ps"):
            lib = cand.parent
            break
    if lib is None:
        return None
    src = (lib / "PDFA_def.ps").read_text(errors="replace")
    icc = None
    for cand in (
        lib.parent / "iccprofiles" / "default_rgb.icc",
        Path("/usr/share/color/icc/ghostscript/default_rgb.icc"),
        Path("/opt/homebrew/share/ghostscript")
        / lib.parent.name
        / "iccprofiles"
        / "default_rgb.icc",
    ):
        if cand.is_file():
            icc = cand
            break
    if icc is None:
        for cand in Path("/").glob("**/ghostscript/**/default_rgb.icc"):
            icc = cand
            break
    if icc is None:
        return None
    src = re.sub(r"/ICCProfile \([^)]*\)", f"/ICCProfile ({icc})", src)
    out = tmp / "PDFA_def.ps"
    out.write_text(src)
    return out


def _gs_pdfa(src: Path, dest: Path, level: int, pdfa_def: Path | None, timeout: int) -> str:
    from .resources import run

    argv = [
        ghostscript._gs(),  # noqa: SLF001
        "-dSAFER",
        "-dNOPAUSE",
        "-dBATCH",
        f"-dPDFA={level}",
        "-dPDFACompatibilityPolicy=1",
        "-sColorConversionStrategy=RGB",
        "-sProcessColorModel=DeviceRGB",
        "-sDEVICE=pdfwrite",
        "-dEmbedAllFonts=true",
        f"-sOutputFile={dest}",
    ]
    if pdfa_def:
        argv.append(str(pdfa_def))
    argv.append(str(src))
    proc = run(argv, timeout=timeout)
    log_text = proc.stdout.decode("utf-8", "replace") + proc.stderr.decode("utf-8", "replace")
    if proc.returncode != 0 or not dest.exists():
        raise ghostscript.GhostscriptError(log_text[-400:])
    return log_text


def to_pdfa(path: Path, dest: Path, *, level: int = 2, timeout: int = 900) -> dict:
    """Convert with Ghostscript to PDF/A-{level}b (sRGB output intent).

    Returns what the result claims afterwards plus Ghostscript's own reasons when it had to
    revert to plain PDF (e.g. a CID font using CID 0). A true validation needs veraPDF."""
    if not ghostscript.available():
        raise ghostscript.GhostscriptError("ghostscript not installed")
    warnings: list[str] = []
    with tempfile.TemporaryDirectory(prefix="funicular-pdfa-") as td:
        tmp = Path(td)
        pdfa_def = _pdfa_def(tmp)
        log_text = _gs_pdfa(path, dest, level, pdfa_def, timeout)
        claims = pdfa_claim(dest)
        if claims is None:
            # Second attempt: normalise fonts/streams with a plain pdfwrite pass first.
            normalised = tmp / "normalised.pdf"
            try:
                ghostscript.repair_pdf(path, normalised, timeout=timeout)
                log_text = _gs_pdfa(normalised, dest, level, pdfa_def, timeout)
                claims = pdfa_claim(dest)
            except ghostscript.GhostscriptError as exc:
                warnings.append(f"normalisation pass failed: {exc}")
        for line in log_text.splitlines():
            if "PDF/A" in line and (
                "not legal" in line or "reverting" in line or "cannot" in line.lower()
            ):
                warnings.append(line.strip())
    return {
        "output": str(dest),
        "claims": claims,
        "output_intent": bool(pdfa_def),
        "bytes": dest.stat().st_size,
        "warnings": sorted(set(warnings)),
    }


# ------------------------------------------------------------------------------------------
# forms and signatures
# ------------------------------------------------------------------------------------------
@dataclass
class FormField:
    page: int
    name: str
    type: str
    value: str
    rect: list[float]
    options: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def list_fields(path: Path, password: str | None = None) -> list[FormField]:
    out: list[FormField] = []
    doc = open_with_password(path, password)
    with doc:
        for page in doc:
            for w in page.widgets():
                opts = []
                try:
                    opts = [
                        str(o if not isinstance(o, list | tuple) else o[0])
                        for o in (w.choice_values or [])
                    ]
                except Exception:  # noqa: BLE001
                    opts = []
                out.append(
                    FormField(
                        page.number + 1,
                        w.field_name or "",
                        w.field_type_string or "",
                        "" if w.field_value is None else str(w.field_value),
                        [round(v, 1) for v in w.rect],
                        opts,
                    )
                )
    return out


def fill_fields(
    path: Path,
    dest: Path,
    values: dict[str, str | bool],
    *,
    flatten: bool = False,
    password: str | None = None,
) -> dict:
    """Set field values by name. Checkboxes accept true/false; radios and combos take a value."""
    doc = open_with_password(path, password)
    filled, missing = [], set(values)
    with doc:
        for page in doc:
            for w in page.widgets():
                name = w.field_name or ""
                if name not in values:
                    continue
                v = values[name]
                if w.field_type == pymupdf.PDF_WIDGET_TYPE_CHECKBOX:
                    w.field_value = w.on_state() if v in (True, "true", "yes", "on", "1") else "Off"
                elif w.field_type == pymupdf.PDF_WIDGET_TYPE_RADIOBUTTON:
                    w.field_value = str(v) == str(w.on_state()) or v is True
                else:
                    w.field_value = "" if v is None else str(v)
                w.update()
                filled.append(name)
                missing.discard(name)
        if flatten:
            doc.bake(annots=True, widgets=True)
        doc.save(dest, garbage=3, deflate=True)
    return {"filled": filled, "missing": sorted(missing), "flattened": flatten, "output": str(dest)}


@dataclass
class SignatureFinding:
    page: int
    kind: str  # field | ink | image
    name: str = ""
    signed: bool | None = None
    rect: list[float] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_SIG_WORDS = re.compile(r"\b(signature|signed|sign here|signatory|firma|unterschrift)\b", re.I)


def detect_signatures(path: Path, password: str | None = None) -> list[SignatureFinding]:
    """Signature fields (with a signed/unsigned verdict) plus heuristics for drawn or pasted
    signatures near a 'Signature' label or signature line."""
    out: list[SignatureFinding] = []
    doc = open_with_password(path, password)
    with doc:
        for page in doc:
            for w in page.widgets():
                if w.field_type == pymupdf.PDF_WIDGET_TYPE_SIGNATURE:
                    signed = None
                    try:
                        signed = bool(doc.xref_get_key(w.xref, "V")[1] not in ("null", ""))
                    except Exception:  # noqa: BLE001
                        signed = None
                    out.append(
                        SignatureFinding(
                            page.number + 1,
                            "field",
                            w.field_name or "",
                            signed,
                            [round(v, 1) for v in w.rect],
                        )
                    )
            anchors = [
                pymupdf.Rect(b[:4])
                for b in page.get_text("blocks")
                if _SIG_WORDS.search(b[4] or "")
            ]
            if not anchors:
                continue
            # ink: many small curved paths clustered near an anchor
            try:
                drawings = page.get_drawings()
            except Exception:  # noqa: BLE001
                drawings = []
            for anchor in anchors:
                zone = pymupdf.Rect(anchor.x0 - 40, anchor.y0 - 90, anchor.x1 + 260, anchor.y1 + 30)
                curves = [
                    d
                    for d in drawings
                    if pymupdf.Rect(d["rect"]).intersects(zone)
                    and any(it[0] == "c" for it in d.get("items", []))
                ]
                strokes = sum(1 for d in curves for it in d.get("items", []) if it[0] == "c")
                if strokes >= 3:
                    r = pymupdf.Rect()
                    for d in curves:
                        r |= pymupdf.Rect(d["rect"])
                    out.append(
                        SignatureFinding(
                            page.number + 1,
                            "ink",
                            "",
                            True,
                            [round(v, 1) for v in r],
                            f"{strokes} curved strokes near a signature label",
                        )
                    )
                    break
                images = [
                    i for i in page.get_image_info() if pymupdf.Rect(i["bbox"]).intersects(zone)
                ]
                for info in images:
                    r = pymupdf.Rect(info["bbox"])
                    if 20 < r.height < 160 and r.width / max(r.height, 1) > 1.5:
                        out.append(
                            SignatureFinding(
                                page.number + 1,
                                "image",
                                "",
                                True,
                                [round(v, 1) for v in r],
                                "small wide image next to a signature label",
                            )
                        )
                        break
    return out


# ------------------------------------------------------------------------------------------
# pagination, headers/footers, outline
# ------------------------------------------------------------------------------------------
@dataclass
class Pagination:
    pages: int
    labels: list[str]
    printed: list[str | None]
    offset: int | None  # printed folio = physical page + offset, when consistent

    def to_dict(self) -> dict:
        return asdict(self)


_NUM = re.compile(r"^\s*(?:page\s+)?(\d{1,4}|[ivxlcdm]{1,7})\s*$", re.I)


def _band_lines(
    page: pymupdf.Page, top_frac: float = 0.12, bottom_frac: float = 0.12
) -> tuple[list[str], list[str]]:
    h = page.rect.height
    top, bottom = [], []
    for b in page.get_text("blocks"):
        y0, y1, text = b[1], b[3], (b[4] or "").strip()
        if not text:
            continue
        for line in text.splitlines():
            if y1 <= h * top_frac:
                top.append(line.strip())
            elif y0 >= h * (1 - bottom_frac):
                bottom.append(line.strip())
    return top, bottom


def pagination(path: Path, password: str | None = None) -> Pagination:
    doc = open_with_password(path, password)
    with doc:
        labels = []
        printed: list[str | None] = []
        for page in doc:
            try:
                labels.append(page.get_label() or "")
            except Exception:  # noqa: BLE001
                labels.append("")
            top, bottom = _band_lines(page)
            folio = None
            for line in bottom + top:
                m = _NUM.match(line)
                if m:
                    folio = m.group(1)
                    break
            printed.append(folio)
        n = doc.page_count
    offsets = Counter()
    for i, f in enumerate(printed, start=1):
        if f and f.isdigit():
            offsets[int(f) - i] += 1
    offset = None
    if offsets:
        best, count = offsets.most_common(1)[0]
        if count >= max(2, n // 3):
            offset = best
    return Pagination(n, labels, printed, offset)


@dataclass
class HeaderFooter:
    headers: list[str]
    footers: list[str]
    pages_checked: int

    def to_dict(self) -> dict:
        return asdict(self)


def _template(line: str) -> str:
    """Digits become '#', punctuation is dropped, so 'p. 12' == 'p 13' and dash variants agree."""
    t = normalize(re.sub(r"\d+", "#", line))
    return re.sub(r"[^\w#]+", " ", t).strip()


def detect_headers_footers(
    path: Path, *, min_share: float = 0.5, password: str | None = None
) -> HeaderFooter:
    """Lines that recur (digits masked) in the top/bottom bands of at least min_share of pages."""
    doc = open_with_password(path, password)
    with doc:
        tops: Counter[str] = Counter()
        bottoms: Counter[str] = Counter()
        surface: dict[str, str] = {}
        n = doc.page_count
        for page in doc:
            t, b = _band_lines(page)
            for line in set(t):
                k = _template(line)
                if k:
                    tops[k] += 1
                    surface.setdefault(k, line)
            for line in set(b):
                k = _template(line)
                if k:
                    bottoms[k] += 1
                    surface.setdefault(k, line)
    need = max(2, int(n * min_share + 0.999))
    headers = [surface[k] for k, c in tops.most_common() if c >= need]
    footers = [surface[k] for k, c in bottoms.most_common() if c >= need]
    return HeaderFooter(headers, footers, n)


def strip_headers_footers(text: str, hf: HeaderFooter) -> str:
    """Remove recurring header/footer lines from extracted text (digits treated as wildcards)."""
    templates = {_template(x) for x in hf.headers + hf.footers if x.strip()}
    if not templates:
        return text
    out = []
    for raw in text.splitlines():
        # MuPDF sometimes glues a header to the previous line with a NUL: split those apart
        parts = [p for p in raw.split("\x00")]
        kept = [p for p in parts if not (p.strip() and _template(p) in templates)]
        if not kept:
            continue
        out.append("\n".join(kept))
    return "\n".join(out)


@dataclass
class OutlineNode:
    level: int
    title: str
    page: int | None
    children: list[OutlineNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "title": self.title,
            "page": self.page,
            "children": [c.to_dict() for c in self.children],
        }


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_PAGE_SEP = re.compile(r"^--- end of page\.page_number=(\d+) ---$")


def _nest(flat: list[tuple[int, str, int | None]]) -> list[OutlineNode]:
    root: list[OutlineNode] = []
    stack: list[OutlineNode] = []
    for level, title, page in flat:
        node = OutlineNode(level, title, page)
        while stack and stack[-1].level >= level:
            stack.pop()
        (stack[-1].children if stack else root).append(node)
        stack.append(node)
    return root


def outline_from_pdf(path: Path, password: str | None = None) -> list[OutlineNode]:
    doc = open_with_password(path, password)
    with doc:
        toc = doc.get_toc(simple=True)
    return _nest([(lvl, title.strip(), page) for lvl, title, page in toc])


def outline_from_markdown(md: str) -> list[OutlineNode]:
    """Heading hierarchy from layout Markdown; page numbers from pymupdf4llm's separators."""
    flat: list[tuple[int, str, int | None]] = []
    pending: list[int] = []
    page = 1
    in_code = False
    for line in md.splitlines():
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = _PAGE_SEP.match(line.strip())
        if m:
            for i in pending:
                flat[i] = (flat[i][0], flat[i][1], int(m.group(1)) + 1)
            pending = []
            page = int(m.group(1)) + 2
            continue
        h = _MD_HEADING.match(line)
        if h:
            title = re.sub(r"[*_`]+", "", h.group(2)).strip()
            if title:
                flat.append((len(h.group(1)), title, None))
                pending.append(len(flat) - 1)
    for i in pending:
        flat[i] = (flat[i][0], flat[i][1], page)
    return _nest(flat)


def outline(path: Path, markdown: str | None = None, password: str | None = None) -> dict:
    pdf_ol = outline_from_pdf(path, password)
    md_ol = outline_from_markdown(markdown) if markdown else []
    chosen = pdf_ol if pdf_ol else md_ol
    return {
        "source": "pdf outline" if pdf_ol else ("markdown headings" if md_ol else "none"),
        "outline": [n.to_dict() for n in chosen],
        "pdf_outline_entries": _count(pdf_ol),
        "markdown_headings": _count(md_ol),
    }


def _count(nodes: list[OutlineNode]) -> int:
    return sum(1 + _count(n.children) for n in nodes)


def copy_original(src: Path, dst: Path) -> Path:
    shutil.copy2(src, dst)
    return dst
