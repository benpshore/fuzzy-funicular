"""Metadata for a PDF: what the file says about itself (XMP/Info, first-page typography), what
identifiers it carries, and a verified record from the bibliographic services."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf
from rapidfuzz import fuzz

from .clients import Fetcher
from .ids import Ids, extract_ids
from .model import Work, clean_title


@dataclass
class Extracted:
    ids: Ids
    pdf_title: str = ""
    pdf_author: str = ""
    pdf_subject: str = ""
    pdf_keywords: str = ""
    pdf_created: str = ""
    guessed_title: str = ""
    first_page_text: str = ""
    year_guess: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ids"] = self.ids.to_dict()
        d.pop("first_page_text", None)
        return d


def _biggest_lines(page: pymupdf.Page, max_lines: int = 3) -> str:
    """Title guess: the largest-font text near the top of page 1 (joined over ≤3 lines)."""
    try:
        d = page.get_text("dict")
    except Exception:  # noqa: BLE001
        return ""
    lines: list[tuple[float, float, str]] = []
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            if len(text) < 4 or not spans:
                continue
            size = max(s.get("size", 0) for s in spans)
            y = line.get("bbox", [0, 0, 0, 0])[1]
            if y > page.rect.height * 0.6:
                continue
            lines.append((size, y, text))
    if not lines:
        return ""
    top = max(size for size, _, _ in lines)
    chosen = [(y, t) for size, y, t in lines if size >= top - 0.6]
    chosen.sort()
    title = " ".join(t for _, t in chosen[:max_lines])
    return clean_title(title)


def extract_from_pdf(path: Path, *, pages: int = 2) -> Extracted:
    with pymupdf.open(path) as doc:
        meta = doc.metadata or {}
        text = "\n".join(doc[i].get_text("text") for i in range(min(pages, doc.page_count)))
        guessed = _biggest_lines(doc[0]) if doc.page_count else ""
    ids = extract_ids(
        text + "\n" + (meta.get("subject") or "") + "\n" + (meta.get("keywords") or "")
    )
    ex = Extracted(
        ids=ids,
        pdf_title=clean_title(meta.get("title") or ""),
        pdf_author=meta.get("author") or "",
        pdf_subject=meta.get("subject") or "",
        pdf_keywords=meta.get("keywords") or "",
        pdf_created=meta.get("creationDate") or "",
        guessed_title=guessed,
        first_page_text=text[:6000],
    )
    m = re.search(r"\b(19[6-9]\d|20[0-4]\d)\b", (ex.pdf_created or "")[2:6] + " " + text[:3000])
    ex.year_guess = int(m.group(1)) if m else None
    # Producer boilerplate masquerading as a title ("Microsoft Word - draft.docx", "untitled")
    if re.match(r"^(microsoft word|untitled|doc\d*|\d+\.pdf|pii:)", ex.pdf_title, re.I):
        ex.pdf_title = ""
    return ex


@dataclass
class Resolution:
    work: Work | None
    confidence: float  # 0..1
    verified: bool
    evidence: list[str] = field(default_factory=list)
    candidates: list[Work] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "work": self.work.to_dict() if self.work else None,
            "confidence": round(self.confidence, 3),
            "verified": self.verified,
            "evidence": self.evidence,
            "candidates": [c.to_dict() for c in self.candidates[:5]],
        }


def title_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(clean_title(a).lower(), clean_title(b).lower()) / 100.0


def _text_supports(work: Work, ex: Extracted) -> tuple[float, list[str]]:
    """How well the fetched record agrees with what the PDF itself shows."""
    ev: list[str] = []
    score = 0.0
    page = ex.first_page_text.lower()
    best_title = max(
        title_similarity(work.title, ex.pdf_title), title_similarity(work.title, ex.guessed_title)
    )
    if best_title >= 0.9:
        score += 0.5
        ev.append(f"title matches PDF ({best_title:.2f})")
    elif work.title and fuzz.partial_ratio(work.title.lower(), page) >= 90:
        score += 0.45
        ev.append("title found on first page")
    elif best_title >= 0.7:
        score += 0.25
        ev.append(f"title roughly matches ({best_title:.2f})")
    if work.first_author and work.first_author.lower() in page:
        score += 0.2
        ev.append(f"first author '{work.first_author}' on first page")
    if work.year and (str(work.year) in page or work.year == ex.year_guess):
        score += 0.15
        ev.append(f"year {work.year} present")
    if work.doi and work.doi in page:
        score += 0.15
        ev.append("DOI printed on first page")
    return min(score, 1.0), ev


def resolve(
    ex: Extracted,
    fetcher: Fetcher,
    *,
    sources: tuple[str, ...] = (
        "crossref",
        "openalex",
        "semanticscholar",
        "pubmed",
        "europepmc",
        "scite",
        "unpaywall",
        "arxiv",
    ),
) -> Resolution:
    """Look the document up by its identifiers first, then by title, and verify the answer
    against the PDF's own first page. Never trusts a lookup the page contradicts."""
    doi = ex.ids.doi
    work: Work | None = None
    evidence: list[str] = []
    candidates: list[Work] = []
    if doi and "crossref" in sources:
        work = fetcher.crossref_work(doi)
        if work:
            evidence.append("Crossref record for the DOI on page 1")
    if work is None and ex.ids.arxiv and "arxiv" in sources:
        work = fetcher.arxiv(ex.ids.arxiv)
        if work:
            evidence.append("arXiv record for the arXiv id on page 1")
    if work is None and ex.ids.pmid and "pubmed" in sources:
        work = fetcher.pubmed_summary(ex.ids.pmid)
        if work:
            evidence.append("PubMed record for the PMID on page 1")
    if work is None:
        query = ex.pdf_title or ex.guessed_title
        if query and len(query) > 12:
            if "crossref" in sources:
                candidates += fetcher.crossref_query(query, rows=5)
            if "semanticscholar" in sources:
                candidates += fetcher.s2_search(query, rows=5)
            if "openalex" in sources and not candidates:
                candidates += fetcher.openalex_search(query, rows=5)
            scored = sorted(
                ((title_similarity(c.title, query), c) for c in candidates), key=lambda t: -t[0]
            )
            if scored and scored[0][0] >= 0.85:
                work = scored[0][1]
                evidence.append(f"title search ({work.source}) similarity {scored[0][0]:.2f}")
    if work is None:
        return Resolution(
            None, 0.0, False, evidence or ["no identifier and no confident title match"], candidates
        )
    # enrich from the other services (never overwrite Crossref's bibliographic fields)
    if work.doi:
        for name, fn in (
            ("openalex", fetcher.openalex_work),
            ("semanticscholar", lambda d: fetcher.s2_paper("DOI:" + d)),
            ("pubmed", fetcher.pubmed_by_doi),
            ("europepmc", lambda d: fetcher.europepmc("DOI:" + d)),
        ):
            if name in sources and name not in work.sources:
                try:
                    other = fn(work.doi)
                except Exception:  # noqa: BLE001 - one flaky service must not sink the lookup
                    other = None
                if other:
                    work.merge(other)
        if "scite" in sources:
            work.scite = fetcher.scite_tallies(work.doi)
        if "unpaywall" in sources and not work.oa_url:
            work.oa_url = fetcher.unpaywall(work.doi)
    score, ev = _text_supports(work, ex)
    evidence += ev
    verified = score >= 0.5
    if not verified:
        evidence.append("record NOT confirmed by the PDF's first page; treat as a guess")
    return Resolution(work, score, verified, evidence, candidates)
