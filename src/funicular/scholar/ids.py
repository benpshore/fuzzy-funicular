"""Identifier extraction from text: DOI, arXiv, PMID, PMCID, ISBN."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>()\[\]{}]+)", re.IGNORECASE)
DOI_URL_RE = re.compile(r"(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s\"'<>()\[\]{}]+)", re.I)
ARXIV_NEW_RE = re.compile(r"\barXiv:\s*(\d{4}\.\d{4,5})(v\d+)?\b", re.I)
ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?", re.I)
ARXIV_OLD_RE = re.compile(r"\barXiv:\s*([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?\b", re.I)
PMID_RE = re.compile(r"\bPMID:?\s*(\d{5,9})\b", re.I)
PMCID_RE = re.compile(r"\b(PMC\d{5,9})\b")
ISBN_RE = re.compile(r"\bISBN(?:-1[03])?:?\s*([\dXx][\d\-\s]{8,16}[\dXx])\b", re.I)
_TRAIL = ".,;:)]}>'\""


def normalize_doi(doi: str) -> str:
    d = doi.strip()
    m = DOI_URL_RE.search(d) or DOI_RE.search(d)
    if m:
        d = m.group(1)
    d = d.rstrip(_TRAIL)
    # PDFs often glue a following word/period: "10.1000/xyz.Received" -> keep up to a sane end
    d = re.sub(r"\.(?:Received|Accepted|Published|Copyright|Available)\b.*$", "", d, flags=re.I)
    return d.lower()


def isbn_valid(raw: str) -> str | None:
    digits = re.sub(r"[\s\-]", "", raw).upper()
    if len(digits) == 10:
        total = 0
        for i, ch in enumerate(digits):
            v = 10 if ch == "X" and i == 9 else (int(ch) if ch.isdigit() else -1)
            if v < 0:
                return None
            total += v * (10 - i)
        return digits if total % 11 == 0 else None
    if len(digits) == 13 and digits.isdigit():
        total = sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(digits))
        return digits if total % 10 == 0 else None
    return None


@dataclass
class Ids:
    doi: str | None = None
    arxiv: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    isbn: str | None = None
    all_dois: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def any(self) -> bool:
        return bool(self.doi or self.arxiv or self.pmid or self.pmcid or self.isbn)


def extract_ids(text: str, *, head_chars: int = 20_000) -> Ids:
    """Identifiers found in text. The first DOI on the first page is the document's own DOI
    far more often than not; later DOIs are usually references."""
    head = text[:head_chars]
    ids = Ids()
    seen: list[str] = []
    for m in list(DOI_URL_RE.finditer(head)) + list(DOI_RE.finditer(head)):
        d = normalize_doi(m.group(1))
        if len(d) < 8 or d in seen:
            continue
        seen.append(d)
    # prefer DOIs that appear before the first "References" heading
    ref_pos = re.search(r"\n\s*(references|bibliography|literature cited)\s*\n", head, re.I)
    if ref_pos:
        before = [d for d in seen if head.lower().find(d) < ref_pos.start()]
        seen = before + [d for d in seen if d not in before]
    ids.all_dois = seen
    ids.doi = seen[0] if seen else None
    m = ARXIV_NEW_RE.search(head) or ARXIV_URL_RE.search(head) or ARXIV_OLD_RE.search(head)
    if m:
        ids.arxiv = m.group(1) + (m.group(2) or "")
    m = PMID_RE.search(head)
    if m:
        ids.pmid = m.group(1)
    m = PMCID_RE.search(head)
    if m:
        ids.pmcid = m.group(1)
    for m in ISBN_RE.finditer(head):
        v = isbn_valid(m.group(1))
        if v:
            ids.isbn = v
            break
    return ids
