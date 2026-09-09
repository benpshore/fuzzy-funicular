"""Reference lists: find the section, split it into entries (numbered or author-year), keep
every raw string verbatim (lossless), extract what can be read off each entry, and resolve
entries to DOIs through Crossref and Semantic Scholar with an explicit confidence."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from rapidfuzz import fuzz

from .clients import Fetcher
from .ids import DOI_RE, DOI_URL_RE, PMID_RE, normalize_doi
from .model import Work

HEADING_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?(references?|bibliography|literature cited|works cited|reference list"
    r"|references and notes)\s*$",
    re.I | re.M,
)
END_RE = re.compile(
    r"^\s*(appendix|supplementary|acknowledg[e]?ments?|author contributions|figure legends?)\b",
    re.I | re.M,
)
NUM_RE = re.compile(r"^\s*(?:\[(\d{1,4})\]|(\d{1,4})[.)])\s+")
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)[a-z]?\b")
# "Hardiman O, Al-Chalabi A, Chio A." / "Miller RG, Mitchell JD, Moore DH."
AUTHORS_VANCOUVER = re.compile(
    r"^(?:[A-ZÀ-Ý][A-Za-zÀ-ÿ'\-]+(?:\s[A-ZÀ-Ý][A-Za-zÀ-ÿ'\-]+)?\s[A-Z]{1,3},?\s*)+"
    r"(?:et al\.?,?\s*)?\.?\s*"
)
# "Hardiman, O., Al-Chalabi, A., & Chio, A."
AUTHORS_APA = re.compile(
    r"^(?:[A-ZÀ-Ý][A-Za-zÀ-ÿ'\-]+,\s(?:[A-Z]\.\s?)+(?:,\s|\s?&\s|\sand\s)?)+(?:et al\.?)?\s*"
)


@dataclass
class RawRef:
    n: int
    raw: str
    doi: str | None = None
    pmid: str | None = None
    year: int | None = None
    first_author: str = ""
    title_guess: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResolvedRef:
    ref: RawRef
    work: Work | None
    confidence: float
    method: str
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "ref": self.ref.to_dict(),
            "work": self.work.to_dict() if self.work else None,
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "note": self.note,
        }


def find_reference_section(text: str) -> str:
    """Return the text of the last References-like section (verbatim, page breaks removed)."""
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return ""
    start = matches[-1].end()
    rest = text[start:]
    end = END_RE.search(rest)
    section = rest[: end.start()] if end and end.start() > 200 else rest
    return section.replace("\f", "\n")


def split_references(section: str) -> list[str]:
    lines = [ln.rstrip() for ln in section.splitlines()]
    # drop running headers / page numbers
    lines = [ln for ln in lines if not re.fullmatch(r"\s*\d{1,4}\s*", ln)]
    numbered = sum(1 for ln in lines if NUM_RE.match(ln))
    entries: list[str] = []
    if numbered >= 3:
        cur = ""
        for ln in lines:
            if NUM_RE.match(ln):
                if cur.strip():
                    entries.append(cur.strip())
                cur = ln
            else:
                cur += (" " if cur and not cur.endswith("-") else "") + ln.strip()
        if cur.strip():
            entries.append(cur.strip())
    else:
        # author-year style: a new entry starts with a capitalised surname after a blank line
        # or after a line that ends with a period / DOI
        cur = ""
        for ln in lines:
            starts = bool(re.match(r"^[A-ZÀ-Ý][A-Za-zÀ-ÿ'\-]+,?\s+[A-Z]", ln)) and (
                not cur or re.search(r"(\.|\d{4}[a-z]?\.?|\))\s*$|doi\S*$", cur.strip(), re.I)
            )
            if not ln.strip():
                if cur.strip():
                    entries.append(cur.strip())
                cur = ""
                continue
            if starts and cur.strip():
                entries.append(cur.strip())
                cur = ln
            else:
                cur += (" " if cur and not cur.endswith("-") else "") + ln.strip()
        if cur.strip():
            entries.append(cur.strip())
    # rejoin words hyphenated across lines
    out = []
    for e in entries:
        e = re.sub(r"(\w)- (\w)", r"\1\2", e)
        e = re.sub(r"\s+", " ", e).strip()
        if len(e) >= 20:
            out.append(e)
    return out


def parse_reference(n: int, raw: str) -> RawRef:
    ref = RawRef(n=n, raw=raw)
    m = DOI_URL_RE.search(raw) or DOI_RE.search(raw)
    if m:
        ref.doi = normalize_doi(m.group(1))
    m = PMID_RE.search(raw)
    if m:
        ref.pmid = m.group(1)
    body = NUM_RE.sub("", raw, count=1)
    years = YEAR_RE.findall(body)
    if years:
        ref.year = int(years[0])
    m = re.match(r"^([A-ZÀ-Ý][A-Za-zÀ-ÿ'\-]+(?:\s[A-ZÀ-Ý][A-Za-zÀ-ÿ'\-]+)?)[,\s]", body)
    if m:
        ref.first_author = m.group(1)
    # title guess: drop a leading author list (Vancouver "Smith J, Doe AB." or APA
    # "Smith, J., Doe, A. B. (2001)."), then take the first sentence-like chunk.
    rest = AUTHORS_VANCOUVER.sub("", body, count=1)
    rest = AUTHORS_APA.sub("", rest, count=1)
    rest = re.sub(r"^\s*\(?(19|20)\d\d[a-z]?\)?\.?\s*", "", rest)
    chunks = [c.strip() for c in re.split(r"(?<=\w\w)[.?!]\s", rest)]
    chunks = [
        c
        for c in chunks
        if len(c) > 15 and not re.match(r"^(19|20)\d\d", c) and not DOI_RE.search(c)
    ]
    if chunks:
        ref.title_guess = chunks[0][:200]
    return ref


def extract_references(text: str) -> list[RawRef]:
    section = find_reference_section(text)
    if not section:
        return []
    return [parse_reference(i + 1, raw) for i, raw in enumerate(split_references(section))]


def resolve_reference(ref: RawRef, fetcher: Fetcher) -> ResolvedRef:
    if ref.doi:
        w = fetcher.crossref_work(ref.doi)
        if w:
            return ResolvedRef(ref, w, 0.98, "doi")
    if ref.pmid:
        w = fetcher.pubmed_summary(ref.pmid)
        if w:
            return ResolvedRef(ref, w, 0.95, "pmid")
    cands = fetcher.crossref_query(ref.raw, rows=3)
    best, best_score = None, 0.0
    for c in cands:
        if not c.title:
            continue
        s = fuzz.partial_ratio(c.title.lower(), ref.raw.lower()) / 100.0
        if ref.year and c.year and abs(c.year - ref.year) > 1:
            s -= 0.3
        if (
            ref.first_author
            and c.first_author
            and ref.first_author.lower() != c.first_author.lower()
        ):
            s -= 0.15
        if s > best_score:
            best, best_score = c, s
    if best is None or best_score < 0.8:
        q = ref.title_guess or ref.raw
        for c in fetcher.s2_search(q, rows=3):
            s = fuzz.partial_ratio(c.title.lower(), ref.raw.lower()) / 100.0
            if ref.year and c.year and abs(c.year - ref.year) > 1:
                s -= 0.3
            if s > best_score:
                best, best_score = c, s
    if best is not None and best_score >= 0.8:
        return ResolvedRef(ref, best, min(best_score, 0.94), "bibliographic search")
    return ResolvedRef(
        ref,
        best,
        best_score if best else 0.0,
        "unresolved",
        "no service returned a confident match; raw string kept",
    )


@dataclass
class ReferenceReport:
    refs: list[ResolvedRef] = field(default_factory=list)

    @property
    def resolved(self) -> int:
        return sum(1 for r in self.refs if r.work and r.confidence >= 0.8)

    def to_dict(self) -> dict:
        return {
            "total": len(self.refs),
            "resolved": self.resolved,
            "refs": [r.to_dict() for r in self.refs],
        }
