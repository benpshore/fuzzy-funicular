from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field


@dataclass
class Author:
    family: str
    given: str = ""

    def short(self) -> str:
        return self.family or self.given

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Work:
    doi: str | None = None
    title: str = ""
    authors: list[Author] = field(default_factory=list)
    year: int | None = None
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    publisher: str = ""
    type: str = ""
    abstract: str = ""
    url: str = ""
    pmid: str | None = None
    pmcid: str | None = None
    arxiv: str | None = None
    openalex_id: str | None = None
    s2_id: str | None = None
    cited_by: int | None = None
    references_count: int | None = None
    referenced_dois: list[str] = field(default_factory=list)
    oa_url: str | None = None
    scite: dict | None = None
    source: str = ""
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @property
    def first_author(self) -> str:
        return self.authors[0].short() if self.authors else ""

    def citation(self) -> str:
        auth = self.first_author + (" et al." if len(self.authors) > 1 else "")
        bits = [
            b for b in [auth, str(self.year) if self.year else "", self.title, self.journal] if b
        ]
        s = ". ".join(bits)
        if self.volume:
            s += f" {self.volume}"
            if self.issue:
                s += f"({self.issue})"
        if self.pages:
            s += f":{self.pages}"
        if self.doi:
            s += f". https://doi.org/{self.doi}"
        return s

    def merge(self, other: Work) -> Work:
        """Fill empty fields from another source; never overwrite a non-empty one."""
        for k, v in other.to_dict().items():
            if k in ("sources", "source"):
                continue
            cur = getattr(self, k)
            if (cur in (None, "", [], {}) or cur is None) and v not in (None, "", [], {}):
                setattr(self, k, getattr(other, k))
        if other.source and other.source not in self.sources:
            self.sources.append(other.source)
        return self


_CLEAN = re.compile(r"\s+")


def clean_title(t: str) -> str:
    t = re.sub(r"<[^>]+>", "", t or "")
    return _CLEAN.sub(" ", t).strip()
