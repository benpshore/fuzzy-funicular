"""Zotero-style file naming from a resolved Work. Dry-run by default."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .model import Work

DEFAULT_TEMPLATE = "{author} - {year} - {title}"
_BAD = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def _clean(s: str, limit: int, *, keep_dots: bool = True) -> str:
    s = _BAD.sub("", s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s[:limit].rstrip()
    return s if keep_dots else s.rstrip(" .")


def build_name(work: Work, template: str = DEFAULT_TEMPLATE, *, max_title: int = 90) -> str:
    authors = work.authors
    if not authors:
        author = "Unknown"
    elif len(authors) == 1:
        author = authors[0].short()
    elif len(authors) == 2:
        author = f"{authors[0].short()} and {authors[1].short()}"
    else:
        author = f"{authors[0].short()} et al."
    fields = {
        "author": _clean(author, 60),
        "first_author": _clean(work.first_author, 40),
        "year": str(work.year) if work.year else "n.d.",
        "title": _clean(work.title, max_title),
        "journal": _clean(work.journal, 60),
        "doi": _clean((work.doi or "").replace("/", "_"), 80),
        "volume": _clean(work.volume, 10),
    }
    try:
        name = template.format(**fields)
    except KeyError, IndexError:
        name = DEFAULT_TEMPLATE.format(**fields)
    return _clean(name, 200, keep_dots=False) or "document"


@dataclass
class RenamePlan:
    source: Path
    target: Path
    changed: bool

    def to_dict(self) -> dict:
        return {"source": str(self.source), "target": str(self.target), "changed": self.changed}


def plan_rename(path: Path, work: Work, template: str = DEFAULT_TEMPLATE) -> RenamePlan:
    name = build_name(work, template) + path.suffix.lower()
    target = path.with_name(name)
    if target == path:
        return RenamePlan(path, path, False)
    n = 2
    while target.exists() and target != path:
        target = path.with_name(f"{build_name(work, template)} ({n}){path.suffix.lower()}")
        n += 1
    return RenamePlan(path, target, True)


def apply_rename(plan: RenamePlan) -> Path:
    if plan.changed:
        plan.source.rename(plan.target)
    return plan.target
