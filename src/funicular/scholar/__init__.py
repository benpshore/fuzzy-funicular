"""Scholarly metadata: identifier extraction, lookup and verification across Crossref,
OpenAlex, Semantic Scholar, PubMed, Europe PMC, Scite, Unpaywall and arXiv; reference-list
extraction and resolution; Zotero-style renaming.

Google Scholar is deliberately absent: it has no API and blocks automated access; anything
claiming to "search Google Scholar" from a server is scraping and breaks within days.
"""

from .ids import Ids, extract_ids, normalize_doi
from .model import Work

__all__ = ["Ids", "Work", "extract_ids", "normalize_doi"]
