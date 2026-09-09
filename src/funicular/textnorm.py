"""Text normalisation for search: diacritics, case, apostrophes and quotes, dashes, ligatures,
mixed scripts. The same function is applied to indexed text and to queries, so "Chio",
"Chiò", "CHIÒ", "Alzheimer's" and "Alzheimer’s" all meet in the middle."""

from __future__ import annotations

import re
import unicodedata

import jellyfish

_APOSTROPHES = "’‘ʼ′`´"
_QUOTES = "“”„«»″"
_DASHES = "‐‑‒–—―−"
_TRANSLATE = str.maketrans(
    {
        **dict.fromkeys(_APOSTROPHES, "'"),
        **dict.fromkeys(_QUOTES, '"'),
        **dict.fromkeys(_DASHES, "-"),
        " ": " ",
        " ": " ",
        " ": " ",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ß": "ss",
        "æ": "ae",
        "œ": "oe",
        "ø": "o",
        "ł": "l",
        "đ": "d",
        "ı": "i",
    }
)
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[\w][\w'\-]*[\w]|[\w]", re.UNICODE)


def strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize(text: str, *, fold_diacritics: bool = True, casefold: bool = True) -> str:
    """Canonical form used for indexing and querying."""
    s = unicodedata.normalize("NFKC", text or "").translate(_TRANSLATE)
    if fold_diacritics:
        s = strip_diacritics(s)
    if casefold:
        s = s.casefold()
    # soft hyphen / zero-width joiners left by PDF extraction
    s = s.replace("­", "").replace("​", "").replace("‍", "")
    # rejoin words hyphenated across a line break: "neuro-\ndegenerative" -> "neurodegenerative"
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    return _WS.sub(" ", s).strip()


def tokens(text: str) -> list[str]:
    return [t.strip("'-") for t in _TOKEN.findall(normalize(text)) if t.strip("'-")]


def phonetic_key(word: str) -> str:
    """Metaphone key on the ASCII-folded word; empty for non-Latin or numeric tokens."""
    w = strip_diacritics(word).lower()
    w = re.sub(r"[^a-z]", "", w)
    if len(w) < 3:
        return ""
    try:
        return jellyfish.metaphone(w)
    except Exception:  # noqa: BLE001
        return ""


def similarity(a: str, b: str) -> float:
    """Jaro-Winkler on normalised strings; 1.0 = identical."""
    return jellyfish.jaro_winkler_similarity(normalize(a), normalize(b))
