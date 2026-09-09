"""Automatic tags. Tagging is not a user job: every processed document gets a handful of
tags from its own text and metadata.

Sources, in order:
1. structural: kind, year (from metadata or the text), language guess, "scanned", "ocr"
2. keyphrases: a RAKE-style scorer over the first part of the text with a scientific twist —
   acronyms, capitalised multi-word terms and hyphenated compounds score higher, generic
   academic words are stop-listed
3. caller-supplied hints (folder names, feed names, Zotero collections)
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

STOP = set(
    """
a an the and or but if then else of in on at to for from by with without about as into onto
over under between among through during before after above below up down out off again
further once here there when where why how all any both each few more most other some such
no nor not only own same so than too very can will just should now is are was were be been
being have has had having do does did doing would could may might must shall this that these
those it its they them their we our you your he she his her i me my who whom which what
et al fig figure table tables figures section sections study studies results result method
methods analysis analyses data using used use based approach approaches paper article
however therefore thus although whereas also within across via per vs versus respectively
significant significantly showed shown show found present presents presented well one two
three first second third new novel important different various several many much high low
higher lower large small total number rate rates level levels time times year years group
groups patients patient case cases model models effect effects value values mean median
associated association compared comparison between overall included including
""".split()
)
STOP |= {w + "s" for w in list(STOP) if len(w) > 3}

WORD = re.compile(r"[^\W\d_](?:[\w'’\-]*[^\W_])?", re.UNICODE)
TOKEN = re.compile(r"[^\W\d_](?:[\w'’\-]*[^\W_])?|\d[\w.,%\-]*|[^\w\s]", re.UNICODE)
YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
ACRONYM = re.compile(r"^[A-Z][A-Z0-9\-]{1,9}$")


@dataclass
class TagResult:
    tags: list[str] = field(default_factory=list)
    keyphrases: list[tuple[str, float]] = field(default_factory=list)
    year: int | None = None
    language: str | None = None

    def to_dict(self) -> dict:
        return {
            "tags": self.tags,
            "keyphrases": [[k, round(s, 2)] for k, s in self.keyphrases],
            "year": self.year,
            "language": self.language,
        }


def _fold(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    ).lower()


def guess_language(text: str) -> str | None:
    """Cheap function-word vote for the handful of languages we see; None when unsure."""
    sample = _fold(text[:6000])
    words = Counter(re.findall(r"[a-z]+", sample))
    if not words:
        return None
    markers = {
        "en": ["the", "and", "of", "with", "that", "were", "which"],
        "de": ["und", "der", "die", "das", "nicht", "mit", "ist"],
        "fr": ["les", "des", "une", "est", "dans", "pour", "avec"],
        "es": ["los", "las", "una", "por", "para", "con", "como"],
        "it": ["della", "che", "per", "con", "una", "sono", "nel"],
        "pt": ["uma", "com", "para", "não", "que", "dos", "das"],
        "nl": ["het", "een", "van", "niet", "met", "zijn", "ook"],
        "la": ["est", "non", "sed", "cum", "quod", "sunt", "atque"],
    }
    scores = {lang: sum(words.get(w, 0) for w in ws) for lang, ws in markers.items()}
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    total = sum(words.values())
    if scores[best] < max(3, total * 0.01):
        return None
    return best


def keyphrases(text: str, *, top: int = 12, max_words: int = 3) -> list[tuple[str, float]]:
    """RAKE-style candidate phrases scored by degree/frequency, boosted for science terms."""
    text = text[:60_000]
    phrases: list[list[str]] = []
    for line in text.splitlines():
        cur: list[str] = []
        for tok in TOKEN.findall(line):
            is_word = bool(WORD.fullmatch(tok))
            low = _fold(tok.strip("'’-")) if is_word else ""
            boundary = (
                not is_word
                or low in STOP
                or len(low) < 2
                or (len(low) < 3 and not ACRONYM.match(tok))
            )
            if boundary:
                if cur:
                    phrases.append(cur)
                    cur = []
                continue
            cur.append(tok)
            if len(cur) >= max_words:
                phrases.append(cur)
                cur = []
        if cur:
            phrases.append(cur)

    freq: Counter[str] = Counter()
    degree: defaultdict[str, int] = defaultdict(int)
    for ph in phrases:
        keys = [_fold(w) for w in ph]
        for k in keys:
            freq[k] += 1
            degree[k] += len(keys) - 1
    word_score = {w: (degree[w] + freq[w]) / freq[w] for w in freq}

    phrase_scores: dict[str, float] = {}
    surface: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for ph in phrases:
        key = " ".join(_fold(w) for w in ph)
        counts[key] += 1
        score = sum(word_score[_fold(w)] for w in ph) / (len(ph) ** 0.5)
        boost = 1.0
        if any(ACRONYM.match(w) for w in ph):
            boost += 0.6
        if len(ph) > 1 and all(w[:1].isupper() for w in ph):
            boost += 0.5
        if any("-" in w for w in ph):
            boost += 0.3
        phrase_scores[key] = score * boost
        surface.setdefault(key, " ".join(ph))
    ranked = []
    for key, sc in phrase_scores.items():
        c = counts[key]
        if c < 2 and len(key.split()) == 1 and not ACRONYM.match(surface[key]):
            continue  # single words must recur
        ranked.append((surface[key], sc * (1 + 0.35 * min(c, 10))))
    ranked.sort(key=lambda kv: -kv[1])
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for s, sc in ranked:
        k = _fold(s)
        if any(k in o or o in k for o in seen):
            continue
        seen.add(k)
        out.append((s, sc))
        if len(out) >= top:
            break
    return out


def top_terms(text: str, *, n: int = 6, min_count: int = 3) -> list[tuple[str, int]]:
    """Frequent single domain terms (drug names, gene symbols, acronyms) the phrase scorer
    under-weights because they travel with different neighbours each time."""
    counts: Counter[str] = Counter()
    surface: dict[str, str] = {}
    for tok in WORD.findall(text[:60_000]):
        low = _fold(tok.strip("'’-"))
        if low in STOP or len(low) < 3:
            continue
        if not (ACRONYM.match(tok) or len(low) >= 5):
            continue
        counts[low] += 1
        # prefer the capitalised / acronym surface form when one exists
        if low not in surface or (tok[:1].isupper() and not surface[low][:1].isupper()):
            surface[low] = tok
    out = [(surface[k], c) for k, c in counts.most_common(50) if c >= min_count]
    return out[:n]


def auto_tags(
    text: str,
    *,
    kind: str = "",
    title: str | None = None,
    metadata: dict | None = None,
    hints: list[str] | None = None,
    scanned: bool = False,
    ocr_used: bool = False,
    limit: int = 10,
) -> TagResult:
    res = TagResult()
    tags: list[str] = []
    if kind:
        tags.append(kind)
    if scanned:
        tags.append("scanned")
    if ocr_used:
        tags.append("ocr")
    meta = metadata or {}
    year = None
    for key in ("year", "issued", "created", "creationDate", "date"):
        v = meta.get(key)
        if v:
            m = YEAR.search(str(v))
            if m:
                year = int(m.group(1))
                break
    if year is None:
        head = (title or "") + "\n" + text[:4000]
        years = [int(y) for y in YEAR.findall(head)]
        if years:
            year = Counter(years).most_common(1)[0][0]
    if year:
        tags.append(str(year))
        res.year = year
    lang = guess_language(text)
    if lang and lang != "en":
        tags.append(f"lang:{lang}")
    res.language = lang
    for h in hints or []:
        h = h.strip()
        if h and h not in tags:
            tags.append(h[:40])
    body = (title or "") + "\n" + text
    kp = keyphrases(body)
    res.keyphrases = kp
    terms = top_terms(body)
    # interleave frequent single terms with the best phrases so both kinds survive the cap
    candidates: list[str] = []
    for i in range(max(len(kp), len(terms))):
        if i < len(terms):
            candidates.append(terms[i][0])
        if i < len(kp):
            candidates.append(kp[i][0])
    for phrase in candidates:
        p = phrase.strip()
        if len(p) > 40:
            continue
        words = p.split()
        # acronyms keep their case; everything else becomes a lowercase tag
        p = " ".join(w if ACRONYM.match(w) else w.lower() for w in words)
        folded = _fold(p)
        if any(folded == _fold(t) or (len(folded) > 3 and folded in _fold(t)) for t in tags):
            continue
        tags.append(p)
        if len(tags) >= limit:
            break
    res.tags = tags[:limit]
    return res
