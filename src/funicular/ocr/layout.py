"""Rebuild a physical-layout text page (the pdftotext -layout look) from positioned lines.

Works for OCR output and for PDF word boxes alike. The algorithm places each line's text on a
character grid whose cell width is the median glyph width observed on the page, so columns,
indentation and table cells land where they sit on the page.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass
class Positioned:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1e-6)

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


def layout_text(items: list[Positioned], *, max_width: int = 400) -> str:
    """Return text with lines and horizontal offsets approximating the source layout."""
    items = [i for i in items if i.text and i.text.strip()]
    if not items:
        return ""
    heights = [i.height for i in items]
    line_h = median(heights)
    char_ws = [(i.x1 - i.x0) / max(len(i.text), 1) for i in items if len(i.text) >= 3]
    char_w = median(char_ws) if char_ws else line_h * 0.5
    char_w = max(char_w, 1e-3)

    # Group into rows by vertical centre proximity.
    items.sort(key=lambda i: (i.yc, i.x0))
    rows: list[list[Positioned]] = []
    for it in items:
        if rows and abs(it.yc - rows[-1][0].yc) <= 0.5 * line_h:
            rows[-1].append(it)
        else:
            rows.append([it])

    out: list[str] = []
    prev_bottom: float | None = None
    for row in rows:
        row.sort(key=lambda i: i.x0)
        top = min(i.y0 for i in row)
        if prev_bottom is not None:
            gap = top - prev_bottom
            blank = int(gap / line_h + 0.4)
            out.extend([""] * max(0, min(blank, 3)))
        line = ""
        for it in row:
            col = int(round(it.x0 / char_w))
            col = min(col, max_width)
            if len(line) < col:
                line += " " * (col - len(line))
            elif line and not line.endswith(" "):
                line += " "
            line += it.text.strip()
        out.append(line.rstrip())
        prev_bottom = max(i.y1 for i in row)
    return "\n".join(out).rstrip() + "\n"
