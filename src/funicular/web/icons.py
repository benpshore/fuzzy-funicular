# ruff: noqa: E501  (inline SVG path data)
"""Emoji and SVG icons per document kind, used by the templates and the CLI."""

from __future__ import annotations

from markupsafe import Markup

EMOJI = {
    "pdf": "📄",
    "image": "🖼️",
    "audio": "🎙️",
    "video": "🎬",
    "office": "📝",
    "docx": "📝",
    "pptx": "📊",
    "xlsx": "📈",
    "html": "🌐",
    "url": "🌐",
    "text": "📃",
    "md": "📃",
    "epub": "📚",
    "archive": "🗜️",
    "folder": "📁",
    "unknown": "❔",
}

# Small inline SVG glyphs (currentColor), 20px. Kept tiny so every row stays light.
_SVG = {
    "pdf": '<path d="M6 2h8l5 5v13a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1zm7 1v5h5"/><path d="M8 13h2a1.5 1.5 0 0 1 0 3H8zm0 3v2m5-5v5m0-5h1.5a2.5 2.5 0 0 1 0 5H13" fill="none" stroke="currentColor" stroke-width="1.2"/>',
    "image": '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="9.5" r="1.8" fill="var(--bg)"/><path d="M4 18l5-5 3 3 3-4 5 6z" fill="var(--bg)"/>',
    "audio": '<path d="M12 3a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3z"/><path d="M6 11a6 6 0 0 0 12 0M12 17v4M9 21h6" fill="none" stroke="currentColor" stroke-width="1.6"/>',
    "video": '<rect x="3" y="5" width="13" height="14" rx="2"/><path d="M16 10l5-3v10l-5-3z"/>',
    "office": '<path d="M6 2h8l5 5v13a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1z"/><path d="M8 12h8M8 15h8M8 18h5" stroke="var(--bg)" stroke-width="1.4"/>',
    "html": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18" fill="none" stroke="var(--bg)" stroke-width="1.2"/>',
    "text": '<path d="M6 2h8l5 5v13a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1z"/><path d="M8 11h8M8 14h8M8 17h6" stroke="var(--bg)" stroke-width="1.4"/>',
    "epub": '<path d="M4 4h7a3 3 0 0 1 3 3v13a2 2 0 0 0-2-2H4zM20 4h-7a3 3 0 0 0-3 3v13a2 2 0 0 1 2-2h8z"/>',
    "archive": '<rect x="4" y="3" width="16" height="18" rx="2"/><path d="M11 3h2v3h-2zm0 4h2v3h-2zm0 4h2v3h-2z" fill="var(--bg)"/>',
    "folder": '<path d="M3 6a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "unknown": '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.4-1 .8-1 1.7M12 17h.01" fill="none" stroke="var(--bg)" stroke-width="1.6" stroke-linecap="round"/>',
}
_ALIAS = {"url": "html", "md": "text", "docx": "office", "pptx": "office", "xlsx": "office"}


def emoji(kind: str) -> str:
    return EMOJI.get(kind, EMOJI["unknown"])


def svg(kind: str, size: int = 20) -> Markup:
    key = _ALIAS.get(kind, kind)
    body = _SVG.get(key, _SVG["unknown"])
    label = EMOJI.get(kind, "")
    return Markup(  # noqa: S704 - constant markup, kind is mapped through a fixed table
        f'<svg class="ic ic-{key}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="currentColor" role="img" aria-label="{key}"><title>{label} {key}</title>{body}</svg>'
    )
