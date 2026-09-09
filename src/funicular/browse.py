"""Server-side folder browsing for the web app, restricted to allowed roots.

Roots default to the user's home, iCloud Drive, and (macOS) mounted volumes; extend with
FUNICULAR_BROWSE_ROOTS (colon separated). A path outside every root is refused before any
filesystem access. Symlinks are resolved and re-checked so a link cannot escape a root.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .archives import BUNDLE_SUFFIXES, SKIP_DIRS, SKIP_FILES, is_archive
from .config import Settings
from .sniff import sniff

ICLOUD = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs"


class BrowseError(PermissionError):
    pass


def allowed_roots(settings: Settings) -> list[Path]:
    roots: list[Path] = [Path.home()]
    if ICLOUD.is_dir():
        roots.append(ICLOUD)
    if sys.platform == "darwin" and Path("/Volumes").is_dir():
        roots.append(Path("/Volumes"))
    for extra in (settings.inbox_dir, settings.archive_dir):
        roots.append(extra)
    env = os.environ.get("FUNICULAR_BROWSE_ROOTS", "")
    for raw in env.split(":"):
        if raw.strip():
            roots.append(Path(raw).expanduser())
    out: list[Path] = []
    for r in roots:
        try:
            rr = r.expanduser().resolve()
        except OSError:
            continue
        if rr not in out:
            out.append(rr)
    return out


def resolve_safe(raw: str | Path, roots: list[Path]) -> Path:
    p = Path(raw).expanduser()
    try:
        real = p.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BrowseError(f"no such path: {raw}") from exc
    for root in roots:
        if real == root or root in real.parents:
            return real
    raise BrowseError("path is outside the folders this app may browse")


@dataclass
class Entry:
    name: str
    path: str
    is_dir: bool
    size: int = 0
    mtime: float = 0.0
    kind: str = ""
    compatible: bool = False
    archive: bool = False
    dataless: bool = False


@dataclass
class Listing:
    path: str
    parent: str | None
    roots: list[str]
    entries: list[Entry] = field(default_factory=list)
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "parent": self.parent,
            "roots": self.roots,
            "truncated": self.truncated,
            "entries": [asdict(e) for e in self.entries],
        }


def listing(settings: Settings, raw: str | None, *, limit: int = 500) -> Listing:
    roots = allowed_roots(settings)
    if not raw:
        # virtual root: the allowed roots themselves
        entries = [Entry(name=str(r), path=str(r), is_dir=True) for r in roots if r.is_dir()]
        return Listing(path="", parent=None, roots=[str(r) for r in roots], entries=entries)
    path = resolve_safe(raw, roots)
    if not path.is_dir():
        raise BrowseError("not a folder")
    from .icloud import is_dataless

    entries: list[Entry] = []
    try:
        names = sorted(os.listdir(path), key=str.casefold)
    except OSError as exc:
        raise BrowseError(f"cannot read folder: {exc}") from exc
    truncated = False
    for name in names:
        if name.startswith(".") or name in SKIP_FILES or name in SKIP_DIRS:
            continue
        p = path / name
        try:
            st = p.stat()
        except OSError:
            continue
        if p.is_dir():
            if name.endswith(BUNDLE_SUFFIXES):
                continue
            entries.append(Entry(name=name, path=str(p), is_dir=True, mtime=st.st_mtime))
        else:
            sn = sniff(p)
            arch = is_archive(p) and sn.kind.value not in ("office", "epub")
            entries.append(
                Entry(
                    name=name,
                    path=str(p),
                    is_dir=False,
                    size=st.st_size,
                    mtime=st.st_mtime,
                    kind="archive" if arch else sn.kind.value,
                    compatible=sn.kind.value != "unknown" or arch,
                    archive=arch,
                    dataless=is_dataless(p),
                )
            )
        if len(entries) >= limit:
            truncated = True
            break
    parent = None
    for root in roots:
        if path != root and root in path.parents:
            parent = str(path.parent)
            break
    return Listing(
        path=str(path),
        parent=parent,
        roots=[str(r) for r in roots],
        entries=entries,
        truncated=truncated,
    )


def now() -> float:
    return time.time()
