"""Locate external binaries. Homebrew paths are searched explicitly so a launchd job
(which gets a minimal PATH) still finds poppler and ghostscript."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# Apple Silicon Homebrew, Intel Homebrew, MacPorts, then the usual Linux locations.
_EXTRA_DIRS = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/opt/local/bin",
    "/usr/bin",
    "/bin",
)


def find_binary(name: str, env_override: str | None = None) -> str | None:
    """Return an absolute path to *name*, honouring an optional env override.

    The override (e.g. FUNICULAR_PDFTOTEXT=/opt/homebrew/bin/pdftotext) wins if it points at
    an executable file; otherwise PATH is searched, then the well-known prefixes.
    """
    if env_override:
        override = os.environ.get(env_override)
        if override:
            p = Path(override).expanduser()
            if p.is_file() and os.access(p, os.X_OK):
                return str(p)
    found = shutil.which(name)
    if found:
        return found
    for d in _EXTRA_DIRS:
        candidate = Path(d) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


class MissingBinaryError(RuntimeError):
    """Raised when a required external tool is not installed."""

    def __init__(self, name: str, hint: str) -> None:
        super().__init__(f"{name} not found. {hint}")
        self.name = name
        self.hint = hint
