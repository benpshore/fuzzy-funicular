"""iCloud Drive: dataless (cloud-only) files and on-demand materialisation.

macOS marks an evicted iCloud file with the ``SF_DATALESS`` flag in ``st_flags``; reading it
triggers a download but can block for a long time, so we ask ``brctl`` to fetch it and wait
with a timeout instead. ``evict`` frees the local copy again once we hold our own copy of the
original and the outputs are in the archive folder.

Everything is a no-op that reports "local" on Linux, so the pipeline code never branches on
platform.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SF_DATALESS = 0x40000000  # <sys/stat.h> on macOS
UF_HIDDEN = 0x00008000


def is_dataless(path: Path) -> bool:
    if sys.platform != "darwin":
        return False
    try:
        st = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return bool(getattr(st, "st_flags", 0) & SF_DATALESS)


def placeholder_target(path: Path) -> Path | None:
    """`.name.icloud` placeholder -> the real file path, else None."""
    if path.name.startswith(".") and path.name.endswith(".icloud"):
        return path.with_name(path.name[1 : -len(".icloud")])
    return None


def _brctl() -> str | None:
    return shutil.which("brctl") if sys.platform == "darwin" else None


def request_download(path: Path) -> bool:
    """Ask iCloud to materialise a file. Returns False when brctl is unavailable."""
    b = _brctl()
    if not b:
        return False
    try:
        subprocess.run([b, "download", str(path)], capture_output=True, timeout=30, check=False)  # noqa: S603
        return True
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.debug("brctl download failed for %s: %s", path, exc)
        return False


def evict(path: Path) -> bool:
    """Remove the local copy, keeping the cloud copy. Never touches non-iCloud paths."""
    b = _brctl()
    if not b or not in_icloud(path):
        return False
    try:
        r = subprocess.run([b, "evict", str(path)], capture_output=True, timeout=30, check=False)  # noqa: S603
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.debug("brctl evict failed for %s: %s", path, exc)
        return False


def in_icloud(path: Path) -> bool:
    try:
        rp = path.expanduser().resolve()
    except OSError:
        return False
    marker = "Library/Mobile Documents/"
    return marker in rp.as_posix()


@dataclass
class Materialised:
    path: Path
    was_dataless: bool
    waited: float


def ensure_local(path: Path, *, timeout: float = 600, poll: float = 1.0) -> Materialised:
    """Return once the file's bytes are on disk. Handles `.x.icloud` placeholders too."""
    target = placeholder_target(path) or path
    if not is_dataless(target) and target.exists():
        return Materialised(target, False, 0.0)
    t0 = time.monotonic()
    request_download(target)
    while time.monotonic() - t0 < timeout:
        if target.exists() and not is_dataless(target):
            return Materialised(target, True, time.monotonic() - t0)
        time.sleep(poll)
    raise TimeoutError(f"iCloud did not deliver {target.name} within {timeout:.0f}s")


def status(path: Path) -> str:
    if placeholder_target(path):
        return "cloud-only"
    if is_dataless(path):
        return "cloud-only"
    return "local" if path.exists() else "missing"
