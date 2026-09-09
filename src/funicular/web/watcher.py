"""Watch the iCloud Drive inbox folder and import whatever lands there.

iCloud specifics handled here:
* Files that are not downloaded locally appear as ``.<name>.icloud`` placeholders. We ask
  ``brctl download`` (macOS) to fetch them and wait for the real file to appear.
* A file being synced grows over several seconds; we only import once its size is stable.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path

from ..config import Settings
from .importer import ImportError_, import_path
from .jobs import JobManager
from .store import Store

log = logging.getLogger(__name__)

IGNORED_PREFIXES = (".", "~$")
IGNORED_NAMES = {"Icon\r", ".DS_Store", "desktop.ini"}


class InboxWatcher:
    def __init__(self, settings: Settings, store: Store, jobs: JobManager) -> None:
        self.settings = settings
        self.store = store
        self.jobs = jobs
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen: dict[Path, tuple[int, float]] = {}

    def start(self) -> None:
        self.settings.inbox_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, name="inbox-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        try:
            from watchfiles import watch
        except ImportError:
            log.error("watchfiles not installed; inbox watching disabled")
            return
        self.scan_once()
        for _changes in watch(
            self.settings.inbox_dir, stop_event=self._stop, debounce=1600, step=200
        ):
            if self._stop.is_set():
                break
            self.scan_once()

    def scan_once(self) -> int:
        """Import every stable, non-placeholder file in the inbox. Returns the count."""
        count = 0
        inbox = self.settings.inbox_dir
        if not inbox.is_dir():
            return 0
        for path in sorted(inbox.iterdir()):
            if not path.is_file():
                continue
            name = path.name
            if name in IGNORED_NAMES:
                continue
            if name.endswith(".icloud") and name.startswith("."):
                self._request_download(path)
                continue
            if name.startswith(IGNORED_PREFIXES):
                continue
            if not self._stable(path):
                continue
            try:
                imported = import_path(self.settings, self.store, self.jobs, path, source="inbox")
                count += 1
                log.info("imported %s as %s", name, imported.doc.id)
                self.store.audit("inbox.import", f"{name} -> {imported.doc.id}")
            except ImportError_ as exc:
                log.warning("inbox: %s: %s", name, exc)
                self._quarantine(path, str(exc))
            except Exception as exc:
                log.exception("inbox: %s failed: %s", name, exc)
        return count

    def _stable(self, path: Path, settle: float = 2.0) -> bool:
        try:
            size = path.stat().st_size
        except OSError:
            return False
        now = time.monotonic()
        prev = self._seen.get(path)
        self._seen[path] = (size, now if (prev is None or prev[0] != size) else prev[1])
        first_seen_at_size = self._seen[path][1]
        return size > 0 and now - first_seen_at_size >= settle

    def _request_download(self, placeholder: Path) -> None:
        brctl = shutil.which("brctl")
        if not brctl:
            return
        real_name = placeholder.name[1 : -len(".icloud")]
        target = placeholder.parent / real_name
        try:
            subprocess.run(
                [brctl, "download", str(target)], capture_output=True, timeout=30, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.debug("brctl download failed: %s", exc)

    def _quarantine(self, path: Path, reason: str) -> None:
        bad = self.settings.inbox_dir / "Unsupported"
        try:
            bad.mkdir(exist_ok=True)
            shutil.move(str(path), bad / path.name)
            (bad / (path.name + ".why.txt")).write_text(reason + "\n", encoding="utf-8")
        except OSError as exc:
            log.warning("could not quarantine %s: %s", path, exc)
