"""Background ingestion with per-document progress broadcast to SSE clients.

A small thread pool runs the CPU-heavy extraction; the event loop only shuffles progress
events. Progress from worker threads crosses to asyncio via ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..config import ExtractSettings, Settings
from ..pipeline import ingest
from .store import Store

log = logging.getLogger(__name__)

STAGE_BASE = {"detect": 0, "ocr": 8, "layout": 45, "markdown": 55, "text": 92, "docling": 0}
STAGE_SPAN = {"detect": 8, "ocr": 37, "layout": 10, "markdown": 37, "text": 6, "docling": 98}
STAGE_LABEL = {
    "queued": "Queued",
    "detect": "Checking pages",
    "ocr": "Recognising text",
    "layout": "Layout text",
    "markdown": "Reading order",
    "text": "Plain text",
    "docling": "Converting",
    "publish": "Saving to iCloud",
    "done": "Done",
    "failed": "Failed",
}


def stage_percent(stage: str, done: int, total: int) -> float:
    base = STAGE_BASE.get(stage, 0)
    span = STAGE_SPAN.get(stage, 100)
    frac = (done / total) if total else 1.0
    return round(min(99.0, base + span * frac), 1)


class JobManager:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="ingest")
        self.loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = threading.Lock()
        self._active: dict[str, float] = {}

    # ------------------------------------------------------------------ lifecycle
    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        # Resume anything that was queued/running when the process last stopped.
        for doc in self.store.list(status="queued") + self.store.list(status="running"):
            self.store.requeue(doc.id)
            self.submit(doc.id)

    def stop(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ pub/sub
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def _publish(self, event: dict[str, Any]) -> None:
        if self.loop is None:
            return

        def _push() -> None:
            for q in list(self._subscribers):
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    # Slow consumer: drop the oldest so progress stays live.
                    try:
                        q.get_nowait()
                        q.put_nowait(event)
                    except asyncio.QueueEmpty, asyncio.QueueFull:
                        log.debug("dropped progress event for a slow SSE client")

        try:
            self.loop.call_soon_threadsafe(_push)
        except RuntimeError:
            pass

    # ------------------------------------------------------------------ jobs
    def submit(self, doc_id: str, extract: ExtractSettings | None = None) -> None:
        with self._lock:
            if doc_id in self._active:
                return
            self._active[doc_id] = time.time()
        self._publish(
            {
                "type": "progress",
                "id": doc_id,
                "status": "queued",
                "stage": "queued",
                "label": STAGE_LABEL["queued"],
                "progress": 0,
            }
        )
        self.pool.submit(self._run, doc_id, extract or self.settings.extract)

    def active_ids(self) -> list[str]:
        with self._lock:
            return list(self._active)

    def _run(self, doc_id: str, extract: ExtractSettings) -> None:
        doc = self.store.get(doc_id)
        if doc is None:
            with self._lock:
                self._active.pop(doc_id, None)
            return
        last = {"t": 0.0}

        def progress(stage: str, done: int, total: int) -> None:
            pct = stage_percent(stage, done, total)
            now = time.monotonic()
            if now - last["t"] < 0.15 and done != total:
                return
            last["t"] = now
            self.store.update_progress(doc_id, stage, pct)
            self._publish(
                {
                    "type": "progress",
                    "id": doc_id,
                    "status": "running",
                    "stage": stage,
                    "label": STAGE_LABEL.get(stage, stage),
                    "progress": pct,
                }
            )

        try:
            original = _original_path(doc.dir)
            if original is None:
                raise FileNotFoundError("original file is missing")
            out_dir = doc.dir / "out"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            progress("detect", 0, 1)
            res = ingest(original, out_dir, extract, progress=progress, stem="document")
            body = ""
            for key in ("text", "markdown", "layout"):
                p = res.outputs.get(key)
                if p and p.exists():
                    body = p.read_text(encoding="utf-8", errors="replace")
                    break
            outputs = {k: str(v.relative_to(doc.dir)) for k, v in res.outputs.items()}
            outputs["original"] = original.name
            title = res.title if doc.title == doc.original_name and res.title else None
            self.store.finish(
                doc_id,
                title=title,
                pages=int(res.stats.get("pages") or 0),
                needs_ocr=res.needs_ocr,
                ocr_used=bool(res.stats.get("ocr_pages")) or "ocr_backend" in res.stats,
                preview=res.text_preview,
                warnings=res.warnings,
                outputs=outputs,
                body_text=body,
            )
            progress("publish", 0, 1)
            try:
                publish_to_archive(self.settings, self.store.get(doc_id))
            except Exception as exc:  # archive is best effort (iCloud may be offline)
                log.warning("archive publish failed for %s: %s", doc_id, exc)
            self._publish(
                {
                    "type": "done",
                    "id": doc_id,
                    "status": "done",
                    "stage": "done",
                    "label": STAGE_LABEL["done"],
                    "progress": 100,
                    "needs_ocr": res.needs_ocr,
                    "warnings": res.warnings,
                }
            )
        except Exception as exc:
            log.exception("ingest failed for %s", doc_id)
            self.store.fail(doc_id, f"{type(exc).__name__}: {exc}")
            self._publish(
                {
                    "type": "failed",
                    "id": doc_id,
                    "status": "failed",
                    "stage": "failed",
                    "label": STAGE_LABEL["failed"],
                    "progress": 0,
                    "error": str(exc)[:300],
                }
            )
        finally:
            with self._lock:
                self._active.pop(doc_id, None)


def _original_path(doc_dir: Path) -> Path | None:
    for p in doc_dir.glob("original.*"):
        return p
    return None


def publish_to_archive(settings: Settings, doc) -> Path | None:
    """Copy original + outputs into the (iCloud) archive folder as a readable bundle.

    Layout: <archive>/<YYYY>/<title (short id)>/  — Files.app friendly, no hidden state."""
    if doc is None or doc.status != "done":
        return None
    root = settings.archive_dir
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    year = time.strftime("%Y", time.localtime(doc.created_at))
    from ..pipeline import safe_stem

    name = f"{safe_stem(doc.title)[:80]} ({doc.id[:6]})"
    dest = root / year / name
    dest.mkdir(parents=True, exist_ok=True)
    for key, rel in doc.outputs.items():
        src = doc.dir / rel
        if not src.is_file():
            continue
        if key == "original":
            target = dest / f"{safe_stem(doc.original_name)}{src.suffix}"
        else:
            target = dest / (safe_stem(doc.title)[:80] + _suffix_for(key, src))
        shutil.copy2(src, target)
    (dest / "funicular.json").write_text(
        json.dumps(doc.to_public(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return dest


def _suffix_for(key: str, src: Path) -> str:
    return {
        "layout": ".layout.txt",
        "markdown": ".md",
        "text": ".txt",
        "report": ".report.json",
        "ocr_pdf": ".ocr.pdf",
    }.get(key, src.suffix)
