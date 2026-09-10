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
from ..estimate import Timings
from ..pipeline import ingest
from ..resources import Cancelled, JobContext, MemoryGuard, SystemMonitor, current_job
from ..search import SearchIndex
from .store import Store

log = logging.getLogger(__name__)

STAGE_BASE = {
    "detect": 0,
    "ocr": 8,
    "layout": 42,
    "markdown": 52,
    "text": 86,
    "docling": 0,
    "index": 90,
    "publish": 97,
}
STAGE_SPAN = {
    "detect": 8,
    "ocr": 34,
    "layout": 10,
    "markdown": 34,
    "text": 4,
    "docling": 88,
    "index": 7,
    "publish": 2,
}
STAGE_LABEL = {
    "queued": "Queued",
    "detect": "Checking pages",
    "ocr": "Recognising text",
    "layout": "Layout text",
    "markdown": "Reading order",
    "text": "Plain text",
    "docling": "Converting",
    "index": "Indexing for search",
    "scholar": "Checking metadata",
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
        self._active: dict[str, JobContext] = {}
        self.monitor = SystemMonitor(settings.data_dir)
        self.guard = MemoryGuard(
            self.contexts,
            job_cap_mb=settings.job_memory_cap_mb,
            job_max_procs=settings.job_max_procs,
            min_free_mb=settings.min_free_mb,
        )
        self.stopping = False
        self._pending_summaries: set[str] = set()
        self._rerun: dict[str, ExtractSettings | None] = {}
        self.timings = Timings(settings.data_dir / "timings.json")
        self.index = SearchIndex(settings.data_dir / "search.sqlite3", embed=settings.embeddings)
        self.fetcher_factory = self._default_fetcher
        self.provider_factory = self._default_provider
        from ..feeds import FeedStore

        self.feeds = FeedStore(settings.data_dir / "feeds.sqlite3")
        self.zotero_factory = self._default_zotero
        self.feed_transport = None  # tests inject an httpx transport for PDF downloads

    # ------------------------------------------------------------------ lifecycle
    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        if not self.guard.is_alive():
            self.guard.start()
        # Resume anything that was queued/running when the process last stopped.
        for doc in self.store.list(status="queued") + self.store.list(status="running"):
            self.store.requeue(doc.id)
            self.submit(doc.id)

    def stop(self, reason: str = "server shutting down") -> None:
        """Clean exit: cancel every job (killing its process tree), stop the guard."""
        self.stopping = True
        for ctx in self.contexts():
            ctx.cancel(reason)
        self.guard.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)

    def _default_provider(self, name: str | None = None):
        from ..llm import get_provider

        return get_provider(name)

    # ------------------------------------------------------------------ summaries / ask
    def submit_summarize(self, doc_id: str, provider_name: str | None = None) -> bool:
        key = f"summary:{doc_id}"
        with self._lock:
            if key in self._active:
                return False
            self._active[key] = JobContext(id=key)
        self._publish(
            {
                "type": "progress",
                "id": doc_id,
                "status": "running",
                "stage": "summarize",
                "label": "Summarising",
                "progress": 5,
            }
        )
        self.pool.submit(self._run_summarize, doc_id, provider_name)
        return True

    def _run_summarize(self, doc_id: str, provider_name: str | None) -> None:
        from ..llm import LLMUnavailable
        from ..summarize import summarize

        key = f"summary:{doc_id}"
        with self._lock:
            ctx = self._active.get(key)
        doc = self.store.get(doc_id)
        if ctx is None or doc is None:
            with self._lock:
                self._active.pop(key, None)
            return
        token = current_job.set(ctx)
        try:
            body = ""
            for k in ("text", "markdown", "layout"):
                if k in doc.outputs and (doc.dir / doc.outputs[k]).is_file():
                    body = (doc.dir / doc.outputs[k]).read_text(encoding="utf-8", errors="replace")
                    break
            if not body.strip():
                raise ValueError("no extracted text to summarise")
            provider = self.provider_factory(provider_name)

            def progress(stage, done, total):
                ctx.check()
                pct = 5 + 90 * (done / total if total else 1)
                self._publish(
                    {
                        "type": "progress",
                        "id": doc_id,
                        "status": "running",
                        "stage": stage,
                        "label": f"Summarising ({done}/{total})",
                        "progress": round(pct, 1),
                    }
                )

            summary = summarize(body, provider, title=doc.title, progress=progress)
            (doc.dir / "summary.json").write_text(
                json.dumps(summary.to_dict(), ensure_ascii=False, indent=1)
            )
            outputs = dict(doc.outputs)
            outputs["summary"] = "summary.json"
            self.store.set_outputs(doc_id, outputs)
            kws = [k for k in (summary.card.get("keywords") or []) if isinstance(k, str)][:6]
            if kws:
                self.store.add_tags(doc_id, [k[:40] for k in kws])
            self.store.audit(
                "summarize",
                f"{doc_id}: {summary.provider}/{summary.model} "
                f"{summary.input_tokens}+{summary.output_tokens} tok",
            )
            self._publish(
                {
                    "type": "done",
                    "id": doc_id,
                    "status": "done",
                    "stage": "done",
                    "label": "Done",
                    "progress": 100,
                    "needs_ocr": doc.needs_ocr,
                    "warnings": doc.warnings,
                    "summary": True,
                }
            )
        except (Cancelled, LLMUnavailable, ValueError) as exc:
            log.info("summary for %s not produced: %s", doc_id, exc)
            self._publish(
                {
                    "type": "done",
                    "id": doc_id,
                    "status": "done",
                    "stage": "done",
                    "label": "Done",
                    "progress": 100,
                    "needs_ocr": doc.needs_ocr,
                    "warnings": doc.warnings + [f"summary: {exc}"],
                }
            )
        except Exception as exc:
            log.exception("summary failed for %s", doc_id)
            self._publish(
                {
                    "type": "done",
                    "id": doc_id,
                    "status": "done",
                    "stage": "done",
                    "label": "Done",
                    "progress": 100,
                    "needs_ocr": doc.needs_ocr,
                    "warnings": doc.warnings + [f"summary failed: {exc}"],
                }
            )
        finally:
            current_job.reset(token)
            with self._lock:
                self._active.pop(key, None)

    def _default_zotero(self):
        from ..zotero import ZoteroLocal

        return ZoteroLocal()

    # ------------------------------------------------------------------ feeds
    def stage_entry(self, entry_id: int, *, summarize: bool | None = None) -> dict:
        """Find an open-access PDF for a feed entry, import it, and queue a summary."""
        from ..feeds import Poller
        from .importer import ImportError_, import_path

        entry = self.feeds.entry(entry_id)
        if not entry:
            raise KeyError(entry_id)
        if entry["status"] in ("imported", "staged") and entry.get("doc_id"):
            return {"status": entry["status"], "doc_id": entry["doc_id"]}
        fetcher = self.fetcher_factory()
        poller = Poller(self.feeds, fetcher, transport=self.feed_transport)
        try:
            url = poller.find_pdf_url(entry)
            if not url:
                self.feeds.set_status(entry_id, "skipped", note="no open-access PDF found")
                return {"status": "skipped", "note": "no open-access PDF found"}
            tmp_dir = self.settings.data_dir / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            dest = tmp_dir / f"feed-{entry_id}.pdf"
            try:
                poller.download_pdf(url, dest)
            except Exception as exc:  # noqa: BLE001
                self.feeds.set_status(entry_id, "failed", note=f"download: {exc}")
                return {"status": "failed", "note": str(exc)[:200]}
            tags = ["feed:" + (self.feeds.get(entry["feed_id"]) or {}).get("name", "feed")[:40]]
            if entry.get("doi"):
                tags.append("doi:" + entry["doi"])
            try:
                imp = import_path(
                    self.settings,
                    self.store,
                    self,
                    dest,
                    source="feed",
                    move=True,
                    tags=tags,
                    skip_duplicates=True,
                    filename=_entry_filename(entry),
                )
            except ImportError_ as exc:
                self.feeds.set_status(entry_id, "failed", note=str(exc))
                return {"status": "failed", "note": str(exc)[:200]}
            self.feeds.set_status(entry_id, "staged", doc_id=imp.doc.id, note=url[:200])
            want_summary = self.settings.feeds_autosummarize if summarize is None else summarize
            if want_summary:
                self._pending_summaries.add(imp.doc.id)
            return {"status": "staged", "doc_id": imp.doc.id, "url": url}
        finally:
            poller.close()
            fetcher.close()

    def poll_feeds(self, feed_id: int | None = None) -> dict[int, int]:
        from ..feeds import Poller

        fetcher = self.fetcher_factory()
        poller = Poller(self.feeds, fetcher, transport=self.feed_transport)
        try:
            if feed_id is not None:
                counts = {feed_id: poller.poll(feed_id)}
            else:
                counts = poller.poll_all()
        finally:
            poller.close()
            fetcher.close()
        for fid, _n in counts.items():
            feed = self.feeds.get(fid)
            if feed and feed.get("auto_stage"):
                for e in self.feeds.entries(fid, status="new", limit=50):
                    self.pool.submit(self._safe_stage, e["id"])
        return counts

    def _safe_stage(self, entry_id: int) -> None:
        try:
            self.stage_entry(entry_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("staging entry %s failed: %s", entry_id, exc)

    # ------------------------------------------------------------------ zotero
    def import_zotero(
        self, collection: str | None, *, extract: ExtractSettings | None = None
    ) -> str:
        """Batch job: copy every PDF attachment in a Zotero collection into the library."""
        bid = self.store.create_batch("zotero", collection or "My Library")
        key = f"batch:{bid}"
        with self._lock:
            self._active[key] = JobContext(id=key)
        self.pool.submit(self._run_zotero, bid, collection, extract or self.settings.extract)
        return bid

    def _run_zotero(self, bid: str, collection: str | None, extract) -> None:
        from .importer import ImportError_, import_path

        key = f"batch:{bid}"
        with self._lock:
            ctx = self._active.get(key)
        if ctx is None:
            return
        token = current_job.set(ctx)
        z = self.zotero_factory()
        imported = skipped = 0
        errors: list[str] = []
        try:
            self.store.update_batch(bid, status="running")
            items = z.items(collection)
            total = sum(1 for it in items if it.attachments)
            self.store.update_batch(bid, total=total)
            done = 0
            tmp_dir = self.settings.data_dir / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            for it in items:
                ctx.check()
                if not it.attachments:
                    continue
                att = it.attachments[0]
                dest = tmp_dir / f"zotero-{att['key']}.pdf"
                got = z.fetch_attachment(att["key"], att.get("filename", ""), dest)
                done += 1
                if not got:
                    skipped += 1
                    errors.append(f"{it.title[:60]}: attachment not available locally")
                    continue
                tags = ["zotero"] + [f"zotero:{c}" for c in it.collections[:3]] + it.tags[:5]
                if it.doi:
                    tags.append("doi:" + it.doi)
                try:
                    imp = import_path(
                        self.settings,
                        self.store,
                        self,
                        dest,
                        source="zotero",
                        move=True,
                        extract=extract,
                        tags=tags,
                        skip_duplicates=True,
                        filename=(att.get("filename") or f"{it.title[:80]}.pdf"),
                    )
                    if imp.duplicate_of is not None and imp.doc.id == imp.duplicate_of.id:
                        skipped += 1
                    else:
                        imported += 1
                        (imp.doc.dir / "zotero.json").write_text(
                            json.dumps(it.to_dict(), ensure_ascii=False)
                        )
                        if imp.doc.title == imp.doc.original_name and it.title:
                            self.store.rename(imp.doc.id, it.title[:200])
                except ImportError_ as exc:
                    errors.append(f"{it.title[:60]}: {exc}")
                self.store.update_batch(bid, done=done, imported=imported, skipped=skipped)
                self._publish(
                    {
                        "type": "batch",
                        "id": bid,
                        "kind": "zotero",
                        "status": "running",
                        "source": collection or "My Library",
                        "done": done,
                        "total": total,
                        "imported": imported,
                        "skipped": skipped,
                    }
                )
            self.store.update_batch(
                bid,
                status="done",
                done=done,
                imported=imported,
                skipped=skipped,
                errors=errors[:50],
            )
            self._publish(
                {
                    "type": "batch",
                    "id": bid,
                    "kind": "zotero",
                    "status": "done",
                    "source": collection or "My Library",
                    "done": done,
                    "total": total,
                    "imported": imported,
                    "skipped": skipped,
                    "errors": len(errors),
                }
            )
        except Cancelled as exc:
            self.store.update_batch(bid, status="failed", errors=[f"cancelled: {exc}"])
        except Exception as exc:
            log.exception("zotero import failed")
            self.store.update_batch(bid, status="failed", errors=[str(exc)[:300]])
            self._publish(
                {
                    "type": "batch",
                    "id": bid,
                    "kind": "zotero",
                    "status": "failed",
                    "source": collection or "My Library",
                    "error": str(exc)[:200],
                }
            )
        finally:
            z.close()
            current_job.reset(token)
            with self._lock:
                self._active.pop(key, None)

    def _default_fetcher(self):
        from ..scholar.clients import Cache, Fetcher

        return Fetcher(cache=Cache(self.settings.data_dir / "scholar-cache.sqlite3"))

    def scholar_lookup(self, doc_id: str, *, online: bool | None = None) -> dict | None:
        """Extract identifiers/metadata from a PDF and (optionally) verify online. Writes
        scholar.json into the document directory and returns it."""
        from ..scholar.metadata import extract_from_pdf, resolve

        doc = self.store.get(doc_id)
        if doc is None or doc.kind != "pdf":
            return None
        original = doc.dir / doc.outputs.get("original", "")
        if not original.is_file():
            return None
        ex = extract_from_pdf(original)
        out: dict[str, Any] = {
            "extracted": ex.to_dict(),
            "resolution": None,
            "checked_at": time.time(),
        }
        online = self.settings.scholar_auto if online is None else online
        if online and (ex.ids.any or ex.pdf_title or ex.guessed_title):
            f = self.fetcher_factory()
            try:
                res = resolve(ex, f)
                out["resolution"] = res.to_dict()
                if res.work and res.verified:
                    extra = [
                        t
                        for t in (
                            f"doi:{res.work.doi}" if res.work.doi else "",
                            str(res.work.year) if res.work.year else "",
                            res.work.journal[:40] if res.work.journal else "",
                        )
                        if t
                    ]
                    if extra:
                        self.store.add_tags(doc_id, extra)
                    if doc.title == doc.original_name and res.work.title:
                        self.store.rename(doc_id, res.work.title[:200])
            except Exception as exc:  # noqa: BLE001 - network is optional here
                out["error"] = str(exc)[:300]
            finally:
                f.close()
        (doc.dir / "scholar.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        outputs = dict(doc.outputs)
        outputs["scholar"] = "scholar.json"
        self.store.set_outputs(doc_id, outputs)
        return out

    def contexts(self) -> list[JobContext]:
        with self._lock:
            return list(self._active.values())

    def cancel(self, doc_id: str, reason: str = "cancelled by user") -> bool:
        with self._lock:
            ctx = self._active.get(doc_id)
        if ctx is None:
            return False
        ctx.cancel(reason)
        return True

    def system(self) -> dict[str, Any]:
        snap = self.monitor.snapshot(self.contexts())
        snap["pressure"] = self.guard.pressure
        snap["last_kill"] = self.guard.last_kill
        snap["workers"] = self.settings.workers
        snap["caps"] = {
            "job_memory_cap_mb": self.settings.job_memory_cap_mb,
            "job_max_procs": self.settings.job_max_procs,
            "min_free_mb": self.settings.min_free_mb,
        }
        return snap

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
        if self.stopping:
            return
        with self._lock:
            if doc_id in self._active:
                # Still running (often just the post-finish steps): run again when it ends.
                self._rerun[doc_id] = extract
                return
            self._active[doc_id] = JobContext(id=doc_id)
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
            return [k for k in self._active if not k.startswith("batch:")]

    # ------------------------------------------------------------------ batches
    def submit_batch(
        self,
        kind: str,
        source: Path,
        *,
        extract: ExtractSettings | None = None,
        recursive: bool = True,
        delete_after: bool = False,
        label: str | None = None,
    ) -> str:
        """Queue an archive unpack or a folder scan. Returns the batch id."""
        label = label or source.name
        bid = self.store.create_batch(kind, label if kind == "archive" else str(source))
        key = f"batch:{bid}"
        with self._lock:
            self._active[key] = JobContext(id=key)
        self._publish(
            {
                "type": "batch",
                "id": bid,
                "kind": kind,
                "status": "queued",
                "source": label,
                "done": 0,
                "total": 0,
                "imported": 0,
                "skipped": 0,
            }
        )
        self.pool.submit(
            self._run_batch,
            bid,
            kind,
            source,
            extract or self.settings.extract,
            recursive,
            delete_after,
            label,
        )
        return bid

    # ------------------------------------------------------------------ compression
    def submit_compress(self, doc_id: str, strength: int, engine: str = "auto") -> bool:
        key = f"compress:{doc_id}"
        with self._lock:
            if key in self._active or doc_id in self._active:
                return False
            self._active[key] = JobContext(id=key)
        self._publish(
            {
                "type": "progress",
                "id": doc_id,
                "status": "running",
                "stage": "compress",
                "label": "Compressing",
                "progress": 5,
            }
        )
        self.pool.submit(self._run_compress, doc_id, strength, engine)
        return True

    def _run_compress(self, doc_id: str, strength: int, engine: str) -> None:
        from ..compress import compress

        key = f"compress:{doc_id}"
        with self._lock:
            ctx = self._active.get(key)
        doc = self.store.get(doc_id)
        if ctx is None or doc is None:
            with self._lock:
                self._active.pop(key, None)
            return
        token = current_job.set(ctx)
        try:
            src = doc.dir / doc.outputs.get("ocr_pdf", doc.outputs.get("original", ""))
            if not src.is_file() or doc.kind != "pdf":
                raise FileNotFoundError("no PDF to compress")
            t0 = time.monotonic()
            res = compress(src, doc.dir / "compressed.pdf", strength, engine=engine)
            self.timings.record(
                f"compress.{res.engine.split()[0]}", time.monotonic() - t0, max(doc.pages, 1)
            )
            outputs = dict(doc.outputs)
            outputs["compressed"] = "compressed.pdf"
            self.store.set_outputs(doc_id, outputs)
            self.store.audit(
                "compress",
                f"{doc_id}: {res.original_bytes} -> {res.output_bytes}"
                f" ({res.engine}, strength {strength})",
            )
            self._publish(
                {
                    "type": "done",
                    "id": doc_id,
                    "status": "done",
                    "stage": "done",
                    "label": "Done",
                    "progress": 100,
                    "needs_ocr": doc.needs_ocr,
                    "warnings": doc.warnings,
                    "compressed": res.to_dict(),
                }
            )
        except Cancelled as exc:
            self._publish(
                {
                    "type": "done",
                    "id": doc_id,
                    "status": "done",
                    "stage": "done",
                    "label": "Done",
                    "progress": 100,
                    "needs_ocr": doc.needs_ocr,
                    "warnings": doc.warnings + [f"compression cancelled: {exc}"],
                }
            )
        except Exception as exc:
            log.exception("compress failed for %s", doc_id)
            self._publish(
                {
                    "type": "done",
                    "id": doc_id,
                    "status": "done",
                    "stage": "done",
                    "label": "Done",
                    "progress": 100,
                    "needs_ocr": doc.needs_ocr,
                    "warnings": doc.warnings + [f"compression failed: {exc}"],
                }
            )
        finally:
            current_job.reset(token)
            with self._lock:
                self._active.pop(key, None)

    def cancel_batch(self, bid: str) -> bool:
        return self.cancel(f"batch:{bid}", "batch cancelled by user")

    def _run_batch(
        self, bid, kind, source: Path, extract, recursive, delete_after, label=None
    ) -> None:
        label = label or source.name
        from .importer import import_archive, import_folder

        key = f"batch:{bid}"
        with self._lock:
            ctx = self._active.get(key)
        if ctx is None:
            return
        token = current_job.set(ctx)
        state = {"done": 0, "total": 0}

        def progress(done: int, total: int) -> None:
            ctx.check()
            state.update(done=done, total=total)
            self.store.update_batch(bid, status="running", done=done, total=total)
            self._publish(
                {
                    "type": "batch",
                    "id": bid,
                    "kind": kind,
                    "status": "running",
                    "source": label,
                    "done": done,
                    "total": total,
                }
            )

        try:
            self.store.update_batch(bid, status="running")
            if kind == "archive":
                res = import_archive(
                    self.settings,
                    self.store,
                    self,
                    source,
                    source="archive",
                    extract=extract,
                    progress=progress,
                    delete_after=delete_after,
                    label=label,
                )
            else:
                res = import_folder(
                    self.settings,
                    self.store,
                    self,
                    source,
                    recursive=recursive,
                    source="folder",
                    extract=extract,
                    progress=progress,
                )
            self.store.update_batch(
                bid,
                status="done",
                total=res.total,
                done=res.total,
                imported=res.imported,
                skipped=res.skipped,
                errors=res.errors,
            )
            self._publish(
                {
                    "type": "batch",
                    "id": bid,
                    "kind": kind,
                    "status": "done",
                    "source": label,
                    "done": res.total,
                    "total": res.total,
                    "imported": res.imported,
                    "skipped": res.skipped,
                    "errors": len(res.errors),
                }
            )
        except Cancelled as exc:
            self.store.update_batch(bid, status="failed", errors=[f"cancelled: {exc}"])
            self._publish(
                {
                    "type": "batch",
                    "id": bid,
                    "kind": kind,
                    "status": "failed",
                    "source": label,
                    "error": str(exc),
                }
            )
        except Exception as exc:
            log.exception("batch %s failed", bid)
            self.store.update_batch(bid, status="failed", errors=[str(exc)[:500]])
            self._publish(
                {
                    "type": "batch",
                    "id": bid,
                    "kind": kind,
                    "status": "failed",
                    "source": label,
                    "error": str(exc)[:300],
                }
            )
        finally:
            current_job.reset(token)
            with self._lock:
                self._active.pop(key, None)

    def _run(self, doc_id: str, extract: ExtractSettings) -> None:
        with self._lock:
            ctx = self._active.get(doc_id)
        doc = self.store.get(doc_id)
        if doc is None or ctx is None:
            with self._lock:
                self._active.pop(doc_id, None)
            return
        token = current_job.set(ctx)
        try:
            self._wait_for_memory(ctx, doc)
            self._run_job(doc_id, doc, ctx, extract)
        finally:
            current_job.reset(token)
            with self._lock:
                self._active.pop(doc_id, None)
                rerun = self._rerun.pop(doc_id, "none")
            if rerun != "none" and not self.stopping:
                self.submit(doc_id, rerun)  # type: ignore[arg-type]

    def _wait_for_memory(self, ctx: JobContext, doc) -> None:
        """Admission control: hold a job while free memory is below what it is likely to need."""
        needed = max(512.0, min(doc.size / 2**20 * 4, self.settings.job_memory_cap_mb / 2))
        waited = 0.0
        while not self.guard.admission_ok(needed) and waited < 600 and not ctx.cancelled:
            if waited == 0:
                self.store.update_progress(doc.id, "queued", 0, status="queued")
                self._publish(
                    {
                        "type": "progress",
                        "id": doc.id,
                        "status": "queued",
                        "stage": "queued",
                        "label": "Waiting for memory",
                        "progress": 0,
                    }
                )
            time.sleep(2)
            waited += 2

    def _run_job(self, doc_id: str, doc, ctx: JobContext, extract: ExtractSettings) -> None:
        last = {"t": 0.0}
        state = {"finished": False}
        clock: dict[str, float] = {}

        def progress(stage: str, done: int, total: int) -> None:
            ctx.check()
            now_m = time.monotonic()
            clock.setdefault(stage, now_m)
            if total and done >= total and stage in clock:
                self.timings.record(_timing_key(stage, extract), now_m - clock.pop(stage), total)
            pct = stage_percent(stage, done, total)
            now = time.monotonic()
            if now - last["t"] < 0.15 and done != total:
                return
            last["t"] = now
            # Once finished, later stages (index, scholar, publish) must not flip the
            # document back to "running": it is already usable.
            status = "done" if state["finished"] else "running"
            self.store.update_progress(doc_id, stage, pct, status=status)
            self._publish(
                {
                    "type": "progress",
                    "id": doc_id,
                    "status": status,
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
            try:
                from ..autotag import auto_tags

                signal = res.stats.get("signal") or {}
                auto = auto_tags(
                    body,
                    kind=doc.kind,
                    title=title or doc.title,
                    metadata={"year": (res.stats.get("info") or {}).get("year")},
                    hints=[t for t in doc.tags if ":" in t],
                    scanned=bool(signal.get("is_scanned_document")),
                    ocr_used=bool(res.stats.get("ocr_pages")) or bool(res.stats.get("ocr_backend")),
                )
                keep = [t for t in doc.tags if t not in auto.tags]
                self.store.set_tags(doc_id, auto.tags + keep)
            except Exception as exc:  # tags are a convenience, never a failure
                log.warning("auto-tag failed for %s: %s", doc_id, exc)
            self.store.finish(
                doc_id,
                title=title,
                pages=int(res.stats.get("pages") or 0),
                needs_ocr=res.needs_ocr,
                ocr_used=bool(res.stats.get("ocr_pages")) or bool(res.stats.get("ocr_backend")),
                preview=res.text_preview,
                warnings=res.warnings,
                outputs=outputs,
                body_text=body,
            )
            state["finished"] = True
            # Everything from here on is additive (search index, iCloud archive, eviction):
            # a cancel or an error must never flip a finished document back to failed.
            try:
                progress("index", 0, 1)
                final = self.store.get(doc_id)
                t_ix = time.monotonic()
                n_chunks = self.index.index_document(
                    doc_id, final.title if final else doc.title, body
                )
                self.timings.record("embed", time.monotonic() - t_ix, max(n_chunks, 1))
            except Cancelled:
                log.info("indexing of %s skipped: cancelled", doc_id)
            except Exception as exc:  # search is additive; never fail the document for it
                log.warning("indexing failed for %s: %s", doc_id, exc)
            if doc_id in self._pending_summaries:
                self._pending_summaries.discard(doc_id)
                self.submit_summarize(doc_id)
            if doc.kind == "pdf":
                try:
                    progress("scholar", 0, 1)
                    self.scholar_lookup(doc_id)
                except Cancelled:
                    log.info("scholar lookup of %s skipped: cancelled", doc_id)
                except Exception as exc:  # noqa: BLE001
                    log.warning("scholar lookup failed for %s: %s", doc_id, exc)
            try:
                progress("publish", 0, 1)
                publish_to_archive(self.settings, self.store.get(doc_id))
                if self.settings.icloud_evict_after:
                    from ..icloud import evict

                    evict(original)
            except Cancelled:
                log.info("archive publish of %s skipped: cancelled", doc_id)
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
        except Cancelled as exc:
            log.info("job %s cancelled: %s", doc_id, exc)
            ctx.kill_children()
            if state["finished"]:
                return
            self.store.fail(doc_id, f"Cancelled: {exc}")
            self._publish(
                {
                    "type": "failed",
                    "id": doc_id,
                    "status": "failed",
                    "stage": "failed",
                    "label": "Cancelled",
                    "progress": 0,
                    "error": str(exc)[:300],
                }
            )
        except Exception as exc:
            log.exception("ingest failed for %s", doc_id)
            ctx.kill_children()
            if state["finished"]:
                return
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


def _entry_filename(entry: dict) -> str:
    from ..pipeline import safe_stem

    base = safe_stem(entry.get("title") or entry.get("doi") or "article")[:100]
    return base + ".pdf"


def _timing_key(stage: str, extract: ExtractSettings) -> str:
    if stage == "ocr":
        backend = extract.ocr_backend
        if backend == "auto":
            import sys

            backend = "macocr" if sys.platform == "darwin" else "tesseract"
        return f"ocr.{backend}"
    return stage


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
        "compressed": ".compressed.pdf",
        "pdfa": ".pdfa.pdf",
        "filled": ".filled.pdf",
        "encrypted": ".encrypted.pdf",
        "stripped": ".stripped.txt",
        "summary": ".summary.json",
        "scholar": ".scholar.json",
        "references": ".references.json",
    }.get(key, src.suffix)
