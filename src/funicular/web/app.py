"""FastAPI application factory for the private Funicular web app."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote, urlparse

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import __version__
from ..archives import build_tar_gz, is_archive
from ..browse import BrowseError, allowed_roots, listing, resolve_safe
from ..config import ExtractSettings, Settings
from .auth import Auth, RateLimiter, User, client_ip
from .importer import ImportError_, import_path, stage_stream
from .jobs import STAGE_LABEL, JobManager
from .store import Store
from .watcher import InboxWatcher

log = logging.getLogger(__name__)
HERE = Path(__file__).parent
PUBLIC_PATHS = ("/auth/", "/healthz", "/static/", "/manifest.webmanifest", "/apple-touch-icon.png")


class ConfigError(SystemExit):
    pass


def validate_settings(s: Settings) -> None:
    host = urlparse(s.public_url).hostname
    loopback = host in ("127.0.0.1", "localhost", "::1")
    problems = []
    if not s.allowed_github_ids:
        problems.append("FUNICULAR_ALLOWED_GITHUB_IDS is empty: nobody could sign in")
    if not s.github_client_id or not s.github_client_secret:
        problems.append("FUNICULAR_GITHUB_CLIENT_ID / FUNICULAR_GITHUB_CLIENT_SECRET are required")
    if urlparse(s.public_url).scheme != "https" and not loopback:
        problems.append("FUNICULAR_PUBLIC_URL must be https unless it is a loopback address")
    if s.host not in ("127.0.0.1", "localhost", "::1") and not s.trust_proxy:
        problems.append(
            "binding to a non-loopback FUNICULAR_HOST requires FUNICULAR_TRUST_PROXY=1 behind a"
            " TLS proxy (Tailscale Serve or Caddy); bind 127.0.0.1 otherwise"
        )
    if problems:
        raise ConfigError("refusing to start:\n  - " + "\n  - ".join(problems))


# ------------------------------------------------------------------------------------------
# middleware
# ------------------------------------------------------------------------------------------
SECURITY_HEADERS = {
    b"content-security-policy": (
        b"default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        b"font-src 'self'; connect-src 'self'; manifest-src 'self'; form-action 'self'; "
        b"base-uri 'none'; frame-ancestors 'none'; object-src 'none'"
    ),
    b"x-content-type-options": b"nosniff",
    b"x-frame-options": b"DENY",
    b"referrer-policy": b"no-referrer",
    b"permissions-policy": b"camera=(), microphone=(), geolocation=(), payment=()",
    b"cross-origin-opener-policy": b"same-origin",
    b"cross-origin-resource-policy": b"same-origin",
}


class SecurityHeaders:
    """Pure ASGI middleware (BaseHTTPMiddleware would buffer the SSE stream)."""

    def __init__(self, app, *, https: bool) -> None:
        self.app = app
        self.https = https

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", [])]
                present = {k.lower() for k, _ in headers}
                for k, v in SECURITY_HEADERS.items():
                    if k not in present:
                        headers.append((k, v))
                if b"cache-control" not in present:
                    headers.append((b"cache-control", b"no-store"))
                if self.https and b"strict-transport-security" not in present:
                    headers.append(
                        (b"strict-transport-security", b"max-age=31536000; includeSubDomains")
                    )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_wrapper)


class BodyLimit:
    """Reject oversized or length-less uploads before any body byte is read."""

    def __init__(self, app, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("method") in ("POST", "PUT", "PATCH"):
            headers = {k.lower(): v for k, v in scope.get("headers", [])}
            cl = headers.get(b"content-length")
            chunked = headers.get(b"transfer-encoding", b"").lower() == b"chunked"
            status = None
            if cl is None and chunked and scope.get("path") == "/upload":
                status, msg = 411, b'{"error":"content-length required"}'
            elif cl is not None:
                try:
                    if int(cl) > self.max_bytes:
                        status, msg = 413, b'{"error":"request too large"}'
                except ValueError:
                    status, msg = 400, b'{"error":"bad content-length"}'
            if status is not None:
                await send(
                    {
                        "type": "http.response.start",
                        "status": status,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(msg)).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": msg})
                return
        await self.app(scope, receive, send)


# ------------------------------------------------------------------------------------------
# app factory
# ------------------------------------------------------------------------------------------
def create_app(settings: Settings, *, validate: bool = True) -> FastAPI:
    if validate:
        validate_settings(settings)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    store = Store(settings.data_dir / "funicular.sqlite3")
    jobs = JobManager(settings, store)
    auth = Auth(settings, store)
    watcher = InboxWatcher(settings, store, jobs) if settings.watch_inbox else None
    upload_limiter = RateLimiter(limit=60, window=60)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Probe acceleration (imports torch when installed) before any job thread starts, so
        # the first import never races a model load in a worker.
        from ..resources import gpu_info

        await asyncio.to_thread(gpu_info)
        jobs.start(asyncio.get_running_loop())
        store.purge_expired(
            idle_seconds=settings.session_idle_minutes * 60,
            absolute_seconds=settings.session_absolute_hours * 3600,
        )
        if watcher:
            watcher.start()
        try:
            yield
        finally:
            if watcher:
                watcher.stop()
            jobs.stop()

    app = FastAPI(
        title="Funicular",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.store = store
    app.state.jobs = jobs
    app.state.auth = auth
    https = urlparse(settings.public_url).scheme == "https"
    app.add_middleware(SecurityHeaders, https=https)
    app.add_middleware(BodyLimit, max_bytes=settings.max_upload_mb * 1024 * 1024 + 1024 * 1024)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    import jinja2

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(HERE / "templates"),
        autoescape=jinja2.select_autoescape(default=True, default_for_string=True),
        undefined=jinja2.StrictUndefined,
    )
    templates = Jinja2Templates(env=env)
    templates.env.filters["human_size"] = human_size
    templates.env.filters["when"] = human_when
    templates.env.globals["stage_label"] = lambda s: STAGE_LABEL.get(s, s)
    templates.env.globals["version"] = __version__
    from .icons import svg as _icon_svg

    templates.env.globals["icon"] = _icon_svg
    templates.env.filters["basename"] = lambda v: Path(str(v)).name

    # -------------------------------------------------------------- auth dependency
    def require_user(request: Request) -> User:
        user = auth.current_user(request)
        if user is None:
            wants_json = "application/json" in request.headers.get("accept", "")
            api = request.url.path.startswith(("/api/", "/events"))
            if request.method == "GET" and not wants_json and not api:
                nxt = quote(
                    request.url.path + ("?" + request.url.query if request.url.query else "")
                )
                raise HTTPException(302, headers={"Location": f"/auth/login?next={nxt}"})
            raise HTTPException(401, "sign in required")
        return user

    def csrf_check(request: Request, user: User, token: str | None) -> None:
        auth.check_csrf(request, user, token)

    def render(
        request: Request, name: str, user: User | None = None, *, status_code: int = 200, **ctx
    ) -> HTMLResponse:
        ctx.setdefault("user", user)
        ctx.setdefault("settings", settings)
        return templates.TemplateResponse(request, name, ctx, status_code=status_code)

    # -------------------------------------------------------------- public
    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict:
        return {"ok": True}

    @app.get("/manifest.webmanifest", include_in_schema=False)
    def manifest() -> Response:
        data = {
            "name": "Funicular",
            "short_name": "Funicular",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#000000",
            "theme_color": "#0a84ff",
            "icons": [
                {"src": "/apple-touch-icon.png", "sizes": "180x180", "type": "image/png"},
                {"src": "/static/icon.svg", "sizes": "any", "type": "image/svg+xml"},
            ],
        }
        return Response(json.dumps(data), media_type="application/manifest+json")

    @app.get("/apple-touch-icon.png", include_in_schema=False)
    def touch_icon() -> FileResponse:
        return FileResponse(HERE / "static" / "apple-touch-icon.png", media_type="image/png")

    @app.get("/auth/login", include_in_schema=False)
    def login(request: Request, next: str = "/", start: str = "") -> Response:  # noqa: A002
        if auth.current_user(request):
            return RedirectResponse("/", status_code=302)
        if start != "1":
            return render(request, "login.html", next=next)
        if not auth.login_limiter.check(client_ip(request)):
            raise HTTPException(429, "too many sign-in attempts")
        return RedirectResponse(auth.login_url(next), status_code=302)

    @app.get("/auth/callback", include_in_schema=False)
    def callback(request: Request, code: str = "", state: str = "", error: str = "") -> Response:
        if error or not code or not state:
            return render(request, "denied.html", reason=error or "missing code", status_code=400)
        try:
            sid, next_path = auth.complete_login(request, code, state)
        except HTTPException as exc:
            return render(request, "denied.html", reason=exc.detail, status_code=exc.status_code)
        response = RedirectResponse(next_path or "/", status_code=302)
        auth.set_session_cookie(response, sid)
        return response

    @app.post("/auth/logout", include_in_schema=False)
    def logout(request: Request, csrf: str = Form("")) -> Response:
        user = auth.current_user(request)
        if user:
            auth.check_csrf(request, user, csrf)
        response = RedirectResponse("/auth/login", status_code=303)
        auth.clear_session(request, response)
        return response

    # -------------------------------------------------------------- pages
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, user: User = Depends(require_user), tag: str | None = None):
        docs = store.list(limit=300, tag=tag)
        batches = [
            b
            for b in store.list_batches(20)
            if b["status"] != "done" or time.time() - b["updated_at"] < 3600
        ]
        return render(
            request,
            "index.html",
            user,
            docs=docs,
            tags=store.all_tags(),
            tag=tag,
            ocr_default=settings.extract.ocr,
            batches=batches,
        )

    @app.get("/search", response_class=HTMLResponse)
    def search(
        request: Request, q: str = "", mode: str = "auto", user: User = Depends(require_user)
    ):
        hits, info = ([], None)
        results = []
        if q.strip():
            hits, info = jobs.index.search(
                q, mode=mode if mode in ("auto", "keyword", "semantic") else "auto"
            )
            ids = list({h.doc_id for h in hits})
            docs = {d.id: d for d in (store.get(i) for i in ids) if d}
            results = [(docs[h.doc_id], h) for h in hits if h.doc_id in docs]
            if not results:  # titles/tags live in the document store, not the chunk index
                results = [(d, None) for d, _snip in store.search(q)]
        return render(
            request,
            "search.html",
            user,
            q=q,
            mode=mode,
            results=results,
            info=info.to_dict() if info else None,
            stats=jobs.index.stats(),
        )

    @app.get("/api/search")
    def api_search(
        q: str = "", mode: str = "auto", limit: int = 20, user: User = Depends(require_user)
    ):
        if not q.strip():
            return {"hits": [], "info": None}
        hits, info = jobs.index.search(q, mode=mode, limit=max(1, min(limit, 100)))
        return {"hits": [h.to_dict() for h in hits], "info": info.to_dict()}

    @app.get("/doc/{doc_id}", response_class=HTMLResponse)
    def doc_page(
        request: Request, doc_id: str, view: str = "layout", user: User = Depends(require_user)
    ):
        doc = store.get(doc_id)
        if not doc:
            raise HTTPException(404)
        view = view if view in ("layout", "markdown", "text") else "layout"
        content = ""
        available = [k for k in ("layout", "markdown", "text") if k in doc.outputs]
        if view not in available and available:
            view = available[0]
        if view in doc.outputs:
            content = _read_capped(doc.dir / doc.outputs[view])
        report = {}
        if "report" in doc.outputs:
            try:
                report = json.loads((doc.dir / doc.outputs["report"]).read_text())
            except OSError, ValueError:
                report = {}
        refs_report = None
        if "references" in doc.outputs:
            try:
                refs_report = json.loads((doc.dir / doc.outputs["references"]).read_text())
            except OSError, ValueError:
                refs_report = None
        return render(
            request,
            "doc.html",
            user,
            doc=doc,
            view=view,
            content=content,
            available=available,
            report=report,
            has_thumb=doc.kind in ("pdf", "image"),
            scholar=_scholar_json(doc),
            refs=refs_report,
            summary=_summary_json(doc),
            providers=_providers(),
        )

    # -------------------------------------------------------------- files
    @app.get("/doc/{doc_id}/file/{key}")
    def doc_file(doc_id: str, key: str, user: User = Depends(require_user), download: int = 0):
        doc = store.get(doc_id)
        if not doc or key not in doc.outputs:
            raise HTTPException(404)
        path = (doc.dir / doc.outputs[key]).resolve()
        if not path.is_file() or doc.dir.resolve() not in path.parents:
            raise HTTPException(404)
        media = "text/plain; charset=utf-8"
        if key in ("original", "ocr_pdf", "compressed"):
            media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if key in ("report", "scholar", "references", "summary"):
            media = "application/json"
        if key == "markdown":
            media = "text/markdown; charset=utf-8"
        if media.startswith("text/html"):
            media = "text/plain; charset=utf-8"  # never render untrusted HTML in our origin
        filename = _download_name(doc, key, path)
        disposition = "attachment" if download or key == "original" else "inline"
        headers = {
            "Content-Disposition": f'{disposition}; filename="{_ascii(filename)}"; '
            f"filename*=UTF-8''{quote(filename)}",
        }
        return FileResponse(path, media_type=media, headers=headers)

    @app.get("/doc/{doc_id}/thumb.png")
    def thumb(doc_id: str, user: User = Depends(require_user)):
        doc = store.get(doc_id)
        if not doc or "original" not in doc.outputs:
            raise HTTPException(404)
        cache = doc.dir / "thumb.png"
        if not cache.exists():
            original = doc.dir / doc.outputs["original"]
            try:
                if doc.kind == "pdf":
                    from ..pdf import render_page_png

                    cache.write_bytes(render_page_png(original, 0, width=560))
                elif doc.kind == "image":
                    from PIL import Image, ImageOps

                    with Image.open(original) as im:
                        im = ImageOps.exif_transpose(im) or im
                        im.thumbnail((560, 800))
                        im.convert("RGB").save(cache, format="PNG", optimize=True)
                else:
                    raise HTTPException(404)
            except HTTPException:
                raise
            except Exception as exc:
                log.warning("thumbnail failed for %s: %s", doc_id, exc)
                raise HTTPException(404) from exc
        return FileResponse(
            cache, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"}
        )

    # -------------------------------------------------------------- actions
    @app.post("/upload")
    async def upload(
        request: Request,
        files: list[UploadFile] = File(...),
        csrf: str = Form(""),
        ocr: str = Form(""),
        user: User = Depends(require_user),
    ):
        csrf_check(request, user, csrf)
        if not upload_limiter.check(f"up:{user.id}"):
            raise HTTPException(429, "slow down")
        extract = _extract_for(settings, ocr)
        created, errors = [], []
        batches = []
        for f in files[:200]:
            name = f.filename or "upload"
            try:
                staged, sha, size = await asyncio.to_thread(stage_stream, settings, f.file, name)
                if _sniff_kind(staged) == "archive" and is_archive(staged):
                    bid = jobs.submit_batch(
                        "archive", staged, extract=extract, delete_after=True, label=name
                    )
                    batches.append({"id": bid, "name": name, "size": size})
                    store.audit("upload.archive", f"{user.login}: {name} -> batch {bid}")
                    continue
                imported = await asyncio.to_thread(
                    import_path,
                    settings,
                    store,
                    jobs,
                    staged,
                    source="upload",
                    move=True,
                    extract=extract,
                    filename=name,
                )
                created.append(
                    imported.doc.to_public()
                    | {"duplicate_of": imported.duplicate_of.id if imported.duplicate_of else None}
                )
                store.audit("upload", f"{user.login}: {name} -> {imported.doc.id}")
            except ImportError_ as exc:
                errors.append({"name": name, "error": str(exc)})
            finally:
                await f.close()
        if "application/json" in request.headers.get("accept", ""):
            return JSONResponse({"created": created, "errors": errors, "batches": batches})
        return RedirectResponse("/", status_code=303)

    # -------------------------------------------------------------- folders / archives
    @app.get("/browse", response_class=HTMLResponse)
    def browse_page(request: Request, path: str = "", user: User = Depends(require_user)):
        try:
            lst = listing(settings, path or None)
        except BrowseError as exc:
            raise HTTPException(403, str(exc)) from exc
        return render(request, "browse.html", user, listing=lst, ocr_default=settings.extract.ocr)

    @app.get("/api/browse")
    def api_browse(path: str = "", user: User = Depends(require_user)):
        try:
            return listing(settings, path or None).to_dict()
        except BrowseError as exc:
            raise HTTPException(403, str(exc)) from exc

    @app.post("/import/folder")
    def import_folder_route(
        request: Request,
        csrf: str = Form(""),
        path: str = Form(""),
        recursive: str = Form("yes"),
        ocr: str = Form(""),
        user: User = Depends(require_user),
    ):
        csrf_check(request, user, csrf)
        try:
            folder = resolve_safe(path, allowed_roots(settings))
        except BrowseError as exc:
            raise HTTPException(403, str(exc)) from exc
        if not folder.is_dir():
            raise HTTPException(400, "not a folder")
        bid = jobs.submit_batch(
            "folder", folder, extract=_extract_for(settings, ocr), recursive=recursive == "yes"
        )
        store.audit("import.folder", f"{user.login}: {folder} -> batch {bid}")
        return _back(request, "/", {"ok": True, "batch": bid})

    @app.post("/import/file")
    def import_file_route(
        request: Request,
        csrf: str = Form(""),
        path: str = Form(""),
        ocr: str = Form(""),
        user: User = Depends(require_user),
    ):
        csrf_check(request, user, csrf)
        try:
            file = resolve_safe(path, allowed_roots(settings))
        except BrowseError as exc:
            raise HTTPException(403, str(exc)) from exc
        if not file.is_file():
            raise HTTPException(400, "not a file")
        extract = _extract_for(settings, ocr)
        if _sniff_kind(file) == "archive" and is_archive(file):
            bid = jobs.submit_batch("archive", file, extract=extract)
            return _back(request, "/", {"ok": True, "batch": bid})
        try:
            imp = import_path(
                settings, store, jobs, file, source="folder", move=False, extract=extract
            )
        except ImportError_ as exc:
            raise HTTPException(400, str(exc)) from exc
        store.audit("import.file", f"{user.login}: {file} -> {imp.doc.id}")
        return _back(request, f"/doc/{imp.doc.id}", {"ok": True, "id": imp.doc.id})

    @app.get("/api/batches")
    def api_batches(user: User = Depends(require_user)):
        return {"batches": store.list_batches()}

    @app.post("/batch/{bid}/cancel")
    def cancel_batch(
        request: Request, bid: str, csrf: str = Form(""), user: User = Depends(require_user)
    ):
        csrf_check(request, user, csrf)
        if not store.get_batch(bid):
            raise HTTPException(404)
        return _back(request, "/", {"ok": jobs.cancel_batch(bid)})

    @app.post("/export")
    def export(
        request: Request,
        csrf: str = Form(""),
        ids: list[str] = Form([]),
        what: str = Form("all"),
        user: User = Depends(require_user),
    ):
        """Download selected documents (outputs and/or originals) as one tar.gz."""
        csrf_check(request, user, csrf)
        ids = [i for i in ids if i][:500]
        if not ids:
            raise HTTPException(400, "select at least one document")
        from ..pipeline import safe_stem

        items: list[tuple[Path, str]] = []
        for doc_id in ids:
            doc = store.get(doc_id)
            if not doc:
                continue
            folder = f"{safe_stem(doc.title)[:60]} ({doc.id[:6]})"
            for key, rel in doc.outputs.items():
                if what == "originals" and key != "original":
                    continue
                if what == "text" and key in ("original", "ocr_pdf"):
                    continue
                items.append((doc.dir / rel, f"{folder}/{_download_name(doc, key, doc.dir / rel)}"))
        tmp_dir = settings.data_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out = tmp_dir / f"export-{int(time.time())}-{user.id}.tar.gz"
        build_tar_gz(items, out)
        store.audit("export", f"{user.login}: {len(ids)} document(s)")
        from starlette.background import BackgroundTask

        return FileResponse(
            out,
            media_type="application/gzip",
            filename="funicular-export.tar.gz",
            background=BackgroundTask(lambda: out.unlink(missing_ok=True)),
        )

    @app.post("/doc/{doc_id}/reprocess")
    def reprocess(
        request: Request,
        doc_id: str,
        csrf: str = Form(""),
        ocr: str = Form("auto"),
        user: User = Depends(require_user),
    ):
        csrf_check(request, user, csrf)
        doc = store.get(doc_id)
        if not doc:
            raise HTTPException(404)
        if doc_id in jobs.active_ids():
            raise HTTPException(409, "already processing")
        store.requeue(doc_id)
        jobs.submit(doc_id, _extract_for(settings, ocr))
        store.audit("reprocess", f"{user.login}: {doc_id} ocr={ocr}")
        return _back(request, f"/doc/{doc_id}", {"ok": True, "id": doc_id})

    @app.post("/doc/{doc_id}/tags")
    def set_tags(
        request: Request,
        doc_id: str,
        csrf: str = Form(""),
        tags: str = Form(""),
        user: User = Depends(require_user),
    ):
        csrf_check(request, user, csrf)
        if not store.get(doc_id):
            raise HTTPException(404)
        names = [t for t in re.split(r"[,\n;]+", tags) if t.strip()][:20]
        clean = store.set_tags(doc_id, names)
        return _back(request, f"/doc/{doc_id}", {"ok": True, "tags": clean})

    @app.post("/doc/{doc_id}/rename")
    def rename(
        request: Request,
        doc_id: str,
        csrf: str = Form(""),
        title: str = Form(""),
        user: User = Depends(require_user),
    ):
        csrf_check(request, user, csrf)
        if not store.get(doc_id):
            raise HTTPException(404)
        title = re.sub(r"\s+", " ", title).strip()[:200]
        if not title:
            raise HTTPException(400, "title required")
        store.rename(doc_id, title)
        try:
            with jobs.index._conn() as c:  # noqa: SLF001 - keep the index title in step
                c.execute("UPDATE docs SET title=? WHERE doc_id=?", (title, doc_id))
        except Exception as exc:  # noqa: BLE001
            log.debug("index title update failed: %s", exc)
        return _back(request, f"/doc/{doc_id}", {"ok": True, "title": title})

    @app.post("/doc/{doc_id}/delete")
    def delete(
        request: Request,
        doc_id: str,
        csrf: str = Form(""),
        confirm: str = Form(""),
        user: User = Depends(require_user),
    ):
        csrf_check(request, user, csrf)
        doc = store.get(doc_id)
        if not doc:
            raise HTTPException(404)
        if confirm != "yes":
            raise HTTPException(400, "confirmation required")
        if doc.status in ("queued", "running"):
            raise HTTPException(409, "still processing; cancel it first or wait")
        store.delete(doc_id)
        jobs.index.remove_document(doc_id)
        import shutil

        shutil.rmtree(doc.dir, ignore_errors=True)
        store.audit("delete", f"{user.login}: {doc_id} ({doc.title})")
        return _back(request, "/", {"ok": True})

    # -------------------------------------------------------------- literature graph
    def _graph_seeds(ids: list[str] | None) -> list[dict]:
        seeds = []
        docs = [store.get(i) for i in ids] if ids else store.list(limit=300)
        for d in docs:
            if not d or d.kind != "pdf":
                continue
            sj = _scholar_json(d) or {}
            w = (sj.get("resolution") or {}).get("work") or {}
            if not (w.get("doi") or w.get("s2_id")):
                continue
            seeds.append(
                {
                    "doc_id": d.id,
                    "doi": w.get("doi"),
                    "s2_id": w.get("s2_id"),
                    "title": w.get("title") or d.title,
                    "year": w.get("year"),
                    "cited_by": w.get("cited_by"),
                }
            )
        return seeds

    @app.get("/graph", response_class=HTMLResponse)
    def graph_page(request: Request, user: User = Depends(require_user)):
        cached = settings.data_dir / "graph.json"
        data = None
        if cached.is_file():
            try:
                data = json.loads(cached.read_text())
            except OSError, ValueError:
                data = None
        return render(request, "graph.html", user, graph=data, seeds=len(_graph_seeds(None)))

    @app.post("/graph/build")
    def graph_build(
        request: Request,
        csrf: str = Form(""),
        ids: list[str] = Form([]),
        max_nodes: int = Form(40),
        user: User = Depends(require_user),
    ):
        csrf_check(request, user, csrf)
        from ..embeddings import get_embedder
        from ..graph import build_graph

        seeds = _graph_seeds([i for i in ids if i] or None)
        if not seeds:
            raise HTTPException(400, "no documents with a verified DOI yet; verify records first")
        f = jobs.fetcher_factory()
        try:
            emb = get_embedder(settings.embeddings) if settings.embeddings != "none" else None
            g = build_graph(seeds[:20], f, max_nodes=max(5, min(max_nodes, 120)), embedder=emb)
        finally:
            f.close()
        data = g.to_dict()
        (settings.data_dir / "graph.json").write_text(json.dumps(data))
        store.audit("graph.build", f"{user.login}: {len(g.nodes)} nodes from {len(seeds)} seeds")
        return _back(request, "/graph", data)

    @app.get("/api/graph")
    def api_graph(user: User = Depends(require_user)):
        cached = settings.data_dir / "graph.json"
        if not cached.is_file():
            return {"nodes": [], "edges": []}
        return json.loads(cached.read_text())

    # -------------------------------------------------------------- feeds
    @app.get("/feeds", response_class=HTMLResponse)
    def feeds_page(
        request: Request,
        feed: int | None = None,
        status: str = "",
        user: User = Depends(require_user),
    ):
        entries = jobs.feeds.entries(feed, status=status or None, limit=200)
        return render(
            request,
            "feeds.html",
            user,
            feeds=jobs.feeds.list(),
            entries=entries,
            feed=feed,
            status=status,
        )

    @app.post("/feeds/add")
    def feeds_add(
        request: Request,
        csrf: str = Form(""),
        kind: str = Form("rss"),
        name: str = Form(""),
        query: str = Form(""),
        auto_stage: str = Form(""),
        user: User = Depends(require_user),
    ):
        csrf_check(request, user, csrf)
        query = query.strip()
        if not query:
            raise HTTPException(400, "feed URL or query required")
        if kind == "rss" and not query.startswith(("http://", "https://")):
            raise HTTPException(400, "RSS feeds need an http(s) URL")
        try:
            fid = jobs.feeds.add(
                kind, name.strip() or query[:80], query, auto_stage=auto_stage == "yes"
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        store.audit("feed.add", f"{user.login}: {kind} {query[:80]}")
        return _back(request, "/feeds", {"ok": True, "id": fid})

    @app.post("/feeds/{fid}/remove")
    def feeds_remove(
        request: Request, fid: int, csrf: str = Form(""), user: User = Depends(require_user)
    ):
        csrf_check(request, user, csrf)
        jobs.feeds.remove(fid)
        return _back(request, "/feeds", {"ok": True})

    @app.post("/feeds/poll")
    def feeds_poll(
        request: Request,
        csrf: str = Form(""),
        fid: int | None = Form(None),
        user: User = Depends(require_user),
    ):
        csrf_check(request, user, csrf)
        counts = jobs.poll_feeds(fid)
        store.audit("feed.poll", f"{user.login}: {counts}")
        return _back(request, "/feeds", {"ok": True, "new": counts})

    @app.post("/feeds/entry/{eid}/stage")
    def feeds_stage(
        request: Request,
        eid: int,
        csrf: str = Form(""),
        summarize: str = Form(""),
        user: User = Depends(require_user),
    ):
        csrf_check(request, user, csrf)
        try:
            out = jobs.stage_entry(eid, summarize=(summarize == "yes") or None)
        except KeyError as exc:
            raise HTTPException(404) from exc
        return _back(request, "/feeds", out)

    @app.post("/feeds/stage-new")
    def feeds_stage_new(
        request: Request,
        csrf: str = Form(""),
        fid: int | None = Form(None),
        user: User = Depends(require_user),
    ):
        csrf_check(request, user, csrf)
        n = 0
        for e in jobs.feeds.entries(fid, status="new", limit=100):
            jobs.pool.submit(jobs._safe_stage, e["id"])  # noqa: SLF001
            n += 1
        return _back(request, "/feeds", {"ok": True, "queued": n})

    @app.get("/api/feeds")
    def api_feeds(user: User = Depends(require_user)):
        return {"feeds": jobs.feeds.list(), "entries": jobs.feeds.entries(limit=100)}

    # -------------------------------------------------------------- zotero
    @app.get("/zotero", response_class=HTMLResponse)
    def zotero_page(request: Request, user: User = Depends(require_user)):
        z = jobs.zotero_factory()
        try:
            ok = z.available()
            cols = z.collections() if ok else []
        finally:
            z.close()
        return render(request, "zotero.html", user, available=ok, collections=cols)

    @app.post("/zotero/import")
    def zotero_import(
        request: Request,
        csrf: str = Form(""),
        collection: str = Form(""),
        ocr: str = Form(""),
        user: User = Depends(require_user),
    ):
        csrf_check(request, user, csrf)
        bid = jobs.import_zotero(collection.strip() or None, extract=_extract_for(settings, ocr))
        store.audit("zotero.import", f"{user.login}: {collection or 'library'} -> {bid}")
        return _back(request, "/", {"ok": True, "batch": bid})

    # -------------------------------------------------------------- summaries / ask
    def _summary_json(doc) -> dict | None:
        p = doc.dir / "summary.json"
        try:
            return json.loads(p.read_text()) if p.is_file() else None
        except OSError, ValueError:
            return None

    @app.post("/doc/{doc_id}/summarize")
    def summarize_route(
        request: Request,
        doc_id: str,
        csrf: str = Form(""),
        provider: str = Form("auto"),
        user: User = Depends(require_user),
    ):
        csrf_check(request, user, csrf)
        doc = store.get(doc_id)
        if not doc:
            raise HTTPException(404)
        if doc.status != "done":
            raise HTTPException(409, "document is still processing")
        if not jobs.submit_summarize(doc_id, None if provider == "auto" else provider):
            raise HTTPException(409, "a summary is already being written")
        store.audit("summarize.start", f"{user.login}: {doc_id} via {provider}")
        return _back(request, f"/doc/{doc_id}", {"ok": True})

    @app.get("/doc/{doc_id}/summary")
    def summary_get(doc_id: str, user: User = Depends(require_user)):
        doc = store.get(doc_id)
        if not doc:
            raise HTTPException(404)
        return _summary_json(doc) or {"card": None}

    def _passages(question: str, doc_ids: list[str] | None, limit: int = 12) -> list[dict]:
        hits, _info = jobs.index.search(question, limit=limit, doc_ids=doc_ids)
        out = []
        with jobs.index._conn() as c:  # noqa: SLF001 - read the chunk text for the answer
            for h in hits:
                row = c.execute(
                    "SELECT text FROM chunks WHERE doc_id=? AND idx=?", (h.doc_id, h.idx)
                ).fetchone()
                if row:
                    out.append(
                        {"doc_id": h.doc_id, "title": h.title, "page": h.page, "text": row["text"]}
                    )
        return out

    @app.post("/ask")
    def ask_route(
        request: Request,
        csrf: str = Form(""),
        question: str = Form(""),
        provider: str = Form("auto"),
        doc_id: str = Form(""),
        user: User = Depends(require_user),
    ):
        """Grounded answer over the library (or one document) with passage citations."""
        csrf_check(request, user, csrf)
        from ..llm import LLMUnavailable
        from ..summarize import ask

        q = question.strip()[:2000]
        if not q:
            raise HTTPException(400, "question required")
        try:
            prov = jobs.provider_factory(None if provider == "auto" else provider)
            answer = ask(q, _passages(q, [doc_id] if doc_id else None), prov)
        except LLMUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        store.audit("ask", f"{user.login}: {prov.name} doc={doc_id or '*'} q={q[:80]}")
        payload = {"question": q, "doc_id": doc_id} | answer.to_dict()
        if "application/json" in request.headers.get("accept", ""):
            return JSONResponse(payload)
        return render(
            request,
            "ask.html",
            user,
            answer=payload,
            providers=_providers(),
            doc=store.get(doc_id) if doc_id else None,
        )

    @app.get("/ask", response_class=HTMLResponse)
    def ask_page(request: Request, doc_id: str = "", user: User = Depends(require_user)):
        return render(
            request,
            "ask.html",
            user,
            answer=None,
            providers=_providers(),
            doc=store.get(doc_id) if doc_id else None,
        )

    def _providers() -> list[dict]:
        from ..llm import describe_providers

        return describe_providers()

    # -------------------------------------------------------------- scholar
    def _scholar_json(doc) -> dict | None:
        p = doc.dir / "scholar.json"
        try:
            return json.loads(p.read_text()) if p.is_file() else None
        except OSError, ValueError:
            return None

    @app.get("/doc/{doc_id}/scholar")
    def scholar_get(doc_id: str, user: User = Depends(require_user)):
        doc = store.get(doc_id)
        if not doc:
            raise HTTPException(404)
        return _scholar_json(doc) or {"extracted": None, "resolution": None}

    @app.post("/doc/{doc_id}/scholar/verify")
    def scholar_verify(
        request: Request, doc_id: str, csrf: str = Form(""), user: User = Depends(require_user)
    ):
        csrf_check(request, user, csrf)
        doc = store.get(doc_id)
        if not doc or doc.kind != "pdf":
            raise HTTPException(404)
        out = jobs.scholar_lookup(doc_id, online=True)
        store.audit("scholar.verify", f"{user.login}: {doc_id}")
        return _back(request, f"/doc/{doc_id}", out or {"error": "no PDF"})

    @app.post("/doc/{doc_id}/scholar/refs")
    def scholar_refs(
        request: Request,
        doc_id: str,
        csrf: str = Form(""),
        resolve: str = Form("yes"),
        user: User = Depends(require_user),
    ):
        """Extract the reference list (lossless) and optionally resolve each entry online."""
        csrf_check(request, user, csrf)
        doc = store.get(doc_id)
        if not doc:
            raise HTTPException(404)
        from ..scholar.refs import (
            ReferenceReport,
            ResolvedRef,
            extract_references,
            resolve_reference,
        )

        text = ""
        for key in ("layout", "text"):
            if key in doc.outputs:
                text = _read_capped(doc.dir / doc.outputs[key], 5_000_000)
                break
        raw = extract_references(text)
        if resolve == "yes":
            f = jobs.fetcher_factory()
            try:
                rep = ReferenceReport([resolve_reference(r, f) for r in raw[:300]])
            finally:
                f.close()
        else:
            rep = ReferenceReport([ResolvedRef(r, None, 0.0, "not resolved") for r in raw])
        (doc.dir / "references.json").write_text(
            json.dumps(rep.to_dict(), ensure_ascii=False, indent=1)
        )
        outputs = dict(doc.outputs)
        outputs["references"] = "references.json"
        store.set_outputs(doc_id, outputs)
        store.audit("scholar.refs", f"{user.login}: {doc_id} {rep.resolved}/{len(rep.refs)}")
        return _back(request, f"/doc/{doc_id}", rep.to_dict())

    @app.post("/doc/{doc_id}/scholar/adopt")
    def scholar_adopt(
        request: Request, doc_id: str, csrf: str = Form(""), user: User = Depends(require_user)
    ):
        """Use the verified record's title as the document title (Zotero-style naming)."""
        csrf_check(request, user, csrf)
        doc = store.get(doc_id)
        if not doc:
            raise HTTPException(404)
        sj = _scholar_json(doc) or {}
        work = (sj.get("resolution") or {}).get("work") or {}
        if not work.get("title"):
            raise HTTPException(400, "no verified record to adopt")
        from ..scholar.model import Author, Work
        from ..scholar.rename import build_name

        w = Work(**{k: v for k, v in work.items() if k != "authors"})
        w.authors = [Author(**a) for a in work.get("authors", [])]
        title = build_name(w)
        store.rename(doc_id, title)
        return _back(request, f"/doc/{doc_id}", {"ok": True, "title": title})

    # -------------------------------------------------------------- preview / compress
    def _plan_reprocess(doc, ocr: str) -> dict:
        from ..estimate import audio_minutes, estimate, pdf_page_count
        from ..resources import gpu_info

        extract = _extract_for(settings, ocr)
        original = doc.dir / doc.outputs.get("original", "")
        pages = doc.pages or (pdf_page_count(original) if doc.kind == "pdf" else 0)
        ocr_pages = None
        if doc.kind == "pdf" and "report" in doc.outputs:
            try:
                rep = json.loads((doc.dir / doc.outputs["report"]).read_text())
                ocr_pages = len(rep["stats"]["signal"]["needs_ocr_pages"])
            except OSError, ValueError, KeyError, TypeError:
                ocr_pages = None
        minutes = (
            audio_minutes(original)
            if doc.kind in ("audio", "video") and original.is_file()
            else 0.0
        )
        est = estimate(
            doc.kind,
            pages=pages,
            size_bytes=doc.size,
            audio_minutes=minutes,
            ocr_pages=ocr_pages,
            settings=extract,
            timings=jobs.timings,
            apple_silicon=gpu_info()["apple_silicon"],
        )
        return {
            "op": "reprocess",
            "ocr": extract.ocr,
            "estimate": est.to_dict(),
            "outputs": ["layout", "markdown", "text", "report"]
            + (
                ["ocr_pdf"]
                if extract.ocr != "off" and (ocr_pages or extract.ocr == "force")
                else []
            ),
        }

    @app.get("/doc/{doc_id}/plan")
    def plan(
        doc_id: str,
        op: str = "reprocess",
        ocr: str = "auto",
        strength: int = 50,
        engine: str = "auto",
        user: User = Depends(require_user),
    ):
        """Dry run: what an operation would do, how long it should take, what it would write."""
        doc = store.get(doc_id)
        if not doc:
            raise HTTPException(404)
        if op == "compress":
            if doc.kind != "pdf":
                raise HTTPException(400, "only PDFs can be compressed")
            from ..compress import preview

            src = doc.dir / doc.outputs.get("ocr_pdf", doc.outputs.get("original", ""))
            try:
                return {"op": "compress"} | preview(src, strength, engine=engine).to_dict()
            except Exception as exc:
                raise HTTPException(400, f"preview failed: {exc}") from exc
        return _plan_reprocess(doc, ocr)

    @app.get("/doc/{doc_id}/compress", response_class=HTMLResponse)
    def compress_page(
        request: Request,
        doc_id: str,
        strength: int = 50,
        engine: str = "auto",
        user: User = Depends(require_user),
    ):
        doc = store.get(doc_id)
        if not doc or doc.kind != "pdf":
            raise HTTPException(404)
        from ..compress import PRESETS, preview

        src = doc.dir / doc.outputs.get("ocr_pdf", doc.outputs.get("original", ""))
        plan_d = None
        error = ""
        try:
            plan_d = preview(src, strength, engine=engine).to_dict()
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        return render(
            request,
            "compress.html",
            user,
            doc=doc,
            strength=strength,
            engine=engine,
            plan=plan_d,
            error=error,
            presets=PRESETS,
        )

    @app.post("/doc/{doc_id}/compress")
    def compress_run(
        request: Request,
        doc_id: str,
        csrf: str = Form(""),
        strength: int = Form(50),
        engine: str = Form("auto"),
        user: User = Depends(require_user),
    ):
        csrf_check(request, user, csrf)
        doc = store.get(doc_id)
        if not doc or doc.kind != "pdf":
            raise HTTPException(404)
        if engine not in ("auto", "pymupdf", "ghostscript"):
            raise HTTPException(400, "bad engine")
        if not jobs.submit_compress(doc_id, max(0, min(strength, 100)), engine):
            raise HTTPException(409, "already processing")
        store.audit("compress.start", f"{user.login}: {doc_id} strength={strength}")
        return _back(request, f"/doc/{doc_id}", {"ok": True})

    # -------------------------------------------------------------- system
    @app.get("/api/system")
    def api_system(user: User = Depends(require_user)):
        return jobs.system()

    @app.get("/system", response_class=HTMLResponse)
    def system_page(request: Request, user: User = Depends(require_user)):
        return render(request, "system.html", user, system=jobs.system())

    @app.post("/doc/{doc_id}/cancel")
    def cancel_job(
        request: Request, doc_id: str, csrf: str = Form(""), user: User = Depends(require_user)
    ):
        csrf_check(request, user, csrf)
        if not store.get(doc_id):
            raise HTTPException(404)
        ok = jobs.cancel(doc_id)
        store.audit("cancel", f"{user.login}: {doc_id} ({'running' if ok else 'idle'})")
        return _back(request, f"/doc/{doc_id}", {"ok": ok})

    @app.post("/api/system/shutdown")
    def shutdown(request: Request, csrf: str = Form(""), user: User = Depends(require_user)):
        """Clean exit: cancel jobs, kill their process trees, stop the server."""
        csrf_check(request, user, csrf)
        store.audit("shutdown", user.login)
        jobs.stop("server stopped from the web app")

        def _exit() -> None:
            import os
            import signal
            import time as _t

            _t.sleep(0.5)
            os.kill(os.getpid(), signal.SIGTERM)

        import threading

        threading.Thread(target=_exit, daemon=True).start()
        return _back(request, "/auth/login", {"ok": True, "stopping": True})

    # -------------------------------------------------------------- JSON + SSE
    @app.get("/api/docs")
    def api_docs(user: User = Depends(require_user)):
        return {"docs": [d.to_public() for d in store.list(limit=300)], "active": jobs.active_ids()}

    @app.get("/api/docs/{doc_id}")
    def api_doc(doc_id: str, user: User = Depends(require_user)):
        doc = store.get(doc_id)
        if not doc:
            raise HTTPException(404)
        return doc.to_public()

    @app.get("/events")
    async def events(request: Request, user: User = Depends(require_user), max_seconds: int = 300):
        """Server-sent progress events. Streams are bounded (EventSource reconnects on its
        own) so a proxy or a lost client can never pin a connection forever; on a real
        disconnect StreamingResponse cancels the generator."""
        q = jobs.subscribe()
        lifetime = max(1, min(max_seconds, 3600))
        started = time.monotonic()

        async def gen() -> AsyncIterator[bytes]:
            try:
                snapshot = {
                    "type": "snapshot",
                    "active": jobs.active_ids(),
                    "docs": [
                        d.to_public()
                        for d in store.list(limit=100)
                        if d.status in ("queued", "running")
                    ],
                    "batches": [
                        b for b in store.list_batches(20) if b["status"] in ("queued", "running")
                    ],
                }
                yield _sse(snapshot)
                while True:
                    remaining = lifetime - (time.monotonic() - started)
                    if remaining <= 0:
                        break
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=min(15, remaining))
                        yield _sse(event)
                    except TimeoutError:
                        yield b": keepalive\n\n"
            finally:
                jobs.unsubscribe(q)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    # -------------------------------------------------------------- errors
    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if exc.status_code == 302 and exc.headers and "Location" in exc.headers:
            return RedirectResponse(exc.headers["Location"], status_code=302)
        if "text/html" in request.headers.get("accept", "") and request.method == "GET":
            return render(
                request,
                "error.html",
                auth.current_user(request),
                status=exc.status_code,
                detail=exc.detail,
                status_code=exc.status_code,
            )
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code, headers=exc.headers)

    return app


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
def _sniff_kind(path: Path) -> str:
    from ..sniff import sniff

    return sniff(path).kind.value


def _extract_for(settings: Settings, ocr: str) -> ExtractSettings:
    base = settings.extract
    if ocr in ("off", "auto", "force"):
        return base.model_copy(update={"ocr": ocr})
    return base


def _back(request: Request, path: str, payload: dict) -> Response:
    if "application/json" in request.headers.get("accept", ""):
        return JSONResponse(payload)
    return RedirectResponse(path, status_code=303)


def _sse(event: dict) -> bytes:
    return f"event: {event.get('type', 'message')}\ndata: {json.dumps(event)}\n\n".encode()


def _read_capped(path: Path, limit: int = 3_000_000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            data = fh.read(limit + 1)
    except OSError:
        return ""
    if len(data) > limit:
        return data[:limit] + "\n\n… (truncated; download the file for the full text)"
    return data


def _download_name(doc, key: str, path: Path) -> str:
    from ..pipeline import safe_stem

    stem = safe_stem(doc.title)[:80]
    suffix = {
        "layout": ".layout.txt",
        "markdown": ".md",
        "text": ".txt",
        "report": ".json",
        "ocr_pdf": ".ocr.pdf",
        "compressed": ".compressed.pdf",
        "scholar": ".scholar.json",
        "references": ".references.json",
        "summary": ".summary.json",
    }.get(key, path.suffix)
    if key == "original":
        return doc.original_name
    return stem + suffix


def _ascii(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name) or "file"


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def human_when(ts: float) -> str:
    delta = time.time() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)} min ago"
    if delta < 86400:
        return f"{int(delta // 3600)} h ago"
    return time.strftime("%d %b %Y", time.localtime(ts))
