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
from ..config import ExtractSettings, Settings
from .auth import Auth, RateLimiter, User, client_ip
from .importer import ImportError_, import_stream
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
        return render(
            request,
            "index.html",
            user,
            docs=docs,
            tags=store.all_tags(),
            tag=tag,
            ocr_default=settings.extract.ocr,
        )

    @app.get("/search", response_class=HTMLResponse)
    def search(request: Request, q: str = "", user: User = Depends(require_user)):
        results = store.search(q) if q.strip() else []
        return render(request, "search.html", user, q=q, results=results)

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
        if key == "original" or key == "ocr_pdf":
            media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if key == "report":
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
        for f in files[:50]:
            name = f.filename or "upload"
            try:
                imported = await asyncio.to_thread(
                    import_stream,
                    settings,
                    store,
                    jobs,
                    f.file,
                    name,
                    source="upload",
                    extract=extract,
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
            return JSONResponse({"created": created, "errors": errors})
        return RedirectResponse("/", status_code=303)

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
        if doc_id in jobs.active_ids():
            raise HTTPException(409, "still processing; try again when it finishes")
        store.delete(doc_id)
        import shutil

        shutil.rmtree(doc.dir, ignore_errors=True)
        store.audit("delete", f"{user.login}: {doc_id} ({doc.title})")
        return _back(request, "/", {"ok": True})

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
