"""Web app tests: auth gate, OAuth flow (GitHub calls stubbed), CSRF, uploads, progress,
search, downloads, deletion, security headers. Runs entirely locally."""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from conftest import needs_poppler

pytest.importorskip("fastapi")
from starlette.testclient import TestClient  # noqa: E402

from funicular.config import Settings  # noqa: E402
from funicular.web import auth as authmod  # noqa: E402
from funicular.web.app import ConfigError, create_app, validate_settings  # noqa: E402
from funicular.web.store import Store, fts_query  # noqa: E402

ALLOWED_ID = 4242


def make_settings(tmp_path: Path, **over) -> Settings:
    base = dict(
        data_dir=tmp_path / "data",
        inbox_dir=tmp_path / "inbox",
        archive_dir=tmp_path / "archive",
        public_url="http://127.0.0.1:8787",
        github_client_id="cid",
        github_client_secret="secret",
        allowed_github_ids=frozenset({ALLOWED_ID}),
        workers=1,
        embeddings="none",  # keep tests offline and fast; semantic search has its own test
    )
    base.update(over)
    return Settings(**base)


@pytest.fixture
def github(monkeypatch):
    """Stub the two outbound GitHub calls. Tests choose which user GitHub 'returns'."""
    state = {"user": (ALLOWED_ID, "ben"), "codes": []}

    def exchange(self, code):
        state["codes"].append(code)
        if code == "bad":
            from fastapi import HTTPException

            raise HTTPException(401, "GitHub did not issue a token: bad_verification_code")
        return "tok"

    monkeypatch.setattr(authmod.Auth, "exchange_code", exchange)
    monkeypatch.setattr(authmod.Auth, "fetch_user", lambda self, tok: state["user"])
    return state


@pytest.fixture
def client(tmp_path, github):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as c:
        yield c


def sign_in(c: TestClient, next_path: str = "/") -> str:
    r = c.get(f"/auth/login?start=1&next={next_path}", follow_redirects=False)
    assert r.status_code == 302
    loc = r.headers["location"]
    assert loc.startswith("https://github.com/login/oauth/authorize?")
    q = parse_qs(urlparse(loc).query)
    assert q["redirect_uri"] == ["http://127.0.0.1:8787/auth/callback"]
    assert q["allow_signup"] == ["false"]
    state = q["state"][0]
    r = c.get(f"/auth/callback?code=good&state={state}", follow_redirects=False)
    assert r.status_code == 302, r.text
    assert r.headers["location"] == next_path
    r = c.get("/")
    assert r.status_code == 200
    return r.text.split('name="csrf" content="')[1].split('"')[0]


def wait_done(c: TestClient, doc_id: str, timeout: float = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = c.get(f"/api/docs/{doc_id}").json()
        if d["status"] in ("done", "failed"):
            return d
        time.sleep(0.2)
    raise AssertionError("document never finished")


# ------------------------------------------------------------------ config
def test_validate_settings_refuses_unsafe_config(tmp_path):
    with pytest.raises(ConfigError):
        validate_settings(make_settings(tmp_path, allowed_github_ids=frozenset()))
    with pytest.raises(ConfigError):
        validate_settings(make_settings(tmp_path, github_client_secret=""))
    with pytest.raises(ConfigError):
        validate_settings(make_settings(tmp_path, public_url="http://funicular.example.net"))
    with pytest.raises(ConfigError):
        validate_settings(make_settings(tmp_path, host="0.0.0.0"))  # noqa: S104
    validate_settings(
        make_settings(
            tmp_path,
            host="0.0.0.0",
            trust_proxy=True,  # noqa: S104
            public_url="https://funicular.example.net",
        )
    )


# ------------------------------------------------------------------ auth
def test_unauthenticated_requests_are_blocked(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"].startswith("/auth/login")
    assert client.get("/api/docs").status_code == 401
    assert client.get("/events", headers={"Accept": "application/json"}).status_code == 401
    assert client.post("/upload").status_code == 401
    r = client.get("/doc/x/file/original", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"].startswith("/auth/login")
    assert client.get("/healthz").json() == {"ok": True}
    assert client.get("/manifest.webmanifest").status_code == 200
    assert client.get("/static/app.css").status_code == 200


def test_login_flow_and_logout(client):
    csrf = sign_in(client, "/search")
    assert csrf
    r = client.get("/api/docs")
    assert r.status_code == 200
    r = client.post("/auth/logout", data={"csrf": csrf}, follow_redirects=False)
    assert r.status_code == 303
    assert client.get("/api/docs").status_code == 401


def test_state_is_single_use_and_bad_state_rejected(client):
    r = client.get("/auth/login?start=1", follow_redirects=False)
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    assert (
        client.get(f"/auth/callback?code=good&state={state}", follow_redirects=False).status_code
        == 302
    )
    r = client.get(f"/auth/callback?code=good&state={state}", follow_redirects=False)
    assert r.status_code == 400
    r = client.get("/auth/callback?code=good&state=forged", follow_redirects=False)
    assert r.status_code == 400
    r = client.get("/auth/callback?error=access_denied", follow_redirects=False)
    assert r.status_code == 400


def test_user_not_on_allowlist_is_denied(client, github):
    github["user"] = (999, "stranger")
    r = client.get("/auth/login?start=1", follow_redirects=False)
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    r = client.get(f"/auth/callback?code=good&state={state}", follow_redirects=False)
    assert r.status_code == 403
    assert "not on the allowlist" in r.text
    assert "set-cookie" not in r.headers
    assert client.get("/api/docs").status_code == 401


def test_github_token_failure(client):
    r = client.get("/auth/login?start=1", follow_redirects=False)
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    r = client.get(f"/auth/callback?code=bad&state={state}", follow_redirects=False)
    assert r.status_code == 401


def test_open_redirect_is_neutralised(client):
    r = client.get("/auth/login?start=1&next=//evil.example", follow_redirects=False)
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    r = client.get(f"/auth/callback?code=good&state={state}", follow_redirects=False)
    assert r.headers["location"] == "/"


def test_session_cookie_flags(client):
    r = client.get("/auth/login?start=1", follow_redirects=False)
    state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
    r = client.get(f"/auth/callback?code=good&state={state}", follow_redirects=False)
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "path=/" in cookie
    assert "secure" not in cookie  # loopback http in tests; https deployments set Secure + __Host-


def test_https_deployment_uses_host_cookie(tmp_path, github):
    app = create_app(make_settings(tmp_path, public_url="https://f.example.net"))
    with TestClient(app, base_url="https://f.example.net") as c:
        r = c.get("/auth/login?start=1", follow_redirects=False)
        state = parse_qs(urlparse(r.headers["location"]).query)["state"][0]
        r = c.get(f"/auth/callback?code=good&state={state}", follow_redirects=False)
        cookie = r.headers["set-cookie"]
        assert cookie.startswith("__Host-funicular_session=")
        assert "Secure" in cookie
        r = c.get("/")
        assert r.headers["strict-transport-security"].startswith("max-age=")


def test_login_rate_limit(client):
    codes = [
        client.get("/auth/login?start=1", follow_redirects=False).status_code for _ in range(12)
    ]
    assert 429 in codes


# ------------------------------------------------------------------ headers + CSRF
def test_security_headers(client):
    sign_in(client)
    r = client.get("/")
    h = r.headers
    csp = h["content-security-policy"]
    assert "default-src 'none'" in csp and "frame-ancestors 'none'" in csp
    assert "'unsafe-inline'" not in csp
    assert h["x-content-type-options"] == "nosniff"
    assert h["x-frame-options"] == "DENY"
    assert h["referrer-policy"] == "no-referrer"
    assert h["cache-control"] == "no-store"
    assert "server" not in h


def test_csrf_required_on_every_post(client, fixtures):
    csrf = sign_in(client)
    pdf = fixtures["scholarly"].read_bytes()
    r = client.post(
        "/upload",
        files=[("files", ("a.pdf", pdf, "application/pdf"))],
        data={"csrf": "nope"},
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 403
    r = client.post(
        "/upload",
        files=[("files", ("a.pdf", pdf, "application/pdf"))],
        headers={"Accept": "application/json", "Origin": "https://evil.example"},
    )
    assert r.status_code == 403
    # header-based token works for fetch()
    r = client.post(
        "/upload",
        files=[("files", ("a.pdf", pdf, "application/pdf"))],
        headers={"Accept": "application/json", "X-CSRF-Token": csrf},
    )
    assert r.status_code == 200
    doc_id = r.json()["created"][0]["id"]
    assert client.post(f"/doc/{doc_id}/delete", data={"confirm": "yes"}).status_code == 403
    wait_done(client, doc_id)


# ------------------------------------------------------------------ documents
@needs_poppler
def test_upload_process_view_download_search_delete(client, fixtures, tmp_path):
    csrf = sign_in(client)
    pdf = fixtures["scholarly"].read_bytes()
    r = client.post(
        "/upload",
        files=[
            ("files", ("scholarly.pdf", pdf, "application/pdf")),
            ("files", ("evil.exe", b"MZ\x90\x00\x03", "application/octet-stream")),
            ("files", ("empty.pdf", b"", "application/pdf")),
        ],
        data={"csrf": csrf, "ocr": "off"},
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["created"]) == 1 and len(body["errors"]) == 2
    doc_id = body["created"][0]["id"]
    assert body["created"][0]["status"] == "queued"

    d = wait_done(client, doc_id)
    assert d["status"] == "done", d
    assert d["title"].startswith("Riluzole")
    assert set(d["outputs"]) >= {"layout", "markdown", "text", "original", "report"}
    assert d["pages"] == 2 and not d["needs_ocr"]

    # library + detail pages
    r = client.get("/")
    assert doc_id in r.text and "Ready" in r.text
    r = client.get(f"/doc/{doc_id}")
    assert r.status_code == 200 and "2.3 Statistical analysis" in r.text
    r = client.get(f"/doc/{doc_id}?view=markdown")
    assert "Riluzole" in r.text
    r = client.get(f"/doc/{doc_id}/thumb.png")
    assert r.status_code == 200 and r.content.startswith(b"\x89PNG")

    # downloads
    r = client.get(f"/doc/{doc_id}/file/layout?download=1")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert r.headers["content-type"].startswith("text/plain")
    r = client.get(f"/doc/{doc_id}/file/original")
    assert r.headers["content-type"] == "application/pdf"
    assert client.get(f"/doc/{doc_id}/file/nope").status_code == 404
    assert client.get("/doc/nope/file/original").status_code == 404

    # tags + rename + search
    r = client.post(
        f"/doc/{doc_id}/tags",
        data={"csrf": csrf, "tags": "MND, registry, MND"},
        headers={"Accept": "application/json"},
    )
    assert r.json()["tags"] == ["MND", "registry"]
    r = client.get("/?tag=MND")
    assert doc_id in r.text
    r = client.post(
        f"/doc/{doc_id}/rename",
        data={"csrf": csrf, "title": "  Renamed   paper "},
        headers={"Accept": "application/json"},
    )
    assert r.json()["title"] == "Renamed paper"
    r = client.get("/search?q=riluzole registry")
    assert "Renamed paper" in r.text
    r = client.get("/search?q=zzzzqqq")
    assert "Nothing matched" in r.text
    r = client.get('/search?q=" OR 1=1 --')
    assert r.status_code == 200

    # archive copy appears shortly after "done" (it is written by a post-finish step)
    deadline = time.time() + 30
    while time.time() < deadline:
        files = [p.name for p in (tmp_path / "archive").rglob("*") if p.is_file()]
        if any(n.endswith(".layout.txt") for n in files) and "scholarly.pdf" in files:
            break
        time.sleep(0.2)
    assert any(n.endswith(".layout.txt") for n in files) and "scholarly.pdf" in files

    # duplicate upload is flagged
    r = client.post(
        "/upload",
        files=[("files", ("again.pdf", pdf, "application/pdf"))],
        data={"csrf": csrf},
        headers={"Accept": "application/json"},
    )
    dup = r.json()["created"][0]
    assert dup["duplicate_of"] == doc_id
    wait_done(client, dup["id"])

    # delete needs confirmation
    r = client.post(
        f"/doc/{doc_id}/delete", data={"csrf": csrf}, headers={"Accept": "application/json"}
    )
    assert r.status_code == 400
    r = client.post(
        f"/doc/{doc_id}/delete",
        data={"csrf": csrf, "confirm": "yes"},
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 200
    assert client.get(f"/api/docs/{doc_id}").status_code == 404
    assert client.get(f"/doc/{doc_id}", follow_redirects=False).status_code == 404


def test_scan_reports_needs_ocr_and_reprocess(client, fixtures, monkeypatch):
    csrf = sign_in(client)
    r = client.post(
        "/upload",
        files=[("files", ("scan.pdf", fixtures["scan"].read_bytes(), "application/pdf"))],
        data={"csrf": csrf, "ocr": "off"},
        headers={"Accept": "application/json"},
    )
    doc_id = r.json()["created"][0]["id"]
    d = wait_done(client, doc_id)
    assert d["needs_ocr"] is True
    r = client.get("/")
    assert "Needs OCR" in r.text
    r = client.post(
        f"/doc/{doc_id}/reprocess",
        data={"csrf": csrf, "ocr": "off"},
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 200
    d = wait_done(client, doc_id)
    assert d["status"] == "done"


def test_progress_events_stream(client, fixtures):
    csrf = sign_in(client)
    r = client.post(
        "/upload",
        files=[("files", ("s.pdf", fixtures["scholarly"].read_bytes(), "application/pdf"))],
        data={"csrf": csrf},
        headers={"Accept": "application/json"},
    )
    doc_id = r.json()["created"][0]["id"]
    seen = []
    deadline = time.time() + 120
    with client.stream("GET", "/events?max_seconds=60") as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        for line in resp.iter_lines():
            if time.time() > deadline:
                raise AssertionError("no terminal event within 120 s")
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            seen.append(ev)
            if (
                ev["type"] == "snapshot"
                and doc_id not in ev["active"]
                and not any(d["id"] == doc_id for d in ev["docs"])
            ):
                break  # finished before we subscribed; nothing more will arrive for it
            if ev.get("type") in ("done", "failed") and ev.get("id") == doc_id:
                break
    assert seen[0]["type"] == "snapshot"
    if len(seen) == 1:
        assert wait_done(client, doc_id)["status"] == "done"
        return
    types = [e["type"] for e in seen]
    assert "progress" in types and types[-1] == "done"
    stages = {e.get("stage") for e in seen if e["type"] == "progress"}
    assert "layout" in stages or "markdown" in stages
    assert all(0 <= e.get("progress", 0) <= 100 for e in seen if e["type"] != "snapshot")


def test_upload_size_limit(tmp_path, github):
    app = create_app(make_settings(tmp_path, max_upload_mb=1))
    with TestClient(app) as c:
        csrf = sign_in(c)
        big = b"%PDF-1.4\n" + b"0" * (2 * 1024 * 1024)
        r = c.post(
            "/upload",
            files=[("files", ("big.pdf", big, "application/pdf"))],
            data={"csrf": csrf},
            headers={"Accept": "application/json"},
        )
        assert r.status_code == 413


def test_html_upload_is_never_served_as_html(client, fixtures):
    csrf = sign_in(client)
    r = client.post(
        "/upload",
        files=[("files", ("article.html", fixtures["html"].read_bytes(), "text/html"))],
        data={"csrf": csrf},
        headers={"Accept": "application/json"},
    )
    doc_id = r.json()["created"][0]["id"]
    wait_done(client, doc_id)
    r = client.get(f"/doc/{doc_id}/file/original")
    assert r.headers["content-type"].startswith("text/plain")
    assert "attachment" in r.headers["content-disposition"]


def test_error_pages_render_for_browsers(client):
    sign_in(client)
    r = client.get("/doc/does-not-exist", headers={"Accept": "text/html"})
    assert r.status_code == 404 and "Back to library" in r.text
    r = client.get("/doc/does-not-exist", headers={"Accept": "application/json"})
    assert r.status_code == 404 and r.json()["error"]


# ------------------------------------------------------------------ store units
def test_fts_query_is_quoted():
    assert fts_query("riluzole registry") == '"riluzole" AND "registry"*'
    assert fts_query('a"b OR 1') == '"a" AND "b" AND "OR" AND "1"*'
    assert fts_query("") == ""
    assert fts_query("   ") == ""


def test_store_sessions_expire(tmp_path):
    st = Store(tmp_path / "s.sqlite3")
    sid, csrf = st.create_session(1, "x")
    assert st.get_session(sid, idle_seconds=3600, absolute_seconds=3600)["csrf"] == csrf
    assert st.get_session(sid, idle_seconds=-1, absolute_seconds=3600) is None
    assert (
        st.get_session(sid, idle_seconds=3600, absolute_seconds=3600) is None
    )  # deleted on expiry
    assert st.consume_state("nope") is None
    state = st.create_state("/x")
    assert st.consume_state(state) == "/x"
    assert st.consume_state(state) is None


def test_inbox_watcher_imports_stable_files(tmp_path, github, fixtures):
    from funicular.web.watcher import InboxWatcher

    settings = make_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as c:
        settings.inbox_dir.mkdir(parents=True, exist_ok=True)
        (settings.inbox_dir / "dropped.pdf").write_bytes(fixtures["scholarly"].read_bytes())
        (settings.inbox_dir / ".hidden.pdf").write_bytes(b"%PDF-")
        (settings.inbox_dir / "bad.xyz").write_bytes(b"\x00\x01")
        w = InboxWatcher(settings, app.state.store, app.state.jobs)
        w._stable = lambda p, settle=0: True  # skip the 2 s settle in tests
        assert w.scan_once() == 1
        assert not (settings.inbox_dir / "dropped.pdf").exists()  # consumed
        assert (settings.inbox_dir / "Unsupported" / "bad.xyz").exists()
        sign_in(c)
        docs = c.get("/api/docs").json()["docs"]
        assert len(docs) == 1 and docs[0]["original_name"] == "dropped.pdf"
        wait_done(c, docs[0]["id"])


def test_system_page_cancel_and_shutdown(client, fixtures, monkeypatch):
    csrf = sign_in(client)
    r = client.get("/api/system")
    assert r.status_code == 200 and r.json()["mem_total_gb"] > 0
    r = client.get("/system")
    assert r.status_code == 200 and "Stop the server" in r.text
    # cancel on an idle document is a no-op
    r = client.post(
        "/upload",
        files=[("files", ("s.pdf", fixtures["scholarly"].read_bytes(), "application/pdf"))],
        data={"csrf": csrf},
        headers={"Accept": "application/json"},
    )
    doc_id = r.json()["created"][0]["id"]
    r = client.post(
        f"/doc/{doc_id}/cancel", data={"csrf": csrf}, headers={"Accept": "application/json"}
    )
    assert r.status_code == 200 and r.json()["ok"] in (True, False)
    wait_done(client, doc_id)
    # shutdown must not actually kill the test process: stub the signal
    import os

    sent = {}
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.setdefault("sig", sig))
    r = client.post(
        "/api/system/shutdown", data={"csrf": csrf}, headers={"Accept": "application/json"}
    )
    assert r.status_code == 200 and r.json()["stopping"] is True
    time.sleep(1)
    assert sent.get("sig") is not None
    assert client.app.state.jobs.stopping is True


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for k, v in members.items():
            zf.writestr(k, v)
    return buf.getvalue()


def wait_batch(c: TestClient, bid: str, timeout: float = 180) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        b = next(b for b in c.get("/api/batches").json()["batches"] if b["id"] == bid)
        if b["status"] in ("done", "failed"):
            return b
        time.sleep(0.2)
    raise AssertionError("batch never finished")


def test_archive_upload_becomes_a_batch(client, fixtures):
    csrf = sign_in(client)
    z = _zip_bytes(
        {
            "Papers/one.pdf": fixtures["scholarly"].read_bytes(),
            "Papers/dup.pdf": fixtures["scholarly"].read_bytes(),
            "notes/n.md": b"# note",
            "junk.bin": b"\0\1",
        }
    )
    r = client.post(
        "/upload",
        files=[("files", ("bundle.zip", z, "application/zip"))],
        data={"csrf": csrf},
        headers={"Accept": "application/json"},
    )
    body = r.json()
    assert body["created"] == [] and len(body["batches"]) == 1
    bid = body["batches"][0]["id"]
    b = wait_batch(client, bid)
    assert b["status"] == "done", b
    assert b["imported"] == 2 and b["skipped"] >= 1  # duplicate pdf + junk skipped
    docs = client.get("/api/docs").json()["docs"]
    assert len(docs) == 2
    for d in docs:
        wait_done(client, d["id"])
    tags = {t for d in docs for t in d["tags"]}
    assert "archive:bundle.zip" in tags and "folder:Papers" in tags
    r = client.get("/")
    assert "Folder and archive imports" in r.text


def test_browse_and_folder_import(client, fixtures, tmp_path, monkeypatch):
    csrf = sign_in(client)
    root = tmp_path / "scan-root"
    (root / "inner").mkdir(parents=True)
    (root / "inner" / "p.pdf").write_bytes(fixtures["scholarly"].read_bytes())
    (root / "m.md").write_text("# m")
    (root / "nested.zip").write_bytes(_zip_bytes({"h.html": fixtures["html"].read_bytes()}))
    monkeypatch.setenv("FUNICULAR_BROWSE_ROOTS", str(root))
    r = client.get("/api/browse")
    assert str(root.resolve()) in r.json()["roots"]
    r = client.get(f"/api/browse?path={root}")
    names = {e["name"]: e for e in r.json()["entries"]}
    assert names["inner"]["is_dir"] and names["nested.zip"]["archive"]
    assert client.get("/api/browse?path=/etc").status_code == 403
    r = client.get(f"/browse?path={root}")
    assert r.status_code == 200 and "Import folder" in r.text
    r = client.post(
        "/import/folder",
        data={"csrf": csrf, "path": str(root), "recursive": "yes"},
        headers={"Accept": "application/json"},
    )
    bid = r.json()["batch"]
    b = wait_batch(client, bid)
    assert b["status"] == "done" and b["imported"] == 3, b
    assert (root / "inner" / "p.pdf").exists()  # originals untouched
    for d in client.get("/api/docs").json()["docs"]:
        wait_done(client, d["id"])
    # single-file import from the browser
    r = client.post(
        "/import/file",
        data={"csrf": csrf, "path": str(root / "m.md")},
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 200
    r = client.post(
        "/import/file",
        data={"csrf": csrf, "path": "/etc/hostname"},
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 403


def test_export_tar_gz(client, fixtures):
    import io
    import tarfile

    csrf = sign_in(client)
    r = client.post(
        "/upload",
        files=[("files", ("s.pdf", fixtures["scholarly"].read_bytes(), "application/pdf"))],
        data={"csrf": csrf},
        headers={"Accept": "application/json"},
    )
    doc_id = r.json()["created"][0]["id"]
    wait_done(client, doc_id)
    r = client.post("/export", data={"csrf": csrf, "ids": [doc_id], "what": "text"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/gzip"
    with tarfile.open(fileobj=io.BytesIO(r.content)) as tf:
        names = tf.getnames()
    assert any(n.endswith(".layout.txt") for n in names) and not any(
        n.endswith("s.pdf") for n in names
    )
    r = client.post("/export", data={"csrf": csrf}, headers={"Accept": "application/json"})
    assert r.status_code == 400


def test_compress_plan_page_and_run(client, fixtures):
    csrf = sign_in(client)
    r = client.post(
        "/upload",
        files=[("files", ("s.pdf", fixtures["scholarly"].read_bytes(), "application/pdf"))],
        data={"csrf": csrf},
        headers={"Accept": "application/json"},
    )
    doc_id = r.json()["created"][0]["id"]
    wait_done(client, doc_id)
    r = client.get(f"/doc/{doc_id}/plan?op=reprocess&ocr=auto")
    assert r.status_code == 200 and r.json()["estimate"]["seconds"] > 0
    r = client.get(f"/doc/{doc_id}/plan?op=compress&strength=70")
    p = r.json()
    assert p["op"] == "compress" and p["pages"] == 2 and 0 < p["ratio"] <= 1
    r = client.get(f"/doc/{doc_id}/compress?strength=70")
    assert r.status_code == 200 and "Compress now" in r.text
    r = client.post(
        f"/doc/{doc_id}/compress",
        data={"csrf": csrf, "strength": "70"},
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 200
    deadline = time.time() + 60
    while time.time() < deadline:
        d = client.get(f"/api/docs/{doc_id}").json()
        if "compressed" in d["outputs"]:
            break
        time.sleep(0.2)
    assert "compressed" in d["outputs"]
    r = client.get(f"/doc/{doc_id}/file/compressed")
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
    assert client.get("/doc/nope/plan").status_code == 404


def test_auto_tags_are_generated(client, fixtures):
    csrf = sign_in(client)
    r = client.post(
        "/upload",
        files=[("files", ("s.pdf", fixtures["scholarly"].read_bytes(), "application/pdf"))],
        data={"csrf": csrf},
        headers={"Accept": "application/json"},
    )
    doc_id = r.json()["created"][0]["id"]
    d = wait_done(client, doc_id)
    tags = [t.lower() for t in d["tags"]]
    assert "pdf" in tags and any("riluzole" in t for t in tags)
    assert not d["needs_ocr"] and "scanned" not in tags


def test_search_page_and_api_use_the_chunk_index(client, fixtures):
    csrf = sign_in(client)
    r = client.post(
        "/upload",
        files=[("files", ("s.pdf", fixtures["scholarly"].read_bytes(), "application/pdf"))],
        data={"csrf": csrf},
        headers={"Accept": "application/json"},
    )
    doc_id = r.json()["created"][0]["id"]
    wait_done(client, doc_id)
    r = client.get("/api/search?q=rilusole")  # misspelt: phonetic expansion
    body = r.json()
    assert body["hits"] and body["hits"][0]["doc_id"] == doc_id
    assert body["info"]["expansions"].get("rilusole") == ["riluzole"]
    r = client.get("/search?q=chio prognostic")
    assert r.status_code == 200 and "[Chiò]" in r.text and "page" in r.text
    r = client.get("/search?q=ALSFRS&mode=keyword")
    assert doc_id in r.text
    r = client.post(
        f"/doc/{doc_id}/delete",
        data={"csrf": csrf, "confirm": "yes"},
        headers={"Accept": "application/json"},
    )
    assert r.status_code == 200
    assert client.get("/api/search?q=riluzole").json()["hits"] == []
