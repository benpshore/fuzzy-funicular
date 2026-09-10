import json

import httpx
import pytest

from funicular import feeds as feedsmod
from funicular import zotero as zmod
from funicular.scholar import clients

RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>J</title>
<item><title>Riluzole in MND: a registry study</title><link>https://doi.org/10.1000/abc.1</link>
<guid>https://doi.org/10.1000/abc.1</guid><pubDate>Mon, 01 Sep 2026 00:00:00 GMT</pubDate>
<description><![CDATA[<p>Registry <b>cohort</b> of riluzole.</p>]]></description></item>
<item><title>Preprint on ALS</title><link>https://arxiv.org/abs/2609.01234v1</link><guid>arx-1</guid></item>
</channel></rss>"""


def test_parse_feed_text_extracts_ids_and_strips_html():
    entries = feedsmod.parse_feed_text(RSS)
    assert len(entries) == 2
    assert (
        entries[0].doi == "10.1000/abc.1"
        and "cohort" in entries[0].summary
        and "<b>" not in entries[0].summary
    )
    assert entries[1].arxiv == "2609.01234v1" and entries[1].guid == "arx-1"
    assert feedsmod.parse_feed_text("not a feed") == []
    assert "search_query=cat%3Aq-bio" in feedsmod.arxiv_query_url("cat:q-bio")


@pytest.fixture
def poller(tmp_path):
    def handler(req: httpx.Request) -> httpx.Response:
        u = req.url
        if u.host == "feed.example":
            return httpx.Response(200, text=RSS)
        if u.host == "arxiv.org" and u.path.startswith("/pdf/"):
            return httpx.Response(200, content=b"%PDF-1.4 fake pdf bytes")
        if u.host == "api.unpaywall.org":
            return httpx.Response(
                200, json={"best_oa_location": {"url_for_pdf": "https://oa.example/paper.pdf"}}
            )
        if u.host == "oa.example":
            return httpx.Response(200, content=b"<html>not a pdf</html>")
        if u.host == "eutils.ncbi.nlm.nih.gov" and "esearch" in u.path:
            return httpx.Response(200, json={"esearchresult": {"idlist": ["111"]}})
        if u.host == "eutils.ncbi.nlm.nih.gov":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "uids": ["111"],
                        "111": {
                            "uid": "111",
                            "title": "PubMed hit.",
                            "source": "J",
                            "pubdate": "2026 Jan",
                            "authors": [],
                            "articleids": [{"idtype": "doi", "value": "10.1000/pm.1"}],
                        },
                    }
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    fetcher = clients.Fetcher(transport=transport)
    fetcher.MIN_INTERVAL = {}
    store = feedsmod.FeedStore(tmp_path / "feeds.sqlite3")
    p = feedsmod.Poller(store, fetcher, transport=transport)
    yield p, store
    p.close()
    fetcher.close()


def test_feed_store_and_polling(poller):
    p, store = poller
    fid = store.add("rss", "Journal", "https://feed.example/rss")
    assert store.list()[0]["name"] == "Journal"
    assert p.poll(fid) == 2
    assert p.poll(fid) == 0  # idempotent
    entries = store.entries(fid)
    assert {e["status"] for e in entries} == {"new"}
    pm = store.add("pubmed", "PM", "riluzole[tiab]")
    assert p.poll(pm) == 1 and store.entries(pm)[0]["pmid"] == "111"
    bad = store.add("rss", "Bad", "https://nope.example/x")
    assert p.poll(bad) == 0 and store.get(bad)["last_status"].startswith("error")
    with pytest.raises(ValueError):
        store.add("mastodon", "x", "y")
    store.remove(bad)
    assert len(store.list()) == 2


def test_find_pdf_and_download(poller, tmp_path):
    p, store = poller
    fid = store.add("rss", "Journal", "https://feed.example/rss")
    p.poll(fid)
    by_title = {e["title"]: e for e in store.entries(fid)}
    arx = by_title["Preprint on ALS"]
    assert p.find_pdf_url(arx) == "https://arxiv.org/pdf/2609.01234v1"
    out = p.download_pdf("https://arxiv.org/pdf/2609.01234v1", tmp_path / "a.pdf")
    assert out.read_bytes().startswith(b"%PDF")
    doi_entry = by_title["Riluzole in MND: a registry study"]
    assert p.find_pdf_url(doi_entry) == "https://oa.example/paper.pdf"
    with pytest.raises(ValueError):
        p.download_pdf("https://oa.example/paper.pdf", tmp_path / "b.pdf")
    assert not (tmp_path / "b.pdf").exists()
    with pytest.raises(ValueError):
        p.download_pdf("https://arxiv.org/pdf/2609.01234v1", tmp_path / "c.pdf", max_bytes=5)


# ---------------------------------------------------------------- zotero
ZOTERO_COLLECTIONS = [{"data": {"key": "C1", "name": "MND"}, "meta": {"numItems": 1}}]
ZOTERO_ITEMS = [
    {
        "data": {
            "key": "I1",
            "itemType": "journalArticle",
            "title": "Riluzole registry",
            "creators": [{"lastName": "Shore", "firstName": "Ben"}],
            "date": "2026-03-01",
            "DOI": "10.1000/ABC.1",
            "publicationTitle": "J Neuro",
            "tags": [{"tag": "riluzole"}],
            "collections": ["C1"],
        }
    }
]
ZOTERO_CHILDREN = [
    {
        "data": {
            "key": "A1",
            "itemType": "attachment",
            "contentType": "application/pdf",
            "filename": "paper.pdf",
            "linkMode": "imported_file",
        }
    }
]


def zotero_transport(storage_has_file: bool):
    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/api/users/0/collections":
            return httpx.Response(200, json=ZOTERO_COLLECTIONS)
        if path.endswith("/items/top"):
            return httpx.Response(200, json=ZOTERO_ITEMS)
        if path == "/api/users/0/items/I1/children":
            return httpx.Response(200, json=ZOTERO_CHILDREN)
        if path == "/api/users/0/items/A1/file":
            return httpx.Response(200, content=b"%PDF-1.4 from api")
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_zotero_items_and_attachment(tmp_path):
    z = zmod.ZoteroLocal(storage=tmp_path / "storage", transport=zotero_transport(False))
    assert z.available()
    cols = z.collections()
    assert cols[0]["name"] == "MND"
    items = z.items("C1")
    it = items[0]
    assert it.doi == "10.1000/abc.1" and it.year == 2026 and it.creators == ["Shore, Ben"]
    assert it.collections == ["MND"] and it.tags == ["riluzole"]
    assert it.attachments[0]["key"] == "A1"
    got = z.fetch_attachment("A1", "paper.pdf", tmp_path / "out.pdf")
    assert got and got.read_bytes().startswith(b"%PDF-1.4 from api")
    (tmp_path / "storage" / "A1").mkdir(parents=True)
    (tmp_path / "storage" / "A1" / "paper.pdf").write_bytes(b"%PDF-1.4 local")
    got = z.fetch_attachment("A1", "paper.pdf", tmp_path / "out2.pdf")
    assert got.read_bytes() == b"%PDF-1.4 local"
    z.close()
    unreachable = zmod.ZoteroLocal(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    assert not unreachable.available()


# ---------------------------------------------------------------- mcp
def test_mcp_server_tools(tmp_path):
    pytest.importorskip("mcp")
    from funicular.config import Settings
    from funicular.mcp_server import build_server

    settings = Settings(
        data_dir=tmp_path / "d",
        inbox_dir=tmp_path / "i",
        archive_dir=tmp_path / "a",
        embeddings="none",
    )
    server = build_server(settings)
    store, index = server._funicular["store"], server._funicular["index"]
    doc = store.create_document(
        title="Riluzole paper",
        original_name="p.pdf",
        kind="pdf",
        size=1,
        sha256="x",
        dir=tmp_path / "d" / "docs" / "x",
    )
    (tmp_path / "d" / "docs" / "x").mkdir(parents=True)
    (tmp_path / "d" / "docs" / "x" / "document.md").write_text(
        "# Riluzole\n\nRiluzole improves survival in MND."
    )
    (tmp_path / "d" / "docs" / "x" / "summary.json").write_text(
        json.dumps({"card": {"one_line": "ok"}})
    )
    store.finish(
        doc.id,
        title=None,
        pages=1,
        needs_ocr=False,
        ocr_used=False,
        preview="",
        warnings=[],
        outputs={"markdown": "document.md"},
        body_text="Riluzole improves survival in MND.",
    )
    index.index_document(
        doc.id, "Riluzole paper", "Riluzole improves survival in MND.", embed=False
    )
    tools = {t.name: t for t in server._tool_manager.list_tools()}
    assert {
        "search",
        "list_documents",
        "get_document",
        "get_summary",
        "get_scholar",
        "get_references",
        "ask",
    } <= set(tools)
    hits = tools["search"].fn("riluzole")
    assert hits and hits[0]["doc_id"] == doc.id
    got = tools["get_document"].fn(doc.id, "markdown", 10)
    assert got["text"].startswith("# Rilu") and "truncated" in got["text"]
    assert tools["get_summary"].fn(doc.id)["card"]["one_line"] == "ok"
    assert tools["get_scholar"].fn(doc.id)["available"] is False
    assert tools["get_document"].fn("nope", "text", 0)["error"] == "not found"
    assert tools["list_documents"].fn(10, None)[0]["id"] == doc.id


def test_in_process_feed_poll_thread(tmp_path):
    """FEEDS_POLL_MINUTES>0 polls on a background thread that survives errors and stops cleanly."""
    import asyncio
    import time

    from funicular.web.jobs import JobManager
    from funicular.web.store import Store
    from test_web import make_settings

    settings = make_settings(tmp_path, feeds_poll_minutes=1)
    store = Store(settings.data_dir / "funicular.sqlite3")
    jm = JobManager(settings, store)
    calls = []

    def fake_poll(feed_id=None):
        calls.append(time.monotonic())
        if len(calls) == 1:
            raise RuntimeError("network down")
        return {1: 2}

    jm.poll_feeds = fake_poll
    jm._poll_interval = 0.02
    loop = asyncio.new_event_loop()
    try:
        jm.start(loop)
        deadline = time.monotonic() + 5
        while len(calls) < 3 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(calls) >= 3, "poller did not keep running after an error"
        jm.stop()
        jm._poller.join(timeout=2)
        assert not jm._poller.is_alive()
    finally:
        loop.close()

    # Off by default: no thread is started.
    jm2 = JobManager(make_settings(tmp_path / "b"), Store(tmp_path / "b" / "f.sqlite3"))
    jm2.start(loop)
    assert jm2._poller is None
    jm2.stop()
