"""Zotero 7 local API (read-only) and local storage.

Enable in Zotero: Settings → Advanced → "Allow other applications on this computer to
communicate with Zotero". The local API answers on http://127.0.0.1:23119/api/ with the same
JSON shapes as the Zotero Web API. Attachments are read from the local storage folder when the
API does not serve file bodies.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)
DEFAULT_BASE = "http://127.0.0.1:23119"
DEFAULT_STORAGE = Path.home() / "Zotero" / "storage"


@dataclass
class ZoteroItem:
    key: str
    item_type: str
    title: str
    creators: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    publication: str = ""
    tags: list[str] = field(default_factory=list)
    collections: list[str] = field(default_factory=list)
    attachments: list[dict] = field(default_factory=list)  # {key, filename, content_type}
    url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ZoteroLocal:
    def __init__(
        self,
        base: str = DEFAULT_BASE,
        storage: Path | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base = base.rstrip("/")
        self.storage = storage or DEFAULT_STORAGE
        self.client = httpx.Client(base_url=self.base, timeout=20, transport=transport)

    def close(self) -> None:
        self.client.close()

    def available(self) -> bool:
        try:
            r = self.client.get("/api/users/0/collections", params={"limit": 1})
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def _get(self, path: str, **params: Any) -> Any:
        r = self.client.get(path, params={"format": "json", **params})
        r.raise_for_status()
        return r.json()

    def collections(self) -> list[dict]:
        out = []
        start = 0
        while True:
            page = self._get("/api/users/0/collections", limit=100, start=start)
            for c in page:
                d = c.get("data", {})
                out.append(
                    {
                        "key": d.get("key"),
                        "name": d.get("name", ""),
                        "parent": d.get("parentCollection") or None,
                        "items": (c.get("meta") or {}).get("numItems", 0),
                    }
                )
            if len(page) < 100:
                break
            start += 100
        return out

    def items(self, collection: str | None = None, *, limit: int = 500) -> list[ZoteroItem]:
        path = (
            f"/api/users/0/collections/{collection}/items/top"
            if collection
            else "/api/users/0/items/top"
        )
        names = {c["key"]: c["name"] for c in self.collections()}
        out: list[ZoteroItem] = []
        start = 0
        while len(out) < limit:
            page = self._get(path, limit=min(100, limit - len(out)), start=start)
            for it in page:
                out.append(parse_item(it, names))
            if len(page) < 100:
                break
            start += 100
        for item in out:
            try:
                kids = self._get(f"/api/users/0/items/{item.key}/children")
            except httpx.HTTPError:
                kids = []
            for k in kids:
                d = k.get("data", {})
                if d.get("itemType") == "attachment" and d.get("contentType") == "application/pdf":
                    item.attachments.append(
                        {
                            "key": d.get("key"),
                            "filename": d.get("filename") or "",
                            "content_type": d.get("contentType"),
                            "link_mode": d.get("linkMode"),
                        }
                    )
        return out

    def attachment_path(self, attachment_key: str, filename: str) -> Path | None:
        """Where Zotero keeps the file on this Mac."""
        folder = self.storage / attachment_key
        if filename and (folder / filename).is_file():
            return folder / filename
        if folder.is_dir():
            for p in folder.iterdir():
                if p.suffix.lower() == ".pdf":
                    return p
        return None

    def fetch_attachment(self, attachment_key: str, filename: str, dest: Path) -> Path | None:
        """Copy the PDF from local storage, else ask the local API for the file body."""
        local = self.attachment_path(attachment_key, filename)
        if local:
            dest.write_bytes(local.read_bytes())
            return dest
        try:
            r = self.client.get(f"/api/users/0/items/{attachment_key}/file")
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                dest.write_bytes(r.content)
                return dest
        except httpx.HTTPError as exc:
            log.debug("zotero file fetch failed: %s", exc)
        return None


def parse_item(it: dict, collection_names: dict[str, str] | None = None) -> ZoteroItem:
    d = it.get("data", {})
    creators = []
    for c in d.get("creators") or []:
        name = c.get("lastName") or c.get("name") or ""
        if c.get("firstName"):
            name = f"{name}, {c['firstName']}"
        if name:
            creators.append(name)
    date = str(d.get("date") or "")
    year = None
    for tok in date.replace("-", " ").replace("/", " ").split():
        if tok.isdigit() and len(tok) == 4:
            year = int(tok)
            break
    names = collection_names or {}
    return ZoteroItem(
        key=d.get("key", ""),
        item_type=d.get("itemType", ""),
        title=d.get("title", ""),
        creators=creators,
        year=year,
        doi=(d.get("DOI") or "").lower() or None,
        publication=d.get("publicationTitle") or d.get("bookTitle") or "",
        tags=[t.get("tag", "") for t in d.get("tags") or [] if t.get("tag")],
        collections=[names.get(k, k) for k in d.get("collections") or []],
        url=d.get("url") or "",
    )
