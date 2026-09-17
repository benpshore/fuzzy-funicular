"""Importer file-claim tests: two concurrent importers (the inbox watcher, a manual folder
import, a re-triggered feed/Zotero stage) must never touch the same underlying file at once.
See FileBusy / _FileClaim in funicular/web/importer.py."""

from __future__ import annotations

import threading
import time

from funicular.web.importer import FileBusy, _FileClaim


def test_file_claim_blocks_concurrent_import(tmp_path):
    target = tmp_path / "paper.pdf"
    target.write_bytes(b"%PDF-1.4 fake")

    results: list[str] = []

    def worker() -> None:
        try:
            with _FileClaim(target):
                time.sleep(0.2)  # hold the claim long enough to force real contention
                results.append("ok")
        except FileBusy:
            results.append("busy")

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    time.sleep(0.05)  # let t1 take the claim before t2 tries
    t2.start()
    t1.join()
    t2.join()

    assert sorted(results) == ["busy", "ok"]


def test_file_claim_released_after_context_exits(tmp_path):
    target = tmp_path / "paper.pdf"
    target.write_bytes(b"%PDF-1.4 fake")

    with _FileClaim(target):
        pass
    # Claim released on exit; a second, sequential claim on the same file must succeed.
    with _FileClaim(target):
        pass


def test_file_claim_released_on_exception(tmp_path):
    target = tmp_path / "paper.pdf"
    target.write_bytes(b"%PDF-1.4 fake")

    try:
        with _FileClaim(target):
            raise ValueError("simulated failure mid-import")
    except ValueError:
        pass
    # A failed import must not permanently poison the file: the claim releases on any
    # exception, not just clean exit.
    with _FileClaim(target):
        pass


def test_file_claim_keys_by_inode_not_path(tmp_path):
    real = tmp_path / "paper.pdf"
    real.write_bytes(b"%PDF-1.4 fake")
    alias = tmp_path / "alias.pdf"
    alias.symlink_to(real)

    with _FileClaim(real):
        try:
            with _FileClaim(alias):
                raise AssertionError("expected FileBusy: alias resolves to the same inode")
        except FileBusy:
            pass
