import os
import sys
import types
from pathlib import Path

import pytest

from funicular import icloud


def test_placeholder_target():
    assert icloud.placeholder_target(Path("/x/.paper.pdf.icloud")) == Path("/x/paper.pdf")
    assert icloud.placeholder_target(Path("/x/paper.pdf")) is None


def test_is_dataless_false_off_macos(tmp_path):
    p = tmp_path / "a"
    p.write_text("x")
    if sys.platform != "darwin":
        assert icloud.is_dataless(p) is False


def test_is_dataless_reads_flag(tmp_path, monkeypatch):
    p = tmp_path / "a"
    p.write_text("x")
    monkeypatch.setattr(icloud.sys, "platform", "darwin")
    real = os.stat(p)
    fake = types.SimpleNamespace(**{k: getattr(real, k) for k in dir(real) if k.startswith("st_")})
    fake.st_flags = icloud.SF_DATALESS
    monkeypatch.setattr(icloud.os, "stat", lambda *a, **k: fake)
    assert icloud.is_dataless(p) is True
    assert icloud.status(p) == "cloud-only"


def test_ensure_local_returns_immediately_for_local_files(tmp_path):
    p = tmp_path / "a.pdf"
    p.write_bytes(b"%PDF-")
    m = icloud.ensure_local(p)
    assert m.path == p and not m.was_dataless and m.waited == 0


def test_ensure_local_waits_for_placeholder(tmp_path, monkeypatch):
    target = tmp_path / "paper.pdf"
    placeholder = tmp_path / ".paper.pdf.icloud"
    placeholder.write_text("{}")
    calls = []

    def fake_download(path):
        calls.append(path)
        target.write_bytes(b"%PDF-")  # "iCloud" delivers the file
        return True

    monkeypatch.setattr(icloud, "request_download", fake_download)
    m = icloud.ensure_local(placeholder, timeout=5, poll=0.01)
    assert m.path == target and m.was_dataless and calls == [target]
    with pytest.raises(TimeoutError):
        monkeypatch.setattr(icloud, "request_download", lambda p: False)
        icloud.ensure_local(tmp_path / ".missing.pdf.icloud", timeout=0.05, poll=0.01)


def test_evict_refuses_non_icloud_paths(tmp_path):
    assert icloud.evict(tmp_path / "x") is False
    assert (
        icloud.in_icloud(Path("/Users/me/Library/Mobile Documents/com~apple~CloudDocs/x")) is True
    )
