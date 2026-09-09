from pathlib import Path

import pytest

from funicular.browse import BrowseError, allowed_roots, listing, resolve_safe
from funicular.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "d", inbox_dir=tmp_path / "inbox", archive_dir=tmp_path / "arch"
    )


def test_roots_include_home_inbox_archive_and_env(tmp_path, monkeypatch):
    (tmp_path / "inbox").mkdir()
    (tmp_path / "arch").mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    monkeypatch.setenv("FUNICULAR_BROWSE_ROOTS", str(extra))
    roots = allowed_roots(_settings(tmp_path))
    assert Path.home().resolve() in roots
    assert (tmp_path / "inbox").resolve() in roots and extra.resolve() in roots


def test_resolve_safe_blocks_escape_and_symlinks(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.txt").write_text("x")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (root / "link").symlink_to(outside)
    roots = [root.resolve()]
    assert resolve_safe(root / "ok.txt", roots) == (root / "ok.txt").resolve()
    with pytest.raises(BrowseError):
        resolve_safe(root / ".." / "outside.txt", roots)
    with pytest.raises(BrowseError):
        resolve_safe(root / "link", roots)
    with pytest.raises(BrowseError):
        resolve_safe("/etc/passwd", roots)
    with pytest.raises(BrowseError):
        resolve_safe(root / "nope", roots)


def test_listing(tmp_path, monkeypatch, fixtures):
    (tmp_path / "inbox").mkdir()
    root = tmp_path / "inbox"
    (root / "Papers").mkdir()
    (root / "a.pdf").write_bytes(fixtures["scholarly"].read_bytes())
    (root / ".hidden.pdf").write_bytes(b"%PDF-")
    (root / "Thing.app").mkdir()
    (root / "z.zip").write_bytes(b"PK\x03\x04" + b"\0" * 30)
    s = _settings(tmp_path)
    top = listing(s, None)
    assert any(e.path == str(root.resolve()) for e in top.entries)
    lst = listing(s, str(root))
    names = [(e.name, e.is_dir) for e in lst.entries]
    assert ("Papers", True) in names and ("a.pdf", False) in names
    assert all(n not in (".hidden.pdf", "Thing.app") for n, _ in names)
    pdf = next(e for e in lst.entries if e.name == "a.pdf")
    assert pdf.compatible and pdf.kind == "pdf" and not pdf.dataless
    z = next(e for e in lst.entries if e.name == "z.zip")
    assert z.archive and z.kind == "archive"
    sub = listing(s, str(root / "Papers"))
    assert sub.parent == str(root.resolve())
    with pytest.raises(BrowseError):
        listing(s, "/")
