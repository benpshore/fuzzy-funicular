import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from funicular.archives import (
    ArchiveError,
    Limits,
    archive_kind,
    build_tar_gz,
    count_compatible,
    extract_archive,
    is_archive,
    iter_compatible,
)
from funicular.sniff import Kind


def _zip(path: Path, members: dict[str, bytes], **kw) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def _tar(path: Path, members: dict[str, bytes], mode="w:gz") -> Path:
    with tarfile.open(path, mode) as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_archive_kind_by_magic(tmp_path, fixtures):
    z = _zip(tmp_path / "a.zip", {"x.txt": b"hi"})
    t = _tar(tmp_path / "a.tgz", {"x.txt": b"hi"})
    assert archive_kind(z) == "zip" and archive_kind(t) == "tar"
    assert is_archive(z) and is_archive(t)
    assert not is_archive(fixtures["scholarly"])
    if "docx" in fixtures:
        assert archive_kind(fixtures["docx"]) == "zip"  # container, but never unpacked (see below)
    plain = tmp_path / "plain.tar"
    _tar(plain, {"x.txt": b"hi"}, mode="w:")
    assert archive_kind(plain) == "tar"
    assert archive_kind(tmp_path / "missing.zip") is None


def test_extract_zip_with_nested_tar_and_junk(tmp_path, fixtures):
    inner = _tar(tmp_path / "inner.tar.gz", {"deep/paper.pdf": fixtures["scholarly"].read_bytes()})
    z = _zip(
        tmp_path / "outer.zip",
        {
            "docs/notes.md": b"# hi",
            "__MACOSX/._notes.md": b"junk",
            "docs/.DS_Store": b"junk",
            "../escape.txt": b"evil",
            "/abs.txt": b"evil",
            "bundle/inner.tar.gz": inner.read_bytes(),
        },
    )
    report = extract_archive(z, tmp_path / "out")
    rels = sorted(f.relpath for f in report.files)
    assert rels == ["bundle/inner.tar.gz/deep/paper.pdf", "docs/notes.md"]
    assert report.nested == 1
    assert any("escape" in s for s in report.skipped) and any("abs" in s for s in report.skipped)
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "out" / "bundle" / "inner.tar.gz").exists()  # nested archive removed


def test_office_zip_container_is_not_unpacked(tmp_path, fixtures):
    if "docx" not in fixtures:
        pytest.skip("no docx fixture")
    z = _zip(tmp_path / "o.zip", {"summary.docx": fixtures["docx"].read_bytes()})
    report = extract_archive(z, tmp_path / "out")
    assert [f.relpath for f in report.files] == ["summary.docx"]


def test_symlink_members_are_skipped(tmp_path):
    t = tmp_path / "links.tar"
    with tarfile.open(t, "w") as tf:
        link = tarfile.TarInfo("evil")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tf.addfile(link)
        ok = tarfile.TarInfo("fine.txt")
        ok.size = 2
        tf.addfile(ok, io.BytesIO(b"ok"))
    report = extract_archive(t, tmp_path / "out")
    assert [f.relpath for f in report.files] == ["fine.txt"]
    assert any("link" in s for s in report.skipped)


def test_total_budget_and_member_cap(tmp_path):
    z = _zip(tmp_path / "big.zip", {"a.bin": b"x" * 5000, "b.bin": b"y" * 5000})
    with pytest.raises(ArchiveError):
        extract_archive(z, tmp_path / "o1", Limits(max_total_bytes=6000))
    report = extract_archive(z, tmp_path / "o2", Limits(max_member_bytes=4000))
    assert report.files == [] and len(report.skipped) == 2


def test_zip_bomb_ratio_is_rejected(tmp_path):
    z = _zip(tmp_path / "bomb.zip", {"zeros.bin": b"\0" * 5_000_000})
    report = extract_archive(z, tmp_path / "o", Limits(max_ratio=100))
    assert report.files == [] and any("ratio" in s for s in report.skipped)


def test_depth_limit(tmp_path):
    lvl2 = _tar(tmp_path / "l2.tgz", {"x.txt": b"deep"})
    lvl1 = _zip(tmp_path / "l1.zip", {"l2.tgz": lvl2.read_bytes()})
    top = _zip(tmp_path / "top.zip", {"l1.zip": lvl1.read_bytes()})
    report = extract_archive(top, tmp_path / "o", Limits(max_depth=2))
    assert report.files == [] and any("too deep" in s for s in report.skipped)
    report = extract_archive(top, tmp_path / "o2", Limits(max_depth=5))
    assert [f.relpath for f in report.files] == ["l1.zip/l2.tgz/x.txt"]


def test_gzip_single_file(tmp_path, fixtures):
    import gzip

    g = tmp_path / "paper.pdf.gz"
    g.write_bytes(gzip.compress(fixtures["scholarly"].read_bytes()))
    assert archive_kind(g) == "gzip"
    report = extract_archive(g, tmp_path / "o")
    assert report.files[0].path.name == "paper.pdf" and report.files[0].size > 1000


def test_iter_compatible_and_count(tmp_path, fixtures):
    (tmp_path / "sub" / ".hidden").mkdir(parents=True)
    (tmp_path / "sub" / "a.pdf").write_bytes(fixtures["scholarly"].read_bytes())
    (tmp_path / "sub" / ".hidden" / "b.pdf").write_bytes(fixtures["scholarly"].read_bytes())
    (tmp_path / "junk.bin").write_bytes(b"\0\1\2")
    (tmp_path / "empty.pdf").write_bytes(b"")
    (tmp_path / "Thing.app").mkdir()
    (tmp_path / "Thing.app" / "c.pdf").write_bytes(fixtures["scholarly"].read_bytes())
    _zip(tmp_path / "z.zip", {"n.md": b"# n"})
    found = list(iter_compatible(tmp_path))
    rel = sorted(f.relpath for f in found)
    assert rel == ["sub/a.pdf", "z.zip"]
    kinds = {f.relpath: f.kind for f in found}
    assert kinds["sub/a.pdf"] is Kind.PDF and kinds["z.zip"] is Kind.ARCHIVE
    assert count_compatible(tmp_path) == {"documents": 1, "archives": 1, "capped": False}
    assert [f.relpath for f in iter_compatible(tmp_path, recursive=False)] == ["z.zip"]


def test_build_tar_gz(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("A")
    out = build_tar_gz([(a, "folder/a.txt"), (tmp_path / "missing", "x")], tmp_path / "e.tgz")
    with tarfile.open(out) as tf:
        assert tf.getnames() == ["folder/a.txt"]
