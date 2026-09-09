from pathlib import Path

from funicular import models


def test_status_reports_missing_models_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCLING_ARTIFACTS_PATH", str(tmp_path / "docling"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    rep = models.status("whisper_turbo", apple_silicon=True)
    names = [m.name for m in rep.models]
    assert any("MLX" in n and "turbo" in n for n in names)
    assert all(not m.present for m in rep.models)
    rep = models.status("whisper_tiny", apple_silicon=False)
    assert any("native" in m.name and "tiny" in m.name for m in rep.models)


def test_status_detects_present_models(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCLING_ARTIFACTS_PATH", str(tmp_path / "docling"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    (tmp_path / "docling" / "ds4sd--docling-models").mkdir(parents=True)
    snap = tmp_path / "hf" / "models--mlx-community--whisper-large-v3-turbo" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (tmp_path / "xdg" / "whisper").mkdir(parents=True)
    (tmp_path / "xdg" / "whisper" / "tiny.pt").write_bytes(b"x")
    rep = models.status("whisper_turbo", apple_silicon=True)
    by = {m.name: m for m in rep.models}
    assert by["docling tableformer"].present
    assert any(m.present for n, m in by.items() if "MLX" in n)
    rep = models.status("whisper_tiny", apple_silicon=False)
    assert any(m.present for n, m in by.items() if "native" in n) or any(
        m.present for m in rep.models if "native" in m.name
    )


def test_fetch_collects_errors_without_network(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCLING_ARTIFACTS_PATH", str(tmp_path / "docling"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    rep = models.fetch(docling=False, asr_model="whisper_tiny", embeddings=True, apple_silicon=True)
    # offline: the MLX snapshot download must fail loudly, not silently
    assert any("whisper" in e for e in rep.errors)
    assert isinstance(rep.to_dict()["models"], list)
    assert Path(models.hf_cache_dir()) == tmp_path / "hf"
    monkeypatch.delenv("HF_HUB_CACHE")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home"))
    assert Path(models.hf_cache_dir()) == tmp_path / "home" / "hub"
