"""Pre-fetch and inspect the machine-learning models the optional engines need, so the first
real document never stalls on a download (and the Mac can work offline afterwards).

* docling: layout, TableFormer (tables), code/formula, picture classifier, RapidOCR
* Whisper: openai-whisper weights (CPU/CUDA) or the MLX repo (Apple Silicon)
* embeddings (semantic search): sentence-transformers or model2vec model
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

WHISPER_MLX_REPOS = {
    "whisper_tiny": "mlx-community/whisper-tiny-mlx",
    "whisper_base": "mlx-community/whisper-base-mlx",
    "whisper_small": "mlx-community/whisper-small-mlx",
    "whisper_medium": "mlx-community/whisper-medium-mlx",
    "whisper_large": "mlx-community/whisper-large-v3-mlx",
    "whisper_turbo": "mlx-community/whisper-large-v3-turbo",
}
WHISPER_NATIVE_NAMES = {
    "whisper_tiny": "tiny",
    "whisper_base": "base",
    "whisper_small": "small",
    "whisper_medium": "medium",
    "whisper_large": "large-v3",
    "whisper_turbo": "turbo",
}
EMBEDDING_MODELS = {
    "sentence-transformers": "BAAI/bge-small-en-v1.5",
    "model2vec": "minishlab/potion-base-8M",
}


@dataclass
class ModelStatus:
    name: str
    present: bool
    path: str = ""
    note: str = ""


@dataclass
class Report:
    models: list[ModelStatus] = field(default_factory=list)
    fetched: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "models": [asdict(m) for m in self.models],
            "fetched": self.fetched,
            "errors": self.errors,
        }


def docling_artifacts_dir() -> Path:
    env = os.environ.get("DOCLING_ARTIFACTS_PATH")
    if env:
        return Path(env).expanduser()
    try:
        from docling.datamodel.settings import settings as dl_settings

        return Path(dl_settings.cache_dir) / "models"
    except Exception:  # noqa: BLE001
        return Path.home() / ".cache" / "docling" / "models"


def hf_cache_dir() -> Path:
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return Path(hub).expanduser()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_present(repo: str) -> Path | None:
    d = hf_cache_dir() / ("models--" + repo.replace("/", "--"))
    snap = d / "snapshots"
    if snap.is_dir() and any(snap.iterdir()):
        return d
    return None


def _whisper_native_present(name: str) -> Path | None:
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper"
    for f in root.glob(f"{name}*.pt"):
        return f
    return None


def status(asr_model: str = "whisper_turbo", *, apple_silicon: bool | None = None) -> Report:
    from .resources import gpu_info

    apple = gpu_info()["apple_silicon"] if apple_silicon is None else apple_silicon
    rep = Report()
    art = docling_artifacts_dir()
    docling_parts = {
        "docling layout": [
            "ds4sd--docling-layout-heron",
            "ds4sd--docling-models",
            "docling-layout-heron",
        ],
        "docling tableformer": ["ds4sd--docling-models", "docling-models"],
        "docling code/formula": ["ds4sd--CodeFormula", "CodeFormula"],
        "docling picture classifier": [
            "ds4sd--DocumentFigureClassifier",
            "DocumentFigureClassifier",
        ],
        "rapidocr": ["RapidOCR", "rapidocr"],
    }
    for label, names in docling_parts.items():
        hit = next((art / n for n in names if (art / n).exists()), None)
        if hit is None:
            hit = next((p for n in names for p in art.glob(f"*{n.split('--')[-1]}*")), None)
        rep.models.append(ModelStatus(label, hit is not None, str(hit or art)))
    key = asr_model.lower().replace("-", "_")
    base = key.replace("_mlx", "").replace("_native", "").replace("_s2t", "")
    if apple and not key.endswith(("_native", "_s2t")):
        repo = WHISPER_MLX_REPOS.get(base, "")
        hit = _hf_present(repo) if repo else None
        rep.models.append(ModelStatus(f"whisper (MLX) {repo}", hit is not None, str(hit or "")))
    else:
        nm = WHISPER_NATIVE_NAMES.get(base, base)
        hit = _whisper_native_present(nm)
        rep.models.append(ModelStatus(f"whisper (native) {nm}", hit is not None, str(hit or "")))
    for backend, repo in EMBEDDING_MODELS.items():
        hit = _hf_present(repo)
        rep.models.append(
            ModelStatus(f"embeddings ({backend}) {repo}", hit is not None, str(hit or ""))
        )
    return rep


def fetch(
    *,
    docling: bool = True,
    asr_model: str | None = "whisper_turbo",
    embeddings: bool = True,
    apple_silicon: bool | None = None,
    progress: bool = False,
) -> Report:
    """Download everything that is missing. Safe to re-run; already-present files are kept."""
    from .resources import gpu_info

    apple = gpu_info()["apple_silicon"] if apple_silicon is None else apple_silicon
    rep = Report()
    if docling:
        try:
            from docling.utils.model_downloader import download_models

            out = download_models(
                output_dir=docling_artifacts_dir(),
                progress=progress,
                with_layout=True,
                with_tableformer=True,
                with_code_formula=True,
                with_picture_classifier=True,
                with_rapidocr=True,
            )
            rep.fetched.append(f"docling models -> {out}")
        except Exception as exc:  # noqa: BLE001
            rep.errors.append(f"docling: {exc}")
    if asr_model:
        key = asr_model.lower().replace("-", "_")
        base = key.replace("_mlx", "").replace("_native", "").replace("_s2t", "")
        try:
            if apple and not key.endswith(("_native", "_s2t")):
                from huggingface_hub import snapshot_download

                repo = WHISPER_MLX_REPOS[base]
                snapshot_download(repo)
                rep.fetched.append(f"whisper MLX {repo}")
            else:
                import whisper  # openai-whisper

                whisper.load_model(WHISPER_NATIVE_NAMES.get(base, base), device="cpu")
                rep.fetched.append(f"whisper native {WHISPER_NATIVE_NAMES.get(base, base)}")
        except Exception as exc:  # noqa: BLE001
            rep.errors.append(f"whisper: {exc}")
    if embeddings:
        for backend, repo in EMBEDDING_MODELS.items():
            try:
                __import__(
                    "sentence_transformers" if backend == "sentence-transformers" else "model2vec"
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("%s not installed, skipping %s: %s", backend, repo, exc)
                continue
            try:
                from huggingface_hub import snapshot_download

                snapshot_download(repo)
                rep.fetched.append(f"embeddings {repo}")
            except Exception as exc:  # noqa: BLE001
                rep.errors.append(f"embeddings {repo}: {exc}")
    rep.models = status(asr_model or "whisper_turbo", apple_silicon=apple).models
    return rep
