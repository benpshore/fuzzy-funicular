"""Sentence embeddings for semantic search.

Backends, chosen automatically:
* sentence-transformers (BAAI/bge-small-en-v1.5) on MPS / CUDA / CPU — best quality
* model2vec (minishlab/potion-base-8M) — static embeddings, no torch, instant on any CPU
* none — FTS-only search still works
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

ST_MODEL = os.environ.get("FUNICULAR_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
M2V_MODEL = os.environ.get("FUNICULAR_M2V_MODEL", "minishlab/potion-base-8M")


@dataclass
class Embedder:
    name: str
    dim: int
    _encode: object

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = np.asarray(self._encode(texts), dtype=np.float32)  # type: ignore[operator]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


_LOCK = threading.Lock()
_CACHE: dict[str, Embedder | None] = {}


def available_backends() -> list[str]:
    out = []
    for mod, name in (
        ("sentence_transformers", "sentence-transformers"),
        ("model2vec", "model2vec"),
    ):
        try:
            __import__(mod)
            out.append(name)
        except Exception as exc:  # noqa: BLE001
            log.debug("%s unavailable: %s", name, exc)
    return out


def get_embedder(preference: str = "auto") -> Embedder | None:
    """Load (once) the best available embedder. Returns None when nothing is installed or the
    model files are missing and cannot be downloaded."""
    key = preference
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        emb = _load(preference)
        _CACHE[key] = emb
        return emb


def _load(preference: str) -> Embedder | None:
    order = {
        "auto": ["sentence-transformers", "model2vec"],
        "sentence-transformers": ["sentence-transformers"],
        "model2vec": ["model2vec"],
        "none": [],
    }.get(preference, ["sentence-transformers", "model2vec"])
    for backend in order:
        try:
            if backend == "sentence-transformers":
                from sentence_transformers import SentenceTransformer

                from .resources import preferred_torch_device

                model = SentenceTransformer(ST_MODEL, device=preferred_torch_device())
                dim = int(model.get_sentence_embedding_dimension() or 384)

                def enc(texts, _m=model):
                    return _m.encode(
                        texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
                    )

                return Embedder(f"sentence-transformers:{ST_MODEL}", dim, enc)
            if backend == "model2vec":
                from model2vec import StaticModel

                model = StaticModel.from_pretrained(M2V_MODEL)
                probe = model.encode(["probe"], use_multiprocessing=False)
                dim = int(np.asarray(probe).shape[1])

                def enc2(texts, _m=model):
                    return _m.encode(list(texts), use_multiprocessing=False)

                return Embedder(f"model2vec:{M2V_MODEL}", dim, enc2)
        except Exception as exc:  # noqa: BLE001 - missing package / no network / no model
            log.info("embedding backend %s unavailable: %s", backend, exc)
            continue
    return None


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float16).tobytes()


def from_blob(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float16).astype(np.float32).reshape(-1, dim)
