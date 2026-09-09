"""Runtime settings, read once from the environment (and a local .env if present).

Every knob has a FUNICULAR_ prefixed environment variable. Nothing here is read from the
network or from user-supplied documents.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

OcrMode = Literal["off", "auto", "force"]
OcrBackend = Literal["auto", "macocr", "macocr-cli", "tesseract"]

_ICLOUD_ROOT = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs"


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"FUNICULAR_{name}", default)


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    v = _env(name)
    if v is None or not v.strip():
        return default
    return int(v)


class ExtractSettings(BaseModel):
    """Options for one extraction run. The CLI and the web app both build one of these."""

    ocr: OcrMode = "off"
    ocr_backend: OcrBackend = "auto"
    ocr_languages: list[str] = Field(default_factory=lambda: ["en-US"])
    ocr_dpi: int = 300
    # Rasterisation DPI used for the scan detector and for OCR of image-only pages.
    detect_dpi: int = 72
    # Pages with fewer than this many non-space characters are treated as text-less.
    min_chars_per_page: int = 25
    # Fraction of the page area covered by raster images above which a text-less page is a scan.
    scan_image_coverage: float = 0.35
    # Try a Ghostscript rewrite when poppler or MuPDF cannot open / fully read the file.
    repair_with_ghostscript: bool = True
    write_markdown: bool = True
    write_layout_text: bool = True
    write_plain_text: bool = True
    write_json: bool = True
    # docling ASR model for audio. "whisper_turbo" auto-selects MLX on Apple Silicon.
    asr_model: str = "whisper_turbo"
    timeout_seconds: int = 900

    @field_validator("ocr_dpi", "detect_dpi")
    @classmethod
    def _dpi_sane(cls, v: int) -> int:
        if not 36 <= v <= 600:
            raise ValueError("dpi must be between 36 and 600")
        return v


class Settings(BaseModel):
    """Process-wide configuration."""

    data_dir: Path
    inbox_dir: Path
    archive_dir: Path
    extract: ExtractSettings = Field(default_factory=ExtractSettings)

    # --- web ---
    host: str = "127.0.0.1"
    port: int = 8787
    # Public origin the browser uses, e.g. https://funicular.tail1234.ts.net. Required for
    # the OAuth redirect URI; must be https unless it is a loopback address.
    public_url: str = "http://127.0.0.1:8787"
    github_client_id: str = ""
    github_client_secret: str = ""
    # Numeric GitHub user IDs allowed to sign in. IDs never change; logins can be renamed.
    allowed_github_ids: frozenset[int] = frozenset()
    session_idle_minutes: int = 60 * 12
    session_absolute_hours: int = 24 * 7
    max_upload_mb: int = 20480
    workers: int = 2
    # Resource guard: per-job process-tree memory cap, process-count cap (fork-bomb guard),
    # and the free-memory floor below which new jobs wait.
    job_memory_cap_mb: int = 8192
    job_max_procs: int = 96
    min_free_mb: int = 1024
    # Folder / archive imports
    max_batch_files: int = 50_000
    icloud_wait_seconds: int = 900
    icloud_evict_after: bool = False
    # Search: embeddings backend (auto | sentence-transformers | model2vec | none)
    embeddings: str = "auto"
    watch_inbox: bool = False
    # Behind Tailscale Serve / Caddy: trust X-Forwarded-Proto from the loopback proxy only.
    trust_proxy: bool = False

    @property
    def pid_file(self) -> Path:
        return self.data_dir / "funicular.pid"

    @classmethod
    def from_env(cls, dotenv: Path | None = None) -> Settings:
        load_dotenv(dotenv or Path.cwd() / ".env", override=False)
        icloud = _ICLOUD_ROOT / "Funicular"
        default_root = icloud if _ICLOUD_ROOT.is_dir() else Path.home() / "Funicular"
        data_dir = Path(_env("DATA_DIR", str(Path.home() / ".funicular"))).expanduser()
        inbox = Path(_env("INBOX_DIR", str(default_root / "Inbox"))).expanduser()
        archive = Path(_env("ARCHIVE_DIR", str(default_root / "Archive"))).expanduser()
        ids = _env("ALLOWED_GITHUB_IDS", "") or ""
        allowed = frozenset(int(x) for x in ids.replace(";", ",").split(",") if x.strip())
        extract = ExtractSettings(
            ocr=_env("OCR", "off"),  # type: ignore[arg-type]
            ocr_backend=_env("OCR_BACKEND", "auto"),  # type: ignore[arg-type]
            ocr_languages=[
                s.strip() for s in (_env("OCR_LANGUAGES", "en-US") or "en-US").split(",")
            ],
            asr_model=_env("ASR_MODEL", "whisper_turbo") or "whisper_turbo",
        )
        return cls(
            data_dir=data_dir,
            inbox_dir=inbox,
            archive_dir=archive,
            extract=extract,
            host=_env("HOST", "127.0.0.1") or "127.0.0.1",
            port=_env_int("PORT", 8787),
            public_url=(_env("PUBLIC_URL", "http://127.0.0.1:8787") or "").rstrip("/"),
            github_client_id=_env("GITHUB_CLIENT_ID", "") or "",
            github_client_secret=_env("GITHUB_CLIENT_SECRET", "") or "",
            allowed_github_ids=allowed,
            session_idle_minutes=_env_int("SESSION_IDLE_MINUTES", 60 * 12),
            session_absolute_hours=_env_int("SESSION_ABSOLUTE_HOURS", 24 * 7),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 20480),
            workers=max(1, min(_env_int("WORKERS", 2), 8)),
            job_memory_cap_mb=_env_int("JOB_MEMORY_CAP_MB", 8192),
            job_max_procs=_env_int("JOB_MAX_PROCS", 96),
            min_free_mb=_env_int("MIN_FREE_MB", 1024),
            max_batch_files=_env_int("MAX_BATCH_FILES", 50_000),
            icloud_wait_seconds=_env_int("ICLOUD_WAIT_SECONDS", 900),
            icloud_evict_after=_env_bool("ICLOUD_EVICT_AFTER", False),
            embeddings=_env("EMBEDDINGS", "auto") or "auto",
            watch_inbox=_env_bool("WATCH_INBOX", False),
            trust_proxy=_env_bool("TRUST_PROXY", False),
        )
