"""Operation cost estimator: how long and how much memory a job will take, before it runs.

Per-stage per-page constants are calibrated defaults (measured on this project's fixtures on
an x86 Linux box and scaled for an M1: see ``BASELINE``). Every completed job records its real
per-page timings into ``timings.json`` under the data dir, and the estimator blends recorded
medians with the baseline, so estimates converge on the actual machine.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import ExtractSettings
from .sniff import Kind

# seconds per unit (page, image, or audio minute) and peak MB per stage
BASELINE: dict[str, tuple[float, float]] = {
    "detect": (0.02, 60),
    "layout": (0.05, 80),  # pdftotext -layout
    "markdown": (0.35, 700),  # pymupdf4llm + layout model
    "text": (0.02, 60),
    "ocr.tesseract": (2.5, 300),
    "ocr.macocr": (0.6, 250),
    "ocr.macocr-cli": (0.9, 250),
    "docling.office": (0.4, 900),  # per document, plus import ~6 s
    "docling.html": (0.3, 900),
    "docling.audio": (6.0, 2500),  # per audio minute, whisper turbo on CPU; MLX ≈ 1.0
    "docling.image": (3.0, 1500),
    "compress.pymupdf": (0.08, 200),
    "compress.ghostscript": (0.5, 300),
    "embed": (0.03, 900),  # per chunk
}
DOCLING_IMPORT_SECONDS = 6.0


@dataclass
class Estimate:
    seconds: float
    peak_mb: float
    units: int
    unit: str
    stages: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Timings:
    """Persistent per-stage per-unit timing samples."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, list[float]] = {}
        try:
            self._data = json.loads(path.read_text())
        except OSError, ValueError:
            self._data = {}

    def record(self, stage: str, seconds: float, units: int) -> None:
        if units <= 0 or seconds <= 0:
            return
        with self._lock:
            samples = self._data.setdefault(stage, [])
            samples.append(round(seconds / units, 4))
            del samples[:-50]
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self._data))
            except OSError:
                pass

    def per_unit(self, stage: str) -> float:
        base = BASELINE.get(stage, (0.5, 300))[0]
        with self._lock:
            samples = list(self._data.get(stage, []))
        if len(samples) >= 3:
            return 0.7 * statistics.median(samples) + 0.3 * base
        return base


def _ocr_stage(settings: ExtractSettings) -> str:
    backend = settings.ocr_backend
    if backend == "auto":
        import sys

        backend = "macocr" if sys.platform == "darwin" else "tesseract"
    return f"ocr.{backend}"


def estimate(
    kind: Kind | str,
    *,
    pages: int = 0,
    size_bytes: int = 0,
    audio_minutes: float = 0.0,
    ocr_pages: int | None = None,
    settings: ExtractSettings | None = None,
    timings: Timings | None = None,
    apple_silicon: bool = False,
) -> Estimate:
    settings = settings or ExtractSettings()
    kind = Kind(kind) if isinstance(kind, str) else kind
    pu = timings.per_unit if timings else (lambda s: BASELINE.get(s, (0.5, 300))[0])
    mb = lambda s: BASELINE.get(s, (0.5, 300))[1]  # noqa: E731
    stages: dict[str, float] = {}
    peak = 100.0
    if kind is Kind.PDF:
        pages = pages or max(1, size_bytes // 60_000)
        for st in ("detect", "layout", "markdown", "text"):
            stages[st] = pu(st) * pages
            peak = max(peak, mb(st))
        if settings.ocr != "off":
            n = pages if settings.ocr == "force" else (ocr_pages if ocr_pages is not None else 0)
            if n:
                st = _ocr_stage(settings)
                stages[st] = pu(st) * n
                peak = max(peak, mb(st))
        total = sum(stages.values()) + 0.3
        return Estimate(round(total, 1), peak, pages, "pages", stages)
    if kind is Kind.IMAGE:
        if settings.ocr == "off":
            return Estimate(0.2, 60, 1, "image", {}, "OCR off: nothing to extract")
        st = _ocr_stage(settings)
        stages[st] = pu(st)
        return Estimate(round(stages[st] + 0.3, 1), mb(st), 1, "image", stages)
    if kind is Kind.AUDIO or kind is Kind.VIDEO:
        minutes = audio_minutes or max(1.0, size_bytes / (1_000_000 * 1.0))  # ~1 MB per min mp3
        st = "docling.audio"
        per = pu(st) * (0.2 if apple_silicon else 1.0)
        stages[st] = per * minutes
        total = stages[st] + DOCLING_IMPORT_SECONDS
        note = "MLX Whisper on Apple Silicon" if apple_silicon else "CPU Whisper"
        return Estimate(round(total, 1), mb(st), int(minutes), "minutes", stages, note)
    if kind in (Kind.OFFICE, Kind.EPUB):
        stages["docling.office"] = pu("docling.office") * max(1, pages or 1)
        return Estimate(
            round(stages["docling.office"] + DOCLING_IMPORT_SECONDS, 1),
            mb("docling.office"),
            max(1, pages or 1),
            "document",
            stages,
        )
    if kind is Kind.HTML:
        stages["docling.html"] = pu("docling.html")
        return Estimate(
            round(stages["docling.html"] + DOCLING_IMPORT_SECONDS, 1),
            mb("docling.html"),
            1,
            "page",
            stages,
        )
    if kind is Kind.TEXT:
        return Estimate(0.1, 50, 1, "file", {})
    return Estimate(0.0, 0, 0, "n/a", {}, "unsupported type")


def pdf_page_count(path: Path) -> int:
    try:
        import pymupdf

        with pymupdf.open(path) as doc:
            return doc.page_count
    except Exception:  # noqa: BLE001
        return 0


def audio_minutes(path: Path) -> float:
    """Duration via ffprobe when available, else a size-based guess."""
    import shutil
    import subprocess

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            out = subprocess.run(  # noqa: S603
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            return float(out.stdout.strip()) / 60.0
        except ValueError, OSError, subprocess.TimeoutExpired:
            pass
    return max(0.5, path.stat().st_size / 1_000_000)


class StageClock:
    """Context helper: time a stage and record it against a unit count."""

    def __init__(self, timings: Timings | None, stage: str, units: int) -> None:
        self.timings, self.stage, self.units = timings, stage, units
        self.t0 = time.monotonic()

    def done(self) -> float:
        dt = time.monotonic() - self.t0
        if self.timings:
            self.timings.record(self.stage, dt, self.units)
        return dt
