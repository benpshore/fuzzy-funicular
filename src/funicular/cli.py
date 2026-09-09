"""Command line entry point: `funicular`."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import __version__
from .config import ExtractSettings, Settings

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Layout-faithful text extraction for scholarly and difficult documents.",
)
console = Console()
err = Console(stderr=True)


def _version(value: bool) -> None:
    if value:
        console.print(f"funicular {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version")
    ] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)


@app.command()
def doctor() -> None:
    """Check Python, packages, poppler, ghostscript, OCR and ASR availability."""
    from .doctor import run_checks

    table = Table(title="funicular doctor", show_lines=False)
    table.add_column("component")
    table.add_column("status")
    table.add_column("detail")
    table.add_column("fix")
    bad = 0
    for c in run_checks():
        table.add_row(
            c.name, "[green]ok" if c.ok else "[red]missing", c.detail, "" if c.ok else c.fix
        )
        bad += 0 if c.ok else 1
    console.print(table)
    if bad:
        err.print(f"[yellow]{bad} component(s) need attention[/]")
        raise typer.Exit(code=1)


@app.command()
def extract(
    sources: Annotated[list[str], typer.Argument(help="Files or http(s) URLs")],
    out: Annotated[Path, typer.Option("-o", "--out", help="Output directory")] = Path("out"),
    ocr: Annotated[str, typer.Option(help="off | auto | force")] = "off",
    ocr_backend: Annotated[
        str, typer.Option(help="auto | macocr | macocr-cli | tesseract")
    ] = "auto",
    ocr_lang: Annotated[
        str, typer.Option(help="Comma-separated BCP-47 codes, e.g. en-US,de-DE")
    ] = "en-US",
    asr_model: Annotated[
        str, typer.Option(help="docling ASR model, e.g. whisper_turbo")
    ] = "whisper_turbo",
    no_repair: Annotated[
        bool, typer.Option(help="Skip Ghostscript repair of unreadable PDFs")
    ] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Print the report as JSON")] = False,
) -> None:
    """Extract layout text, Markdown and plain text from each source into --out."""
    from .pipeline import ingest

    settings = ExtractSettings(
        ocr=ocr,  # type: ignore[arg-type]
        ocr_backend=ocr_backend,  # type: ignore[arg-type]
        ocr_languages=[s.strip() for s in ocr_lang.split(",") if s.strip()],
        asr_model=asr_model,
        repair_with_ghostscript=not no_repair,
    )
    failures = 0
    reports = []
    with Progress(
        TextColumn("[bold]{task.fields[name]}"),
        BarColumn(),
        TextColumn("{task.fields[stage]}"),
        TimeElapsedColumn(),
        console=err,
        transient=json_out,
    ) as progress:
        for src in sources:
            task = progress.add_task("", total=100, name=Path(src).name[:40], stage="queued")

            def cb(stage: str, done: int, total: int, task=task) -> None:
                pct = _stage_percent(stage, done, total)
                progress.update(task, completed=pct, stage=stage)

            try:
                r = ingest(src, out, settings, progress=cb)
                progress.update(task, completed=100, stage="done")
                reports.append(r.to_dict())
                if not json_out:
                    _print_result(r)
            except Exception as exc:
                failures += 1
                progress.update(task, stage=f"[red]failed: {exc}")
                reports.append({"source": src, "error": str(exc)})
    if json_out:
        console.print_json(json.dumps(reports))
    if failures:
        raise typer.Exit(code=1)


_STAGE_BASE = {"detect": 0, "ocr": 10, "layout": 45, "markdown": 55, "text": 90, "docling": 0}
_STAGE_SPAN = {"detect": 10, "ocr": 35, "layout": 10, "markdown": 35, "text": 8, "docling": 98}


def _stage_percent(stage: str, done: int, total: int) -> float:
    base = _STAGE_BASE.get(stage, 0)
    span = _STAGE_SPAN.get(stage, 100)
    frac = (done / total) if total else 1.0
    return min(100.0, base + span * frac)


def _print_result(r) -> None:
    outs = ", ".join(f"{k}={v.name}" for k, v in r.outputs.items())
    console.print(f"[green]✓[/] {r.source} [{r.kind}] {r.seconds:.1f}s → {outs}")
    for w in r.warnings:
        console.print(f"   [yellow]![/] {w}")


@app.command()
def detect(pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Report per-page scan / text / garble signals for a PDF."""
    from .detect import analyse_pdf

    sig = analyse_pdf(pdf)
    table = Table(title=str(pdf))
    for col in ("page", "chars", "img cover", "scan", "text-over-scan", "garbled", "needs OCR"):
        table.add_column(col)
    for p in sig.pages:
        table.add_row(
            str(p.number),
            str(p.chars),
            f"{p.image_coverage:.0%}",
            "yes" if p.is_scan else "",
            "yes" if p.text_over_scan else "",
            "yes" if p.garbled else "",
            "[red]yes" if p.needs_ocr else "",
        )
    console.print(table)
    if sig.error:
        err.print(f"[red]{sig.error}")
    console.print(
        f"scanned document: {sig.is_scanned_document}; pages needing OCR: {sig.needs_ocr_pages}"
    )


@app.command()
def serve(
    host: Annotated[
        str | None, typer.Option(help="Bind address (default from env, 127.0.0.1)")
    ] = None,
    port: Annotated[int | None, typer.Option(help="Port (default 8787)")] = None,
    watch: Annotated[bool, typer.Option(help="Also watch the iCloud inbox folder")] = False,
) -> None:
    """Run the private web app (requires: uv sync --extra web)."""
    try:
        import uvicorn
    except ImportError as exc:
        err.print("[red]web extra not installed. Run: uv sync --extra web")
        raise typer.Exit(code=2) from exc
    from .web.app import create_app

    settings = Settings.from_env()
    if host:
        settings = settings.model_copy(update={"host": host})
    if port:
        settings = settings.model_copy(update={"port": port})
    if watch:
        settings = settings.model_copy(update={"watch_inbox": True})
    application = create_app(settings)
    uvicorn.run(
        application,
        host=settings.host,
        port=settings.port,
        log_level="info",
        proxy_headers=settings.trust_proxy,
        forwarded_allow_ips="127.0.0.1" if settings.trust_proxy else None,
        server_header=False,
        date_header=False,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
