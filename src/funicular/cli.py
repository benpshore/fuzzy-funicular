"""Command line entry point: `funicular`."""

from __future__ import annotations

import json
import logging
import os
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
def compress(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    out: Annotated[Path | None, typer.Option("-o", "--out")] = None,
    strength: Annotated[
        int, typer.Option(min=0, max=100, help="0 = gentlest, 100 = smallest")
    ] = 50,
    engine: Annotated[str, typer.Option(help="auto | pymupdf | ghostscript")] = "auto",
    preview: Annotated[bool, typer.Option(help="Estimate only; write nothing")] = False,
) -> None:
    """Shrink a PDF (images re-encoded, text untouched). --preview shows the expected result."""
    from .compress import compress as _compress
    from .compress import preview as _preview
    from .web.app import human_size

    plan = _preview(pdf, strength, engine=engine)
    console.print(
        f"{pdf.name}: {human_size(plan.original_bytes)} → ≈{human_size(plan.estimated_bytes)}"
        f" ({plan.to_dict()['ratio']:.0%}) in ≈{plan.estimated_seconds}s via {plan.engine}"
        f" [{plan.level.label}]" + (f" — {plan.note}" if plan.note else "")
    )
    if preview:
        return
    out = out or pdf.with_name(pdf.stem + ".compressed.pdf")
    res = _compress(pdf, out, strength, engine=engine)
    console.print(
        f"[green]✓[/] {out} {human_size(res.original_bytes)} → {human_size(res.output_bytes)}"
        f" ({res.to_dict()['ratio']:.0%}) in {res.seconds:.1f}s via {res.engine}"
    )


@app.command()
def estimate(
    sources: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False)],
    ocr: Annotated[str, typer.Option(help="off | auto | force")] = "off",
) -> None:
    """Estimate time and memory before extracting (no work is done)."""
    from .estimate import audio_minutes, pdf_page_count
    from .estimate import estimate as _estimate
    from .resources import gpu_info
    from .sniff import sniff

    settings = ExtractSettings(ocr=ocr)  # type: ignore[arg-type]
    table = Table(title="work estimate")
    for col in ("file", "kind", "units", "≈ seconds", "peak MB", "note"):
        table.add_column(col)
    total = 0.0
    for src in sources:
        kind = sniff(src).kind
        pages = pdf_page_count(src) if kind.value == "pdf" else 0
        minutes = audio_minutes(src) if kind.value in ("audio", "video") else 0.0
        e = _estimate(
            kind,
            pages=pages,
            size_bytes=src.stat().st_size,
            audio_minutes=minutes,
            settings=settings,
            apple_silicon=gpu_info()["apple_silicon"],
        )
        total += e.seconds
        table.add_row(
            src.name, kind.value, f"{e.units} {e.unit}", f"{e.seconds}", f"{e.peak_mb:.0f}", e.note
        )
    console.print(table)
    console.print(f"total ≈ {total:.0f} s")


scholar_app = typer.Typer(
    help="Scholarly metadata: identifiers, verified records, references, renaming."
)
app.add_typer(scholar_app, name="scholar")


def _fetcher():
    from .config import Settings
    from .scholar.clients import Cache, Fetcher

    data_dir = Settings.from_env().data_dir
    return Fetcher(cache=Cache(data_dir / "scholar-cache.sqlite3"))


@scholar_app.command("meta")
def scholar_meta(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    resolve: Annotated[bool, typer.Option(help="Look the record up online and verify it")] = True,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Identifiers and metadata for a PDF; verified against Crossref & co."""
    from .scholar.metadata import extract_from_pdf
    from .scholar.metadata import resolve as _resolve

    ex = extract_from_pdf(pdf)
    out: dict = {"extracted": ex.to_dict()}
    if resolve:
        f = _fetcher()
        try:
            out["resolution"] = _resolve(ex, f).to_dict()
        finally:
            f.close()
    if json_out:
        console.print_json(json.dumps(out))
        return
    console.print(f"[bold]{pdf.name}[/]")
    console.print(f"  ids: {ex.ids.to_dict()}")
    console.print(f"  title guess: {ex.guessed_title or ex.pdf_title or '-'}")
    res = out.get("resolution")
    if res:
        w = res["work"]
        if w:
            state = "green]verified" if res["verified"] else "yellow]unverified"
            console.print(
                f"  [{state}[/] ({res['confidence']:.2f}): {w['title']} — {w['journal']}"
                f" {w['year']} — doi:{w['doi']}"
            )
            console.print(
                f"  sources: {', '.join(w['sources'])}; cited by {w.get('cited_by')};"
                f" OA: {w.get('oa_url') or '-'}"
            )
        for e in res["evidence"]:
            console.print(f"    · {e}")


@scholar_app.command("refs")
def scholar_refs(
    source: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="PDF or extracted .txt")
    ],
    resolve: Annotated[bool, typer.Option(help="Resolve each reference online")] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Extract the reference list losslessly; optionally resolve each entry to a DOI."""
    from .scholar.refs import ReferenceReport, ResolvedRef, extract_references, resolve_reference

    if source.suffix.lower() == ".pdf":
        import pymupdf

        with pymupdf.open(source) as doc:
            text = "\f".join(p.get_text("text") for p in doc)
    else:
        text = source.read_text(encoding="utf-8", errors="replace")
    raw = extract_references(text)
    if resolve:
        f = _fetcher()
        try:
            rep = ReferenceReport([resolve_reference(r, f) for r in raw])
        finally:
            f.close()
    else:
        rep = ReferenceReport([ResolvedRef(r, None, 0.0, "not resolved") for r in raw])
    if json_out:
        console.print_json(json.dumps(rep.to_dict()))
        return
    for r in rep.refs:
        mark = "[green]✓[/]" if r.work and r.confidence >= 0.8 else "[yellow]?[/]"
        console.print(f"{mark} {r.ref.n:>3}. {r.ref.raw[:110]}")
        if r.work and r.work.doi:
            console.print(f"       → doi:{r.work.doi} ({r.method}, {r.confidence:.2f})")
    console.print(f"{len(rep.refs)} references, {rep.resolved} resolved")


@scholar_app.command("rename")
def scholar_rename(
    pdfs: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False)],
    template: Annotated[
        str, typer.Option(help="e.g. '{author} - {year} - {title}'")
    ] = "{author} - {year} - {title}",
    apply: Annotated[bool, typer.Option(help="Actually rename (default is a dry run)")] = False,
    min_confidence: Annotated[float, typer.Option()] = 0.5,
) -> None:
    """Zotero-style renaming from verified metadata. Dry run unless --apply."""
    from .scholar.metadata import extract_from_pdf
    from .scholar.metadata import resolve as _resolve
    from .scholar.rename import apply_rename, plan_rename

    f = _fetcher()
    try:
        for pdf in pdfs:
            res = _resolve(extract_from_pdf(pdf), f)
            if not res.work or res.confidence < min_confidence:
                console.print(f"[yellow]skip[/] {pdf.name}: no confident metadata")
                continue
            plan = plan_rename(pdf, res.work, template)
            if not plan.changed:
                console.print(f"[dim]same[/] {pdf.name}")
                continue
            if apply:
                apply_rename(plan)
                console.print(f"[green]renamed[/] {pdf.name} → {plan.target.name}")
            else:
                console.print(f"[cyan]would rename[/] {pdf.name} → {plan.target.name}")
    finally:
        f.close()


models_app = typer.Typer(help="Pre-fetch and inspect ML models (docling, Whisper, embeddings).")
app.add_typer(models_app, name="models")


@models_app.command("status")
def models_status(
    asr_model: Annotated[str, typer.Option(help="ASR model to check")] = "whisper_turbo",
) -> None:
    """Which models are already on disk."""
    from .models import status

    table = Table(title="models")
    table.add_column("model")
    table.add_column("present")
    table.add_column("location")
    for m in status(asr_model).models:
        table.add_row(m.name, "[green]yes" if m.present else "[yellow]no", m.path)
    console.print(table)


@models_app.command("fetch")
def models_fetch(
    asr_model: Annotated[
        str, typer.Option(help="ASR model to fetch (none to skip)")
    ] = "whisper_turbo",
    no_docling: Annotated[bool, typer.Option(help="Skip docling layout/table models")] = False,
    no_embeddings: Annotated[bool, typer.Option(help="Skip embedding models")] = False,
) -> None:
    """Download missing models now, so the first document does not wait (needs network once)."""
    from .models import fetch

    rep = fetch(
        docling=not no_docling,
        asr_model=None if asr_model.lower() in ("none", "") else asr_model,
        embeddings=not no_embeddings,
        progress=True,
    )
    for f in rep.fetched:
        console.print(f"[green]✓[/] {f}")
    for e in rep.errors:
        err.print(f"[red]![/] {e}")
    if rep.errors:
        raise typer.Exit(code=1)


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
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.pid_file.write_text(str(os.getpid()))
    try:
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
    finally:
        settings.pid_file.unlink(missing_ok=True)


@app.command()
def stop(
    force: Annotated[bool, typer.Option(help="SIGKILL instead of a clean SIGTERM")] = False,
) -> None:
    """Stop a running `funicular serve` cleanly (jobs cancelled, helper processes killed)."""
    import signal

    settings = Settings.from_env()
    pid_file = settings.pid_file
    if not pid_file.exists():
        err.print(f"[yellow]no pid file at {pid_file}; is the server running?")
        raise typer.Exit(code=1)
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError as exc:
        err.print("[red]pid file is unreadable")
        raise typer.Exit(code=1) from exc
    from .resources import kill_tree

    if force:
        n = kill_tree(pid, grace=0.5)
        console.print(f"killed {n} process(es)")
    else:
        try:
            os.kill(pid, signal.SIGTERM)
            console.print(f"sent SIGTERM to {pid}")
        except ProcessLookupError:
            err.print("[yellow]process already gone")
    pid_file.unlink(missing_ok=True)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
