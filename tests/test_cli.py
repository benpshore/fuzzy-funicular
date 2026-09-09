import json

from typer.testing import CliRunner

from conftest import needs_poppler
from funicular.cli import app

runner = CliRunner()


def test_version():
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0 and "funicular" in r.output


def test_doctor_runs():
    r = runner.invoke(app, ["doctor"])
    assert "pymupdf" in r.output and "poppler" in r.output


@needs_poppler
def test_extract_json(fixtures, tmp_path):
    r = runner.invoke(app, ["extract", str(fixtures["scholarly"]), "-o", str(tmp_path), "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output[r.output.index("[") :])
    assert data[0]["kind"] == "pdf"
    assert (tmp_path / "scholarly.layout.txt").exists()


def test_extract_failure_exit_code(tmp_path):
    r = runner.invoke(app, ["extract", "nope.pdf", "-o", str(tmp_path)])
    assert r.exit_code == 1


def test_detect_command(fixtures):
    r = runner.invoke(app, ["detect", str(fixtures["mixed"])])
    assert r.exit_code == 0
    assert "pages needing OCR: [2]" in r.output
