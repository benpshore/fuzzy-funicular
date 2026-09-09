"""Exercise the Apple Vision glue on any platform by injecting fake `Vision` / `ocrmac` modules.

This validates language negotiation, coordinate conversion (Vision's bottom-left normalised
boxes -> top-left pixels) and the result parsing. It does not, and cannot, validate Apple's
recognition engine itself; that needs a Mac.
"""

from __future__ import annotations

import sys
import types

import pytest
from PIL import Image

from funicular.ocr.backends import MacVisionBackend, OcrUnavailableError, vision_results_to_lines


class _Rect:
    def __init__(self, x, y, w, h):
        self.origin = types.SimpleNamespace(x=x, y=y)
        self.size = types.SimpleNamespace(width=w, height=h)


class _Obs:
    def __init__(self, text, conf, box):
        self._text, self._conf, self._box = text, conf, box

    def topCandidates_(self, n):  # noqa: N802 - Objective-C selector style
        return [types.SimpleNamespace(string=lambda: self._text)]

    def confidence(self):
        return self._conf

    def boundingBox(self):  # noqa: N802
        return _Rect(*self._box)


class _Request:
    last = None

    def __init__(self):
        self.level = None
        self.langs = None
        self.auto = None
        self.correction = None
        self._results = []
        _Request.last = self

    @classmethod
    def alloc(cls):
        return cls()

    def init(self):
        return self

    def setRecognitionLevel_(self, level):  # noqa: N802
        self.level = level

    def setUsesLanguageCorrection_(self, v):  # noqa: N802
        self.correction = v

    def setAutomaticallyDetectsLanguage_(self, v):  # noqa: N802
        self.auto = v

    def supportedRecognitionLanguagesAndReturnError_(self, _err):  # noqa: N802
        return (["en-US", "de-DE"], None)

    def setRecognitionLanguages_(self, langs):  # noqa: N802
        self.langs = list(langs)

    def results(self):
        return self._results


class _Handler:
    fail = False

    @classmethod
    def alloc(cls):
        return cls()

    def initWithData_options_(self, data, _opts):  # noqa: N802
        assert data.startswith(b"\x89PNG")
        return self

    def performRequests_error_(self, reqs, _err):  # noqa: N802
        if _Handler.fail:
            return (False, "boom")
        for r in reqs:
            r._results = [
                _Obs("Hello", 0.9, (0.10, 0.80, 0.30, 0.10)),  # top-left region
                _Obs("world", 0.8, (0.55, 0.80, 0.30, 0.10)),  # same row, right column
                _Obs("", 0.5, (0.1, 0.1, 0.1, 0.1)),  # empty: dropped
            ]
        return True


@pytest.fixture
def fake_vision(monkeypatch):
    mod = types.ModuleType("Vision")
    mod.VNRecognizeTextRequest = _Request
    mod.VNImageRequestHandler = _Handler
    monkeypatch.setitem(sys.modules, "Vision", mod)
    monkeypatch.setitem(sys.modules, "ocrmac", None)  # force the direct path
    _Handler.fail = False
    return mod


def test_direct_vision_path_converts_coordinates(fake_vision):
    backend = MacVisionBackend(["en-US", "xx-XX"])
    img = Image.new("RGB", (1000, 500), "white")
    lines = backend.recognize(img)
    assert [ln.text for ln in lines] == ["Hello", "world"]
    hello = lines[0]
    # x: 0.10*1000 .. 0.40*1000 ; y: Vision origin bottom-left 0.80..0.90 -> top-left 50..100
    assert (round(hello.x0), round(hello.x1)) == (100, 400)
    assert (round(hello.y0), round(hello.y1)) == (50, 100)
    assert hello.confidence == pytest.approx(0.9)
    req = _Request.last
    assert req.level == 0  # accurate
    assert req.langs == ["en-US"]  # unsupported xx-XX dropped
    assert req.correction is True


def test_no_languages_enables_auto_detection(fake_vision):
    backend = MacVisionBackend([], recognition_level="fast")
    backend.recognize(Image.new("RGB", (10, 10)))
    assert _Request.last.level == 1 and _Request.last.auto is True


def test_vision_failure_surfaces(fake_vision):
    _Handler.fail = True
    with pytest.raises(OcrUnavailableError):
        MacVisionBackend(["en-US"]).recognize(Image.new("RGB", (10, 10)))


def test_layout_from_vision_boxes(fake_vision):
    lines = MacVisionBackend(["en-US"]).recognize(Image.new("RGB", (1000, 500)))
    from funicular.ocr.layout import layout_text

    row = layout_text(list(lines)).splitlines()[0]
    # Indentation mirrors the x position; both words sit on one row, in order, spaced apart.
    assert row.lstrip().startswith("Hello") and "world" in row
    assert row.index("world") - row.index("Hello") > 6


def test_ocrmac_path_and_language_fallback(monkeypatch):
    calls = []

    class FakeOCR:
        def __init__(self, image, **kw):
            calls.append(kw)
            if kw.get("language_preference") == ["zz-ZZ"]:
                raise ValueError("Invalid language preference")

        def recognize(self, px=False):
            assert px
            return [("Text", 0.95, (1.0, 2.0, 30.0, 12.0))]

    mod = types.ModuleType("ocrmac")
    sub = types.ModuleType("ocrmac.ocrmac")
    sub.OCR = FakeOCR
    mod.ocrmac = sub
    monkeypatch.setitem(sys.modules, "ocrmac", mod)
    monkeypatch.setitem(sys.modules, "ocrmac.ocrmac", sub)
    lines = MacVisionBackend(["zz-ZZ"]).recognize(Image.new("RGB", (40, 20)))
    assert lines[0].text == "Text" and lines[0].x1 == 30.0
    assert calls[0]["language_preference"] == ["zz-ZZ"] and calls[1]["language_preference"] is None


def test_unavailable_without_frameworks(monkeypatch):
    monkeypatch.setitem(sys.modules, "ocrmac", None)
    monkeypatch.setitem(sys.modules, "Vision", None)
    with pytest.raises(OcrUnavailableError):
        MacVisionBackend(["en-US"])


def test_results_parser_handles_text_only_observations():
    class Old:
        def topCandidates_(self, n):
            raise AttributeError

        def text(self):
            return "legacy"

        def boundingBox(self):  # noqa: N802
            return _Rect(0, 0, 1, 1)

        def confidence(self):
            return 1.0

    out = vision_results_to_lines([Old()], 10, 10)
    assert out[0].text == "legacy" and (out[0].y0, out[0].y1) == (0.0, 10.0)
