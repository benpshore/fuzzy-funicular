from funicular.ocr.layout import Positioned, layout_text


def test_columns_land_in_separate_horizontal_positions():
    items = [
        Positioned("Left one", 0, 0, 80, 10),
        Positioned("Right one", 200, 0, 290, 10),
        Positioned("Left two", 0, 12, 80, 22),
        Positioned("Right two", 200, 12, 290, 22),
    ]
    out = layout_text(items).splitlines()
    assert len(out) == 2
    assert out[0].startswith("Left one")
    assert "Right one" in out[0]
    assert out[0].index("Right one") > 15
    # Column alignment is stable across rows.
    assert out[0].index("Right one") == out[1].index("Right two")


def test_vertical_gaps_become_blank_lines():
    items = [Positioned("A", 0, 0, 10, 10), Positioned("B", 0, 40, 10, 50)]
    out = layout_text(items)
    assert out.split("\n")[:5] == ["A", "", "", "", "B"]  # three empty line slots between


def test_empty_and_whitespace_only():
    assert layout_text([]) == ""
    assert layout_text([Positioned("   ", 0, 0, 5, 5)]) == ""


def test_width_is_capped():
    items = [Positioned("far", 10_000_000, 0, 10_000_010, 10), Positioned("near", 0, 0, 40, 10)]
    line = layout_text(items).splitlines()[0]
    assert len(line) < 500
