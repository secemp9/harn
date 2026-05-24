from __future__ import annotations

from harnify_tui.components.truncated_text import TruncatedText
from harnify_tui.utils import visibleWidth

RED = "\x1b[31m"
BLUE = "\x1b[34m"
RESET = "\x1b[0m"


def strip_sgr(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_truncated_text_pads_output_to_width() -> None:
    text = TruncatedText("Hello world", 1, 0)
    lines = text.render(50)

    assert len(lines) == 1
    assert visibleWidth(lines[0]) == 50


def test_truncated_text_pads_vertical_padding_lines() -> None:
    text = TruncatedText("Hello", 0, 2)
    lines = text.render(40)

    assert len(lines) == 5
    for line in lines:
        assert visibleWidth(line) == 40


def test_truncated_text_truncates_long_text_and_preserves_ansi() -> None:
    long_text = f"{RED}This is a very long red text that will be truncated{RESET}"
    text = TruncatedText(long_text, 1, 0)
    lines = text.render(20)

    assert len(lines) == 1
    assert visibleWidth(lines[0]) == 20
    assert "\x1b[0m..." in lines[0]


def test_truncated_text_handles_styled_text_that_fits() -> None:
    styled_text = f"{RED}Hello{RESET} {BLUE}world{RESET}"
    text = TruncatedText(styled_text, 1, 0)
    lines = text.render(40)

    assert len(lines) == 1
    assert visibleWidth(lines[0]) == 40
    assert "\x1b[" in lines[0]


def test_truncated_text_only_shows_first_line() -> None:
    text = TruncatedText("First line\nSecond line\nThird line", 1, 0)
    lines = text.render(40)

    assert len(lines) == 1
    assert visibleWidth(lines[0]) == 40
    stripped = strip_sgr(lines[0]).strip()
    assert "First line" in stripped
    assert "Second line" not in stripped
