from __future__ import annotations

from harnify_tui.components.box import Box
from harnify_tui.components.text import Text
from harnify_tui.utils import visibleWidth


def test_text_wraps_and_pads_lines_to_width() -> None:
    text = Text("hello world this is wrapped", paddingX=1, paddingY=1)
    lines = text.render(12)

    assert len(lines) >= 3
    for line in lines:
        assert visibleWidth(line) == 12


def test_text_returns_empty_for_blank_content() -> None:
    text = Text("   ", paddingX=1, paddingY=1)
    assert text.render(20) == []


def test_box_applies_padding_and_background() -> None:
    def bg_fn(value: str) -> str:
        return f"[{value}]"

    text = Text("hello", paddingX=0, paddingY=0)
    box = Box(paddingX=1, paddingY=1, bgFn=bg_fn)
    box.addChild(text)

    lines = box.render(10)

    assert len(lines) == 3
    assert all(line.startswith("[") and line.endswith("]") for line in lines)
