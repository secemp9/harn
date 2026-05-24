from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from harnify_tui.terminal_image import (
    ImageDimensions,
    deleteAllKittyImages,
    deleteKittyImage,
    detectCapabilities,
    encodeKitty,
    hyperlink,
    isImageLine,
    renderImage,
    resetCapabilitiesCache,
    setCapabilities,
    setCellDimensions,
)


@contextmanager
def patched_env(**updates: str | None) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_is_image_line_detects_kitty_and_iterm_sequences() -> None:
    assert isImageLine("\x1b]1337;File=inline=1:data==\x07") is True
    assert isImageLine("prefix \x1b_Ga=T,f=100;AAAA\x1b\\ suffix") is True
    assert isImageLine("\x1b[31mplain text\x1b[0m") is False


def test_detect_capabilities_disables_hyperlinks_under_tmux() -> None:
    with patched_env(TMUX="/tmp/tmux", TERM_PROGRAM="ghostty", TERM=None, COLORTERM=None):
        caps = detectCapabilities()
    assert caps.hyperlinks is False
    assert caps.images is None


def test_encode_kitty_and_delete_commands_follow_upstream_sequences() -> None:
    sequence = encodeKitty("AAAA", {"columns": 2, "rows": 2, "moveCursor": False})
    assert sequence.startswith("\x1b_Ga=T,f=100,q=2,C=1,c=2,r=2;")
    assert deleteKittyImage(42) == "\x1b_Ga=d,d=I,i=42,q=2\x1b\\"
    assert deleteAllKittyImages() == "\x1b_Ga=d,d=A,q=2\x1b\\"


def test_render_image_honors_max_height_by_reducing_width() -> None:
    setCapabilities({"images": "kitty", "trueColor": True, "hyperlinks": True})
    setCellDimensions({"widthPx": 10, "heightPx": 10})
    try:
        result = renderImage(
            "AAAA",
            ImageDimensions(widthPx=10, heightPx=100),
            {"maxWidthCells": 10, "maxHeightCells": 5},
        )
        assert result is not None
        assert result.rows == 5
        assert ",c=1,r=5" in result.sequence
    finally:
        resetCapabilitiesCache()
        setCellDimensions({"widthPx": 9, "heightPx": 18})


def test_hyperlink_wraps_text_in_osc8_sequences() -> None:
    assert hyperlink("click me", "https://example.com") == "\x1b]8;;https://example.com\x1b\\click me\x1b]8;;\x1b\\"
