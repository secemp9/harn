from __future__ import annotations

from collections import OrderedDict

from harnify_tui import utils as utils_module
from harnify_tui.utils import (
    normalizeTerminalOutput,
    sliceByColumn,
    truncateToWidth,
    visibleWidth,
    wrapTextWithAnsi,
)


def test_truncate_to_width_handles_large_unicode_and_preserves_resets() -> None:
    text = "🙂界" * 1000
    truncated = truncateToWidth(text, 40, "…")

    assert visibleWidth(truncated) <= 40
    assert truncated.endswith("…\x1b[0m")


def test_truncate_to_width_preserves_kept_ansi_prefix() -> None:
    text = f"\x1b[31m{'hello ' * 50}\x1b[0m"
    truncated = truncateToWidth(text, 20, "…")

    assert visibleWidth(truncated) <= 20
    assert "\x1b[31m" in truncated
    assert truncated.endswith("\x1b[0m…\x1b[0m")


def test_visible_width_counts_tabs_and_regional_indicators() -> None:
    assert visibleWidth("\t\x1b[31m界\x1b[0m") == 5
    assert visibleWidth("🇨") == 2
    assert visibleWidth("🇨🇳") == 2


def test_normalize_terminal_output_only_changes_terminal_form() -> None:
    assert normalizeTerminalOutput("ำ") == "ํา"
    assert normalizeTerminalOutput("ຳ") == "ໍາ"
    assert visibleWidth(normalizeTerminalOutput("ำabc")) == visibleWidth("ำabc")


def test_wrap_text_with_ansi_keeps_styles_on_continuation_lines() -> None:
    underline_on = "\x1b[4m"
    underline_off = "\x1b[24m"
    url = "https://example.com/very/long/path/that/will/wrap"
    text = f"read this thread {underline_on}{url}{underline_off}"

    wrapped = wrapTextWithAnsi(text, 40)

    assert wrapped[0] == "read this thread"
    assert wrapped[1].startswith(underline_on)
    assert underline_off not in wrapped[0]


def test_slice_by_column_avoids_wide_character_overflow_when_strict() -> None:
    assert sliceByColumn("ab界cd", 0, 3, True) == "ab"
    assert sliceByColumn("ab界cd", 2, 2, True) == "界"


def test_utils_module_exports_match_ts_surface() -> None:
    assert utils_module.__all__ == [
        "applyBackgroundToLine",
        "extractAnsiCode",
        "extractSegments",
        "getSegmenter",
        "isPunctuationChar",
        "isWhitespaceChar",
        "normalizeTerminalOutput",
        "sliceByColumn",
        "sliceWithWidth",
        "truncateToWidth",
        "visibleWidth",
        "wrapTextWithAnsi",
    ]
    assert not hasattr(utils_module, "visible_width")
    assert not hasattr(utils_module, "wrap_text_with_ansi")
    assert not hasattr(utils_module, "ActiveHyperlink")


def test_is_whitespace_char_matches_js_test_semantics() -> None:
    assert utils_module.isWhitespaceChar("a b") is True
    assert utils_module.isWhitespaceChar("abc") is False


def test_visible_width_cache_uses_fifo_eviction_like_ts(monkeypatch) -> None:
    monkeypatch.setattr(utils_module, "_width_cache", OrderedDict())
    monkeypatch.setattr(utils_module, "_WIDTH_CACHE_SIZE", 2)

    assert utils_module.visibleWidth("é") == 1
    assert utils_module.visibleWidth("ß") == 1
    assert utils_module.visibleWidth("é") == 1
    assert utils_module.visibleWidth("ø") == 1

    assert list(utils_module._width_cache.keys()) == ["ß", "ø"]
