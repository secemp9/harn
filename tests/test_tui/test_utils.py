from __future__ import annotations

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
