from __future__ import annotations

from harnify_tui.components.editor import segmentWithMarkers, wordWrapLine
from harnify_tui.utils import visibleWidth


def test_word_wrap_line_wraps_at_word_boundaries() -> None:
    chunks = wordWrapLine("hello world test", 11)

    assert [chunk.text for chunk in chunks] == ["hello ", "world test"]


def test_word_wrap_line_does_not_start_next_line_with_leading_whitespace() -> None:
    chunks = wordWrapLine("Word1 Word2 Word3 Word4 Word5 Word6", 18)

    for chunk in chunks[1:]:
        assert not chunk.text[:1].isspace()


def test_word_wrap_line_breaks_long_words_at_character_level() -> None:
    chunks = wordWrapLine("Check https://example.com/very/long/path/that/exceeds/width here", 29)

    assert all(visibleWidth(chunk.text) <= 29 for chunk in chunks)


def test_word_wrap_line_handles_empty_string() -> None:
    chunks = wordWrapLine("", 10)

    assert len(chunks) == 1
    assert chunks[0].text == ""


def test_word_wrap_line_force_breaks_when_wide_char_after_wrap_opportunity_still_overflows() -> None:
    chunks = wordWrapLine("a b界", 3)

    assert [chunk.text for chunk in chunks] == ["a ", "b界"]


def test_segment_with_markers_merges_only_valid_markers() -> None:
    text = "before [paste #1 +12 lines] after [paste #9 1234 chars]"
    merged = segmentWithMarkers(text, {1})

    merged_segments = [segment.segment for segment in merged]
    assert "[paste #1 +12 lines]" in merged_segments
    assert "[paste #9 1234 chars]" not in merged_segments


def test_word_wrap_line_splits_oversized_atomic_segments_across_multiple_chunks() -> None:
    marker = "[paste #1 +20 lines]"
    line = f"A{marker}B"
    chunks = wordWrapLine(line, 10, segmentWithMarkers(line, {1}))

    assert all(visibleWidth(chunk.text) <= 10 for chunk in chunks)
    reconstructed = "".join(line[chunk.startIndex : chunk.endIndex] for chunk in chunks)
    assert reconstructed == line


def test_word_wrap_line_wraps_normally_after_oversized_atomic_segment() -> None:
    marker = "[paste #1 +20 lines]"
    line = f"{marker} hello world"
    chunks = wordWrapLine(line, 10, segmentWithMarkers(line, {1}))

    assert all(visibleWidth(chunk.text) <= 10 for chunk in chunks)
    assert chunks[-1].text == "world"
    reconstructed = "".join(line[chunk.startIndex : chunk.endIndex] for chunk in chunks)
    assert reconstructed == line
