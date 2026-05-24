from __future__ import annotations

from harnify_agent.harness.utils.truncate import truncate_head, truncate_line, truncate_tail


def test_truncate_counts_utf8_bytes() -> None:
    content = "aé🙂\nb"
    result = truncate_head(content, {"maxBytes": 100, "maxLines": 10})

    assert result.truncated is False
    assert result.totalBytes == 9
    assert result.outputBytes == 9


def test_truncate_head_preserves_complete_lines() -> None:
    result = truncate_head("éé\nabc", {"maxBytes": 4, "maxLines": 10})

    assert result.content == "éé"
    assert result.truncated is True
    assert result.truncatedBy == "bytes"
    assert result.outputBytes == 4
    assert result.firstLineExceedsLimit is False


def test_truncate_head_reports_first_line_exceeds_limit() -> None:
    result = truncate_head("éé\nabc", {"maxBytes": 3, "maxLines": 10})

    assert result.content == ""
    assert result.truncated is True
    assert result.truncatedBy == "bytes"
    assert result.firstLineExceedsLimit is True


def test_truncate_tail_respects_utf8_boundaries() -> None:
    result = truncate_tail("aé🙂b", {"maxBytes": 5, "maxLines": 10})

    assert result.content == "🙂b"
    assert result.truncated is True
    assert result.truncatedBy == "bytes"
    assert result.lastLinePartial is True
    assert result.outputBytes == 5


def test_truncate_tail_handles_oversized_lines_and_multibyte_drop() -> None:
    oversized = truncate_tail(f'{"X" * 300_000}\n', {"maxBytes": 1024, "maxLines": 100})
    assert oversized.content == "X" * 1024
    assert oversized.outputBytes == 1024
    assert oversized.outputLines == 1
    assert oversized.lastLinePartial is True
    assert oversized.truncatedBy == "bytes"

    dropped = truncate_tail("abc🙂", {"maxBytes": 3, "maxLines": 10})
    assert dropped.content == ""
    assert dropped.truncated is True
    assert dropped.truncatedBy == "bytes"
    assert dropped.lastLinePartial is True
    assert dropped.outputBytes == 0


def test_truncate_tail_handles_surrogate_edge_cases_and_line_truncation() -> None:
    assert truncate_tail("👩‍💻", {"maxBytes": 4, "maxLines": 10}).content == "💻"
    assert truncate_tail("a\ud83d", {"maxBytes": 3, "maxLines": 10}).content == "�"
    assert truncate_tail("\ud83d\ude42", {"maxBytes": 4, "maxLines": 10}).content == "🙂"

    assert truncate_line("short") == {"text": "short", "wasTruncated": False}
    assert truncate_line("x" * 10, 5) == {"text": "xxxxx... [truncated]", "wasTruncated": True}


def test_truncate_preserves_zero_limits_and_untruncated_content() -> None:
    head_zero = truncate_head("one\ntwo", {"maxBytes": 10, "maxLines": 0})
    assert head_zero.content == ""
    assert head_zero.truncatedBy == "lines"
    assert head_zero.maxLines == 0

    tail_zero = truncate_tail("one\ntwo", {"maxBytes": 0, "maxLines": 10})
    assert tail_zero.content == ""
    assert tail_zero.truncatedBy == "bytes"
    assert tail_zero.maxBytes == 0

    unpaired = truncate_head("a\ud83d", {"maxBytes": 10, "maxLines": 10})
    assert unpaired.truncated is False
    assert unpaired.content == "a\ud83d"
