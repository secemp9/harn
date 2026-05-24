from __future__ import annotations

import re

import pytest
from harnify_tui import DefaultTextStyle, Markdown, MarkdownTheme, resetCapabilitiesCache, setCapabilities, visibleWidth

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
OSC8_RE = re.compile(r"\x1b\]8;;[^\x1b]*\x1b\\")


def style(code: str):
    return lambda text: f"\x1b[{code}m{text}\x1b[0m"


TEST_MARKDOWN_THEME = MarkdownTheme(
    heading=style("36"),
    link=style("34"),
    linkUrl=style("2"),
    code=style("33"),
    codeBlock=style("32"),
    codeBlockBorder=style("2"),
    quote=style("3"),
    quoteBorder=style("2"),
    hr=style("2"),
    listBullet=style("36"),
    bold=style("1"),
    italic=style("3"),
    strikethrough=style("9"),
    underline=style("4"),
)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", OSC8_RE.sub("", text))


@pytest.fixture(autouse=True)
def reset_terminal_capabilities() -> None:
    resetCapabilitiesCache()
    yield
    resetCapabilitiesCache()


def test_markdown_renders_nested_lists_and_wrapped_items() -> None:
    markdown = Markdown(
        "- parent\n  - alpha beta gamma delta epsilon\n- [ ] beep\n- [x] boop",
        0,
        0,
        TEST_MARKDOWN_THEME,
    )

    lines = [strip_ansi(line).rstrip() for line in markdown.render(24)]

    assert lines == [
        "- parent",
        "    - alpha beta gamma",
        "      delta epsilon",
        "- [ ] beep",
        "- [x] boop",
    ]


def test_markdown_renders_blockquotes_and_code_blocks_inside_list_items() -> None:
    quote_markdown = Markdown("- > alpha beta gamma delta epsilon zeta", 0, 0, TEST_MARKDOWN_THEME)
    quote_lines = [strip_ansi(line).rstrip() for line in quote_markdown.render(24)]
    assert quote_lines == ["- │ alpha beta gamma", "  │ delta epsilon zeta"]

    code_markdown = Markdown("- ```ts\n  alpha beta gamma delta epsilon zeta\n  ```", 0, 0, TEST_MARKDOWN_THEME)
    code_lines = [strip_ansi(line).rstrip() for line in code_markdown.render(24)]
    assert code_lines == ["- ```ts", "    alpha beta gamma", "  delta epsilon zeta", "  ```"]


def test_markdown_renders_tables_with_borders_and_wrapping() -> None:
    markdown = Markdown(
        "| Name | Description |\n| --- | --- |\n| Alpha | alpha beta gamma delta epsilon |\n| Beta | short |",
        0,
        0,
        TEST_MARKDOWN_THEME,
    )

    lines = markdown.render(36)
    plain_lines = [strip_ansi(line).rstrip() for line in lines]

    assert plain_lines[0].startswith("┌─")
    assert any("│" in line for line in plain_lines)
    assert any("Alpha" in line for line in plain_lines)
    assert any("Beta" in line for line in plain_lines)
    assert all(visibleWidth(line) <= 36 for line in lines)


def test_markdown_uses_strict_strikethrough_rules() -> None:
    markdown = Markdown("~~strikethrough~~ and ~single-tilde~", 0, 0, TEST_MARKDOWN_THEME)

    lines = markdown.render(80)
    joined_output = "\n".join(lines)
    joined_plain = " ".join(strip_ansi(line) for line in lines)

    assert "\x1b[9m" in joined_output
    assert "~single-tilde~" in joined_plain


def test_markdown_links_without_hyperlinks_match_upstream_behavior() -> None:
    setCapabilities({"images": None, "trueColor": False, "hyperlinks": False})

    email_md = Markdown("Contact user@example.com for help", 0, 0, TEST_MARKDOWN_THEME)
    email_plain = " ".join(strip_ansi(line) for line in email_md.render(80))
    assert "user@example.com" in email_plain
    assert "mailto:" not in email_plain

    url_md = Markdown("Visit https://example.com for more", 0, 0, TEST_MARKDOWN_THEME)
    url_plain = " ".join(strip_ansi(line) for line in url_md.render(80))
    assert url_plain.count("https://example.com") == 1

    link_md = Markdown("[click here](https://example.com)", 0, 0, TEST_MARKDOWN_THEME)
    link_plain = " ".join(strip_ansi(line) for line in link_md.render(80))
    assert "click here" in link_plain
    assert "(https://example.com)" in link_plain


def test_markdown_links_with_hyperlinks_emit_osc8_sequences() -> None:
    setCapabilities({"images": None, "trueColor": False, "hyperlinks": True})

    markdown = Markdown("Visit https://example.com for more", 0, 0, TEST_MARKDOWN_THEME)
    joined = "".join(markdown.render(80))

    assert "\x1b]8;;https://example.com\x1b\\" in joined
    raw_plain = "".join(
        line.replace("\x1b]8;;https://example.com\x1b\\", "").replace("\x1b]8;;\x1b\\", "")
        for line in markdown.render(80)
    )
    assert "(https://example.com)" not in strip_ansi(raw_plain)


def test_markdown_blockquotes_use_quote_style_not_default_message_color() -> None:
    markdown = Markdown(
        "> Quote with **bold** and `code`",
        0,
        0,
        TEST_MARKDOWN_THEME,
        DefaultTextStyle(color=style("35")),
    )

    lines = markdown.render(80)
    plain_lines = [strip_ansi(line) for line in lines]
    all_output = "\n".join(lines)

    assert any(line.startswith("│ ") for line in plain_lines)
    assert "Quote with" in " ".join(plain_lines)
    assert "\x1b[1m" in all_output
    assert "\x1b[33m" in all_output
    assert "\x1b[3m" in all_output
    assert "\x1b[35m" not in all_output


def test_markdown_reapplies_heading_style_after_inline_code() -> None:
    markdown = Markdown("### Why `sourceInfo` should not be optional", 0, 0, TEST_MARKDOWN_THEME)

    joined_output = "\n".join(markdown.render(80))
    after_code_index = joined_output.index("should not be optional")
    preceding_chunk = joined_output[max(0, after_code_index - 40) : after_code_index]

    assert "\x1b[33m" in joined_output
    assert "\x1b[1m" in preceding_chunk
    assert "\x1b[36m" in preceding_chunk


def test_markdown_adds_single_blank_line_after_blocks_without_trailing_blank_line() -> None:
    heading_markdown = Markdown("# Hello\n\nThis is a paragraph", 0, 0, TEST_MARKDOWN_THEME)
    heading_lines = [strip_ansi(line).rstrip() for line in heading_markdown.render(80)]
    heading_index = next(index for index, line in enumerate(heading_lines) if "Hello" in line)
    assert heading_lines[heading_index + 1] == ""
    assert heading_lines[heading_index + 2] == "This is a paragraph"

    code_markdown = Markdown("```py\nprint(1)\n```\n\nAfter", 0, 0, TEST_MARKDOWN_THEME)
    code_lines = [strip_ansi(line).rstrip() for line in code_markdown.render(80)]
    closing_index = code_lines.index("```")
    assert code_lines[closing_index + 1] == ""
    assert code_lines[closing_index + 2] == "After"

    tail_markdown = Markdown("> quote", 0, 0, TEST_MARKDOWN_THEME)
    tail_lines = [strip_ansi(line).rstrip() for line in tail_markdown.render(80)]
    assert tail_lines[-1] != ""


def test_markdown_keeps_html_like_tags_visible() -> None:
    markdown = Markdown(
        "This is text with <thinking>hidden content</thinking> that should be visible",
        0,
        0,
        TEST_MARKDOWN_THEME,
    )

    joined_plain = " ".join(strip_ansi(line) for line in markdown.render(80))
    assert "hidden content" in joined_plain or "<thinking>" in joined_plain
