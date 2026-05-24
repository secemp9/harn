from __future__ import annotations

from harnify_tui.components.select_list import (
    SelectItem,
    SelectList,
    SelectListLayoutOptions,
    SelectListTheme,
)
from harnify_tui.utils import visibleWidth

TEST_THEME = SelectListTheme(
    selectedPrefix=lambda text: text,
    selectedText=lambda text: text,
    description=lambda text: text,
    scrollInfo=lambda text: text,
    noMatch=lambda text: text,
)


def visible_index_of(line: str, text: str) -> int:
    index = line.index(text)
    assert index != -1
    return visibleWidth(line[:index])


def test_select_list_normalizes_multiline_descriptions() -> None:
    items = [SelectItem(value="test", label="test", description="Line one\nLine two\nLine three")]
    select_list = SelectList(items, 5, TEST_THEME)

    rendered = select_list.render(100)

    assert rendered
    assert "\n" not in rendered[0]
    assert "Line one Line two Line three" in rendered[0]


def test_select_list_keeps_descriptions_aligned_when_primary_text_truncates() -> None:
    items = [
        SelectItem(value="short", label="short", description="short description"),
        SelectItem(
            value="very-long-command-name-that-needs-truncation",
            label="very-long-command-name-that-needs-truncation",
            description="long description",
        ),
    ]
    select_list = SelectList(items, 5, TEST_THEME)

    rendered = select_list.render(80)

    assert visible_index_of(rendered[0], "short description") == visible_index_of(rendered[1], "long description")


def test_select_list_respects_minimum_primary_column_width() -> None:
    items = [
        SelectItem(value="a", label="a", description="first"),
        SelectItem(value="bb", label="bb", description="second"),
    ]
    select_list = SelectList(
        items,
        5,
        TEST_THEME,
        SelectListLayoutOptions(minPrimaryColumnWidth=12, maxPrimaryColumnWidth=20),
    )

    rendered = select_list.render(80)

    assert rendered[0].index("first") == 14
    assert rendered[1].index("second") == 14


def test_select_list_respects_maximum_primary_column_width() -> None:
    items = [
        SelectItem(
            value="very-long-command-name-that-needs-truncation",
            label="very-long-command-name-that-needs-truncation",
            description="first",
        ),
        SelectItem(value="short", label="short", description="second"),
    ]
    select_list = SelectList(
        items,
        5,
        TEST_THEME,
        SelectListLayoutOptions(minPrimaryColumnWidth=12, maxPrimaryColumnWidth=20),
    )

    rendered = select_list.render(80)

    assert visible_index_of(rendered[0], "first") == 22
    assert visible_index_of(rendered[1], "second") == 22


def test_select_list_allows_custom_primary_truncation() -> None:
    items = [
        SelectItem(
            value="very-long-command-name-that-needs-truncation",
            label="very-long-command-name-that-needs-truncation",
            description="first",
        ),
        SelectItem(value="short", label="short", description="second"),
    ]
    select_list = SelectList(
        items,
        5,
        TEST_THEME,
        SelectListLayoutOptions(
            minPrimaryColumnWidth=12,
            maxPrimaryColumnWidth=12,
            truncatePrimary=lambda context: (
                context.text
                if len(context.text) <= context.maxWidth
                else f"{context.text[: max(0, context.maxWidth - 1)]}…"
            ),
        ),
    )

    rendered = select_list.render(80)

    assert "…" in rendered[0]
    assert visible_index_of(rendered[0], "first") == visible_index_of(rendered[1], "second")
