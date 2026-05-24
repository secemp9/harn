from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest
from harnify_tui.autocomplete import CombinedAutocompleteProvider, SlashCommand


async def get_suggestions(
    provider: CombinedAutocompleteProvider,
    line: str,
    *,
    cursor_col: int | None = None,
    force: bool = False,
) -> object:
    return await provider.getSuggestions(
        [line],
        0,
        len(line) if cursor_col is None else cursor_col,
        {"signal": asyncio.Event(), "force": force},
    )


@pytest.mark.asyncio
async def test_command_completion_uses_fuzzy_matching() -> None:
    provider = CombinedAutocompleteProvider(
        [SlashCommand(name="model", description="Change model")],
        "/tmp",
    )

    result = await get_suggestions(provider, "/mod")

    assert result is not None
    assert result.prefix == "/mod"
    assert [item.value for item in result.items] == ["model"]


@pytest.mark.asyncio
async def test_argument_completion_returns_none_when_command_has_no_completion_handler() -> None:
    provider = CombinedAutocompleteProvider([SlashCommand(name="model", description="Change model")], "/tmp")

    result = await get_suggestions(provider, "/model high")

    assert result is None


@pytest.mark.asyncio
async def test_direct_path_completion_preserves_dot_slash(tmp_path: Path) -> None:
    (tmp_path / "update.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    provider = CombinedAutocompleteProvider([], str(tmp_path))

    result = await get_suggestions(provider, "./up", force=True)

    assert result is not None
    assert "./update.sh" in [item.value for item in result.items]


@pytest.mark.asyncio
async def test_quoted_completion_does_not_duplicate_closing_quote(tmp_path: Path) -> None:
    folder = tmp_path / "my folder"
    folder.mkdir()
    (folder / "test.txt").write_text("content", encoding="utf-8")
    provider = CombinedAutocompleteProvider([], str(tmp_path))
    line = '"my folder/te"'
    cursor_col = len(line) - 1

    result = await get_suggestions(provider, line, cursor_col=cursor_col, force=True)

    assert result is not None
    item = next(entry for entry in result.items if entry.value == '"my folder/test.txt"')
    applied = provider.applyCompletion([line], 0, cursor_col, item, result.prefix)
    assert applied["lines"][0] == '"my folder/test.txt"'


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("fd") is None, reason="fd not installed")
async def test_fd_file_completion_quotes_space_paths(tmp_path: Path) -> None:
    folder = tmp_path / "my folder"
    folder.mkdir()
    (folder / "test.txt").write_text("content", encoding="utf-8")
    provider = CombinedAutocompleteProvider([], str(tmp_path), shutil.which("fd"))

    result = await get_suggestions(provider, "@my")

    assert result is not None
    assert '@"my folder/"' in [item.value for item in result.items]
