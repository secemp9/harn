from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import harnify_tui.autocomplete as autocomplete
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


class DummyAbortSignal:
    def __init__(self) -> None:
        self.aborted = False
        self._listeners: list[object] = []

    def addEventListener(self, event: str, callback: object, _options: dict[str, object] | None = None) -> None:
        if event != "abort":
            return
        if self.aborted:
            if callable(callback):
                callback()
            return
        self._listeners.append(callback)

    def removeEventListener(self, event: str, callback: object) -> None:
        if event != "abort":
            return
        self._listeners = [listener for listener in self._listeners if listener is not callback]

    def abort(self) -> None:
        if self.aborted:
            return
        self.aborted = True
        listeners = list(self._listeners)
        self._listeners.clear()
        for callback in listeners:
            if callable(callback):
                callback()


def test_autocomplete_module_exports_match_ts_surface() -> None:
    assert autocomplete.__all__ == [
        "AutocompleteItem",
        "AutocompleteProvider",
        "AutocompleteSuggestions",
        "CombinedAutocompleteProvider",
        "SlashCommand",
    ]


def test_combined_autocomplete_provider_keeps_null_fd_path_by_default() -> None:
    provider = CombinedAutocompleteProvider([], "/tmp")

    assert provider.fdPath is None


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
async def test_duck_typed_command_and_future_argument_completion_match_ts() -> None:
    loop = asyncio.get_running_loop()

    def get_argument_completions(prefix: str) -> asyncio.Future[list[autocomplete.AutocompleteItem] | None]:
        future: asyncio.Future[list[autocomplete.AutocompleteItem] | None] = loop.create_future()
        future.set_result([autocomplete.AutocompleteItem(value=f"{prefix}-opus", label=f"{prefix}-opus")])
        return future

    command = SimpleNamespace(
        name="model",
        description="Change model",
        argumentHint="<name>",
        getArgumentCompletions=get_argument_completions,
    )
    provider = CombinedAutocompleteProvider([command], "/tmp")

    command_result = await get_suggestions(provider, "/mod")
    assert command_result is not None
    assert [(item.value, item.description) for item in command_result.items] == [("model", "<name> — Change model")]

    argument_result = await get_suggestions(provider, "/model claude")
    assert argument_result is not None
    assert argument_result.prefix == "claude"
    assert [item.value for item in argument_result.items] == ["claude-opus"]


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


@pytest.mark.asyncio
async def test_walk_directory_with_fd_aborts_running_process(tmp_path: Path) -> None:
    fake_fd = tmp_path / "fake-fd"
    fake_fd.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, time\n"
        "time.sleep(1)\n"
        "sys.stdout.write('slow.txt\\n')\n",
        encoding="utf-8",
    )
    fake_fd.chmod(0o755)
    signal = DummyAbortSignal()

    started = time.monotonic()
    task = asyncio.create_task(autocomplete.walk_directory_with_fd(str(tmp_path), str(fake_fd), "", 100, signal))
    await asyncio.sleep(0.05)
    signal.abort()
    result = await task
    elapsed = time.monotonic() - started

    assert result == []
    assert elapsed < 0.5
