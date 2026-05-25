from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

import harnify_tui.components.editor as editor_module
import pytest
from harnify_tui.autocomplete import (
    AutocompleteItem,
    AutocompleteSuggestions,
    CombinedAutocompleteProvider,
    SlashCommand,
)
from harnify_tui.components.editor import Editor, EditorTheme
from harnify_tui.components.select_list import SelectListTheme


def _identity(text: str) -> str:
    return text


DEFAULT_SELECT_LIST_THEME = SelectListTheme(
    selectedPrefix=_identity,
    selectedText=_identity,
    description=_identity,
    scrollInfo=_identity,
    noMatch=_identity,
)

DEFAULT_EDITOR_THEME = EditorTheme(borderColor=_identity, selectList=DEFAULT_SELECT_LIST_THEME)


def apply_completion_replace_prefix(
    lines: list[str],
    cursor_line: int,
    cursor_col: int,
    item: AutocompleteItem,
    prefix: str,
) -> dict[str, object]:
    line = lines[cursor_line] if cursor_line < len(lines) else ""
    before = line[: cursor_col - len(prefix)]
    after = line[cursor_col:]
    new_lines = list(lines)
    new_lines[cursor_line] = before + item.value + after
    return {
        "lines": new_lines,
        "cursorLine": cursor_line,
        "cursorCol": cursor_col - len(prefix) + len(item.value),
    }


@dataclass
class DummyTerminal:
    rows: int = 24


@dataclass
class DummyTUI:
    terminal: DummyTerminal = field(default_factory=DummyTerminal)
    render_requests: int = 0

    def requestRender(self) -> None:
        self.render_requests += 1


class MockAutocompleteProvider:
    def __init__(self, get_suggestions: Any, apply_completion: Any | None = None) -> None:
        self._get_suggestions = get_suggestions
        self._apply_completion = apply_completion or apply_completion_replace_prefix

    async def getSuggestions(
        self,
        lines: list[str],
        cursorLine: int,
        cursorCol: int,
        options: dict[str, object],
    ) -> AutocompleteSuggestions | None:
        return await self._get_suggestions(lines, cursorLine, cursorCol, options)

    def applyCompletion(
        self,
        lines: list[str],
        cursorLine: int,
        cursorCol: int,
        item: AutocompleteItem,
        prefix: str,
    ) -> dict[str, object]:
        return self._apply_completion(lines, cursorLine, cursorCol, item, prefix)

    def shouldTriggerFileCompletion(self, _lines: list[str], _cursorLine: int, _cursorCol: int) -> bool:
        return True


def create_editor(*, rows: int = 24) -> tuple[Editor, DummyTUI]:
    tui = DummyTUI(terminal=DummyTerminal(rows=rows))
    return Editor(tui, DEFAULT_EDITOR_THEME), tui


async def flush_editor(delay: float = 0.0) -> None:
    await asyncio.sleep(delay)
    await asyncio.sleep(0)


def move_right(editor: Editor, count: int) -> None:
    for _ in range(count):
        editor.handleInput("\x1b[C")


def test_editor_module_exports_match_ts_surface() -> None:
    assert editor_module.__all__ == [
        "Editor",
        "EditorOptions",
        "EditorTheme",
        "TextChunk",
        "wordWrapLine",
    ]


def test_history_up_arrow_does_nothing_when_empty() -> None:
    editor, _tui = create_editor()

    editor.handleInput("\x1b[A")

    assert editor.getText() == ""


def test_history_cycles_and_down_returns_to_empty() -> None:
    editor, _tui = create_editor()
    editor.addToHistory("first")
    editor.addToHistory("second")
    editor.addToHistory("third")

    editor.handleInput("\x1b[A")
    assert editor.getText() == "third"

    editor.handleInput("\x1b[A")
    assert editor.getText() == "second"

    editor.handleInput("\x1b[A")
    assert editor.getText() == "first"

    editor.handleInput("\x1b[B")
    assert editor.getText() == "second"

    editor.handleInput("\x1b[B")
    assert editor.getText() == "third"

    editor.handleInput("\x1b[B")
    assert editor.getText() == ""


def test_history_skips_empty_and_consecutive_duplicates() -> None:
    editor, _tui = create_editor()
    editor.addToHistory("")
    editor.addToHistory("   ")
    editor.addToHistory("same")
    editor.addToHistory("same")
    editor.addToHistory("other")

    editor.handleInput("\x1b[A")
    assert editor.getText() == "other"

    editor.handleInput("\x1b[A")
    assert editor.getText() == "same"

    editor.handleInput("\x1b[A")
    assert editor.getText() == "same"


def test_editor_history_navigation_exits_when_typing() -> None:
    editor, _tui = create_editor()
    editor.addToHistory("older")
    editor.addToHistory("newer")

    editor.handleInput("\x1b[A")
    assert editor.getText() == "newer"

    editor.handleInput("\x1b[A")
    assert editor.getText() == "older"

    editor.handleInput("x")
    assert editor.getText() == "olderx"


def test_insert_text_at_cursor_is_atomic_for_undo() -> None:
    editor, _tui = create_editor()

    editor.insertTextAtCursor("hello")
    editor.insertTextAtCursor("\nworld")
    assert editor.getText() == "hello\nworld"

    editor.handleInput("\x1b[45;5u")
    assert editor.getText() == "hello"


def test_history_mode_uses_cursor_movement_when_editor_has_content() -> None:
    editor, _tui = create_editor()
    editor.addToHistory("history item")
    editor.setText("line1\nline2")

    editor.handleInput("\x1b[A")
    editor.handleInput("X")

    assert editor.getText() == "line1X\nline2"


def test_public_state_accessors_return_cursor_and_defensive_line_copy() -> None:
    editor, _tui = create_editor()
    assert editor.getCursor() == {"line": 0, "col": 0}

    editor.setText("a\nb")
    assert editor.getCursor() == {"line": 1, "col": 1}
    lines = editor.getLines()
    assert lines == ["a", "b"]

    lines[0] = "mutated"
    assert editor.getLines() == ["a", "b"]


def test_ctrl_w_saves_deleted_text_to_kill_ring_and_ctrl_y_yanks_it() -> None:
    editor, _tui = create_editor()
    editor.setText("foo bar baz")

    editor.handleInput("\x17")
    assert editor.getText() == "foo bar "

    editor.handleInput("\x01")
    editor.handleInput("\x19")
    assert editor.getText() == "bazfoo bar "


def test_alt_y_cycles_through_kill_ring_after_yank() -> None:
    editor, _tui = create_editor()
    editor.setText("first")
    editor.handleInput("\x17")
    editor.setText("second")
    editor.handleInput("\x17")
    editor.setText("third")
    editor.handleInput("\x17")

    editor.handleInput("\x19")
    assert editor.getText() == "third"

    editor.handleInput("\x1by")
    assert editor.getText() == "second"

    editor.handleInput("\x1by")
    assert editor.getText() == "first"

    editor.handleInput("\x1by")
    assert editor.getText() == "third"


def test_undo_tracks_newlines_as_separate_units() -> None:
    editor, _tui = create_editor()
    for char in "hello":
        editor.handleInput(char)
    editor.handleInput("\n")
    for char in "world":
        editor.handleInput(char)

    assert editor.getText() == "hello\nworld"

    editor.handleInput("\x1b[45;5u")
    assert editor.getText() == "hello\n"

    editor.handleInput("\x1b[45;5u")
    assert editor.getText() == "hello"

    editor.handleInput("\x1b[45;5u")
    assert editor.getText() == ""


def test_undo_exits_history_browsing_mode_back_to_pre_history_state() -> None:
    editor, _tui = create_editor()
    editor.addToHistory("hello")

    for char in "world":
        editor.handleInput(char)
    assert editor.getText() == "world"

    editor.handleInput("\x17")
    assert editor.getText() == ""

    editor.handleInput("\x1b[A")
    assert editor.getText() == "hello"

    editor.handleInput("\x1b[45;5u")
    assert editor.getText() == ""

    editor.handleInput("\x1b[45;5u")
    assert editor.getText() == "world"


@pytest.mark.asyncio
async def test_force_tab_single_suggestion_auto_applies_without_menu() -> None:
    editor, _tui = create_editor()

    async def get_suggestions(
        lines: list[str],
        _cursor_line: int,
        cursor_col: int,
        options: dict[str, object],
    ) -> AutocompleteSuggestions | None:
        if not options.get("force"):
            return None
        prefix = (lines[0] if lines else "")[:cursor_col]
        if prefix != "Work":
            return None
        return AutocompleteSuggestions(
            items=[AutocompleteItem(value="Workspace/", label="Workspace/")],
            prefix="Work",
        )

    editor.setAutocompleteProvider(MockAutocompleteProvider(get_suggestions))
    for char in "Work":
        editor.handleInput(char)

    editor.handleInput("\t")
    await flush_editor(0.01)

    assert editor.getText() == "Workspace/"
    assert editor.isShowingAutocomplete() is False

    editor.handleInput("\x1b[45;5u")
    assert editor.getText() == "Work"


@pytest.mark.asyncio
async def test_force_tab_multiple_suggestions_shows_menu_then_accepts_first() -> None:
    editor, _tui = create_editor()

    async def get_suggestions(
        lines: list[str],
        _cursor_line: int,
        cursor_col: int,
        options: dict[str, object],
    ) -> AutocompleteSuggestions | None:
        if not options.get("force"):
            return None
        prefix = (lines[0] if lines else "")[:cursor_col]
        if prefix != "src":
            return None
        return AutocompleteSuggestions(
            items=[
                AutocompleteItem(value="src/", label="src/"),
                AutocompleteItem(value="src.txt", label="src.txt"),
            ],
            prefix="src",
        )

    editor.setAutocompleteProvider(MockAutocompleteProvider(get_suggestions))
    for char in "src":
        editor.handleInput(char)

    editor.handleInput("\t")
    await flush_editor(0.01)

    assert editor.getText() == "src"
    assert editor.isShowingAutocomplete() is True

    editor.handleInput("\t")
    assert editor.getText() == "src/"
    assert editor.isShowingAutocomplete() is False


@pytest.mark.asyncio
async def test_force_mode_keeps_suggestions_open_while_typing() -> None:
    editor, _tui = create_editor()
    all_files = [
        AutocompleteItem(value="readme.md", label="readme.md"),
        AutocompleteItem(value="package.json", label="package.json"),
        AutocompleteItem(value="src/", label="src/"),
        AutocompleteItem(value="dist/", label="dist/"),
    ]

    async def get_suggestions(
        lines: list[str],
        _cursor_line: int,
        cursor_col: int,
        options: dict[str, object],
    ) -> AutocompleteSuggestions | None:
        prefix = (lines[0] if lines else "")[:cursor_col]
        should_match = bool(options.get("force")) or "/" in prefix or prefix.startswith(".")
        if not should_match:
            return None
        filtered = [item for item in all_files if item.value.lower().startswith(prefix.lower())]
        if not filtered:
            return None
        return AutocompleteSuggestions(items=filtered, prefix=prefix)

    editor.setAutocompleteProvider(MockAutocompleteProvider(get_suggestions))

    editor.handleInput("\t")
    await flush_editor(0.01)
    assert editor.isShowingAutocomplete() is True

    editor.handleInput("r")
    await flush_editor(0.01)
    assert editor.getText() == "r"
    assert editor.isShowingAutocomplete() is True

    editor.handleInput("e")
    await flush_editor(0.01)
    assert editor.getText() == "re"
    assert editor.isShowingAutocomplete() is True

    editor.handleInput("\t")
    assert editor.getText() == "readme.md"
    assert editor.isShowingAutocomplete() is False


@pytest.mark.asyncio
async def test_at_autocomplete_is_debounced_while_typing() -> None:
    editor, _tui = create_editor()
    calls = 0

    async def get_suggestions(
        lines: list[str],
        _cursor_line: int,
        cursor_col: int,
        _options: dict[str, object],
    ) -> AutocompleteSuggestions | None:
        nonlocal calls
        calls += 1
        prefix = (lines[0] if lines else "")[:cursor_col]
        return AutocompleteSuggestions(
            items=[AutocompleteItem(value="@main.ts", label="main.ts")],
            prefix=prefix,
        )

    editor.setAutocompleteProvider(MockAutocompleteProvider(get_suggestions))
    for char in "@mai":
        editor.handleInput(char)

    assert calls == 0
    assert editor.isShowingAutocomplete() is False

    await flush_editor(0.05)

    assert calls == 1
    assert editor.isShowingAutocomplete() is True


@pytest.mark.asyncio
async def test_hash_autocomplete_is_debounced_while_typing() -> None:
    editor, _tui = create_editor()
    calls = 0

    async def get_suggestions(
        lines: list[str],
        _cursor_line: int,
        cursor_col: int,
        _options: dict[str, object],
    ) -> AutocompleteSuggestions | None:
        nonlocal calls
        calls += 1
        prefix = (lines[0] if lines else "")[:cursor_col]
        return AutocompleteSuggestions(
            items=[AutocompleteItem(value="#2983", label="#2983")],
            prefix=prefix,
        )

    editor.setAutocompleteProvider(MockAutocompleteProvider(get_suggestions))
    for char in "#298":
        editor.handleInput(char)

    assert calls == 0
    assert editor.isShowingAutocomplete() is False

    await flush_editor(0.05)

    assert calls == 1
    assert editor.isShowingAutocomplete() is True


@pytest.mark.asyncio
async def test_typing_continues_aborts_active_autocomplete_request() -> None:
    editor, _tui = create_editor()
    aborts = 0

    async def get_suggestions(
        lines: list[str],
        _cursor_line: int,
        cursor_col: int,
        options: dict[str, object],
    ) -> AutocompleteSuggestions | None:
        nonlocal aborts
        prefix = (lines[0] if lines else "")[:cursor_col]
        if prefix == "@mai":
            future: asyncio.Future[AutocompleteSuggestions | None] = asyncio.get_running_loop().create_future()

            def on_abort() -> None:
                nonlocal aborts
                aborts += 1
                if not future.done():
                    future.set_result(None)

            signal = options["signal"]
            signal.addEventListener("abort", on_abort, {"once": True})
            return await future

        return AutocompleteSuggestions(
            items=[AutocompleteItem(value="@main.ts", label="main.ts")],
            prefix=prefix,
        )

    editor.setAutocompleteProvider(MockAutocompleteProvider(get_suggestions))
    for char in "@mai":
        editor.handleInput(char)

    await flush_editor(0.03)
    editor.handleInput("n")
    await flush_editor(0.06)

    assert aborts == 1


@pytest.mark.asyncio
async def test_new_autocomplete_request_waits_for_previous_abort_cleanup() -> None:
    editor, _tui = create_editor()
    events: list[str] = []
    loop = asyncio.get_running_loop()

    async def get_suggestions(
        lines: list[str],
        _cursor_line: int,
        cursor_col: int,
        options: dict[str, object],
    ) -> AutocompleteSuggestions | None:
        prefix = (lines[0] if lines else "")[:cursor_col]
        if prefix == "/":
            events.append("first-start")
            future: asyncio.Future[AutocompleteSuggestions | None] = loop.create_future()

            def finish_first() -> None:
                events.append("first-finish")
                if not future.done():
                    future.set_result(None)

            def on_abort() -> None:
                events.append("first-abort")
                loop.call_later(0.02, finish_first)

            signal = options["signal"]
            signal.addEventListener("abort", on_abort, {"once": True})
            return await future

        events.append("second-start")
        return AutocompleteSuggestions(
            items=[AutocompleteItem(value="help", label="help")],
            prefix=prefix,
        )

    editor.setAutocompleteProvider(MockAutocompleteProvider(get_suggestions))
    editor.handleInput("/")
    await flush_editor(0.01)

    editor.handleInput("h")
    await flush_editor(0.06)

    assert events == ["first-start", "first-abort", "first-finish", "second-start"]


@pytest.mark.asyncio
async def test_backspacing_empty_slash_context_hides_autocomplete() -> None:
    editor, _tui = create_editor()

    async def get_suggestions(
        lines: list[str],
        _cursor_line: int,
        cursor_col: int,
        _options: dict[str, object],
    ) -> AutocompleteSuggestions | None:
        prefix = (lines[0] if lines else "")[:cursor_col]
        if not prefix.startswith("/"):
            return None
        return AutocompleteSuggestions(
            items=[
                AutocompleteItem(value="model", label="model", description="Change model"),
                AutocompleteItem(value="help", label="help", description="Show help"),
            ],
            prefix=prefix,
        )

    editor.setAutocompleteProvider(MockAutocompleteProvider(get_suggestions))
    editor.handleInput("/")
    await flush_editor(0.01)
    assert editor.isShowingAutocomplete() is True

    editor.handleInput("\x7f")
    await flush_editor(0.01)

    assert editor.getText() == ""
    assert editor.isShowingAutocomplete() is False


@pytest.mark.asyncio
async def test_exact_typed_slash_argument_is_retained_on_enter() -> None:
    editor, _tui = create_editor()

    async def get_suggestions(
        lines: list[str],
        _cursor_line: int,
        cursor_col: int,
        _options: dict[str, object],
    ) -> AutocompleteSuggestions | None:
        before_cursor = (lines[0] if lines else "")[:cursor_col]
        match = re.match(r"^/argtest\s+(\S+)$", before_cursor)
        if match is None:
            return None
        argument_text = match.group(1)
        all_arguments = [
            AutocompleteItem(value="one", label="one"),
            AutocompleteItem(value="two", label="two"),
            AutocompleteItem(value="three", label="three"),
        ]
        filtered = [item for item in all_arguments if item.value.startswith(argument_text)]
        if not filtered:
            return None
        return AutocompleteSuggestions(items=filtered, prefix=argument_text)

    editor.setAutocompleteProvider(MockAutocompleteProvider(get_suggestions))
    for char in "/argtest two":
        editor.handleInput(char)

    await flush_editor(0.01)
    assert editor.isShowingAutocomplete() is True

    editor.handleInput("\r")
    assert editor.getText() == "/argtest two"


@pytest.mark.asyncio
async def test_prefix_slash_argument_selects_first_prefix_match_on_enter() -> None:
    editor, _tui = create_editor()

    async def get_suggestions(
        lines: list[str],
        _cursor_line: int,
        cursor_col: int,
        _options: dict[str, object],
    ) -> AutocompleteSuggestions | None:
        before_cursor = (lines[0] if lines else "")[:cursor_col]
        match = re.match(r"^/argtest\s+(\S+)$", before_cursor)
        if match is None:
            return None
        argument_text = match.group(1)
        all_arguments = [
            AutocompleteItem(value="two", label="two"),
            AutocompleteItem(value="three", label="three"),
            AutocompleteItem(value="twelve", label="twelve"),
        ]
        filtered = [item for item in all_arguments if item.value.startswith(argument_text)]
        if not filtered:
            return None
        return AutocompleteSuggestions(items=filtered, prefix=argument_text)

    editor.setAutocompleteProvider(MockAutocompleteProvider(get_suggestions))
    for char in "/argtest t":
        editor.handleInput(char)

    await flush_editor(0.01)
    assert editor.isShowingAutocomplete() is True

    editor.handleInput("\r")
    assert editor.getText() == "/argtest two"


@pytest.mark.asyncio
async def test_unique_prefix_match_is_selected_before_full_exact_match() -> None:
    editor, _tui = create_editor()

    async def get_suggestions(
        lines: list[str],
        _cursor_line: int,
        cursor_col: int,
        _options: dict[str, object],
    ) -> AutocompleteSuggestions | None:
        before_cursor = (lines[0] if lines else "")[:cursor_col]
        match = re.match(r"^/argtest\s+(\S+)$", before_cursor)
        if match is None:
            return None
        return AutocompleteSuggestions(
            items=[
                AutocompleteItem(value="one", label="one"),
                AutocompleteItem(value="two", label="two"),
                AutocompleteItem(value="three", label="three"),
            ],
            prefix=match.group(1),
        )

    editor.setAutocompleteProvider(MockAutocompleteProvider(get_suggestions))
    for char in "/argtest tw":
        editor.handleInput(char)

    await flush_editor(0.01)
    assert editor.isShowingAutocomplete() is True

    editor.handleInput("\r")
    assert editor.getText() == "/argtest two"


@pytest.mark.asyncio
async def test_async_slash_argument_completion_with_combined_provider() -> None:
    editor, _tui = create_editor()
    provider = CombinedAutocompleteProvider(
        [
            SlashCommand(
                name="load-skills",
                description="Load skills",
                getArgumentCompletions=lambda prefix: (
                    asyncio.sleep(0, result=[AutocompleteItem(value="skill-a", label="skill-a")])
                    if prefix.startswith("s")
                    else asyncio.sleep(0, result=None)
                ),
            ),
        ],
        ".",
    )
    editor.setAutocompleteProvider(provider)
    editor.setText("/load-skills ")

    editor.handleInput("s")
    await flush_editor(0.01)
    assert editor.isShowingAutocomplete() is True

    editor.handleInput("\t")
    assert editor.getText() == "/load-skills skill-a"
    assert editor.isShowingAutocomplete() is False


@pytest.mark.asyncio
async def test_invalid_slash_argument_completion_results_are_ignored() -> None:
    editor, _tui = create_editor()
    provider = CombinedAutocompleteProvider(
        [
            SlashCommand(
                name="load-skills",
                description="Load skills",
                getArgumentCompletions=lambda _prefix: "not-an-array",
            )
        ],
        ".",
    )
    editor.setAutocompleteProvider(provider)
    editor.setText("/load-skills ")

    editor.handleInput("s")
    await flush_editor(0.01)

    assert editor.isShowingAutocomplete() is False
    assert editor.getText() == "/load-skills s"


@pytest.mark.asyncio
async def test_command_without_argument_completer_does_not_show_argument_completion() -> None:
    editor, _tui = create_editor()
    provider = CombinedAutocompleteProvider(
        [
            SlashCommand(name="help", description="Show help"),
            SlashCommand(
                name="model",
                description="Switch model",
                getArgumentCompletions=lambda _prefix: [AutocompleteItem(value="claude-opus", label="claude-opus")],
            ),
        ],
        ".",
    )
    editor.setAutocompleteProvider(provider)

    editor.handleInput("/")
    editor.handleInput("h")
    editor.handleInput("e")
    await flush_editor(0.01)
    assert editor.isShowingAutocomplete() is True

    editor.handleInput("\t")
    assert editor.getText() == "/help "
    assert editor.isShowingAutocomplete() is False


@pytest.mark.asyncio
async def test_undoes_single_line_bracketed_paste_atomically() -> None:
    editor, _tui = create_editor()
    editor.setText("hello world")
    editor.handleInput("\x01")
    move_right(editor, 5)

    editor.handleInput("\x1b[200~beep boop\x1b[201~")
    assert editor.getText() == "hellobeep boop world"

    editor.handleInput("\x1b[45;5u")
    assert editor.getText() == "hello world"
