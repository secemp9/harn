from __future__ import annotations

import importlib
import re
import time
from datetime import UTC, datetime, timedelta

from harnify_coding_agent.core.keybindings import KeybindingsManager
from harnify_coding_agent.core.session_manager import SessionInfo
from harnify_coding_agent.modes.interactive.components.countdown_timer import CountdownTimer
from harnify_coding_agent.modes.interactive.components.dynamic_border import DynamicBorder
from harnify_coding_agent.modes.interactive.components.keybinding_hints import (
    KeyTextFormatOptions,
    format_key_text,
    key_display_text,
    key_text,
    raw_key_hint,
)
from harnify_coding_agent.modes.interactive.components.session_selector_search import (
    filter_and_sort_sessions,
    has_session_name,
    parse_search_query,
)
from harnify_coding_agent.modes.interactive.components.show_images_selector import ShowImagesSelectorComponent
from harnify_coding_agent.modes.interactive.components.theme_selector import ThemeSelectorComponent
from harnify_coding_agent.modes.interactive.components.thinking_selector import ThinkingSelectorComponent
from harnify_coding_agent.modes.interactive.components.user_message_selector import (
    UserMessageItem,
    UserMessageSelectorComponent,
)
from harnify_tui import setKeybindings

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
interactive_theme_module = importlib.import_module("harnify_coding_agent.modes.interactive.theme.theme")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def setup_function() -> None:
    setKeybindings(KeybindingsManager())
    interactive_theme_module.init_theme("dark")


def test_theme_helpers_load_builtin_themes() -> None:
    names = [item["name"] for item in interactive_theme_module.get_available_themes_with_paths()]
    assert "dark" in names
    assert "light" in names

    switched = interactive_theme_module.set_theme("light")
    assert switched["success"] is True
    assert interactive_theme_module.theme.name == "light"


def test_keybinding_hints_format_text(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    assert format_key_text("alt+enter/ctrl+p", KeyTextFormatOptions(capitalize=True)) == "Option+Enter/Ctrl+P"

    monkeypatch.setattr("sys.platform", "linux")
    assert key_text("app.message.followUp") == "alt+enter"
    assert key_display_text("app.message.followUp") == "Alt+Enter"
    assert "follow-up" in _strip_ansi(raw_key_hint("alt+enter", "queue follow-up"))


def test_dynamic_border_renders_width_with_theme_color() -> None:
    border = DynamicBorder()
    rendered = border.render(5)
    assert len(rendered) == 1
    assert _strip_ansi(rendered[0]) == "─────"


def test_show_images_selector_preselects_and_confirms_current_value() -> None:
    selected: list[bool] = []
    cancelled: list[bool] = []
    component = ShowImagesSelectorComponent(False, selected.append, lambda: cancelled.append(True))

    assert component.getSelectList().getSelectedItem() is not None
    assert component.getSelectList().getSelectedItem().value == "no"

    component.handleInput("\r")
    assert selected == [False]
    assert cancelled == []


def test_thinking_selector_cycles_and_confirms_selection() -> None:
    selected: list[str] = []
    component = ThinkingSelectorComponent(
        "medium",
        ["minimal", "medium", "high"],
        selected.append,
        lambda: None,
    )

    assert component.getSelectList().getSelectedItem() is not None
    assert component.getSelectList().getSelectedItem().value == "medium"

    component.handleInput("\x1b[B")
    component.handleInput("\r")
    assert selected == ["high"]


def test_countdown_timer_ticks_and_expires() -> None:
    ticks: list[int] = []
    expired: list[bool] = []
    render_calls: list[bool] = []

    class FakeTui:
        def requestRender(self) -> None:
            render_calls.append(True)

    timer = CountdownTimer(1000, FakeTui(), ticks.append, lambda: expired.append(True))
    try:
        time.sleep(1.2)
    finally:
        timer.dispose()

    assert ticks[0] == 1
    assert ticks[-1] == 0
    assert expired == [True]
    assert render_calls


def test_theme_selector_previews_and_confirms() -> None:
    previewed: list[str] = []
    selected: list[str] = []
    component = ThemeSelectorComponent("dark", selected.append, lambda: None, previewed.append)

    assert component.getSelectList().getSelectedItem() is not None
    assert component.getSelectList().getSelectedItem().value == "dark"

    component.handleInput("\x1b[B")
    component.handleInput("\r")
    assert previewed
    assert selected == [component.getSelectList().getSelectedItem().value]


def test_session_selector_search_parses_and_filters() -> None:
    now = datetime.now(UTC)
    sessions = [
        SessionInfo(
            path="/tmp/a.jsonl",
            id="sess-a",
            cwd="/repo/a",
            created=now - timedelta(hours=2),
            modified=now - timedelta(minutes=30),
            messageCount=2,
            firstMessage="alpha",
            allMessagesText="fix parser issue",
            name="Parser Fix",
        ),
        SessionInfo(
            path="/tmp/b.jsonl",
            id="sess-b",
            cwd="/repo/b",
            created=now - timedelta(hours=1),
            modified=now - timedelta(minutes=5),
            messageCount=3,
            firstMessage="beta",
            allMessagesText="refactor ui theme selector",
            name=None,
        ),
    ]

    parsed = parse_search_query('parser "fix"')
    assert [token.kind for token in parsed.tokens] == ["fuzzy", "phrase"]
    assert has_session_name(sessions[0]) is True
    assert has_session_name(sessions[1]) is False

    named_only = filter_and_sort_sessions(sessions, "parser", "relevance", "named")
    assert [session.id for session in named_only] == ["sess-a"]

    regex_sorted = filter_and_sort_sessions(sessions, "re:theme", "recent", "all")
    assert [session.id for session in regex_sorted] == ["sess-b"]


def test_user_message_selector_selects_and_auto_cancels_empty_messages() -> None:
    selected: list[str] = []
    cancelled: list[bool] = []
    component = UserMessageSelectorComponent(
        [
            UserMessageItem(id="m1", text="first"),
            UserMessageItem(id="m2", text="second"),
        ],
        selected.append,
        lambda: cancelled.append(True),
    )

    assert component.getMessageList().selectedIndex == 1
    component.handleInput("\r")
    assert selected == ["m2"]
    assert cancelled == []

    empty_cancelled: list[bool] = []
    UserMessageSelectorComponent([], lambda _entry: None, lambda: empty_cancelled.append(True))
    time.sleep(0.15)
    assert empty_cancelled == [True]
