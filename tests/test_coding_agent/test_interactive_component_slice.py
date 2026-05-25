from __future__ import annotations

import importlib
import json
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from harnify_coding_agent.core.keybindings import KeybindingsManager
from harnify_coding_agent.core.session_manager import SessionInfo
from harnify_coding_agent.modes.interactive.components.countdown_timer import CountdownTimer
from harnify_coding_agent.modes.interactive.components.daxnuts import DaxnutsComponent
from harnify_coding_agent.modes.interactive.components.dynamic_border import DynamicBorder
from harnify_coding_agent.modes.interactive.components.earendil_announcement import (
    EarendilAnnouncementComponent,
)
from harnify_coding_agent.modes.interactive.components.extension_editor import (
    ExtensionEditorComponent,
)
from harnify_coding_agent.modes.interactive.components.extension_input import (
    ExtensionInputComponent,
)
from harnify_coding_agent.modes.interactive.components.extension_selector import (
    ExtensionSelectorComponent,
)
from harnify_coding_agent.modes.interactive.components.keybinding_hints import (
    KeyTextFormatOptions,
    formatKeyText,
    keyDisplayText,
    keyText,
    rawKeyHint,
)
from harnify_coding_agent.modes.interactive.components.session_selector_search import (
    filterAndSortSessions,
    hasSessionName,
    parseSearchQuery,
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
countdown_timer_module = importlib.import_module(
    "harnify_coding_agent.modes.interactive.components.countdown_timer"
)
daxnuts_module = importlib.import_module("harnify_coding_agent.modes.interactive.components.daxnuts")
dynamic_border_module = importlib.import_module(
    "harnify_coding_agent.modes.interactive.components.dynamic_border"
)
earendil_announcement_module = importlib.import_module(
    "harnify_coding_agent.modes.interactive.components.earendil_announcement"
)
extension_editor_module = importlib.import_module(
    "harnify_coding_agent.modes.interactive.components.extension_editor"
)
extension_input_module = importlib.import_module(
    "harnify_coding_agent.modes.interactive.components.extension_input"
)
extension_selector_module = importlib.import_module(
    "harnify_coding_agent.modes.interactive.components.extension_selector"
)
keybinding_hints_module = importlib.import_module(
    "harnify_coding_agent.modes.interactive.components.keybinding_hints"
)
session_selector_search_module = importlib.import_module(
    "harnify_coding_agent.modes.interactive.components.session_selector_search"
)
show_images_selector_module = importlib.import_module(
    "harnify_coding_agent.modes.interactive.components.show_images_selector"
)
theme_selector_module = importlib.import_module(
    "harnify_coding_agent.modes.interactive.components.theme_selector"
)
thinking_selector_module = importlib.import_module(
    "harnify_coding_agent.modes.interactive.components.thinking_selector"
)
user_message_selector_module = importlib.import_module(
    "harnify_coding_agent.modes.interactive.components.user_message_selector"
)


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def setup_function() -> None:
    setKeybindings(KeybindingsManager())
    interactive_theme_module.init_theme("dark")


class FakeTimer:
    def __init__(self, interval: float, callback) -> None:  # noqa: ANN001
        self.interval = interval
        self.callback = callback
        self.started = False
        self.cancelled = False
        self.daemon = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True


def test_theme_helpers_load_builtin_themes() -> None:
    names = [item["name"] for item in interactive_theme_module.get_available_themes_with_paths()]
    assert "dark" in names
    assert "light" in names

    switched = interactive_theme_module.set_theme("light")
    assert switched["success"] is True
    assert interactive_theme_module.theme.name == "light"


def test_theme_module_exports_match_ts_surface() -> None:
    assert interactive_theme_module.__all__ == [
        "Theme",
        "ThemeBg",
        "ThemeColor",
        "ThemeInfo",
        "TerminalTheme",
        "RgbColor",
        "TerminalThemeDetection",
        "TerminalThemeDetectionOptions",
        "getAvailableThemes",
        "getAvailableThemesWithPaths",
        "loadThemeFromPath",
        "getThemeByName",
        "getThemeForRgbColor",
        "parseOsc11BackgroundColor",
        "detectTerminalBackground",
        "getDefaultTheme",
        "theme",
        "setRegisteredThemes",
        "initTheme",
        "setTheme",
        "setThemeInstance",
        "onThemeChange",
        "stopThemeWatcher",
        "getResolvedThemeColors",
        "isLightTheme",
        "getThemeExportColors",
        "highlightCode",
        "getLanguageFromPath",
        "getMarkdownTheme",
        "getSelectListTheme",
        "getEditorTheme",
        "getSettingsListTheme",
    ]


def test_theme_helpers_match_ts_detection_and_name_contracts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    custom_theme_dir = tmp_path / "themes"
    custom_theme_dir.mkdir()
    payload = interactive_theme_module.load_theme_json("dark")
    payload = {
        "name": "Oceanic Next",
        "vars": dict(payload.get("vars") or {}),
        "colors": dict(payload["colors"]),
    }
    (custom_theme_dir / "oceanic-next.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(interactive_theme_module, "get_custom_themes_dir", lambda: str(custom_theme_dir))
    interactive_theme_module.set_registered_themes([])

    names = interactive_theme_module.get_available_themes()
    assert "Oceanic Next" in names
    assert "oceanic-next" not in names

    detection = interactive_theme_module.detectTerminalBackground({"env": {"COLORFGBG": "15;0"}})
    assert detection == {
        "theme": "dark",
        "source": "COLORFGBG",
        "detail": "background color index 0",
        "confidence": "high",
    }

    assert interactive_theme_module.getThemeForRgbColor({"r": 255, "g": 255, "b": 255}) == "light"
    assert interactive_theme_module.parseOsc11BackgroundColor("\x1b]11;rgb:ffff/ffff/ffff\x07") == {
        "r": 255,
        "g": 255,
        "b": 255,
    }
    assert interactive_theme_module.isLightTheme("light") is True
    assert interactive_theme_module.isLightTheme("dark") is False
    assert interactive_theme_module.getThemeByName("__missing_theme__") is None


def test_theme_settings_list_cursor_matches_ts_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(interactive_theme_module.sys, "platform", "win32")
    settings_theme = interactive_theme_module.getSettingsListTheme()
    assert _strip_ansi(settings_theme.cursor) == "→ "


def test_keybinding_hints_format_text(monkeypatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    assert formatKeyText("alt+enter/ctrl+p", KeyTextFormatOptions(capitalize=True)) == "Option+Enter/Ctrl+P"

    monkeypatch.setattr("sys.platform", "linux")
    assert keyText("app.message.followUp") == "alt+enter"
    assert keyDisplayText("app.message.followUp") == "Alt+Enter"
    assert "follow-up" in _strip_ansi(rawKeyHint("alt+enter", "queue follow-up"))


def test_keybinding_hints_module_exports_match_ts_surface() -> None:
    assert keybinding_hints_module.__all__ == [
        "KeyTextFormatOptions",
        "formatKeyText",
        "keyDisplayText",
        "keyHint",
        "keyText",
        "rawKeyHint",
    ]


def test_session_selector_search_module_exports_match_ts_surface() -> None:
    assert session_selector_search_module.__all__ == [
        "MatchResult",
        "NameFilter",
        "ParsedSearchQuery",
        "SortMode",
        "filterAndSortSessions",
        "getSessionSearchText",
        "hasSessionName",
        "matchSession",
        "parseSearchQuery",
    ]


def test_dynamic_border_renders_width_with_theme_color() -> None:
    border = DynamicBorder()
    rendered = border.render(5)
    assert len(rendered) == 1
    assert _strip_ansi(rendered[0]) == "─────"


def test_dynamic_border_module_exports_match_ts_surface() -> None:
    assert dynamic_border_module.__all__ == ["DynamicBorder"]


def test_show_images_selector_preselects_and_confirms_current_value() -> None:
    selected: list[bool] = []
    cancelled: list[bool] = []
    component = ShowImagesSelectorComponent(False, selected.append, lambda: cancelled.append(True))

    assert component.getSelectList().getSelectedItem() is not None
    assert component.getSelectList().getSelectedItem().value == "no"

    component.handleInput("\r")
    assert selected == [False]
    assert cancelled == []


def test_show_images_selector_module_exports_match_ts_surface() -> None:
    assert show_images_selector_module.__all__ == ["ShowImagesSelectorComponent"]


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


def test_theme_selector_module_exports_match_ts_surface() -> None:
    assert theme_selector_module.__all__ == ["ThemeSelectorComponent"]


def test_thinking_selector_module_exports_match_ts_surface() -> None:
    assert thinking_selector_module.__all__ == ["ThinkingSelectorComponent"]


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


def test_countdown_timer_zero_timeout_matches_ts_tick_sequence() -> None:
    ticks: list[int] = []
    expired: list[bool] = []

    timer = CountdownTimer(0, None, ticks.append, lambda: expired.append(True))
    try:
        time.sleep(1.2)
    finally:
        timer.dispose()

    assert ticks == [0, -1]
    assert expired == [True]


def test_countdown_timer_module_exports_match_ts_surface() -> None:
    assert countdown_timer_module.__all__ == ["CountdownTimer"]


def test_daxnuts_component_renders_and_stops_animation(monkeypatch) -> None:
    timers: list[FakeTimer] = []
    render_calls: list[bool | None] = []

    class FakeUi:
        def requestRender(self, force: bool | None = None) -> None:
            render_calls.append(force)

    def timer_factory(interval: float, callback):  # noqa: ANN001
        timer = FakeTimer(interval, callback)
        timers.append(timer)
        return timer

    monkeypatch.setattr(daxnuts_module.threading, "Timer", timer_factory)

    ui = FakeUi()
    component = DaxnutsComponent(ui)
    rendered = [_strip_ansi(line) for line in component.render(80)]

    assert len(rendered) == 25
    assert "▓" * 32 in rendered[1]
    assert len(timers) == 1
    assert timers[0].started is True

    component.dispose()

    assert timers[0].cancelled is True


def test_daxnuts_component_advances_tick_and_requests_render(monkeypatch) -> None:
    timers: list[FakeTimer] = []
    render_calls: list[bool | None] = []

    class FakeUi:
        def requestRender(self, force: bool | None = None) -> None:
            render_calls.append(force)

    def timer_factory(interval: float, callback):  # noqa: ANN001
        timer = FakeTimer(interval, callback)
        timers.append(timer)
        return timer

    monkeypatch.setattr(daxnuts_module.threading, "Timer", timer_factory)

    ui = FakeUi()
    component = DaxnutsComponent(ui)

    timers[0].callback()

    assert component.tick == 1
    assert render_calls == [None]
    assert len(timers) == 2
    assert timers[1].started is True

    component.dispose()


def test_daxnuts_module_exports_match_ts_surface() -> None:
    assert daxnuts_module.__all__ == ["DaxnutsComponent"]


def test_earendil_announcement_renders_banner_without_image(monkeypatch) -> None:
    monkeypatch.setattr(earendil_announcement_module, "_load_image_base64", lambda: None)
    component = EarendilAnnouncementComponent()

    rendered = _strip_ansi("\n".join(component.render(80)))

    assert "pi has joined Earendil" in rendered
    assert "Read the blog post:" in rendered
    assert "https://mariozechner.at/posts/2026-04-08-ive-sold-out/" in rendered


def test_earendil_announcement_module_exports_match_ts_surface() -> None:
    assert earendil_announcement_module.__all__ == ["EarendilAnnouncementComponent"]


def test_extension_editor_routes_cancel_and_external_shortcuts(monkeypatch) -> None:
    class FakeUi:
        def requestRender(self, force: bool | None = None) -> None:
            del force

    cancelled: list[bool] = []
    scheduled: list[bool] = []
    forwarded: list[str] = []
    component = ExtensionEditorComponent(
        FakeUi(),
        KeybindingsManager(),
        "Title",
        None,
        lambda _value: None,
        lambda: cancelled.append(True),
    )
    monkeypatch.setattr(component, "_schedule_external_editor", lambda: scheduled.append(True))
    monkeypatch.setattr(component.editor, "handleInput", lambda data: forwarded.append(data))

    component.handleInput("\x1b")
    component.handleInput("\x07")
    component.handleInput("x")

    assert cancelled == [True]
    assert scheduled == [True]
    assert forwarded == ["x"]


def test_extension_editor_module_exports_match_ts_surface() -> None:
    assert extension_editor_module.__all__ == ["ExtensionEditorComponent"]


def test_extension_input_ignores_placeholder_and_routes_submit_cancel() -> None:
    submitted: list[str] = []
    cancelled: list[bool] = []
    component = ExtensionInputComponent(
        "Prompt",
        "placeholder text",
        submitted.append,
        lambda: cancelled.append(True),
    )
    component.input.setValue("typed")

    rendered = _strip_ansi("\n".join(component.render(80)))
    assert "placeholder text" not in rendered

    component.handleInput("\n")
    component.handleInput("\x1b")

    assert submitted == ["typed"]
    assert cancelled == [True]


def test_extension_input_module_exports_match_ts_surface() -> None:
    assert extension_input_module.__all__ == ["ExtensionInputComponent"]


def test_extension_selector_renders_and_handles_navigation_callbacks() -> None:
    selected: list[str] = []
    cancelled: list[bool] = []
    toggled: list[bool] = []
    component = ExtensionSelectorComponent(
        "Choose",
        ["alpha", "beta"],
        selected.append,
        lambda: cancelled.append(True),
        {"onToggleToolsExpanded": lambda: toggled.append(True)},
    )

    initial = _strip_ansi("\n".join(component.render(80)))
    assert "→ alpha" in initial

    component.handleInput("j")
    moved = _strip_ansi("\n".join(component.render(80)))
    assert "→ beta" in moved

    component.handleInput("\x0f")
    component.handleInput("\n")
    component.handleInput("\x1b")

    assert toggled == [True]
    assert selected == ["beta"]
    assert cancelled == [True]


def test_extension_selector_module_exports_match_ts_surface() -> None:
    assert extension_selector_module.__all__ == ["ExtensionSelectorComponent"]


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

    parsed = parseSearchQuery('parser "fix"')
    assert [token.kind for token in parsed.tokens] == ["fuzzy", "phrase"]
    assert hasSessionName(sessions[0]) is True
    assert hasSessionName(sessions[1]) is False

    named_only = filterAndSortSessions(sessions, "parser", "relevance", "named")
    assert [session.id for session in named_only] == ["sess-a"]

    regex_sorted = filterAndSortSessions(sessions, "re:theme", "recent", "all")
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


def test_user_message_selector_module_exports_match_ts_surface() -> None:
    assert user_message_selector_module.__all__ == [
        "UserMessageItem",
        "UserMessageSelectorComponent",
    ]


def test_user_message_selector_preserves_double_space_from_blank_lines() -> None:
    component = UserMessageSelectorComponent(
        [UserMessageItem(id="m1", text="alpha\n\nbeta")],
        lambda _entry: None,
        lambda: None,
    )

    rendered = _strip_ansi("\n".join(component.render(120)))
    assert "alpha  beta" in rendered
