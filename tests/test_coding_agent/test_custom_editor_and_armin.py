from __future__ import annotations

import re
from typing import Any

from harnify_coding_agent.core.keybindings import KeybindingsManager
from harnify_coding_agent.modes.interactive.components import armin as armin_module
from harnify_coding_agent.modes.interactive.components.armin import DISPLAY_HEIGHT, ArminComponent
from harnify_coding_agent.modes.interactive.components.custom_editor import CustomEditor
from harnify_coding_agent.modes.interactive.theme.theme import get_editor_theme, init_theme
from harnify_tui import setKeybindings

_ANSI_RE = re.compile(r"\x1b(?:\[[0-9;]*m|\]8;;.*?\x07)", re.DOTALL)


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class FakeUi:
    def __init__(self) -> None:
        self.render_calls: list[bool | None] = []

    def requestRender(self, force: bool | None = None) -> None:
        self.render_calls.append(force)


class FakeTimer:
    def __init__(self, interval: float, callback: Any) -> None:
        self.interval = interval
        self.callback = callback
        self.started = False
        self.cancelled = False
        self.daemon = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True


def setup_function() -> None:
    setKeybindings(KeybindingsManager())
    init_theme("dark")


def test_custom_editor_prioritizes_extension_and_app_shortcuts() -> None:
    ui = FakeUi()
    editor = CustomEditor(ui, get_editor_theme(), KeybindingsManager())
    events: list[str] = []

    def handle_extension(data: str) -> bool:
        if data == "EXT":
            events.append("extension")
            return True
        return False

    editor.onExtensionShortcut = handle_extension
    editor.onPasteImage = lambda: events.append("paste-image")
    editor.onEscape = lambda: events.append("escape")
    editor.onCtrlD = lambda: events.append("exit")
    editor.onAction("app.model.select", lambda: events.append("model-select"))

    editor.handleInput("EXT")
    editor.handleInput("\x16")
    editor.handleInput("\x0c")
    editor.handleInput("\x1b")
    editor.handleInput("\x04")

    assert events == ["extension", "paste-image", "model-select", "escape", "exit"]

    editor.setText("body")
    editor.handleInput("\x04")

    assert events == ["extension", "paste-image", "model-select", "escape", "exit"]


def test_custom_editor_allows_escape_to_cancel_autocomplete() -> None:
    ui = FakeUi()
    editor = CustomEditor(ui, get_editor_theme(), KeybindingsManager())
    events: list[str] = []
    editor.onEscape = lambda: events.append("escape")
    editor.autocompleteState = "force"
    editor.autocompleteList = object()

    editor.handleInput("\x1b")

    assert events == []
    assert editor.autocompleteState is None
    assert editor.autocompleteList is None


def test_armin_component_renders_and_stops_animation(monkeypatch) -> None:
    timers: list[FakeTimer] = []

    def timer_factory(interval: float, callback: Any) -> FakeTimer:
        timer = FakeTimer(interval, callback)
        timers.append(timer)
        return timer

    monkeypatch.setattr(armin_module.random, "choice", lambda _effects: "typewriter")
    monkeypatch.setattr(armin_module.threading, "Timer", timer_factory)

    ui = FakeUi()
    component = ArminComponent(ui)

    lines = [_strip_ansi(line) for line in component.render(40)]

    assert len(lines) == DISPLAY_HEIGHT + 1
    assert lines[-1].strip() == "ARMIN SAYS HI"
    assert len(timers) == 1
    assert timers[0].started is True

    component.dispose()

    assert timers[0].cancelled is True


def test_armin_component_advances_frame_and_requests_render(monkeypatch) -> None:
    monkeypatch.setattr(armin_module.threading, "Timer", FakeTimer)

    ui = FakeUi()
    component = ArminComponent(ui, effect="typewriter")
    component.stopAnimation()
    initial_version = component.gridVersion

    done = component._advance_frame()

    assert done is False
    assert int(component.effectState["pos"]) == 3
    assert component.currentGrid[0][:3] == component.finalGrid[0][:3]
    assert component.gridVersion == initial_version + 1
    assert ui.render_calls == [None]
