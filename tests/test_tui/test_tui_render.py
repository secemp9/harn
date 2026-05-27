from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

from harnify_tui import tui as tui_module
from harnify_tui.terminal_image import deleteKittyImage, encodeKitty
from harnify_tui.tui import TUI


@dataclass
class FakeTerminal:
    columns: int = 40
    rows: int = 10
    writes: list[str] = field(default_factory=list)
    input_handler: Any | None = None
    resize_handler: Any | None = None
    stopped: bool = False

    def start(self, on_input: Any, on_resize: Any) -> None:
        self.input_handler = on_input
        self.resize_handler = on_resize

    def stop(self) -> None:
        self.stopped = True

    async def drainInput(self, maxMs: int = 1000, idleMs: int = 50) -> None:
        return None

    def write(self, data: str) -> None:
        self.writes.append(data)

    def hideCursor(self) -> None:
        self.write("\x1b[?25l")

    def showCursor(self) -> None:
        self.write("\x1b[?25h")

    def clear_writes(self) -> None:
        self.writes.clear()

    def resize(self, columns: int, rows: int) -> None:
        self.columns = columns
        self.rows = rows
        if self.resize_handler is not None:
            self.resize_handler()


class DemoComponent:
    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines = lines or []

    def render(self, _width: int) -> list[str]:
        return list(self.lines)

    def invalidate(self) -> None:
        return None


def wait_for_tui_idle(tui: TUI, timeout: float = 0.25) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not tui.renderRequested and tui.renderTimer is None:
            time.sleep(0.01)
            if not tui.renderRequested and tui.renderTimer is None:
                return
        time.sleep(0.001)
    raise AssertionError("TUI did not become idle in time")


def test_tui_differential_render_updates_changed_lines_without_full_redraw() -> None:
    terminal = FakeTerminal()
    tui = TUI(terminal)
    component = DemoComponent(["Line 0", "Line 1", "Line 2"])
    tui.addChild(component)

    tui.start()
    wait_for_tui_idle(tui)
    initial_redraws = tui.fullRedraws
    terminal.clear_writes()

    component.lines = ["Line 0", "CHANGED", "Line 2"]
    tui.requestRender()
    wait_for_tui_idle(tui)

    writes = "".join(terminal.writes)
    assert "\x1b[2J" not in writes
    assert "CHANGED" in writes
    assert tui.fullRedraws == initial_redraws


def test_tui_width_change_triggers_full_redraw() -> None:
    terminal = FakeTerminal()
    tui = TUI(terminal)
    component = DemoComponent(["Line 0", "Line 1"])
    tui.addChild(component)

    tui.start()
    wait_for_tui_idle(tui)
    initial_redraws = tui.fullRedraws
    terminal.clear_writes()

    terminal.resize(60, 10)
    wait_for_tui_idle(tui)

    writes = "".join(terminal.writes)
    assert "\x1b[2J" in writes
    assert tui.fullRedraws > initial_redraws


def test_tui_overlay_composition_works_with_short_content() -> None:
    terminal = FakeTerminal(columns=80, rows=24)
    tui = TUI(terminal)
    tui.addChild(DemoComponent(["Line 1", "Line 2", "Line 3"]))
    overlay = DemoComponent(["OVERLAY_TOP", "OVERLAY_MID", "OVERLAY_BOT"])

    tui.showOverlay(overlay)
    tui.start()
    wait_for_tui_idle(tui)

    assert any("OVERLAY" in line for line in tui.previousLines)


def test_tui_overlay_uses_live_options_reference_like_ts() -> None:
    terminal = FakeTerminal(columns=80, rows=24)
    tui = TUI(terminal)
    options: dict[str, Any] = {}
    overlay = DemoComponent(["OVERLAY"])

    tui.showOverlay(overlay, options)
    assert tui.hasOverlay() is True

    options["visible"] = lambda _width, _height: False
    assert tui.hasOverlay() is False


def test_tui_deletes_changed_kitty_image_before_redrawing_new_placement() -> None:
    terminal = FakeTerminal()
    tui = TUI(terminal)
    component = DemoComponent()
    tui.addChild(component)

    old_image = encodeKitty("AAAA", {"columns": 2, "rows": 2, "imageId": 42, "moveCursor": False})
    component.lines = ["top", old_image]
    tui.start()
    wait_for_tui_idle(tui)
    terminal.clear_writes()

    new_image = encodeKitty("BBBB", {"columns": 2, "rows": 1, "imageId": 42, "moveCursor": False})
    component.lines = [new_image, ""]
    tui.requestRender()
    wait_for_tui_idle(tui)

    writes = "".join(terminal.writes)
    delete_index = writes.index(deleteKittyImage(42))
    draw_index = writes.index(new_image)
    assert delete_index < draw_index


def test_tui_input_listeners_follow_set_semantics() -> None:
    terminal = FakeTerminal()
    tui = TUI(terminal)
    calls: list[str] = []

    def listener(data: str) -> None:
        calls.append(data)
        return None

    tui.addInputListener(listener)
    tui.addInputListener(listener)
    tui.handleInput("x")

    assert calls == ["x"]


def test_tui_input_listener_iteration_matches_ts_live_set_behavior() -> None:
    terminal = FakeTerminal()
    tui = TUI(terminal)
    calls: list[str] = []

    def second(_data: str) -> None:
        calls.append("second")
        return None

    def third(_data: str) -> None:
        calls.append("third")
        return None

    def first(_data: str) -> None:
        calls.append("first")
        tui.removeInputListener(second)
        tui.addInputListener(third)
        return None

    tui.addInputListener(first)
    tui.addInputListener(second)
    tui.handleInput("x")

    assert calls == ["first", "third"]


def test_tui_request_render_is_deferred_and_coalesced() -> None:
    terminal = FakeTerminal()
    tui = TUI(terminal)
    component = DemoComponent(["one"])
    tui.addChild(component)

    tui.start()
    wait_for_tui_idle(tui)
    terminal.clear_writes()

    component.lines = ["two"]
    tui.requestRender()
    tui.requestRender()

    assert terminal.writes == []
    wait_for_tui_idle(tui)
    writes = "".join(terminal.writes)
    assert "two" in writes


def test_tui_resolve_overlay_layout_matches_ts_nullish_and_invalid_fallbacks() -> None:
    terminal = FakeTerminal(columns=100, rows=40)
    tui = TUI(terminal)

    zero_width = tui.resolveOverlayLayout({"width": 0}, 4, terminal.columns, terminal.rows)
    invalid_position = tui.resolveOverlayLayout(
        {"anchor": "bottom-right", "width": 10, "row": "bad", "col": "bad"},
        4,
        terminal.columns,
        terminal.rows,
    )

    assert zero_width.width == 1
    assert invalid_position.row == 18
    assert invalid_position.col == 45


def test_tui_debug_redraw_log_matches_ts_hook(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HARNIFY_DEBUG_REDRAW", "1")
    log_dir = tmp_path / ".harnify" / "agent"
    log_dir.mkdir(parents=True)

    terminal = FakeTerminal()
    tui = TUI(terminal)
    tui.addChild(DemoComponent(["one"]))

    tui.start()
    wait_for_tui_idle(tui)

    log_text = (log_dir / "harnify-debug.log").read_text(encoding="utf-8")
    assert "fullRender: first render" in log_text


def test_tui_debug_buffer_dump_matches_ts_hook(monkeypatch: Any) -> None:
    monkeypatch.setenv("HARNIFY_TUI_DEBUG", "1")
    debug_dir = Path("/tmp/tui")
    before = set(debug_dir.glob("render-*.log")) if debug_dir.exists() else set()

    terminal = FakeTerminal()
    tui = TUI(terminal)
    component = DemoComponent(["one"])
    tui.addChild(component)

    tui.start()
    wait_for_tui_idle(tui)
    component.lines = ["two"]
    tui.requestRender()
    wait_for_tui_idle(tui)

    after = set(debug_dir.glob("render-*.log"))
    created = after - before
    assert created
    for path in created:
        text = path.read_text(encoding="utf-8")
        assert "firstChanged:" in text
        assert "=== buffer ===" in text
        path.unlink()


def test_tui_module_exports_match_ts_surface() -> None:
    assert tui_module.__all__ == [
        "Component",
        "Container",
        "CURSOR_MARKER",
        "Focusable",
        "isFocusable",
        "OverlayAnchor",
        "OverlayHandle",
        "OverlayMargin",
        "OverlayOptions",
        "SizeValue",
        "TUI",
        "visibleWidth",
    ]
    assert not hasattr(tui_module, "extract_kitty_image_ids")
    assert not hasattr(tui_module, "parse_size_value")
    assert not hasattr(tui_module, "OverlayLayout")
