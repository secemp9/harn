from __future__ import annotations

from dataclasses import dataclass, field
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


def test_tui_differential_render_updates_changed_lines_without_full_redraw() -> None:
    terminal = FakeTerminal()
    tui = TUI(terminal)
    component = DemoComponent(["Line 0", "Line 1", "Line 2"])
    tui.addChild(component)

    tui.start()
    initial_redraws = tui.fullRedraws
    terminal.clear_writes()

    component.lines = ["Line 0", "CHANGED", "Line 2"]
    tui.requestRender()

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
    initial_redraws = tui.fullRedraws
    terminal.clear_writes()

    terminal.resize(60, 10)

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

    assert any("OVERLAY" in line for line in tui.previousLines)


def test_tui_deletes_changed_kitty_image_before_redrawing_new_placement() -> None:
    terminal = FakeTerminal()
    tui = TUI(terminal)
    component = DemoComponent()
    tui.addChild(component)

    old_image = encodeKitty("AAAA", {"columns": 2, "rows": 2, "imageId": 42, "moveCursor": False})
    component.lines = ["top", old_image]
    tui.start()
    terminal.clear_writes()

    new_image = encodeKitty("BBBB", {"columns": 2, "rows": 1, "imageId": 42, "moveCursor": False})
    component.lines = [new_image, ""]
    tui.requestRender()

    writes = "".join(terminal.writes)
    delete_index = writes.index(deleteKittyImage(42))
    draw_index = writes.index(new_image)
    assert delete_index < draw_index


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
