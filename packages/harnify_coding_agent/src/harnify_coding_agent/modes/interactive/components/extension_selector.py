"""Generic bordered selector used by interactive extension dialogs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from harnify_tui import Container, SelectItem, SelectList, Spacer, Text, getKeybindings

from harnify_coding_agent.modes.interactive.theme.theme import get_select_list_theme, theme

from .countdown_timer import CountdownTimer
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint, raw_key_hint


class ExtensionSelectorComponent(Container):
    wantsKeyRelease = False

    def __init__(
        self,
        title: str,
        options: list[str],
        onSelect: Callable[[str], None],
        onCancel: Callable[[], None],
        opts: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._focused = False
        self.onSelectCallback = onSelect
        self.onCancelCallback = onCancel
        self.onToggleToolsExpanded = (opts or {}).get("onToggleToolsExpanded")
        self.baseTitle = title
        self.countdown: CountdownTimer | None = None

        self.addChild(DynamicBorder())
        self.addChild(Spacer(1))

        self.titleText = Text(theme.fg("accent", theme.bold(title)), 1, 0)
        self.addChild(self.titleText)
        self.addChild(Spacer(1))

        timeout = (opts or {}).get("timeout")
        tui = (opts or {}).get("tui")
        if isinstance(timeout, int) and timeout > 0 and tui is not None:
            self.countdown = CountdownTimer(
                timeout,
                tui,
                lambda seconds: self.titleText.setText(
                    theme.fg("accent", theme.bold(f"{self.baseTitle} ({seconds}s)"))
                ),
                self.onCancelCallback,
            )

        items = [SelectItem(value=option, label=option) for option in options]
        self.selectList = SelectList(items, len(items) or 1, get_select_list_theme())
        self.selectList.onSelect = lambda item: self.onSelectCallback(item.value)
        self.selectList.onCancel = self.onCancelCallback
        self.addChild(self.selectList)

        self.addChild(Spacer(1))
        self.addChild(
            Text(
                raw_key_hint("↑↓", "navigate")
                + "  "
                + key_hint("tui.select.confirm", "select")
                + "  "
                + key_hint("tui.select.cancel", "cancel"),
                1,
                0,
            )
        )
        self.addChild(Spacer(1))
        self.addChild(DynamicBorder())

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value

    def handleInput(self, data: str) -> None:
        kb = getKeybindings()
        if kb.matches(data, "app.tools.expand"):
            callback = self.onToggleToolsExpanded
            if callable(callback):
                callback()
            return
        if data == "k":
            self.selectList.setSelectedIndex(max(0, self.selectList.selectedIndex - 1))
            return
        if data == "j":
            max_index = max(0, len(self.selectList.filteredItems) - 1)
            self.selectList.setSelectedIndex(min(max_index, self.selectList.selectedIndex + 1))
            return
        self.selectList.handleInput(data)

    def dispose(self) -> None:
        if self.countdown is not None:
            self.countdown.dispose()


__all__ = ["ExtensionSelectorComponent"]
