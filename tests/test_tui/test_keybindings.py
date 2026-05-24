from __future__ import annotations

from harnify_tui.keybindings import (
    TUI_KEYBINDINGS,
    KeybindingConflict,
    KeybindingsManager,
)


def test_keybindings_do_not_evict_selector_confirm_when_input_submit_is_rebound() -> None:
    keybindings = KeybindingsManager(
        TUI_KEYBINDINGS,
        {
            "tui.input.submit": ["enter", "ctrl+enter"],
        },
    )

    assert keybindings.getKeys("tui.input.submit") == ["enter", "ctrl+enter"]
    assert keybindings.getKeys("tui.select.confirm") == ["enter"]


def test_keybindings_do_not_evict_cursor_bindings_when_another_action_reuses_key() -> None:
    keybindings = KeybindingsManager(
        TUI_KEYBINDINGS,
        {
            "tui.select.up": ["up", "ctrl+p"],
        },
    )

    assert keybindings.getKeys("tui.select.up") == ["up", "ctrl+p"]
    assert keybindings.getKeys("tui.editor.cursorUp") == ["up"]


def test_keybindings_report_direct_user_conflicts_without_eviction() -> None:
    keybindings = KeybindingsManager(
        TUI_KEYBINDINGS,
        {
            "tui.input.submit": "ctrl+x",
            "tui.select.confirm": "ctrl+x",
        },
    )

    assert keybindings.getConflicts() == [
        KeybindingConflict(
            key="ctrl+x",
            keybindings=["tui.input.submit", "tui.select.confirm"],
        )
    ]
    assert keybindings.getKeys("tui.editor.cursorLeft") == ["left", "ctrl+b"]
