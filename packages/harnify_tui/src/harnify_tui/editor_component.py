"""Interface contract for custom editor components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from harnify_tui.autocomplete import AutocompleteProvider
from harnify_tui.tui import Component


@runtime_checkable
class EditorComponent(Component, Protocol):
    onSubmit: Callable[[str], None] | None
    onChange: Callable[[str], None] | None
    addToHistory: Callable[[str], None] | None
    insertTextAtCursor: Callable[[str], None] | None
    getExpandedText: Callable[[], str] | None
    setAutocompleteProvider: Callable[[AutocompleteProvider], None] | None
    borderColor: Callable[[str], str] | None
    setPaddingX: Callable[[int], None] | None
    setAutocompleteMaxVisible: Callable[[int], None] | None

    def getText(self) -> str: ...

    def setText(self, text: str) -> None: ...

    def handleInput(self, data: str) -> None: ...


def get_expanded_text(component: EditorComponent) -> str:
    getter = getattr(component, "getExpandedText", None)
    if callable(getter):
        return getter()
    return component.getText()


getExpandedText = get_expanded_text

__all__ = ["EditorComponent", "getExpandedText", "get_expanded_text"]
