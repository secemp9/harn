"""Generic bordered multi-line editor used by interactive extension dialogs."""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from harnify_tui import Container, Spacer, Text, getKeybindings

from harnify_coding_agent.core.keybindings import KeybindingsManager
from harnify_coding_agent.modes.interactive.theme.theme import get_editor_theme, theme

from .custom_editor import CustomEditor
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint


class ExtensionEditorComponent(Container):
    wantsKeyRelease = False

    def __init__(
        self,
        tui,
        keybindings: KeybindingsManager,
        title: str,
        prefill: str | None,
        onSubmit: Callable[[str], None],
        onCancel: Callable[[], None],
        options=None,
    ) -> None:
        super().__init__()
        self._focused = False
        self.tui = tui
        self.keybindings = keybindings
        self.onSubmitCallback = onSubmit
        self.onCancelCallback = onCancel

        self.addChild(DynamicBorder())
        self.addChild(Spacer(1))
        self.addChild(Text(theme.fg("accent", title), 1, 0))
        self.addChild(Spacer(1))

        self.editor = CustomEditor(tui, get_editor_theme(), keybindings, options)
        if prefill:
            self.editor.setText(prefill)
        self.editor.onSubmit = self.onSubmitCallback
        self.editor.onEscape = self.onCancelCallback
        self.editor.onAction("app.editor.external", self._schedule_external_editor)
        self.addChild(self.editor)

        self.addChild(Spacer(1))
        has_external_editor = bool(os.environ.get("VISUAL") or os.environ.get("EDITOR"))
        hint = (
            key_hint("tui.select.confirm", "submit")
            + "  "
            + key_hint("tui.input.newLine", "newline")
            + "  "
            + key_hint("tui.select.cancel", "cancel")
        )
        if has_external_editor:
            hint += "  " + key_hint("app.editor.external", "external editor")
        self.addChild(Text(hint, 1, 0))
        self.addChild(Spacer(1))
        self.addChild(DynamicBorder())

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self.editor.focused = value

    def handleInput(self, data: str) -> None:
        if getKeybindings().matches(data, "tui.select.cancel"):
            self.onCancelCallback()
            return
        self.editor.handleInput(data)

    def _schedule_external_editor(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.openExternalEditor())

    async def openExternalEditor(self) -> None:
        editor_cmd = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if not editor_cmd:
            return

        current_text = self.editor.getText()
        temp_file = Path(tempfile.gettempdir()) / f"harnify-extension-editor-{os.getpid()}-{id(self)}.md"
        temp_file.write_text(current_text, encoding="utf-8")

        stop = getattr(self.tui, "stop", None)
        if callable(stop):
            stop()

        try:
            args = shlex.split(editor_cmd)
            if not args:
                return
            editor = args[0]
            editor_args = [*args[1:], str(temp_file)]
            if sys.platform == "win32":
                command = subprocess.list2cmdline([editor, *editor_args])
                return_code = await asyncio.to_thread(subprocess.run, command, check=False, shell=True)
            else:
                return_code = await asyncio.to_thread(
                    subprocess.run,
                    [editor, *editor_args],
                    check=False,
                    shell=False,
                )
            if return_code.returncode == 0:
                new_content = temp_file.read_text(encoding="utf-8").removesuffix("\n")
                self.editor.setText(new_content)
        finally:
            try:
                temp_file.unlink()
            except OSError:
                pass
            start = getattr(self.tui, "start", None)
            if callable(start):
                start()
            request_render = getattr(self.tui, "requestRender", None)
            if callable(request_render):
                request_render(True)


__all__ = ["ExtensionEditorComponent"]
