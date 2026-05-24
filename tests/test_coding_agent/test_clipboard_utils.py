from __future__ import annotations

import types

import pytest
from harnify_coding_agent.utils import clipboard as clipboard_utils


class _NativeClipboard:
    def __init__(self, callback=None) -> None:
        self.callback = callback
        self.calls: list[str] = []

    async def set_text(self, text: str) -> None:
        if self.callback is not None:
            await self.callback(text)
        self.calls.append(text)


@pytest.mark.asyncio
async def test_copy_to_clipboard_local_native_success_skips_osc52(monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[str] = []
    native = _NativeClipboard()
    commands: list[tuple[list[str], str]] = []

    monkeypatch.setattr(clipboard_utils, "_get_native_clipboard", lambda: native)
    monkeypatch.setattr(clipboard_utils, "_platform", lambda: "darwin")
    monkeypatch.setattr(clipboard_utils, "_run_command", lambda command, text: commands.append((command, text)))
    monkeypatch.setattr(clipboard_utils, "emit_osc52", lambda text, writer=None: writes.append(text) or True)

    await clipboard_utils.copy_to_clipboard("hello")

    assert native.calls == ["hello"]
    assert writes == []
    assert commands == []


@pytest.mark.asyncio
async def test_copy_to_clipboard_remote_native_success_emits_osc52(monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[str] = []

    async def _on_set_text(_text: str) -> None:
        assert writes == []

    native = _NativeClipboard(callback=_on_set_text)
    monkeypatch.setattr(clipboard_utils, "_get_native_clipboard", lambda: native)
    monkeypatch.setattr(clipboard_utils, "_platform", lambda: "darwin")
    monkeypatch.setattr(clipboard_utils, "emit_osc52", lambda text, writer=None: writes.append(text) or True)
    monkeypatch.setenv("SSH_CONNECTION", "client server")

    await clipboard_utils.copy_to_clipboard("hello")

    assert native.calls == ["hello"]
    assert writes == ["hello"]


@pytest.mark.asyncio
async def test_copy_to_clipboard_shell_fallback_and_osc52(monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[str] = []
    commands: list[tuple[list[str], str]] = []

    async def _raise(_text: str) -> None:
        raise RuntimeError("native failed")

    native = _NativeClipboard(callback=_raise)
    monkeypatch.setattr(clipboard_utils, "_get_native_clipboard", lambda: native)
    monkeypatch.setattr(clipboard_utils, "_platform", lambda: "darwin")
    monkeypatch.setattr(clipboard_utils, "_run_command", lambda command, text: commands.append((command, text)))
    monkeypatch.setattr(clipboard_utils, "emit_osc52", lambda text, writer=None: writes.append(text) or True)

    await clipboard_utils.copy_to_clipboard("hello")

    assert commands == [(["pbcopy"], "hello")]
    assert writes == []

    commands.clear()
    monkeypatch.setattr(
        clipboard_utils,
        "_run_command",
        lambda command, text: (_ for _ in ()).throw(RuntimeError("pbcopy failed")),
    )

    await clipboard_utils.copy_to_clipboard("world")
    assert writes == ["world"]


@pytest.mark.asyncio
async def test_copy_to_clipboard_linux_xclip_and_xsel_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[tuple[list[str], str]] = []

    monkeypatch.setattr(clipboard_utils, "_get_native_clipboard", lambda: None)
    monkeypatch.setattr(clipboard_utils, "_platform", lambda: "linux")
    monkeypatch.setattr(clipboard_utils, "is_wayland_session", lambda env=None: False)
    monkeypatch.setenv("DISPLAY", ":0")

    def run_command(command: list[str], text: str) -> None:
        commands.append((command, text))

    monkeypatch.setattr(clipboard_utils, "_run_command", run_command)

    await clipboard_utils.copy_to_clipboard("hello")

    assert commands == [(["xclip", "-selection", "clipboard"], "hello")]

    commands.clear()

    def run_command_with_fallback(command: list[str], text: str) -> None:
        commands.append((command, text))
        if command[0] == "xclip":
            raise RuntimeError("xclip failed")

    monkeypatch.setattr(clipboard_utils, "_run_command", run_command_with_fallback)

    await clipboard_utils.copy_to_clipboard("world")

    assert commands == [
        (["xclip", "-selection", "clipboard"], "world"),
        (["xsel", "--clipboard", "--input"], "world"),
    ]


@pytest.mark.asyncio
async def test_copy_to_clipboard_rejects_oversized_osc52_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _raise(_text: str) -> None:
        raise RuntimeError("native failed")

    native = _NativeClipboard(callback=_raise)
    monkeypatch.setattr(clipboard_utils, "_get_native_clipboard", lambda: native)
    monkeypatch.setattr(clipboard_utils, "_platform", lambda: "darwin")
    monkeypatch.setattr(
        clipboard_utils,
        "_run_command",
        lambda command, text: (_ for _ in ()).throw(RuntimeError("pbcopy failed")),
    )

    with pytest.raises(RuntimeError, match="Failed to copy to clipboard"):
        await clipboard_utils.copy_to_clipboard("x" * 80_000)


def test_emit_osc52_and_remote_session_helpers() -> None:
    buffer: list[str] = []
    writer = types.SimpleNamespace(write=lambda chunk: buffer.append(chunk))

    assert clipboard_utils.emit_osc52("hello", writer=writer) is True
    assert buffer and buffer[0].startswith("\x1b]52;c;")
    assert clipboard_utils.is_remote_session({}) is False
    assert clipboard_utils.is_remote_session({"SSH_CONNECTION": "remote"}) is True
