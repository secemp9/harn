from __future__ import annotations

import signal
from dataclasses import dataclass, field
from typing import Any

import pytest
from harnify_tui.terminal import ProcessTerminal


@dataclass
class FakeStdin:
    isRaw: bool = False
    resumed: bool = False
    paused: bool = False
    raw_calls: list[bool] = field(default_factory=list)

    def setRawMode(self, value: bool) -> None:
        self.isRaw = value
        self.raw_calls.append(value)

    def resume(self) -> None:
        self.resumed = True

    def pause(self) -> None:
        self.paused = True

    def fileno(self) -> int:
        return 0


@dataclass
class FakeStdout:
    writes: list[str] = field(default_factory=list)
    columns: int | None = None
    rows: int | None = None

    def write(self, data: str) -> None:
        self.writes.append(data)

    def flush(self) -> None:
        return None


@dataclass
class FakeLoop:
    readers: dict[int, Any] = field(default_factory=dict)
    removed: list[int] = field(default_factory=list)

    def add_reader(self, fd: int, callback: Any) -> None:
        self.readers[fd] = callback

    def remove_reader(self, fd: int) -> None:
        self.removed.append(fd)
        self.readers.pop(fd, None)


def test_process_terminal_dimension_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = FakeStdout(columns=None, rows=None)
    monkeypatch.setenv("COLUMNS", "123")
    monkeypatch.setenv("LINES", "45")

    terminal = ProcessTerminal(stdin=FakeStdin(), stdout=stdout, loop=FakeLoop())

    assert terminal.columns == 123
    assert terminal.rows == 45


def test_process_terminal_start_and_stop_leave_terminal_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    stdin = FakeStdin()
    stdout = FakeStdout()
    loop = FakeLoop()
    signal_calls: list[Any] = []
    kill_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(signal, "getsignal", lambda _sig: "previous")
    monkeypatch.setattr(signal, "signal", lambda sig, handler: signal_calls.append((sig, handler)))
    monkeypatch.setattr("os.kill", lambda pid, sig: kill_calls.append((pid, sig)))

    terminal = ProcessTerminal(stdin=stdin, stdout=stdout, loop=loop)
    received: list[str] = []

    terminal.start(received.append, lambda: received.append("<resize>"))
    assert stdin.raw_calls == [True]
    assert stdin.resumed is True
    assert stdout.writes[:2] == ["\x1b[?2004h", "\x1b[?u"]
    assert 0 in loop.readers
    assert signal_calls
    assert kill_calls

    assert terminal.stdinBuffer is not None
    terminal.stdinBuffer.process("\x1b[?1u")
    assert terminal.kittyProtocolActive is True
    assert "\x1b[>7u" in stdout.writes

    terminal.stop()
    assert stdin.paused is True
    assert stdin.raw_calls[-1] is False
    assert 0 in loop.removed
    assert "\x1b[?2004l" in stdout.writes
    assert "\x1b[<u" in stdout.writes


@pytest.mark.asyncio
async def test_process_terminal_drain_input_disables_protocols_and_restores_handler() -> None:
    stdin = FakeStdin()
    stdout = FakeStdout()
    terminal = ProcessTerminal(stdin=stdin, stdout=stdout, loop=FakeLoop())

    terminal.inputHandler = lambda _data: None
    terminal._kittyProtocolActive = True
    terminal._modifyOtherKeysActive = True
    original_handler = terminal.inputHandler

    await terminal.drainInput(maxMs=5, idleMs=1)

    assert terminal.inputHandler is original_handler
    assert "\x1b[<u" in stdout.writes
    assert "\x1b[>4;0m" in stdout.writes
