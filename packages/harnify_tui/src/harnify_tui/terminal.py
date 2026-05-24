"""Terminal lifecycle and ANSI control helpers for the TUI package."""

from __future__ import annotations

import asyncio
import ctypes
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from harnify_tui.keys import setKittyProtocolActive
from harnify_tui.stdin_buffer import StdinBuffer

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows import path.
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

TERMINAL_PROGRESS_KEEPALIVE_MS = 1000
TERMINAL_PROGRESS_ACTIVE_SEQUENCE = "\x1b]9;4;3\x07"
TERMINAL_PROGRESS_CLEAR_SEQUENCE = "\x1b]9;4;0;\x07"
KITTY_RESPONSE_PATTERN = __import__("re").compile(r"^\x1b\[\?(\d+)u$")


class Terminal(Protocol):
    def start(self, onInput: Any, onResize: Any) -> None: ...
    def stop(self) -> None: ...
    async def drainInput(self, maxMs: int = 1000, idleMs: int = 50) -> None: ...
    def write(self, data: str) -> None: ...


class ProcessTerminal:
    def __init__(
        self,
        stdin: Any | None = None,
        stdout: Any | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.stdin = stdin if stdin is not None else sys.stdin
        self.stdout = stdout if stdout is not None else sys.stdout
        self.loop = loop
        self.wasRaw = False
        self.inputHandler: Any | None = None
        self.resizeHandler: Any | None = None
        self._kittyProtocolActive = False
        self._modifyOtherKeysActive = False
        self.stdinBuffer: StdinBuffer | None = None
        self.stdinDataHandler: Any | None = None
        self.progressTimer: threading.Timer | None = None
        self._progressActive = False
        self._previousSigwinchHandler: Any | None = None
        self._previousTermiosSettings: list[Any] | None = None
        self._lastStdinActivityMs = self._now_ms()
        self._readerInstalled = False
        self.writeLogPath = self._resolve_write_log_path()

    @property
    def kittyProtocolActive(self) -> bool:
        return self._kittyProtocolActive

    def start(self, onInput: Any, onResize: Any) -> None:
        self.inputHandler = onInput
        self.resizeHandler = onResize
        self._lastStdinActivityMs = self._now_ms()

        self.wasRaw = bool(getattr(self.stdin, "isRaw", False))
        if hasattr(self.stdin, "setRawMode"):
            self.stdin.setRawMode(True)
        else:
            self._enter_raw_mode()

        if hasattr(self.stdin, "resume"):
            self.stdin.resume()

        self.write("\x1b[?2004h")
        self._install_resize_handler()
        self._refresh_dimensions()
        self.enableWindowsVTInput()
        self.queryAndEnableKittyProtocol()

    def queryAndEnableKittyProtocol(self) -> None:
        self.setupStdinBuffer()
        self._install_reader()
        self.write("\x1b[?u")
        timer = threading.Timer(0.150, self._enable_modify_other_keys_if_needed)
        timer.daemon = True
        timer.start()

    def setupStdinBuffer(self) -> None:
        self.stdinBuffer = StdinBuffer({"timeout": 10})

        def on_data(sequence: str) -> None:
            if not self._kittyProtocolActive:
                match = KITTY_RESPONSE_PATTERN.match(sequence)
                if match is not None:
                    self._kittyProtocolActive = True
                    setKittyProtocolActive(True)
                    self.write("\x1b[>7u")
                    return
            if self.inputHandler is not None:
                self.inputHandler(sequence)

        def on_paste(content: str) -> None:
            if self.inputHandler is not None:
                self.inputHandler(f"\x1b[200~{content}\x1b[201~")

        self.stdinBuffer.on("data", on_data)
        self.stdinBuffer.on("paste", on_paste)
        self.stdinDataHandler = lambda data: self.stdinBuffer.process(data)

    def enableWindowsVTInput(self) -> None:
        if sys.platform != "win32":
            return
        try:
            ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200
            STD_INPUT_HANDLE = -10
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(STD_INPUT_HANDLE)
            if handle in {0, -1}:
                return
            mode = ctypes.c_uint()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return
            kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_INPUT)
        except Exception:
            return

    async def drainInput(self, maxMs: int = 1000, idleMs: int = 50) -> None:
        if self._kittyProtocolActive:
            self.write("\x1b[<u")
            self._kittyProtocolActive = False
            setKittyProtocolActive(False)
        if self._modifyOtherKeysActive:
            self.write("\x1b[>4;0m")
            self._modifyOtherKeysActive = False

        previous_handler = self.inputHandler
        self.inputHandler = None
        end_time = self._now_ms() + maxMs

        try:
            while True:
                now = self._now_ms()
                time_left = end_time - now
                if time_left <= 0:
                    break
                if now - self._lastStdinActivityMs >= idleMs:
                    break
                await asyncio.sleep(min(idleMs, time_left) / 1000.0)
        finally:
            self.inputHandler = previous_handler

    def stop(self) -> None:
        if self.clearProgressInterval():
            self.write(TERMINAL_PROGRESS_CLEAR_SEQUENCE)

        self.write("\x1b[?2004l")
        if self._kittyProtocolActive:
            self.write("\x1b[<u")
            self._kittyProtocolActive = False
            setKittyProtocolActive(False)
        if self._modifyOtherKeysActive:
            self.write("\x1b[>4;0m")
            self._modifyOtherKeysActive = False

        if self.stdinBuffer is not None:
            self.stdinBuffer.destroy()
            self.stdinBuffer = None

        self._remove_reader()
        self.stdinDataHandler = None
        self.inputHandler = None
        self._restore_resize_handler()

        if hasattr(self.stdin, "pause"):
            self.stdin.pause()

        if hasattr(self.stdin, "setRawMode"):
            self.stdin.setRawMode(self.wasRaw)
        else:
            self._restore_raw_mode()

    def write(self, data: str) -> None:
        self.stdout.write(data)
        flush = getattr(self.stdout, "flush", None)
        if callable(flush):
            flush()
        if self.writeLogPath:
            try:
                Path(self.writeLogPath).parent.mkdir(parents=True, exist_ok=True)
                with open(self.writeLogPath, "a", encoding="utf-8") as handle:
                    handle.write(data)
            except OSError:
                pass

    @property
    def columns(self) -> int:
        return int(getattr(self.stdout, "columns", None) or os.environ.get("COLUMNS") or 80)

    @property
    def rows(self) -> int:
        return int(getattr(self.stdout, "rows", None) or os.environ.get("LINES") or 24)

    def moveBy(self, lines: int) -> None:
        if lines > 0:
            self.write(f"\x1b[{lines}B")
        elif lines < 0:
            self.write(f"\x1b[{-lines}A")

    def hideCursor(self) -> None:
        self.write("\x1b[?25l")

    def showCursor(self) -> None:
        self.write("\x1b[?25h")

    def clearLine(self) -> None:
        self.write("\x1b[K")

    def clearFromCursor(self) -> None:
        self.write("\x1b[J")

    def clearScreen(self) -> None:
        self.write("\x1b[2J\x1b[H")

    def setTitle(self, title: str) -> None:
        self.write(f"\x1b]0;{title}\x07")

    def setProgress(self, active: bool) -> None:
        if active:
            self._progressActive = True
            self.write(TERMINAL_PROGRESS_ACTIVE_SEQUENCE)
            if self.progressTimer is None:
                self._schedule_progress_keepalive()
            return

        self._progressActive = False
        self.clearProgressInterval()
        self.write(TERMINAL_PROGRESS_CLEAR_SEQUENCE)

    def clearProgressInterval(self) -> bool:
        if self.progressTimer is None:
            return False
        self.progressTimer.cancel()
        self.progressTimer = None
        return True

    def _schedule_progress_keepalive(self) -> None:
        if not self._progressActive:
            return

        def tick() -> None:
            if not self._progressActive:
                self.progressTimer = None
                return
            self.write(TERMINAL_PROGRESS_ACTIVE_SEQUENCE)
            self._schedule_progress_keepalive()

        self.progressTimer = threading.Timer(TERMINAL_PROGRESS_KEEPALIVE_MS / 1000.0, tick)
        self.progressTimer.daemon = True
        self.progressTimer.start()

    def _resolve_write_log_path(self) -> str:
        raw = os.environ.get("PI_TUI_WRITE_LOG", "")
        if not raw:
            return ""
        try:
            path = Path(raw)
            if path.is_dir():
                from datetime import datetime

                now = datetime.now()
                ts = now.strftime("%Y-%m-%d_%H-%M-%S")
                return str(path / f"tui-{ts}-{os.getpid()}.log")
        except OSError:
            pass
        return raw

    def _install_resize_handler(self) -> None:
        if not hasattr(signal, "SIGWINCH") or self.resizeHandler is None:
            return
        self._previousSigwinchHandler = signal.getsignal(signal.SIGWINCH)

        def handler(_signum: int, _frame: Any) -> None:
            if self.resizeHandler is not None:
                self.resizeHandler()

        signal.signal(signal.SIGWINCH, handler)

    def _restore_resize_handler(self) -> None:
        if not hasattr(signal, "SIGWINCH"):
            return
        if self._previousSigwinchHandler is not None:
            signal.signal(signal.SIGWINCH, self._previousSigwinchHandler)
            self._previousSigwinchHandler = None
        self.resizeHandler = None

    def _refresh_dimensions(self) -> None:
        if sys.platform == "win32" or not hasattr(signal, "SIGWINCH"):
            return
        try:
            os.kill(os.getpid(), signal.SIGWINCH)
        except OSError:
            return

    def _enable_modify_other_keys_if_needed(self) -> None:
        if not self._kittyProtocolActive and not self._modifyOtherKeysActive:
            self.write("\x1b[>4;2m")
            self._modifyOtherKeysActive = True

    def _install_reader(self) -> None:
        if self._readerInstalled:
            return
        fd = self._stdin_fileno()
        if fd is None:
            return
        loop = self._event_loop()
        add_reader = getattr(loop, "add_reader", None)
        if callable(add_reader):
            add_reader(fd, self._handle_stdin_ready)
            self._readerInstalled = True

    def _remove_reader(self) -> None:
        if not self._readerInstalled:
            return
        fd = self._stdin_fileno()
        if fd is None:
            self._readerInstalled = False
            return
        loop = self._event_loop()
        remove_reader = getattr(loop, "remove_reader", None)
        if callable(remove_reader):
            remove_reader(fd)
        self._readerInstalled = False

    def _handle_stdin_ready(self) -> None:
        data = self._read_stdin_data()
        if not data:
            return
        self._lastStdinActivityMs = self._now_ms()
        if self.stdinDataHandler is not None:
            self.stdinDataHandler(data)

    def _read_stdin_data(self) -> str | bytes:
        fd = self._stdin_fileno()
        if fd is not None:
            try:
                return os.read(fd, 4096)
            except OSError:
                return b""
        reader = getattr(self.stdin, "read", None)
        if callable(reader):
            return reader(4096)
        return ""

    def _stdin_fileno(self) -> int | None:
        try:
            return int(self.stdin.fileno())
        except Exception:
            return None

    def _enter_raw_mode(self) -> None:
        if sys.platform == "win32" or termios is None or tty is None:
            return
        fd = self._stdin_fileno()
        is_tty = getattr(self.stdin, "isatty", None)
        if fd is None or (callable(is_tty) and not is_tty()):
            return
        self._previousTermiosSettings = termios.tcgetattr(fd)
        tty.setraw(fd)

    def _restore_raw_mode(self) -> None:
        if termios is None or self._previousTermiosSettings is None:
            return
        fd = self._stdin_fileno()
        if fd is None:
            return
        termios.tcsetattr(fd, termios.TCSADRAIN, self._previousTermiosSettings)
        self._previousTermiosSettings = None

    def _event_loop(self) -> asyncio.AbstractEventLoop:
        if self.loop is not None:
            return self.loop
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.get_event_loop_policy().get_event_loop()
        return self.loop

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)


__all__ = ["ProcessTerminal", "Terminal"]
