"""Stdout takeover helpers for print and RPC modes."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class _StdoutTakeoverState:
    raw_stdout: Any
    raw_stderr: Any
    original_stdout: Any


class _RedirectedStdout:
    def __init__(self, raw_stdout: Any, raw_stderr: Any) -> None:
        self._raw_stdout = raw_stdout
        self._raw_stderr = raw_stderr

    def write(self, chunk: str) -> int:
        return self._raw_stderr.write(str(chunk))

    def flush(self) -> None:
        flush = getattr(self._raw_stderr, "flush", None)
        if callable(flush):
            flush()

    def writelines(self, lines: list[str]) -> None:
        for line in lines:
            self.write(line)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw_stdout, name)


_stdout_takeover_state: _StdoutTakeoverState | None = None


def take_over_stdout() -> None:
    global _stdout_takeover_state
    if _stdout_takeover_state is not None:
        return

    raw_stdout = sys.stdout
    raw_stderr = sys.stderr
    sys.stdout = _RedirectedStdout(raw_stdout, raw_stderr)
    _stdout_takeover_state = _StdoutTakeoverState(
        raw_stdout=raw_stdout,
        raw_stderr=raw_stderr,
        original_stdout=raw_stdout,
    )


def restore_stdout() -> None:
    global _stdout_takeover_state
    if _stdout_takeover_state is None:
        return

    sys.stdout = _stdout_takeover_state.original_stdout
    _stdout_takeover_state = None


def is_stdout_taken_over() -> bool:
    return _stdout_takeover_state is not None


def write_raw_stdout(text: str) -> None:
    if _stdout_takeover_state is not None:
        _stdout_takeover_state.raw_stdout.write(text)
        return
    sys.stdout.write(text)


async def flush_raw_stdout() -> None:
    stream = _stdout_takeover_state.raw_stdout if _stdout_takeover_state is not None else sys.stdout
    flush = getattr(stream, "flush", None)
    if callable(flush):
        flush()


takeOverStdout = take_over_stdout
restoreStdout = restore_stdout
isStdoutTakenOver = is_stdout_taken_over
writeRawStdout = write_raw_stdout
flushRawStdout = flush_raw_stdout

__all__ = [
    "flushRawStdout",
    "flush_raw_stdout",
    "isStdoutTakenOver",
    "is_stdout_taken_over",
    "restoreStdout",
    "restore_stdout",
    "takeOverStdout",
    "take_over_stdout",
    "writeRawStdout",
    "write_raw_stdout",
]
