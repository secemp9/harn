"""Strict JSONL framing helpers for RPC mode and client."""

from __future__ import annotations

import codecs
import dataclasses
import json
import threading
from collections.abc import Callable, Generator
from typing import Any


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return to_jsonable(vars(value))
    return value


class JsonlLineBuffer:
    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""

    def feed(self, chunk: bytes | str) -> list[str]:
        self._buffer += chunk if isinstance(chunk, str) else self._decoder.decode(chunk, final=False)
        lines: list[str] = []
        while True:
            newline_index = self._buffer.find("\n")
            if newline_index < 0:
                break
            line = self._buffer[:newline_index]
            self._buffer = self._buffer[newline_index + 1 :]
            lines.append(line[:-1] if line.endswith("\r") else line)
        return lines

    def end(self) -> list[str]:
        trailing = self._decoder.decode(b"", final=True)
        if trailing:
            self._buffer += trailing
        if not self._buffer:
            return []
        line = self._buffer[:-1] if self._buffer.endswith("\r") else self._buffer
        self._buffer = ""
        return [line]


def serialize_json_line(value: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False) + "\n"


def iter_jsonl_lines(stream: Any, chunk_size: int = 4096) -> Generator[str, None, None]:
    reader = JsonlLineBuffer()
    source = getattr(stream, "buffer", stream)
    while True:
        chunk = source.read(chunk_size)
        if not chunk:
            break
        yield from reader.feed(chunk)
    yield from reader.end()


def attach_jsonl_line_reader(
    stream: Any,
    on_line: Callable[[str], None],
    *,
    chunk_size: int = 4096,
) -> Callable[[], None]:
    stop_event = threading.Event()
    reader = JsonlLineBuffer()
    source = getattr(stream, "buffer", stream)

    def _run() -> None:
        while not stop_event.is_set():
            chunk = source.read(chunk_size)
            if not chunk:
                break
            for line in reader.feed(chunk):
                if stop_event.is_set():
                    return
                on_line(line)
        if stop_event.is_set():
            return
        for line in reader.end():
            if stop_event.is_set():
                return
            on_line(line)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    def detach() -> None:
        stop_event.set()

    return detach


attachJsonlLineReader = attach_jsonl_line_reader
serializeJsonLine = serialize_json_line

__all__ = [
    "JsonlLineBuffer",
    "attachJsonlLineReader",
    "attach_jsonl_line_reader",
    "iter_jsonl_lines",
    "serializeJsonLine",
    "serialize_json_line",
    "to_jsonable",
]
