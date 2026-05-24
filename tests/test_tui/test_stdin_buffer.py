from __future__ import annotations

import asyncio

import pytest
from harnify_tui.stdin_buffer import StdinBuffer


@pytest.fixture
def buffer() -> StdinBuffer:
    return StdinBuffer({"timeout": 10})


@pytest.fixture
def emitted_sequences(buffer: StdinBuffer) -> list[str]:
    sequences: list[str] = []
    buffer.on("data", lambda sequence: sequences.append(sequence))
    return sequences


def test_stdin_buffer_passes_regular_characters_through(buffer: StdinBuffer, emitted_sequences: list[str]) -> None:
    buffer.process("hello 世界")
    assert emitted_sequences == ["h", "e", "l", "l", "o", " ", "世", "界"]


def test_stdin_buffer_passes_complete_escape_sequences(buffer: StdinBuffer, emitted_sequences: list[str]) -> None:
    mouse = "\x1b[<35;20;5m"
    up_arrow = "\x1b[A"
    ss3 = "\x1bOA"

    buffer.process(mouse)
    buffer.process(up_arrow)
    buffer.process(ss3)

    assert emitted_sequences == [mouse, up_arrow, ss3]


def test_stdin_buffer_buffers_partial_escape_sequences(buffer: StdinBuffer, emitted_sequences: list[str]) -> None:
    buffer.process("\x1b")
    buffer.process("[<35")
    assert emitted_sequences == []
    assert buffer.getBuffer() == "\x1b[<35"

    buffer.process(";20;5m")
    assert emitted_sequences == ["\x1b[<35;20;5m"]
    assert buffer.getBuffer() == ""


@pytest.mark.asyncio
async def test_stdin_buffer_flushes_incomplete_sequence_after_timeout(
    buffer: StdinBuffer,
    emitted_sequences: list[str],
) -> None:
    buffer.process("\x1b[<35")
    assert emitted_sequences == []
    await asyncio.sleep(0.03)
    assert emitted_sequences == ["\x1b[<35"]


def test_stdin_buffer_handles_mixed_content(buffer: StdinBuffer, emitted_sequences: list[str]) -> None:
    buffer.process("abc\x1b[A")
    buffer.process("\x1b[Bxy")
    assert emitted_sequences == ["a", "b", "c", "\x1b[A", "\x1b[B", "x", "y"]


def test_stdin_buffer_handles_batched_kitty_events(buffer: StdinBuffer, emitted_sequences: list[str]) -> None:
    buffer.process("\x1b[97u\x1b[97;1:3u\x1b[98u\x1b[98;1:3u")
    assert emitted_sequences == ["\x1b[97u", "\x1b[97;1:3u", "\x1b[98u", "\x1b[98;1:3u"]


def test_stdin_buffer_splits_esc_esc_csi_for_wezterm_escape_release(
    buffer: StdinBuffer,
    emitted_sequences: list[str],
) -> None:
    buffer.process("\x1b\x1b[27;129:3u")
    assert emitted_sequences == ["\x1b", "\x1b[27;129:3u"]


def test_stdin_buffer_drops_duplicate_plain_character_after_kitty_printable(
    buffer: StdinBuffer,
    emitted_sequences: list[str],
) -> None:
    buffer.process("\x1b[224uà")
    buffer.process("\x1b[64u")
    buffer.process("@")
    buffer.process("\x1b[97ub")

    assert emitted_sequences == ["\x1b[224u", "\x1b[64u", "\x1b[97u", "b"]


def test_stdin_buffer_emits_bracketed_paste_as_single_paste_event() -> None:
    buffer = StdinBuffer({"timeout": 10})
    paste_events: list[str] = []
    data_events: list[str] = []
    buffer.on("paste", lambda content: paste_events.append(content))
    buffer.on("data", lambda sequence: data_events.append(sequence))

    buffer.process("a\x1b[200~hello\nworld\x1b[201~b")

    assert paste_events == ["hello\nworld"]
    assert data_events == ["a", "b"]
