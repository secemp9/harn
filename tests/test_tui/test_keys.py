from __future__ import annotations

import pytest
from harnify_tui.keys import (
    Key,
    decodeKittyPrintable,
    decodePrintableKey,
    matchesKey,
    parseKey,
    setKittyProtocolActive,
)


@pytest.fixture(autouse=True)
def reset_kitty_protocol() -> None:
    setKittyProtocolActive(False)
    yield
    setKittyProtocolActive(False)


def test_keys_match_kitty_alternate_keys_for_non_latin_layouts() -> None:
    setKittyProtocolActive(True)
    assert matchesKey("\x1b[1089::99;5u", "ctrl+c") is True
    assert matchesKey("\x1b[1074::100;5u", "ctrl+d") is True
    assert matchesKey("\x1b[1103::122;5u", "ctrl+z") is True
    assert matchesKey("\x1b[1079::112;6u", "ctrl+shift+p") is True
    assert parseKey("\x1b[1089::99;5u") == "ctrl+c"


def test_keys_match_super_and_digit_kitty_bindings() -> None:
    setKittyProtocolActive(True)
    assert matchesKey("\x1b[107;9u", "super+k") is True
    assert matchesKey("\x1b[13;9u", "super+enter") is True
    assert matchesKey("\x1b[107;13u", Key.ctrlSuper("k")) is True
    assert matchesKey("\x1b[49u", "1") is True
    assert matchesKey("\x1b[49;5u", "ctrl+1") is True
    assert parseKey("\x1b[107;14u") == "shift+ctrl+super+k"
    assert parseKey("\x1b[49;5u") == "ctrl+1"


def test_keys_normalize_kitty_keypad_functional_keys() -> None:
    setKittyProtocolActive(True)
    assert matchesKey("\x1b[57400u", "1") is True
    assert matchesKey("\x1b[57410u", "/") is True
    assert matchesKey("\x1b[57417u", "left") is True
    assert matchesKey("\x1b[57426u", "delete") is True
    assert parseKey("\x1b[57399u") == "0"
    assert parseKey("\x1b[57409u") == "."
    assert parseKey("\x1b[57413u") == "+"
    assert parseKey("\x1b[57423u") == "home"


def test_keys_match_modify_other_keys_variants() -> None:
    assert matchesKey("\x1b[27;5;99~", "ctrl+c") is True
    assert matchesKey("\x1b[27;5;13~", "ctrl+enter") is True
    assert matchesKey("\x1b[27;2;9~", "shift+tab") is True
    assert matchesKey("\x1b[27;3;127~", "alt+backspace") is True
    assert matchesKey("\x1b[27;5;47~", "ctrl+/") is True
    assert matchesKey("\x1b[27;2;49~", "shift+1") is True
    assert matchesKey("\x1b[27;2;69~", "shift+e") is True
    assert parseKey("\x1b[27;6;69~") == "shift+ctrl+e"
    assert parseKey("\x1b[27;7;104~") == "ctrl+alt+h"


def test_keys_match_legacy_ctrl_symbol_and_alt_prefix_sequences() -> None:
    assert matchesKey("\x03", "ctrl+c") is True
    assert matchesKey("\x1c", "ctrl+\\") is True
    assert matchesKey("\x1d", "ctrl+]") is True
    assert matchesKey("\x1f", "ctrl+-") is True
    assert matchesKey("\x1b\x1b", "ctrl+alt+[") is True
    assert matchesKey("\x1b\x1c", "ctrl+alt+\\") is True
    assert matchesKey("\x1b ", "alt+space") is True
    assert matchesKey("\x1bB", "alt+left") is True
    assert matchesKey("\x1bF", "alt+right") is True
    assert parseKey("\x1ba") == "alt+a"
    assert parseKey("\x1b1") == "alt+1"
    assert parseKey("\x1b\x03") == "ctrl+alt+c"


def test_keys_treat_linefeed_and_alt_sequences_differently_when_kitty_is_active() -> None:
    assert matchesKey("\n", "enter") is True
    assert parseKey("\n") == "enter"

    setKittyProtocolActive(True)
    assert matchesKey("\n", "shift+enter") is True
    assert matchesKey("\n", "enter") is False
    assert parseKey("\n") == "shift+enter"
    assert matchesKey("\x1b ", "alt+space") is False
    assert parseKey("\x1b ") is None


def test_keys_handle_raw_backspace_windows_terminal_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)

    assert matchesKey("\x7f", "backspace") is True
    assert matchesKey("\x08", "backspace") is True
    assert matchesKey("\x08", "ctrl+backspace") is False
    assert matchesKey("\x08", "ctrl+h") is True
    assert parseKey("\x08") == "backspace"

    monkeypatch.setenv("WT_SESSION", "test-session")
    assert matchesKey("\x08", "ctrl+backspace") is True
    assert matchesKey("\x08", "backspace") is False
    assert parseKey("\x08") == "ctrl+backspace"

    monkeypatch.setenv("SSH_CONNECTION", "1 2 3 4")
    monkeypatch.setenv("SSH_CLIENT", "1 2 3")
    monkeypatch.setenv("SSH_TTY", "/dev/pts/1")
    assert matchesKey("\x08", "ctrl+backspace") is False
    assert matchesKey("\x08", "backspace") is True
    assert parseKey("\x08") == "backspace"


def test_keys_parse_special_keys_legacy_sequences_and_ss3() -> None:
    assert parseKey("\x1b") == "escape"
    assert parseKey("\t") == "tab"
    assert parseKey("\r") == "enter"
    assert parseKey("\x00") == "ctrl+space"
    assert parseKey("1") == "1"
    assert parseKey("\x1b[A") == "up"
    assert parseKey("\x1bOB") == "down"
    assert parseKey("\x1bOC") == "right"
    assert parseKey("\x1bOH") == "home"
    assert parseKey("\x1b[24~") == "f12"
    assert parseKey("\x1b[E") == "clear"
    assert parseKey("\x1bp") == "alt+up"
    assert parseKey("\x1b[[5~") == "pageUp"


def test_keys_decode_printable_sequences() -> None:
    assert decodeKittyPrintable("\x1b[57399u") == "0"
    assert decodeKittyPrintable("\x1b[57409u") == "."
    assert decodeKittyPrintable("\x1b[57417u") is None
    assert decodePrintableKey("\x1b[27;2;69~") == "E"
    assert decodePrintableKey("\x1b[27;2;196~") == "Ä"
    assert decodePrintableKey("\x1b[27;2;32~") == " "
    assert decodePrintableKey("\x1b[27;2;13~") is None
    assert decodePrintableKey("\x1b[27;6;69~") is None


def test_keys_prefer_codepoint_for_latin_and_symbol_layout_remaps() -> None:
    setKittyProtocolActive(True)
    assert matchesKey("\x1b[107::118;5u", "ctrl+k") is True
    assert matchesKey("\x1b[107::118;5u", "ctrl+v") is False
    assert matchesKey("\x1b[47::91;5u", "ctrl+/") is True
    assert matchesKey("\x1b[47::91;5u", "ctrl+[") is False
    assert parseKey("\x1b[107::118;5u") == "ctrl+k"
    assert parseKey("\x1b[47::91;5u") == "ctrl+/"
