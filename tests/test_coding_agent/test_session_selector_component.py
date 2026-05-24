from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta

import pytest
from harnify_coding_agent.core.keybindings import KeybindingsManager
from harnify_coding_agent.core.session_manager import SessionInfo
from harnify_coding_agent.modes.interactive.components.session_selector import (
    SessionList,
    SessionSelectorComponent,
)
from harnify_coding_agent.modes.interactive.theme.theme import init_theme
from harnify_tui import setKeybindings

_ANSI_RE = re.compile(r"\x1b(?:\[[0-9;]*m|\]8;;.*?\x07)", re.DOTALL)


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def setup_function() -> None:
    setKeybindings(KeybindingsManager())
    init_theme("dark")


@pytest.mark.asyncio
async def test_session_selector_loads_current_scope_and_toggles_all_scope() -> None:
    now = datetime.now(UTC)
    current_session = SessionInfo(
        path="/tmp/current.jsonl",
        id="sess-current",
        cwd="/repo/current",
        created=now - timedelta(hours=4),
        modified=now - timedelta(minutes=5),
        messageCount=3,
        firstMessage="current work",
        allMessagesText="fix parser issue",
        name="Parser Fix",
    )
    all_session = SessionInfo(
        path="/tmp/other.jsonl",
        id="sess-other",
        cwd="/repo/other",
        created=now - timedelta(hours=6),
        modified=now - timedelta(minutes=30),
        messageCount=8,
        firstMessage="other work",
        allMessagesText="theme selector refactor",
        name="Theme Refactor",
    )

    progress: list[tuple[str, int, int]] = []
    render_calls: list[bool] = []
    selected: list[str] = []

    async def load_current(on_progress=None):
        if on_progress is not None:
            on_progress(1, 1)
        progress.append(("current", 1, 1))
        return [current_session]

    async def load_all(on_progress=None):
        if on_progress is not None:
            on_progress(1, 2)
            on_progress(2, 2)
        progress.extend([("all", 1, 2), ("all", 2, 2)])
        return [current_session, all_session]

    component = SessionSelectorComponent(
        load_current,
        load_all,
        selected.append,
        lambda: None,
        lambda: None,
        lambda: render_calls.append(True),
    )

    await asyncio.sleep(0)
    output = _strip_ansi("\n".join(component.render(140)))
    assert "Parser Fix" in output
    assert "Theme Refactor" not in output

    component.handleInput("\t")
    await asyncio.sleep(0)
    output = _strip_ansi("\n".join(component.render(140)))
    assert "Theme Refactor" in output
    assert "/repo/other" in output

    component.handleInput("\r")
    assert selected == [component.getSessionList().getSelectedSessionPath()]
    assert ("current", 1, 1) in progress
    assert ("all", 2, 2) in progress
    assert render_calls


@pytest.mark.asyncio
async def test_session_selector_rename_mode_and_current_delete_guard() -> None:
    now = datetime.now(UTC)
    sessions = [
        SessionInfo(
            path="/tmp/current.jsonl",
            id="sess-current",
            cwd="/repo/current",
            created=now - timedelta(hours=2),
            modified=now - timedelta(minutes=1),
            messageCount=2,
            firstMessage="current work",
            allMessagesText="active session",
            name="Current Session",
        ),
        SessionInfo(
            path="/tmp/other.jsonl",
            id="sess-other",
            cwd="/repo/current",
            created=now - timedelta(hours=3),
            modified=now - timedelta(minutes=10),
            messageCount=4,
            firstMessage="other work",
            allMessagesText="rename me",
            name="Other Session",
        ),
    ]
    renamed: list[tuple[str, str]] = []

    async def load_current(on_progress=None):
        if on_progress is not None:
            on_progress(1, len(sessions))
        return sessions

    async def rename_session(session_path: str, next_name: str) -> None:
        renamed.append((session_path, next_name))
        for session in sessions:
            if session.path == session_path:
                session.name = next_name

    component = SessionSelectorComponent(
        load_current,
        load_current,
        lambda _session_path: None,
        lambda: None,
        lambda: None,
        lambda: None,
        {"renameSession": rename_session},
        currentSessionFilePath="/tmp/current.jsonl",
    )

    await asyncio.sleep(0)

    component.handleInput("\x04")
    output = _strip_ansi("\n".join(component.render(140)))
    assert "Cannot delete the currently active session" in output

    component.handleInput("\x1b[B")
    component.handleInput("\x12")
    assert component.mode == "rename"

    component.renameInput.setValue("Renamed Session")
    component.renameInput.cursor = len(component.renameInput.getValue())
    component.handleInput("\r")
    await asyncio.sleep(0)

    assert renamed == [("/tmp/other.jsonl", "Renamed Session")]
    assert component.mode == "list"
    output = _strip_ansi("\n".join(component.render(140)))
    assert "Renamed Session" in output


@pytest.mark.asyncio
async def test_session_selector_status_message_auto_hides_and_discards_stale_all_scope_loads() -> None:
    now = datetime.now(UTC)
    current_session = SessionInfo(
        path="/tmp/current.jsonl",
        id="sess-current",
        cwd="/repo/current",
        created=now - timedelta(hours=1),
        modified=now - timedelta(minutes=1),
        messageCount=1,
        firstMessage="current",
        allMessagesText="current",
        name="Current",
    )
    stale_session = SessionInfo(
        path="/tmp/stale.jsonl",
        id="sess-stale",
        cwd="/repo/stale",
        created=now - timedelta(hours=2),
        modified=now - timedelta(minutes=5),
        messageCount=2,
        firstMessage="stale",
        allMessagesText="stale",
        name="Stale",
    )
    fresh_session = SessionInfo(
        path="/tmp/fresh.jsonl",
        id="sess-fresh",
        cwd="/repo/fresh",
        created=now - timedelta(hours=3),
        modified=now - timedelta(minutes=10),
        messageCount=3,
        firstMessage="fresh",
        allMessagesText="fresh",
        name="Fresh",
    )

    stale_gate = asyncio.Event()
    fresh_gate = asyncio.Event()
    render_calls: list[bool] = []
    all_calls = {"count": 0}

    async def load_current(on_progress=None):
        del on_progress
        return [current_session]

    async def load_all(on_progress=None):
        del on_progress
        all_calls["count"] += 1
        if all_calls["count"] == 1:
            await stale_gate.wait()
            return [stale_session]
        await fresh_gate.wait()
        return [fresh_session]

    component = SessionSelectorComponent(
        load_current,
        load_all,
        lambda _session_path: None,
        lambda: None,
        lambda: None,
        lambda: render_calls.append(True),
    )

    await asyncio.sleep(0)
    component.header.setStatusMessage(("info", "temporary"), 10)
    assert component.header.statusMessage == ("info", "temporary")
    await asyncio.sleep(0.02)
    assert component.header.statusMessage is None
    assert render_calls

    component.scope = "all"
    first = asyncio.create_task(component.loadScope("all", "toggle"))
    await asyncio.sleep(0)
    second = asyncio.create_task(component.loadScope("all", "refresh"))
    await asyncio.sleep(0)
    fresh_gate.set()
    await second
    stale_gate.set()
    await first

    output = _strip_ansi("\n".join(component.render(140)))
    assert "Fresh" in output
    assert "Stale" not in output


def test_session_list_ctrl_c_uses_exit_callback() -> None:
    now = datetime.now(UTC)
    session_list = SessionList(
        [
            SessionInfo(
                path="/tmp/demo.jsonl",
                id="sess-demo",
                cwd="/repo/demo",
                created=now - timedelta(hours=1),
                modified=now - timedelta(minutes=2),
                messageCount=1,
                firstMessage="demo",
                allMessagesText="demo",
                name=None,
            )
        ],
        False,
        "threaded",
        "all",
        KeybindingsManager(),
    )
    exit_calls: list[bool] = []
    session_list.onExit = lambda: exit_calls.append(True)

    session_list.handleInput("\x03")
    assert exit_calls == [True]
